"""
fetch_missing_synonyms.py

Fetch synonym data from PubChem for SMILES that are missing from your synonym cache.

Reads:
  ~/ChemLLM/GPM/synonyms_cache.json (existing raw cache, canonical SMILES keys)
  data/moleculenet_property_scaffold/seed_{0,1,2}/*/{task}_test.csv

Determines missing canonical SMILES (test molecules across all seeds, minus
those already in cache), fetches each from PubChem via REST API.

Writes (appends to existing file, atomic):
  ~/ChemLLM/GPM/synonyms_cache.json

Output format matches your existing pipeline:
  {canonical_smiles: {"cid": int, "iupac_name": str, "synonyms": [str, ...]}}

Resumable: writes to disk every 500 successful fetches.

After this finishes, run your existing build_synonym_cache.py to LLM-filter
the new entries:
    python build_synonym_cache.py

Run:
    python fetch_missing_synonyms.py             # full fetch
    python fetch_missing_synonyms.py --limit 50  # smoke test
"""

import argparse
import json
import os
import sys
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")


REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # scripts/data_preparation/ -> scripts/ -> MolE-RAG/
SCAFFOLD_DIR = REPO_ROOT / "data" / "moleculenet_property_scaffold"
RAW_CACHE_PATH = REPO_ROOT / "caches" / "synonyms_cache.json"

DATASETS = ["bbbp", "bace", "hiv", "tox21", "sider", "toxcast",
            "clintox", "esol", "freesolv", "lipo"]
SEEDS = [0, 1, 2]

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
TIMEOUT = 15
MAX_RETRIES = 3


def canon(smi: str) -> str:
    if not smi:
        return smi
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol else smi


def pubchem_smiles_to_cid_iupac(session: requests.Session, smiles: str) -> Optional[Tuple[int, str]]:
    """Get PubChem CID + IUPAC name for a SMILES. Returns (cid, iupac) or None."""
    url = f"{PUBCHEM_BASE}/compound/smiles/property/IUPACName/JSON"
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.post(url, data={"smiles": smiles}, timeout=TIMEOUT)
            if resp.status_code == 404:
                return None
            if resp.status_code == 503:
                time.sleep(2 ** attempt)
                continue
            if resp.status_code != 200:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(1)
                    continue
                return None
            data = resp.json()
            props = data.get("PropertyTable", {}).get("Properties", [])
            if not props:
                return None
            prop = props[0]
            cid = prop.get("CID")
            iupac = (prop.get("IUPACName") or "").strip()
            return (cid, iupac) if cid else None
        except (requests.RequestException, json.JSONDecodeError):
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
                continue
            return None
    return None


def pubchem_cid_to_synonyms(session: requests.Session, cid: int) -> List[str]:
    """Get synonyms list for a CID. Returns [] on any failure."""
    url = f"{PUBCHEM_BASE}/compound/cid/{cid}/synonyms/JSON"
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(url, timeout=TIMEOUT)
            if resp.status_code == 404:
                return []
            if resp.status_code == 503:
                time.sleep(2 ** attempt)
                continue
            if resp.status_code != 200:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(1)
                    continue
                return []
            data = resp.json()
            info = data.get("InformationList", {}).get("Information", [])
            if not info:
                return []
            return [str(s).strip() for s in info[0].get("Synonym", []) if str(s).strip()]
        except (requests.RequestException, json.JSONDecodeError):
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
                continue
            return []
    return []


def fetch_one(session: requests.Session, smiles: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Fetch full entry for one SMILES. Returns (smiles, entry) or (smiles, None)."""
    res = pubchem_smiles_to_cid_iupac(session, smiles)
    if res is None:
        # Not found in PubChem - store empty entry so we don't retry next run
        return smiles, {"cid": None, "iupac_name": "", "synonyms": []}
    cid, iupac = res
    syns = pubchem_cid_to_synonyms(session, cid)
    return smiles, {"cid": cid, "iupac_name": iupac, "synonyms": syns}


def gather_test_smiles_union() -> List[str]:
    """Union of canonical test SMILES across all 3 seeds, all datasets."""
    print("Gathering test SMILES across all seeds...")
    all_canon = set()
    for seed in SEEDS:
        for ds in DATASETS:
            csv_path = SCAFFOLD_DIR / f"seed_{seed}" / ds / f"{ds}_test.csv"
            if not csv_path.exists():
                print(f"  WARN: missing {csv_path}")
                continue
            df = pd.read_csv(csv_path)
            for s in df["smiles"].dropna().unique():
                all_canon.add(canon(s))
    print(f"  Union: {len(all_canon)} unique canonical test SMILES")
    return sorted(all_canon)


def save_atomic(data: Dict[str, Any], path: Path):
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4,
                    help="Parallel PubChem workers (rate limit is 5 req/sec)")
    ap.add_argument("--checkpoint_every", type=int, default=500)
    ap.add_argument("--limit", type=int, default=None,
                    help="Stop after fetching this many SMILES (for testing)")
    ap.add_argument("--cache_path", default=str(RAW_CACHE_PATH))
    args = ap.parse_args()

    cache_path = Path(args.cache_path)

    # Load existing cache (assume already canonical after canonicalize_synonyms.py)
    if cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)
        print(f"Loaded existing cache: {len(cache)} entries from {cache_path}")
    else:
        cache = {}
        print(f"No existing cache, starting fresh at {cache_path}")

    # Get target SMILES list
    target_smiles = gather_test_smiles_union()
    missing = [s for s in target_smiles if s not in cache]
    print(f"Already in cache: {len(target_smiles) - len(missing)}")
    print(f"Need to fetch:    {len(missing)}")

    if args.limit is not None:
        missing = missing[: args.limit]
        print(f"LIMIT applied: only fetching {len(missing)}")

    if not missing:
        print("Nothing to do.")
        return

    print(f"\nFetching with {args.workers} workers...")
    print(f"Estimated time: {len(missing) * 0.5 / args.workers / 60:.1f} min")

    session = requests.Session()
    session.headers.update({"User-Agent": "MCRAG-research/1.0"})

    save_lock = Lock()
    t0 = time.time()
    done = 0
    n_with_syns = 0
    n_not_found = 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(fetch_one, session, smi): smi for smi in missing}
        for fut in as_completed(futures):
            smi, entry = fut.result()
            if entry is not None:
                cache[smi] = entry
                if entry.get("synonyms"):
                    n_with_syns += 1
                if entry.get("cid") is None:
                    n_not_found += 1
            done += 1

            if done % args.checkpoint_every == 0:
                with save_lock:
                    save_atomic(cache, cache_path)
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (len(missing) - done) / rate / 60
                print(f"  {done:>6}/{len(missing)}  "
                      f"rate={rate:5.2f}/s  "
                      f"with_syns={n_with_syns}  "
                      f"not_found={n_not_found}  "
                      f"ETA {eta:5.1f} min")

    save_atomic(cache, cache_path)
    elapsed = time.time() - t0
    print(f"\nDone. Wrote {len(cache)} total entries to {cache_path}")
    print(f"  fetched          : {done}")
    print(f"  with synonyms    : {n_with_syns}")
    print(f"  not found in PC  : {n_not_found}")
    print(f"  time             : {elapsed/60:.1f} min")
    print(f"\nNext step: run your existing LLM filter:")
    print(f"  python build_synonym_cache.py")


if __name__ == "__main__":
    main()
