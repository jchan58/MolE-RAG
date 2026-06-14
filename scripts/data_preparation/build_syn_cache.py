#!/usr/bin/env python
"""
build_synonym_cache.py — pre-compute LLM-filtered synonyms for every SMILES.

Reads:  ~/ChemLLM/GPM/synonyms_cache.json
          {smiles: {"cid": int, "iupac_name": str, "synonyms": [str, ...]}}

Writes: ~/ChemLLM/GPM/llm_filtered_synonyms.json
          {smiles: {"good_syns": [...paper-friendly names...],
                    "iupac_name": str}}

gpt-4o-mini keeps drug names, trivial names, and chemically descriptive names;
drops catalog/registry codes (NSC, CHEMBL, NCI60, ...) and CAS numbers.

Resumable, parallelized (16 concurrent calls), atomic checkpoints every 500.

Run:
  # small test first (~$0.0005, ~30 sec)
  python build_synonym_cache.py --limit 20

  # full pre-computation (typical: $3-5, ~1 hr)
  python build_synonym_cache.py
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Tuple

from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------------
# Local paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # scripts/data_preparation/ -> scripts/ -> MolE-RAG/

ENV_PATH    = REPO_ROOT / ".env"
INPUT_PATH  = REPO_ROOT / "caches" / "synonyms_cache.json"
OUTPUT_PATH = REPO_ROOT / "caches" / "llm_filtered_synonyms.json"

load_dotenv(str(ENV_PATH))

MODEL = "gpt-4o-mini"

PROMPT_TEMPLATE = """You are filtering chemical compound synonyms for a literature search.

Some names appear in research papers (e.g., "aspirin", "acetylsalicylic acid", "Mithramycin", "1,4-dioxane", trade/brand names, IUPAC-style descriptive names).

Others are catalog / registry / database identifiers that NEVER appear in papers:
  NSC641763, CHEMBL1992168, NCI60_014117, SCHEMBL11290222, AKOS006364538,
  DTXSID8025256, MLS002703041, SMR001566849, CCG-54069, BDBM115153, PD027570,
  HMS1670M09, SR-01000643191-1, Oprea1_312826, RefChem:193734, BSPBio_...
  CAS numbers like 5226-19-7.

Return ONLY the paper-friendly names from the candidates. Drop registry codes,
database IDs, supplier codes, CAS numbers, and lab notebook codes.

Return however many paper-friendly names you find — could be zero, could be many.
Order them by how commonly they appear in chemistry literature, most common first.

IUPAC name (reference only, do not include in output): {iupac}
Candidate synonyms: {syns}

Respond with JSON only, no prose: {{"keep": [...]}}. Empty list [] if none are paper-friendly."""


def filter_one(client: OpenAI, smiles: str, entry: Dict[str, Any]
               ) -> Tuple[str, Dict[str, Any]]:
    iupac = (entry.get("iupac_name") or "").strip()
    syns = entry.get("synonyms") or []

    if not syns:
        return smiles, {"good_syns": [], "iupac_name": iupac, "n_in": 0, "n_out": 0}

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You output JSON only. No prose, no markdown."},
                {"role": "user", "content": PROMPT_TEMPLATE.format(
                    iupac=iupac, syns=json.dumps(syns)
                )},
            ],
            max_tokens=800,
            temperature=0,
        )
        text = resp.choices[0].message.content.strip()
        for fence in ("```json", "```"):
            if text.startswith(fence):
                text = text[len(fence):].strip()
        if text.endswith("```"):
            text = text[:-3].strip()
        parsed = json.loads(text)
        keep = parsed.get("keep", [])
        if not isinstance(keep, list):
            keep = []
        keep = [str(x).strip() for x in keep if str(x).strip()]
        return smiles, {
            "good_syns": keep,
            "iupac_name": iupac,
            "n_in": len(syns),
            "n_out": len(keep),
        }
    except Exception as e:
        return smiles, {
            "good_syns": [],
            "iupac_name": iupac,
            "n_in": len(syns),
            "n_out": 0,
            "error": str(e)[:300],
        }


def save_atomic(out: Dict[str, Any], path: Path):
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default=str(INPUT_PATH))
    ap.add_argument("--output", default=str(OUTPUT_PATH))
    ap.add_argument("--workers", type=int, default=16,
                    help="parallel OpenAI calls")
    ap.add_argument("--checkpoint_every", type=int, default=500,
                    help="save to disk every N completions")
    ap.add_argument("--limit", type=int, default=None,
                    help="for testing: only filter this many SMILES")
    args = ap.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or api_key.startswith("sk-proj-REPLACE"):
        raise RuntimeError(f"OPENAI_API_KEY not set or still placeholder in {ENV_PATH}.")
    client = OpenAI(api_key=api_key)

    in_path = Path(args.input)
    out_path = Path(args.output)

    with open(in_path) as f:
        raw = json.load(f)
    print(f"Input  : {in_path}  ({len(raw)} SMILES)")

    out: Dict[str, Any] = {}
    if out_path.exists():
        with open(out_path) as f:
            out = json.load(f)
        print(f"Resume : {out_path}  ({len(out)} already done)")

    todo = {s: e for s, e in raw.items() if s not in out}
    if args.limit is not None:
        todo = dict(list(todo.items())[:args.limit])
        print(f"LIMIT applied: filtering {len(todo)} SMILES")
    else:
        print(f"To do  : {len(todo)} SMILES")

    if not todo:
        print("Nothing to do.")
        return

    t0 = time.time()
    done = 0
    errors = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(filter_one, client, s, e): s for s, e in todo.items()}
        for fut in as_completed(futures):
            smi, result = fut.result()
            out[smi] = result
            done += 1
            if "error" in result:
                errors += 1
            if done % args.checkpoint_every == 0:
                save_atomic(out, out_path)
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (len(todo) - done) / rate / 60
                print(f"  {done:>6}/{len(todo)}  rate={rate:5.1f}/s  "
                      f"errors={errors}  ETA {eta:5.1f} min")

    save_atomic(out, out_path)
    elapsed = time.time() - t0
    print(f"\nDone. Wrote {len(out)} entries to {out_path}")
    print(f"  total time : {elapsed/60:.1f} min")
    print(f"  errors     : {errors}")

    # quick sanity stats
    no_syns = 0
    keep_counts = []
    for r in out.values():
        if r.get("n_in", 0) == 0:
            no_syns += 1
        else:
            keep_counts.append(len(r.get("good_syns", [])))
    print(f"  no raw syns           : {no_syns}")
    if keep_counts:
        keep_counts.sort()
        n = len(keep_counts)
        avg = sum(keep_counts) / n
        median = keep_counts[n // 2]
        p90 = keep_counts[int(n * 0.9)]
        mx = keep_counts[-1]
        zero = sum(1 for c in keep_counts if c == 0)
        print(f"  filtered to 0 names   : {zero}")
        print(f"  good_syns count avg/median/p90/max : {avg:.1f} / {median} / {p90} / {mx}")


if __name__ == "__main__":
    main()
