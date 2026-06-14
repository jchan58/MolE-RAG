#!/usr/bin/env python
"""
compute_feature_correlation.py — rank RDKit descriptors per task by correlation.

For each MoleculeNet task (and sub-task for multi-task datasets):
  1. Load <task>_train.csv and <task>_valid.csv → compute all RDKit descriptors
  2. Compute correlation between each descriptor and the label:
       - Classification: point-biserial (Pearson between continuous & binary)
       - Regression: Pearson
  3. Rank features by |correlation|, also store p-values
  4. Save top-K per task to caches/task_rdkit_features.json

Why correlation (vs SHAP/RF):
  - 100x faster (no model fitting)
  - Descriptive statistic, no leakage concerns
  - Easy to interpret: sign tells you direction of effect
  - Defensible as a first-pass feature ranking method

Usage:
  python compute_feature_correlation.py                  # all tasks
  python compute_feature_correlation.py --tasks bbbp bace
  python compute_feature_correlation.py --top_k 20       # default 15
  python compute_feature_correlation.py --overwrite      # rebuild cache

Cache: caches/task_rdkit_features.json
  {
    "bbbp": {
      "task_type": "classification",
      "method": "pointbiserial",
      "n_total": 1835,
      "top_features": [
        {"name": "MolLogP", "score": 0.412, "abs_score": 0.412,
         "p_value": 1.2e-78, "rank": 1},
        ...
      ]
    },
    ...
  }
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Paths (resolved relative to repo root)
# ---------------------------------------------------------------------------
REPO_ROOT     = Path(__file__).resolve().parent.parent.parent  # src/subdir/ -> src/ -> MolE-RAG/
FEATURE_CACHE = REPO_ROOT / "caches" / "task_rdkit_features.json"

DATASET_TASK_TYPE = {
    "bbbp":     "classification",
    "bace":     "classification",
    "clintox":  "classification",
    "hiv":      "classification",
    "tox21":    "classification",
    "sider":    "classification",
    "toxcast":  "classification",
    "esol":     "regression",
    "lipo":     "regression",
    "freesolv": "regression",
}


# =========================
# RDKit descriptor computation
# =========================
def get_rdkit_descriptor_list():
    """Return list of (name, function) for all RDKit Descriptors.descList."""
    from rdkit.Chem import Descriptors
    return [(name, fn) for name, fn in Descriptors.descList]


def compute_descriptors_for_smiles(smiles_list: List[str]
                                   ) -> Tuple[np.ndarray, List[str], List[int]]:
    """Compute all RDKit descriptors for a list of SMILES.
    Returns (matrix [n_valid, n_features], feature_names, valid_indices)."""
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")  # suppress RDKit warnings

    desc_list = get_rdkit_descriptor_list()
    feature_names = [name for name, _ in desc_list]
    feature_fns = [fn for _, fn in desc_list]

    rows = []
    valid_idx = []
    for i, smi in enumerate(smiles_list):
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            vals = [fn(mol) for fn in feature_fns]
        except Exception:
            continue
        # Replace NaN/inf with 0 (some descriptors fail on edge-case mols)
        vals = [v if (isinstance(v, (int, float)) and math.isfinite(v)) else 0.0
                for v in vals]
        rows.append(vals)
        valid_idx.append(i)

    if not rows:
        return np.zeros((0, len(feature_names))), feature_names, []
    return np.array(rows, dtype=np.float64), feature_names, valid_idx


# =========================
# CSV loading + label extraction
# =========================
def load_split_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def valid_label_str(value: Any) -> bool:
    if value is None: return False
    s = str(value).strip().lower()
    return s not in {"", "nan", "none", "null"}


def extract_smiles_labels_tasks(rows: List[Dict[str, Any]]
                                ) -> List[Tuple[str, Optional[str], str]]:
    """Return list of (smiles, sub_task_or_None, label_str) tuples with non-null labels."""
    out = []
    for r in rows:
        smi = (r.get("smiles") or "").strip()
        lab = r.get("label")
        if not smi or not valid_label_str(lab):
            continue
        task = (r.get("task") or "").strip() or None
        out.append((smi, task, str(lab).strip()))
    return out


# =========================
# Correlation per task
# =========================
def correlation_rank(X: np.ndarray, y: np.ndarray,
                     feature_names: List[str],
                     task_type: str,
                     top_k: int = 15) -> Dict[str, Any]:
    """Compute correlation between each feature column and label y.

    For binary classification: point-biserial (which is mathematically
    identical to Pearson between a continuous and a {0,1}-valued variable).
    For regression: Pearson.

    Returns dict with top_features (sorted by |correlation| desc).
    """
    n_features = X.shape[1]
    scores = np.zeros(n_features)
    p_values = np.zeros(n_features)

    # Use scipy for proper p-values
    from scipy.stats import pearsonr, pointbiserialr
    corr_fn = pointbiserialr if task_type == "classification" else pearsonr

    for j in range(n_features):
        col = X[:, j]
        # Skip features with zero variance (constant across all molecules)
        if np.std(col) < 1e-12:
            scores[j] = 0.0
            p_values[j] = 1.0
            continue
        try:
            if task_type == "classification":
                # point-biserial expects (continuous, binary)
                r, p = corr_fn(col, y)
            else:
                r, p = corr_fn(col, y)
            scores[j] = r if math.isfinite(r) else 0.0
            p_values[j] = p if math.isfinite(p) else 1.0
        except Exception:
            scores[j] = 0.0
            p_values[j] = 1.0

    # Rank by absolute correlation
    abs_scores = np.abs(scores)
    ranking = np.argsort(-abs_scores)[:top_k]
    top_features = []
    for r, idx in enumerate(ranking, start=1):
        if abs_scores[idx] < 1e-9:
            break  # don't list zero-correlation features
        top_features.append({
            "name": feature_names[idx],
            "score": float(scores[idx]),
            "abs_score": float(abs_scores[idx]),
            "p_value": float(p_values[idx]),
            "rank": r,
        })
    return {"top_features": top_features}


# =========================
# Per-task entry point
# =========================
def process_task(dataset_name: str, top_k: int) -> Dict[str, Any]:
    """Compute feature correlation for one MoleculeNet task. For multi-task
    datasets (tox21/sider/toxcast) compute per sub-task and store under
    f'{dataset}::{sub_task}' keys.
    """
    base = REPO_ROOT / "data" / dataset_name
    train_path = base / f"{dataset_name}_train.csv"
    valid_path = base / f"{dataset_name}_valid.csv"
    if not train_path.exists():
        print(f"  [{dataset_name}] no train CSV at {train_path}, skipping")
        return {}

    task_type = DATASET_TASK_TYPE.get(dataset_name, "classification")
    train_rows = load_split_csv(train_path)
    valid_rows = load_split_csv(valid_path)

    # Combine train+valid for correlation (no leakage — correlation is
    # a descriptive statistic, not a fitted model). More data = more
    # stable correlation estimates.
    all_data = extract_smiles_labels_tasks(train_rows + valid_rows)
    if not all_data:
        print(f"  [{dataset_name}] empty after filtering")
        return {}

    # Group by sub-task
    by_task: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for smi, sub_task, lab in all_data:
        key = sub_task or dataset_name
        by_task[key].append((smi, lab))

    results: Dict[str, Any] = {}
    sub_tasks = sorted(by_task.keys())
    print(f"  [{dataset_name}] {len(sub_tasks)} sub-task(s) to process")

    for st_idx, st in enumerate(sub_tasks, start=1):
        data = by_task[st]
        print(f"    [{st_idx}/{len(sub_tasks)}] sub-task '{st}' "
              f"(n={len(data)})...", flush=True)
        if len(data) < 30:
            print(f"      too few rows ({len(data)}), skip")
            continue

        smiles_list = [s for s, _ in data]
        labels_raw = [lab for _, lab in data]

        # Compute descriptors
        print(f"      computing RDKit descriptors for {len(smiles_list)} molecules...",
              flush=True)
        X, feature_names, valid_idx = compute_descriptors_for_smiles(smiles_list)
        y_raw = [labels_raw[i] for i in valid_idx]

        # Parse labels
        try:
            if task_type == "classification":
                y = np.array([int(round(float(v))) for v in y_raw])
            else:
                y = np.array([float(v) for v in y_raw])
        except ValueError:
            print(f"      label parse error, skip")
            continue

        if X.shape[0] < 30:
            print(f"      too few valid molecules after descriptor compute, skip")
            continue
        if task_type == "classification" and len(set(y.tolist())) < 2:
            print(f"      only one class, skip")
            continue

        # Compute correlations
        print(f"      computing correlations across "
              f"{X.shape[1]} features...", flush=True)
        try:
            rank_result = correlation_rank(X, y, feature_names, task_type, top_k)
        except Exception as e:
            print(f"      correlation failed: {e}")
            continue

        cache_key = (dataset_name if st == dataset_name
                     else f"{dataset_name}::{st}")
        results[cache_key] = {
            **rank_result,
            "task_type": task_type,
            "method": ("pointbiserial" if task_type == "classification"
                       else "pearson"),
            "n_total": int(X.shape[0]),
        }
        if rank_result["top_features"]:
            top_3 = ", ".join(
                f"{f['name']}({f['score']:+.3f})"
                for f in rank_result["top_features"][:3]
            )
            print(f"      done. Top-3: {top_3}", flush=True)
        else:
            print(f"      done. No features above threshold.", flush=True)
    return results


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=list(DATASET_TASK_TYPE.keys()))
    ap.add_argument("--top_k", type=int, default=15,
                    help="number of top features to keep per task (default 15)")
    ap.add_argument("--overwrite", action="store_true",
                    help="recompute even if task is already in the cache")
    ap.add_argument("--out", default=str(FEATURE_CACHE))
    args = ap.parse_args()

    cache_path = Path(args.out)
    if cache_path.exists() and not args.overwrite:
        with open(cache_path) as f:
            cache = json.load(f)
        print(f"Loaded existing feature cache: {len(cache)} entries")
    else:
        cache = {}
        if cache_path.exists():
            print(f"--overwrite set, rebuilding cache from scratch")

    for dataset_name in args.tasks:
        print(f"\n=== {dataset_name} ===")
        if not args.overwrite:
            prefix_keys = [k for k in cache.keys()
                           if k == dataset_name or k.startswith(f"{dataset_name}::")]
            if prefix_keys:
                print(f"  [{dataset_name}] already cached ({len(prefix_keys)} keys), "
                      f"skipping. Use --overwrite to rebuild.")
                continue

        results = process_task(dataset_name, args.top_k)
        cache.update(results)
        # Save after each dataset (resilient to interruption)
        tmp = cache_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(cache, f, indent=2)
        os.replace(tmp, cache_path)
        print(f"  Saved {len(results)} keys → {cache_path}")

    print(f"\nFinal cache size: {len(cache)} entries")
    print(f"Cache file: {cache_path}")


if __name__ == "__main__":
    main()
