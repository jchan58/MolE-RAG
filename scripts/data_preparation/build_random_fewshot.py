"""
build_random_fewshot_pools.py

Generate fewshot_random_top5.csv for each (dataset, seed) — per-test-molecule
random 5 train examples in the SAME format as fewshot_global_structural_top5.csv.

This is the "random baseline" version of rebuild_fewshot_pools.py: instead of
choosing the top-5 most structurally similar training molecules, we choose 5
uniformly at random from the training pool that have a label for the target task.

Selections are deterministic via a hash of (dataset, scaffold_seed, test_smiles,
test_task), so re-runs are reproducible.

Output columns (match fewshot_global_structural_top5.csv exactly):
    source_row_index, fewshot_rank, similarity_score, retrieval_method,
    selected_global_method, selected_global_top_k, test_task, test_smiles,
    dataset, task, task_type, metric, smiles, label, split,
    _train_original_index

Run:
    python scripts/data_preparation/build_random_fewshot.py
"""

# %% Imports
import hashlib
import random
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# %% Configuration
REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # scripts/data_preparation/ -> scripts/ -> MolE-RAG/
SCAFFOLD_DIR = REPO_ROOT / "data" / "moleculenet_property_scaffold"
SEEDS = [0, 1, 2]
TOP_K = 5

# Per-dataset metadata (matches rebuild_fewshot_pools.py)
DATASETS = {
    "bbbp":     {"task_type": "classification", "metric": "macro_f1"},
    "bace":     {"task_type": "classification", "metric": "macro_f1"},
    "hiv":      {"task_type": "classification", "metric": "macro_f1"},
    "tox21":    {"task_type": "classification", "metric": "macro_f1"},
    "toxcast":  {"task_type": "classification", "metric": "macro_f1"},
    "sider":    {"task_type": "classification", "metric": "macro_f1"},
    "clintox":  {"task_type": "classification", "metric": "macro_f1"},
    "esol":     {"task_type": "regression",     "metric": "mae"},
    "freesolv": {"task_type": "regression",     "metric": "mae"},
    "lipo":     {"task_type": "regression",     "metric": "mae"},
}


# %% Helpers
def get_rng(dataset: str, scaffold_seed: int,
            test_smiles: str, test_task: str) -> random.Random:
    """Return a deterministic Random instance for this (test_smi, test_task) cell."""
    key = f"{dataset}::seed{scaffold_seed}::{test_smiles}::{test_task}"
    h = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:16], 16)
    return random.Random(h)


# %% Core processing
def process_dataset_seed(dataset_name: str, seed: int) -> bool:
    """Build random fewshot CSV for one (dataset, seed). Returns True on success."""
    cfg = DATASETS[dataset_name]

    seed_dir = SCAFFOLD_DIR / f"seed_{seed}" / dataset_name
    train_path = seed_dir / f"{dataset_name}_train.csv"
    test_path = seed_dir / f"{dataset_name}_test.csv"
    out_path = seed_dir / "fewshot_random_top5.csv"

    if not train_path.exists():
        print(f"  [SKIP] Missing train CSV: {train_path}")
        return False
    if not test_path.exists():
        print(f"  [SKIP] Missing test CSV: {test_path}")
        return False

    t_start = time.time()
    print(f"\n[{dataset_name} seed={seed}]")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"  train: {len(train_df)} long-format rows, {train_df['task'].nunique()} task(s)")
    print(f"  test : {len(test_df)} long-format rows, {test_df['smiles'].nunique()} unique molecules")

    # Build per-task candidate pool: {task: [(train_long_row_idx, train_smiles, label), ...]}
    # Deduplicate by smiles within each task (keep first occurrence's label)
    print(f"  Building per-task candidate index...")
    task_candidates: Dict[str, List[Tuple[int, str, float]]] = {}
    for task_name, group in train_df.groupby("task", sort=False):
        seen_smi = set()
        candidates = []
        for orig_idx, row in group.iterrows():
            smi = row["smiles"]
            if smi in seen_smi:
                continue
            seen_smi.add(smi)
            candidates.append((orig_idx, smi, row["label"]))
        task_candidates[task_name] = candidates
    if task_candidates:
        avg_n = np.mean([len(v) for v in task_candidates.values()])
        print(f"    {len(task_candidates)} task(s), avg {avg_n:.0f} candidates/task")

    # For each test row, pick TOP_K random candidates from this task's pool
    print(f"  Sampling top-{TOP_K} random neighbors per test row...")
    output_rows = []
    n_skipped_no_candidates = 0

    for source_row_index, test_row in test_df.iterrows():
        test_smi = test_row["smiles"]
        test_task = test_row["task"]

        candidates = task_candidates.get(test_task, [])
        if not candidates:
            n_skipped_no_candidates += 1
            continue

        rng = get_rng(dataset_name, seed, test_smi, test_task)
        n_to_sample = min(TOP_K, len(candidates))
        sampled = rng.sample(candidates, k=n_to_sample)

        for rank, (train_orig_idx, train_smi, label) in enumerate(sampled, start=1):
            output_rows.append({
                "source_row_index":      source_row_index,
                "fewshot_rank":          rank,
                "similarity_score":      0.0,         # not applicable for random
                "retrieval_method":      "random",
                "selected_global_method": "random",
                "selected_global_top_k": TOP_K,
                "test_task":             test_task,
                "test_smiles":           test_smi,
                "dataset":               dataset_name,
                "task":                  test_task,
                "task_type":             cfg["task_type"],
                "metric":                cfg["metric"],
                "smiles":                train_smi,
                "label":                 label,
                "split":                 "train",
                "_train_original_index": train_orig_idx,
            })

    out_df = pd.DataFrame(output_rows)
    out_df.to_csv(out_path, index=False)
    print(f"  Saved {len(output_rows)} rows -> {out_path}")
    if n_skipped_no_candidates > 0:
        print(f"  [WARN] {n_skipped_no_candidates} test rows had no candidates for their task")
    print(f"  Time: {time.time()-t_start:.1f}s")
    return True


# %% Main
def main():
    print("=" * 70)
    print("Build random fewshot pools")
    print("=" * 70)
    print(f"Scaffold dir : {SCAFFOLD_DIR}")
    print(f"Seeds        : {SEEDS}")
    print(f"Top-K        : {TOP_K}")
    print(f"Datasets     : {list(DATASETS.keys())}")
    print("=" * 70)

    succeeded = []
    failed = []
    for seed in SEEDS:
        for dataset_name in DATASETS:
            try:
                ok = process_dataset_seed(dataset_name, seed)
                if ok:
                    succeeded.append((dataset_name, seed))
            except Exception as e:
                print(f"  [ERROR] {dataset_name} seed={seed}: {e}")
                traceback.print_exc()
                failed.append((dataset_name, seed))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Succeeded: {len(succeeded)}")
    print(f"Failed:    {len(failed)}")
    if failed:
        for ds, seed in failed:
            print(f"    {ds} seed={seed}")
    print("\nDone.")


if __name__ == "__main__":
    main()