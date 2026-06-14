"""
rebuild_fewshot_pools.py

Regenerate fewshot_global_structural_top5.csv for each (task, seed)
using the new scaffold-split train/test CSVs.

For each (dataset, seed):
  1. Load new train and test CSVs from data/moleculenet_property_scaffold/seed_{N}/{dataset}/
  2. Compute the dataset's "best" fingerprint (from fingerprint_sweep results)
     for all train and test molecules
  3. For each (test_smiles, test_task) row, find the top-5 nearest train molecules
     that have a non-null label for that task, sorted by Tanimoto similarity
  4. Write fewshot_global_structural_top5.csv next to the train/test CSVs

Output CSV format (matches the existing format):
  source_row_index, fewshot_rank, similarity_score, retrieval_method,
  selected_global_method, selected_global_top_k, test_task, test_smiles,
  dataset, task, task_type, metric, smiles, label, split, _train_original_index

Usage:
  python rebuild_fewshot_pools.py
"""

# %% Imports
import os
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")


# %% Configuration

REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # scripts/data_preparation/ -> scripts/ -> MolE-RAG/
SCAFFOLD_DIR = REPO_ROOT / "data" / "moleculenet_property_scaffold"
SEEDS = [0, 1, 2]
TOP_K = 5

# Best fingerprint per dataset (from your fingerprint_sweep results)
DATASET_BEST_FP = {
    "bbbp":     "ecfp2",
    "bace":     "topological_torsion",
    "hiv":      "rdkit",
    "tox21":    "atom_pair",
    "sider":    "fcfp2",
    "toxcast":  "atom_pair",
    "clintox":  "fcfp2",
    "esol":     "atom_pair",
    "freesolv": "maccs",
    "lipo":     "atom_pair",
}

# Per-dataset task type / metric (for output CSV columns)
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


# %% Fingerprint computation

def compute_fp(mol, method: str):
    """Compute fingerprint for a single molecule using specified method."""
    if mol is None:
        return None
    if method == "ecfp2":
        return AllChem.GetMorganFingerprintAsBitVect(mol, 1, nBits=2048)
    if method == "ecfp4":
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    if method == "ecfp6":
        return AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048)
    if method == "fcfp2":
        return AllChem.GetMorganFingerprintAsBitVect(mol, 1, nBits=2048, useFeatures=True)
    if method == "fcfp4":
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, useFeatures=True)
    if method == "fcfp6":
        return AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048, useFeatures=True)
    if method == "maccs":
        return MACCSkeys.GenMACCSKeys(mol)
    if method == "rdkit":
        return Chem.RDKFingerprint(mol)
    if method == "topological_torsion":
        return rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=2048)
    if method == "atom_pair":
        return rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=2048)
    raise ValueError(f"Unknown fingerprint method: {method}")


def fps_for_smiles_list(smiles_list: List[str], method: str) -> Tuple[List, List[bool]]:
    """Compute fingerprints for a list of SMILES. Returns (fps, valid_mask)."""
    fps = []
    valid = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(None)
            valid.append(False)
            continue
        fp = compute_fp(mol, method)
        fps.append(fp)
        valid.append(fp is not None)
    return fps, valid


# %% Core processing

def process_dataset_seed(dataset_name: str, seed: int) -> bool:
    """Build fewshot CSV for one (dataset, seed). Returns True on success."""
    cfg = DATASETS[dataset_name]
    method = DATASET_BEST_FP[dataset_name]

    seed_dir = SCAFFOLD_DIR / f"seed_{seed}" / dataset_name
    train_path = seed_dir / f"{dataset_name}_train.csv"
    test_path = seed_dir / f"{dataset_name}_test.csv"
    out_path = seed_dir / "fewshot_global_structural_top5.csv"

    if not train_path.exists():
        print(f"  [SKIP] Missing train CSV: {train_path}")
        return False
    if not test_path.exists():
        print(f"  [SKIP] Missing test CSV: {test_path}")
        return False

    t_start = time.time()
    print(f"\n[{dataset_name} seed={seed}]  method={method}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"  train: {len(train_df)} long-format rows, {train_df['smiles'].nunique()} unique molecules, "
          f"{train_df['task'].nunique()} task(s)")
    print(f"  test : {len(test_df)} long-format rows, {test_df['smiles'].nunique()} unique molecules")

    # ---- Compute train fingerprints (once per unique SMILES) ----
    unique_train_smiles = train_df["smiles"].drop_duplicates().tolist()
    smiles_to_train_idx = {s: i for i, s in enumerate(unique_train_smiles)}

    t0 = time.time()
    print(f"  Computing {len(unique_train_smiles)} train fingerprints...")
    train_fps, train_valid_mask = fps_for_smiles_list(unique_train_smiles, method)
    n_valid_train = sum(train_valid_mask)
    print(f"    {n_valid_train}/{len(unique_train_smiles)} valid ({time.time()-t0:.1f}s)")

    # Precompute the list of valid train fps and their original indices for BulkTanimoto
    valid_train_indices = [i for i, v in enumerate(train_valid_mask) if v]
    valid_train_fps_list = [train_fps[i] for i in valid_train_indices]

    # ---- Build per-task candidate index ----
    # task_candidates[task] = list of (train_unique_idx, label) for molecules
    # that have a non-null label for this task
    print(f"  Building per-task candidate index...")
    task_candidates: Dict[str, List[Tuple[int, float]]] = {}
    for task_name, group in train_df.groupby("task", sort=False):
        seen_train_idx = set()
        candidates = []
        for _, row in group.iterrows():
            smi = row["smiles"]
            if smi not in smiles_to_train_idx:
                continue
            ti = smiles_to_train_idx[smi]
            if not train_valid_mask[ti]:
                continue
            if ti in seen_train_idx:
                continue
            candidates.append((ti, row["label"]))
            seen_train_idx.add(ti)
        task_candidates[task_name] = candidates
    print(f"    {len(task_candidates)} task(s), avg {np.mean([len(v) for v in task_candidates.values()]):.0f} candidates/task")

    # ---- Build mapping (train_smiles, train_task) -> original row index ----
    # for the _train_original_index output column
    train_long_idx: Dict[Tuple[str, str], int] = {}
    for orig_idx, row in train_df.iterrows():
        key = (row["smiles"], row["task"])
        # Keep first occurrence if duplicates
        if key not in train_long_idx:
            train_long_idx[key] = orig_idx

    # ---- Compute test fingerprints (once per unique SMILES) ----
    unique_test_smiles = test_df["smiles"].drop_duplicates().tolist()
    t0 = time.time()
    print(f"  Computing {len(unique_test_smiles)} test fingerprints...")
    test_fps, test_valid_mask = fps_for_smiles_list(unique_test_smiles, method)
    test_smi_to_fp = {
        s: fp for s, fp, v in zip(unique_test_smiles, test_fps, test_valid_mask) if v
    }
    print(f"    {len(test_smi_to_fp)}/{len(unique_test_smiles)} valid ({time.time()-t0:.1f}s)")

    # ---- For each unique test SMILES: compute sims once, then process all rows ----
    print(f"  Computing similarities and top-{TOP_K} retrieval...")
    output_rows = []
    test_grouped = test_df.groupby("smiles", sort=False)

    n_done = 0
    t_sim = time.time()
    for test_smi, group in test_grouped:
        test_fp = test_smi_to_fp.get(test_smi)
        if test_fp is None:
            continue

        # One BulkTanimoto call per unique test molecule against all valid train fps
        sims = DataStructs.BulkTanimotoSimilarity(test_fp, valid_train_fps_list)
        # Map: train_unique_idx (in full unique_train_smiles list) -> similarity
        sim_lookup: Dict[int, float] = {
            valid_train_indices[i]: sims[i] for i in range(len(sims))
        }

        for source_row_index, test_row in group.iterrows():
            test_task = test_row["task"]
            candidates = task_candidates.get(test_task, [])
            if not candidates:
                continue

            # Score candidates and pick top-K
            scored = [
                (ti, label, sim_lookup.get(ti, 0.0))
                for ti, label in candidates
            ]
            scored.sort(key=lambda x: -x[2])
            top_k = scored[:TOP_K]

            for rank, (train_idx, label, sim) in enumerate(top_k, start=1):
                train_smi = unique_train_smiles[train_idx]
                train_orig_idx = train_long_idx.get((train_smi, test_task), -1)
                output_rows.append({
                    "source_row_index": source_row_index,
                    "fewshot_rank": rank,
                    "similarity_score": sim,
                    "retrieval_method": method,
                    "selected_global_method": method,
                    "selected_global_top_k": TOP_K,
                    "test_task": test_task,
                    "test_smiles": test_smi,
                    "dataset": dataset_name,
                    "task": test_task,
                    "task_type": cfg["task_type"],
                    "metric": cfg["metric"],
                    "smiles": train_smi,
                    "label": label,
                    "split": "train",
                    "_train_original_index": train_orig_idx,
                })

        n_done += 1
        if n_done % 100 == 0:
            elapsed = time.time() - t_sim
            rate = n_done / elapsed if elapsed > 0 else 0
            print(f"    Processed {n_done}/{len(unique_test_smiles)} unique test SMILES "
                  f"({rate:.1f}/sec)")

    out_df = pd.DataFrame(output_rows)
    out_df.to_csv(out_path, index=False)
    print(f"  Saved {len(output_rows)} rows -> {out_path}")
    print(f"  Total time: {time.time()-t_start:.1f}s")
    return True


# %% Main

def main():
    print("=" * 70)
    print("Rebuild structural fewshot pools")
    print("=" * 70)
    print(f"Scaffold dir : {SCAFFOLD_DIR}")
    print(f"Seeds        : {SEEDS}")
    print(f"Top-K        : {TOP_K}")
    print(f"Best fingerprint per dataset:")
    for ds, fp in DATASET_BEST_FP.items():
        print(f"    {ds:10s} -> {fp}")
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
        print(f"Failures:")
        for ds, seed in failed:
            print(f"    {ds} seed={seed}")
    print("\nDone.")


if __name__ == "__main__":
    main()