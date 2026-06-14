"""
make_scaffold_splits.py

Generate scaffold_balanced 80/10/10 splits for 10 MoleculeNet tasks
across 3 seeds (0, 1, 2) for variance estimation.

Data source: KANO's bundled CSVs (set KANO_DATA_DIR env var, or default external/KANO/data/)
    (matches KANO, GODE, MolRAG, GROVER exactly)

The scaffold_split_balanced() function below is a faithful reimplementation
of chemprop's scaffold_split(balanced=True), the same function used by
KANO (Fang et al. 2023), GODE (Jiang et al. 2025), and GROVER (Rong et al. 2020).

Output structure:
    data/moleculenet_property_scaffold/
    ├── seed_0/
    │   ├── bbbp/
    │   │   ├── bbbp_train.csv
    │   │   ├── bbbp_valid.csv
    │   │   └── bbbp_test.csv
    │   ├── bace/
    │   └── ... (10 datasets)
    ├── seed_1/
    └── seed_2/

Each CSV has columns: smiles, label, task, dataset, task_type, metric, split

Usage:
    python make_scaffold_splits.py

Or in Jupyter:
    %run make_scaffold_splits.py
"""

# %% Imports
import os
import warnings
import random
import traceback
from collections import defaultdict
from typing import List, Tuple, Dict

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")


# %% Configuration
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # scripts/data_preparation/ -> scripts/ -> MolE-RAG/

KANO_DATA_DIR = os.environ.get("KANO_DATA_DIR", str(REPO_ROOT / "external" / "KANO" / "data"))
OUT_DIR = str(REPO_ROOT / "data" / "moleculenet_property_scaffold")

SEEDS = [0, 1, 2]
SPLIT_SIZES = (0.8, 0.1, 0.1)  # train, valid, test

# Maps user's task name -> {csv filename in KANO repo, task type, metric}
DATASETS = {
    # Classification
    "bbbp":     {"csv": "bbbp.csv",     "task_type": "classification", "metric": "macro_f1"},
    "bace":     {"csv": "bace.csv",     "task_type": "classification", "metric": "macro_f1"},
    "hiv":      {"csv": "hiv.csv",      "task_type": "classification", "metric": "macro_f1"},
    "tox21":    {"csv": "tox21.csv",    "task_type": "classification", "metric": "macro_f1"},
    "toxcast":  {"csv": "toxcast.csv",  "task_type": "classification", "metric": "macro_f1"},
    "sider":    {"csv": "sider.csv",    "task_type": "classification", "metric": "macro_f1"},
    "clintox":  {"csv": "clintox.csv",  "task_type": "classification", "metric": "macro_f1"},
    # Regression
    "esol":     {"csv": "esol.csv",     "task_type": "regression",     "metric": "mae"},
    "freesolv": {"csv": "freesolv.csv", "task_type": "regression",     "metric": "mae"},
    "lipo":     {"csv": "lipo.csv",     "task_type": "regression",     "metric": "mae"},
}


# %% Chemprop scaffold_balanced reimplementation

def generate_scaffold(smiles: str, include_chirality: bool = False) -> str:
    """
    Generate Bemis-Murcko scaffold for a molecule.
    Matches chemprop's generate_scaffold() exactly.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol, includeChirality=include_chirality
    )
    return scaffold


def scaffold_to_indices(smiles_list: List[str]) -> Dict[str, List[int]]:
    """Group molecule indices by their scaffolds."""
    scaffold_groups: Dict[str, List[int]] = defaultdict(list)
    for i, smiles in enumerate(smiles_list):
        scaffold = generate_scaffold(smiles)
        scaffold_groups[scaffold].append(i)
    return scaffold_groups


def scaffold_split_balanced(
    smiles_list: List[str],
    sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Faithful reimplementation of chemprop's scaffold_split(balanced=True).

    Algorithm (from chemprop/data/scaffold.py):
        1. Compute Bemis-Murcko scaffold for each molecule
        2. Group molecules by scaffold
        3. Separate scaffold groups into "big" (>half val/test size) and "small"
        4. Shuffle both lists with random.seed(seed)
        5. Concatenate (big sets first) and greedily assign to train/val/test

    Returns:
        (train_indices, val_indices, test_indices) into the input smiles_list.
    """
    n = len(smiles_list)
    train_size = sizes[0] * n
    val_size = sizes[1] * n
    test_size = sizes[2] * n

    scaffold_groups = scaffold_to_indices(smiles_list)
    index_sets = list(scaffold_groups.values())

    big_index_sets: List[List[int]] = []
    small_index_sets: List[List[int]] = []
    for index_set in index_sets:
        if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
            big_index_sets.append(index_set)
        else:
            small_index_sets.append(index_set)

    random.seed(seed)
    random.shuffle(big_index_sets)
    random.shuffle(small_index_sets)
    index_sets = big_index_sets + small_index_sets

    train: List[int] = []
    val: List[int] = []
    test: List[int] = []
    for index_set in index_sets:
        if len(train) + len(index_set) <= train_size:
            train.extend(index_set)
        elif len(val) + len(index_set) <= val_size:
            val.extend(index_set)
        else:
            test.extend(index_set)

    return train, val, test


# %% Data loading from KANO CSVs

def find_smiles_column(df: pd.DataFrame) -> str:
    """
    Locate the SMILES column in a KANO CSV. They use chemprop format where
    the SMILES column is named 'smiles' (case-insensitive) or sits at column 0.
    """
    for col in df.columns:
        if col.strip().lower() == "smiles":
            return col
    # Fallback: assume first column (chemprop default)
    return df.columns[0]


def load_dataset_wide_from_kano(
    dataset_name: str, csv_filename: str
) -> Tuple[List[str], pd.DataFrame]:
    """
    Load a KANO-bundled CSV in wide format.

    KANO CSV format (chemprop-style):
        smiles, task1, task2, ...
    where missing labels are empty strings (-> NaN in pandas).

    Returns:
        tasks: list of task column names
        wide_df: DataFrame with columns ['smiles', task1, task2, ...]
                 Filtered to keep only molecules with valid RDKit-parseable SMILES.
    """
    csv_path = os.path.join(KANO_DATA_DIR, csv_filename)
    print(f"  Loading {dataset_name} from {csv_path}...")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Could not find {csv_path}. "
            f"Make sure KANO data is available at KANO_DATA_DIR ({KANO_DATA_DIR}), "
            f"or set the KANO_DATA_DIR environment variable."
        )

    raw_df = pd.read_csv(csv_path)
    smiles_col = find_smiles_column(raw_df)

    # Task columns are everything except smiles (and any compound-name column if present)
    skip_cols = {smiles_col}
    for col in raw_df.columns:
        if col.strip().lower() in {"mol_id", "num", "name", "compound_name"}:
            skip_cols.add(col)
    tasks = [c for c in raw_df.columns if c not in skip_cols]

    # Rename smiles column to canonical name
    raw_df = raw_df.rename(columns={smiles_col: "smiles"})

    # Filter invalid SMILES
    rows = []
    n_dropped = 0
    for _, row in raw_df.iterrows():
        smiles = row["smiles"]
        if not isinstance(smiles, str) or smiles == "":
            n_dropped += 1
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            n_dropped += 1
            continue
        rows.append(row)

    wide_df = pd.DataFrame(rows).reset_index(drop=True)

    print(
        f"    Loaded {len(wide_df)} valid molecules, "
        f"{len(tasks)} tasks "
        f"({n_dropped} invalid SMILES dropped)"
    )
    return tasks, wide_df


def wide_to_long_split(
    wide_df: pd.DataFrame,
    tasks: List[str],
    indices: List[int],
    dataset_name: str,
    task_type: str,
    metric: str,
    split_name: str,
) -> pd.DataFrame:
    """
    Convert wide-format split to long-format (one row per molecule x task).

    For multitask datasets (Tox21, SIDER, ToxCast, ClinTox), each molecule
    produces multiple rows. Rows with NaN labels (KANO uses empty strings
    for missing labels in multitask datasets) are dropped.
    """
    rows = []
    sub_df = wide_df.iloc[indices].reset_index(drop=True)

    for _, row in sub_df.iterrows():
        for task in tasks:
            label = row[task]
            if pd.isna(label):
                continue
            # Cast to float; classification labels stored as 0/1 ints in KANO CSVs
            try:
                label_val = float(label)
            except (TypeError, ValueError):
                continue

            rows.append(
                {
                    "smiles": row["smiles"],
                    "label": label_val,
                    "task": task,
                    "dataset": dataset_name,
                    "task_type": task_type,
                    "metric": metric,
                    "split": split_name,
                }
            )

    return pd.DataFrame(rows)


# %% Main processing

def process_dataset(
    dataset_name: str, cfg: dict, seeds: List[int]
) -> List[dict]:
    """Process one dataset across all seeds. Returns summary stats."""
    print(f"\n[{dataset_name}]")

    tasks, wide_df = load_dataset_wide_from_kano(dataset_name, cfg["csv"])
    smiles_list = wide_df["smiles"].tolist()

    summaries = []
    for seed in seeds:
        print(f"  Splitting with seed={seed}...")

        train_idx, val_idx, test_idx = scaffold_split_balanced(
            smiles_list,
            sizes=SPLIT_SIZES,
            seed=seed,
        )

        # Sanity checks
        assert len(set(train_idx) & set(val_idx)) == 0, "train/val overlap!"
        assert len(set(train_idx) & set(test_idx)) == 0, "train/test overlap!"
        assert len(set(val_idx) & set(test_idx)) == 0, "val/test overlap!"
        assert len(train_idx) + len(val_idx) + len(test_idx) == len(smiles_list)

        out_folder = os.path.join(OUT_DIR, f"seed_{seed}", dataset_name)
        os.makedirs(out_folder, exist_ok=True)

        for split_name, indices in [
            ("train", train_idx),
            ("valid", val_idx),
            ("test", test_idx),
        ]:
            long_df = wide_to_long_split(
                wide_df,
                tasks,
                indices,
                dataset_name,
                cfg["task_type"],
                cfg["metric"],
                split_name,
            )
            out_path = os.path.join(out_folder, f"{dataset_name}_{split_name}.csv")
            long_df.to_csv(out_path, index=False)

        n_total = len(wide_df)
        summary = {
            "dataset": dataset_name,
            "seed": seed,
            "n_molecules_total": n_total,
            "n_molecules_train": len(train_idx),
            "n_molecules_valid": len(val_idx),
            "n_molecules_test": len(test_idx),
            "train_pct": round(len(train_idx) / n_total * 100, 1),
            "val_pct": round(len(val_idx) / n_total * 100, 1),
            "test_pct": round(len(test_idx) / n_total * 100, 1),
        }
        summaries.append(summary)
        print(
            f"    seed={seed}: train={len(train_idx)} ({summary['train_pct']}%), "
            f"val={len(val_idx)} ({summary['val_pct']}%), "
            f"test={len(test_idx)} ({summary['test_pct']}%)"
        )

    return summaries


# %% Run

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70)
    print("Scaffold_balanced split generation (from KANO CSVs)")
    print("=" * 70)
    print(f"Data source:      {KANO_DATA_DIR}")
    print(f"Output directory: {OUT_DIR}")
    print(f"Seeds:            {SEEDS}")
    print(f"Split sizes:      {SPLIT_SIZES}")
    print(f"Datasets:         {list(DATASETS.keys())}")
    print("=" * 70)

    all_summaries = []
    failed = []

    for dataset_name, cfg in DATASETS.items():
        try:
            summaries = process_dataset(dataset_name, cfg, SEEDS)
            all_summaries.extend(summaries)
        except Exception as e:
            print(f"  [ERROR] Failed on {dataset_name}: {e}")
            traceback.print_exc()
            failed.append(dataset_name)

    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        summary_path = os.path.join(OUT_DIR, "summary.csv")
        summary_df.to_csv(summary_path, index=False)

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(summary_df.to_string(index=False))
        print(f"\nSummary saved: {summary_path}")

    if failed:
        print(f"\n[WARN] Failed datasets: {failed}")

    print("\nDone.")


if __name__ == "__main__":
    main()