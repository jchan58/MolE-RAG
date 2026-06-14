#!/usr/bin/env python3
"""Extract best fingerprint per task from validation sweep CSVs."""

import pandas as pd
import glob
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # scripts/data_preparation/ -> scripts/ -> MolE-RAG/
ROOT = REPO_ROOT

# Find all summary CSVs
csvs = sorted(glob.glob(str(ROOT / "**/*fingerprint_sweep_summary.csv"), recursive=True))
print(f"Found {len(csvs)} summary CSVs\n")

rows = []
for csv_path in csvs:
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        continue

    # Identify metric column
    if "rmse" in df.columns and df["task_type"].iloc[0] == "regression":
        metric = "rmse"
        ascending = True   # lower RMSE is better
    elif "roc_auc" in df.columns:
        metric = "roc_auc"
        ascending = False
    elif "auc" in df.columns:
        metric = "auc"
        ascending = False
    elif "accuracy" in df.columns:
        metric = "accuracy"
        ascending = False
    else:
        print(f"⚠️  Unknown metric in {csv_path}, columns: {df.columns.tolist()}")
        continue

    # Sort and pick best
    df_sorted = df.sort_values(metric, ascending=ascending).reset_index(drop=True)
    best = df_sorted.iloc[0]
    second = df_sorted.iloc[1] if len(df_sorted) > 1 else None

    # Get dataset name from path
    rel_path = Path(csv_path).relative_to(ROOT)
    dataset = rel_path.parts[0]

    row = {
        "Dataset": dataset,
        "Task": best["task"],
        "Type": best["task_type"],
        "Best FP": best["method"],
        "k": int(best["k"]),
        "Best Score": float(best[metric]),
        "Runner-up FP": second["method"] if second is not None else "-",
        "Runner-up Score": float(second[metric]) if second is not None else None,
        "Δ": float(best[metric] - second[metric]) if second is not None else 0,
        "n_eval": int(best["n_eval"]),
    }
    rows.append(row)

result = pd.DataFrame(rows)
print(f"Extracted best-FP for {len(result)} tasks\n")

# Show summary by dataset
print("=== Best FP frequency per dataset ===")
freq = result.groupby(["Dataset", "Best FP"]).size().unstack(fill_value=0)
print(freq)
print()

# Save full table
out_csv = ROOT / "textrag" / "best_fingerprint_per_task.csv"
result.to_csv(out_csv, index=False)
print(f"Saved full table to {out_csv}")

# Print preview
print("\n=== First 30 rows ===")
print(result.head(30).to_string(index=False))