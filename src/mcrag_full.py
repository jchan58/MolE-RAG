#!/usr/bin/env python
"""
mcrag_full.py — the full MCRAG pipeline (with --seed + --fewshot_pool support).

Combines all three pillars into a single inference run:
  1. Text retrieval     — hybrid BM25 (LLM task keywords + LLM-filtered synonyms)
  2. Structure retrieval — k-NN over training set using per-task best fingerprint
  3. Context injection   — synonyms + functional groups + RDKit descriptors

Auto-runs SMILES baseline for any model missing one in baseline_results/.
Retrieval caching saves BM25 results to disk after first run (per-seed).

Seed handling:
  --seed N   read test/fewshot from data/moleculenet_property_scaffold/seed_{N}/{dataset}/
             write outputs to mcrag_full_results/seed_{N}/ and baseline_results/seed_{N}/
             retrieval cache filename includes _seed{N} suffix
             (text retrieval still hits the same corpus, but cache is keyed by row index
              and row indices differ across seeds, so per-seed cache files are required)
  no --seed  use old random-split paths (backward-compatible)

Fewshot pool handling:
  --fewshot_pool structural  (default) — use fewshot_global_structural_top5.csv (best-FP neighbors)
  --fewshot_pool random      — use fewshot_random_top5.csv (random training examples)

CLI:
  python mcrag_full.py --dataset bbbp --models gpt-4o-mini --seed 0
  python mcrag_full.py --dataset bbbp --models gpt-4o-mini --seed 0 --fewshot_pool random
  python mcrag_full.py --dataset bbbp --no_baseline   # skip auto-baseline check
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths (resolved relative to repo root)
# ---------------------------------------------------------------------------
REPO_ROOT     = Path(__file__).resolve().parent.parent  # src/ -> MolE-RAG/
CHEMRAG_ROOT  = REPO_ROOT / "external" / "ChemRAG"
TEXTRAG_DIR   = REPO_ROOT / "src"
ENV_PATH      = REPO_ROOT / ".env"

DEFAULT_CORPUS_PATH       = CHEMRAG_ROOT / "corpus" / "chemrag_full_corpus.jsonl"
DEFAULT_INDEX_DIR         = CHEMRAG_ROOT / "index"
DEFAULT_SYNONYM_CACHE     = REPO_ROOT / "caches" / "llm_filtered_synonyms.json"

if str(TEXTRAG_DIR) not in sys.path:
    sys.path.insert(0, str(TEXTRAG_DIR))

from prompt_blocks import (  # noqa: E402
    load_syn_cache, build_prompt_injection, availability_check,
    DEFAULT_SYNONYM_CACHE as PB_DEFAULT_SYNONYM_CACHE,
)
import chemrag_retrieval as cr  # noqa: E402

load_dotenv(str(ENV_PATH))


# ===========================================================================
# Per-dataset config
# ===========================================================================
DATASET_CONFIG = cr.DATASET_CONFIG
BASELINE_DIR_NAME = "baseline_results"

# Sane ranges for regression predictions (drop values outside these as garbage).
REGRESSION_SANE_RANGE = {
    "esol":     (-15.0, 15.0),
    "freesolv": (-30.0, 10.0),
    "lipo":     (-5.0,  10.0),
}

def is_sane_regression(value, dataset_name):
    """Return True if value is within the sane range for this regression task."""
    if value is None:
        return False
    lo, hi = REGRESSION_SANE_RANGE.get(dataset_name, (float("-inf"), float("inf")))
    return lo <= value <= hi


# Per-task guidance for regression prompts.
REGRESSION_TASK_INFO = {
    "esol": {
        "unit_long":   "log-molar aqueous solubility (log mol/L)",
        "valid_range": "between -10 and 2",
        "example":     "-2.34",
    },
    "freesolv": {
        "unit_long":   "hydration free energy in kcal/mol",
        "valid_range": "between -25 and 5",
        "example":     "-5.12",
    },
    "lipo": {
        "unit_long":   "octanol-water partition coefficient at pH 7.4 (logD)",
        "valid_range": "between -2 and 5",
        "example":     "2.45",
    },
}


# ===========================================================================
# Retrieval cache helpers (seed-aware)
# ===========================================================================
def get_retrieval_cache_path(base_dir: Path, dataset_name: str,
                              retriever: str, k: int,
                              seed: Optional[int] = None) -> Path:
    """Return the path for the on-disk retrieval cache for a given run config.

    When seed is set, includes a _seed{N} suffix. The cache is keyed by row
    index in the test CSV, and different scaffold seeds put different molecules
    at the same row index, so per-seed cache files are required for correctness.
    """
    cache_dir = base_dir / "retrieval_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    seed_tag = f"_seed{seed}" if seed is not None else ""
    return cache_dir / f"retrieval_{dataset_name}_hybrid_{retriever}_k{k}{seed_tag}.json"


def load_retrieval_cache(cache_path: Path) -> Optional[Dict[int, List[Dict[str, Any]]]]:
    if not cache_path.exists():
        return None
    print(f"  Loading cached retrievals from {cache_path}")
    with open(cache_path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def save_retrieval_cache(cache_path: Path,
                          retrieval_cache: Dict[int, List[Dict[str, Any]]]):
    tmp = cache_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(retrieval_cache, f)
    os.replace(tmp, cache_path)
    print(f"  Saved retrieval cache -> {cache_path}")


# ===========================================================================
# Structural fewshot CSV loader
# ===========================================================================
def load_structural_fewshot(path: Path) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    """Load a structural fewshot CSV from an explicit path."""
    if not path.exists():
        print(f"  WARNING: no structural fewshot CSV at {path}")
        return {}
    print(f"  Loading structural fewshot from {path}")
    out: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_smi = (row.get("test_smiles") or "").strip()
            task = (row.get("task") or "").strip()
            neighbor = {
                "smiles":            (row.get("smiles") or "").strip(),
                "label":             row.get("label"),
                "similarity_score":  row.get("similarity_score"),
                "fewshot_rank":      row.get("fewshot_rank"),
                "retrieval_method":  row.get("retrieval_method", ""),
            }
            out[(test_smi, task)].append(neighbor)
    for key in out:
        try:
            out[key].sort(key=lambda d: int(d.get("fewshot_rank") or 999))
        except ValueError:
            pass
    if out:
        first_key = next(iter(out.keys()))
        n_neighbors = len(out[first_key])
        sample_method = out[first_key][0].get("retrieval_method", "?")
        print(f"  Loaded {len(out)} (test_smi, task) entries x {n_neighbors} neighbors each")
        print(f"  Retrieval method: {sample_method}")
    return dict(out)


def build_structural_block(neighbors: List[Dict[str, Any]],
                            task_type: str,
                            pool_label: str = "fingerprint") -> str:
    if not neighbors:
        return ""
    method_name = neighbors[0].get("retrieval_method", pool_label)
    # Header text depends on whether neighbors are similarity-based or random
    if method_name == "random":
        header = "Randomly sampled training molecules:"
    else:
        header = f"Structurally similar training molecules (by {method_name} fingerprint):"
    label_word = "true value" if task_type == "regression" else "label"
    lines = [header]
    for n in neighbors:
        rank = n.get("fewshot_rank", "?")
        smi = n.get("smiles", "?")
        sim = n.get("similarity_score")
        try:
            sim_str = f"{float(sim):.3f}" if sim is not None else "?"
        except (TypeError, ValueError):
            sim_str = "?"
        lab = n.get("label")
        if task_type == "regression":
            try:
                lab_str = f"{float(lab):.3f}"
            except (TypeError, ValueError):
                lab_str = str(lab)
        else:
            try:
                lab_int = int(round(float(lab)))
                lab_str = "Yes" if lab_int == 1 else "No"
            except (TypeError, ValueError):
                lab_str = str(lab)
        # For random, suppress the meaningless sim score
        if method_name == "random":
            lines.append(f"  Example {rank}: SMILES={smi}  {label_word}={lab_str}")
        else:
            lines.append(f"  Example {rank} (sim={sim_str}): SMILES={smi}  {label_word}={lab_str}")
    return "\n".join(lines)


# ===========================================================================
# SMILES baseline prompts
# ===========================================================================
def build_smiles_baseline_prompt(smiles: str, dataset_name: str, task_name: str) -> str:
    desc = cr.task_description(dataset_name, task_name)
    task_type = DATASET_CONFIG[dataset_name]["task_type"]
    if task_type == "classification":
        return (f"Task: Predict {desc}.\n\n"
                f"SMILES: {smiles}\n\n"
                f"Reply with EXACTLY ONE WORD: either 'Yes' or 'No'. "
                f"Output nothing else. No explanation, no punctuation, no context.")
    info = REGRESSION_TASK_INFO.get(dataset_name, {})
    unit_long   = info.get("unit_long", "value")
    valid_range = info.get("valid_range", "(any real number)")
    example_val = info.get("example", "0.0")
    return (f"Task: Predict {desc}.\n\n"
            f"SMILES: {smiles}\n\n"
            f"Predict the {unit_long}.\n"
            f"Valid predictions are real numbers {valid_range}.\n\n"
            f"Output EXACTLY ONE number (example: {example_val}). No units, no explanation.\n\n"
            f"ANSWER: ")


def build_smiles_baseline_prompt_strict(smiles: str, dataset_name: str, task_name: str) -> str:
    desc = cr.task_description(dataset_name, task_name)
    task_type = DATASET_CONFIG[dataset_name]["task_type"]
    if task_type == "classification":
        return (f"Question: Does this molecule have the following property? {desc}\n\n"
                f"SMILES: {smiles}\n\n"
                f"You MUST answer with exactly one of these two words: Yes OR No.\n"
                f"EVEN IF YOU ARE UNCERTAIN, choose the most likely answer.\n"
                f"Do not refuse. Do not explain. Do not add anything else.\n\n"
                f"Answer:")
    info = REGRESSION_TASK_INFO.get(dataset_name, {})
    unit_long   = info.get("unit_long", "value")
    valid_range = info.get("valid_range", "(any real number)")
    example_val = info.get("example", "0.0")
    return (f"Question: Predict the {unit_long} for: {desc}\n\n"
            f"SMILES: {smiles}\n\n"
            f"You MUST output exactly one number {valid_range}.\n"
            f"Example valid answer: {example_val}\n"
            f"EVEN IF YOU ARE UNCERTAIN, give your best numeric estimate.\n"
            f"Do not refuse. Do not explain. Do not include units.\n\n"
            f"Answer:")


def run_smiles_baseline_for_model(model_name: str, dataset_name: str,
                                    valid_examples: List, task_type: str,
                                    output_dir: Path, args) -> None:
    """Run SMILES-only baseline for one model -> baseline_results/"""
    model_short = cr.get_model_short(model_name)
    out_name = f"{model_short}_{dataset_name}_smiles_only_zero_shot.jsonl"
    out_path = output_dir / out_name

    if out_path.exists() and not args.overwrite_baseline:
        print(f"  [SKIP-BASELINE] {out_name} already exists")
        return

    print(f"\n  >>> Running SMILES baseline for {model_name} on {dataset_name}")
    cr.load_model(model_name)
    try:
        eff_batch_size = (args.openai_batch_size
                          if cr._model_cfg["format"] == "openai"
                          else args.batch_size)
        max_new = 10 if task_type == "classification" else 30

        batch_meta = []
        for idx, example in valid_examples:
            smiles    = example.get(cr.SMILES_COL, "").strip()
            true_label = example.get(cr.LABEL_COL, None)
            task_name = cr.get_task_name(example, dataset_name)
            user_prompt = build_smiles_baseline_prompt(smiles, dataset_name, task_name)
            batch_meta.append((idx, smiles, true_label, task_name, user_prompt))

        final_outputs = []
        for bs in range(0, len(batch_meta), eff_batch_size):
            batch        = batch_meta[bs: bs + eff_batch_size]
            user_prompts = [b[-1] for b in batch]
            model_outputs = cr.run_inference_batch(
                cr.SYSTEM_PROMPT, user_prompts,
                max_new_tokens=max_new,
                openai_max_workers=args.openai_batch_size,
            )
            for (idx, smiles, true_label, task_name, _), out in zip(batch, model_outputs):
                if task_type == "classification":
                    predicted_label = cr.parse_binary_output(out)
                    predicted_value = None
                else:
                    predicted_label = None
                    predicted_value = cr.parse_regression_output(out)
                final_outputs.append({
                    "id": idx, "smiles": smiles, "true_label": true_label,
                    "dataset": dataset_name, "task": task_name, "assay": task_name,
                    "condition": "smiles_baseline", "mode": "smiles",
                    "model": model_name, "task_type": task_type,
                    "model_output": out,
                    "predicted_label": predicted_label,
                    "predicted_value": predicted_value,
                    "retry_used": False, "retry_output": None,
                })
            done = min(bs + eff_batch_size, len(batch_meta))
            print(f"    [baseline] {done}/{len(batch_meta)} done")

        if args.retry_nulls:
            if task_type == "classification":
                null_idx = [i for i, r in enumerate(final_outputs)
                            if r["predicted_label"] is None]
            else:
                null_idx = [i for i, r in enumerate(final_outputs)
                            if r["predicted_value"] is None
                            or not is_sane_regression(r["predicted_value"], dataset_name)]
            n_first_null = len(null_idx)
            if n_first_null > 0:
                print(f"  Baseline retrying {n_first_null} nulls/garbage...")
                retry_prompts = [
                    build_smiles_baseline_prompt_strict(
                        final_outputs[i]["smiles"], dataset_name, final_outputs[i]["task"])
                    for i in null_idx
                ]
                retry_max = 5 if task_type == "classification" else 20
                n_recovered = 0
                for rs in range(0, len(retry_prompts), eff_batch_size):
                    rp = retry_prompts[rs: rs + eff_batch_size]
                    ri = null_idx[rs: rs + eff_batch_size]
                    retry_outs = cr.run_inference_batch(
                        cr.SYSTEM_PROMPT_STRICT, rp,
                        max_new_tokens=retry_max,
                        openai_max_workers=args.openai_batch_size,
                    )
                    for orig_idx, retry_out in zip(ri, retry_outs):
                        row = final_outputs[orig_idx]
                        row["retry_used"]   = True
                        row["retry_output"] = retry_out
                        if task_type == "classification":
                            p = cr.parse_binary_output(retry_out)
                            if p is not None:
                                row["predicted_label"] = p
                                n_recovered += 1
                        else:
                            p = cr.parse_regression_output(retry_out)
                            if p is not None and is_sane_regression(p, dataset_name):
                                row["predicted_value"] = p
                                n_recovered += 1
                            elif p is not None:
                                row["predicted_value"] = None
                print(f"  Baseline retry recovered {n_recovered}/{n_first_null}")

        with open(out_path, "w", encoding="utf-8") as f:
            for row in final_outputs:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  [BASELINE SAVED] {len(final_outputs)} rows -> {out_path}")
    finally:
        cr.unload_model()


# ===========================================================================
# Prompt construction (full MCRAG)
# ===========================================================================
SYSTEM_PROMPT = cr.SYSTEM_PROMPT
SYSTEM_PROMPT_STRICT = cr.SYSTEM_PROMPT_STRICT


def build_user_prompt_full(smiles: str, dataset_name: str, task_name: str,
                            retrieved_docs: List[Dict[str, Any]],
                            structural_block: str = "",
                            synonyms_block: str = "",
                            fgs_block: str = "",
                            rdkit_block: str = "") -> str:
    desc = cr.task_description(dataset_name, task_name)
    retrieved_block = cr.build_retrieved_block(retrieved_docs)
    task_type = DATASET_CONFIG[dataset_name]["task_type"]
    sections = [f"Retrieved chemistry context:\n{retrieved_block}"]
    if structural_block: sections.append(structural_block)
    if synonyms_block:   sections.append(synonyms_block)
    if fgs_block:        sections.append(fgs_block)
    if rdkit_block:      sections.append(rdkit_block)
    body = "\n\n".join(sections)
    if task_type == "classification":
        return (f"Task: Predict {desc}.\n\n"
                f"{body}\n\n"
                f"Now predict the answer for this molecule.\n"
                f"SMILES: {smiles}\n\n"
                f"Reply with EXACTLY ONE WORD: either 'Yes' or 'No'. "
                f"Output nothing else. No explanation, no punctuation, no context.")
    info = REGRESSION_TASK_INFO.get(dataset_name, {})
    unit_long   = info.get("unit_long", "value")
    valid_range = info.get("valid_range", "(any real number)")
    example_val = info.get("example", "0.0")
    return (f"Task: Predict {desc}.\n\n"
            f"{body}\n\n"
            f"==========================================\n"
            f"PREDICTION TASK FOR TARGET MOLECULE\n"
            f"==========================================\n"
            f"Target SMILES: {smiles}\n\n"
            f"Predict the {unit_long} for THIS target molecule.\n"
            f"Valid predictions are real numbers {valid_range}.\n\n"
            f"CRITICAL: The values shown in the reference blocks above (such as "
            f"MolWt, TopoPSA, NumHeavyAtoms, NumHDonors, molecular weight, polar "
            f"surface area, CIDs, etc.) are PROPERTIES of the molecule, NOT the answer. "
            f"DO NOT echo any of those numbers. Those values are typically 10 to 500, "
            f"while the correct answer is {valid_range}.\n\n"
            f"Output EXACTLY ONE number (example: {example_val}). No units, no descriptor "
            f"names, no explanation.\n\n"
            f"ANSWER: ")


def build_user_prompt_full_strict(smiles: str, dataset_name: str, task_name: str,
                                   retrieved_docs: List[Dict[str, Any]],
                                   structural_block: str = "",
                                   synonyms_block: str = "",
                                   fgs_block: str = "",
                                   rdkit_block: str = "") -> str:
    desc = cr.task_description(dataset_name, task_name)
    short_docs = retrieved_docs[:3] if retrieved_docs else []
    short_lines = []
    for i, d in enumerate(short_docs, start=1):
        text = d.get("contents", "").strip().replace("\n", " ")
        if len(text) > 400:
            text = text[:400] + " ..."
        short_lines.append(f"[{i}] {text}")
    short_block = "\n\n".join(short_lines) if short_lines else "No documents retrieved."
    sections = [f"Retrieved evidence:\n{short_block}"]
    if structural_block: sections.append(structural_block)
    if synonyms_block:   sections.append(synonyms_block)
    if fgs_block:        sections.append(fgs_block)
    if rdkit_block:      sections.append(rdkit_block)
    body = "\n\n".join(sections)
    task_type = DATASET_CONFIG[dataset_name]["task_type"]
    if task_type == "classification":
        return (f"Question: Does this molecule have the following property? {desc}\n\n"
                f"{body}\n\n"
                f"SMILES: {smiles}\n\n"
                f"You MUST answer with exactly one of these two words: Yes OR No.\n"
                f"EVEN IF YOU ARE UNCERTAIN, choose the most likely answer.\n"
                f"Do not refuse. Do not explain. Do not add anything else.\n\n"
                f"Answer:")
    info = REGRESSION_TASK_INFO.get(dataset_name, {})
    unit_long   = info.get("unit_long", "value")
    valid_range = info.get("valid_range", "(any real number)")
    example_val = info.get("example", "0.0")
    return (f"Question: Predict the {unit_long} for: {desc}\n\n"
            f"{body}\n\n"
            f"Target SMILES: {smiles}\n\n"
            f"You MUST output exactly one number {valid_range}.\n"
            f"Example valid answer: {example_val}\n\n"
            f"DO NOT output reference values such as MolWt, TopoPSA, NumHeavyAtoms, "
            f"or any descriptor value. Those are properties, not the answer. "
            f"The correct answer is a number {valid_range}.\n\n"
            f"Answer:")


# ===========================================================================
# Main
# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASET_CONFIG.keys()))
    ap.add_argument("--retriever", default="bm25",
                    choices=["bm25", "contriever", "specter", "e5", "rrf"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--rrf_pool_size", type=int, default=100)
    ap.add_argument("--rrf_k", type=int, default=60)
    ap.add_argument("--corpus_path", default=str(DEFAULT_CORPUS_PATH))
    ap.add_argument("--index_dir", default=str(DEFAULT_INDEX_DIR))

    ap.add_argument("--no_text_retrieval",      action="store_true")
    ap.add_argument("--no_structure_retrieval", action="store_true")
    ap.add_argument("--no_synonyms",            action="store_true")
    ap.add_argument("--no_fgs",                 action="store_true")
    ap.add_argument("--no_rdkit",               action="store_true")

    ap.add_argument("--synonym_cache", default=str(DEFAULT_SYNONYM_CACHE))
    ap.add_argument("--models", nargs="+", default=cr.DEFAULT_MODELS_TO_RUN)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=cr.DEFAULT_BATCH_SIZE)
    ap.add_argument("--openai_batch_size", type=int, default=8)

    ap.add_argument("--retry_nulls", dest="retry_nulls", action="store_true", default=True)
    ap.add_argument("--no_retry_nulls", dest="retry_nulls", action="store_false")
    ap.add_argument("--overwrite", action="store_true",
                    help="overwrite existing MCRAG inference output files")
    ap.add_argument("--overwrite_retrieval", action="store_true")
    ap.add_argument("--retrieval_only", action="store_true")

    # SMILES baseline auto-generation
    ap.add_argument("--no_baseline", action="store_true",
                    help="skip auto-running SMILES baselines for missing models")
    ap.add_argument("--overwrite_baseline", action="store_true",
                    help="re-run SMILES baselines even if they exist")

    # Scaffold split seed support
    ap.add_argument("--seed", type=int, default=None,
                    help="If set, read test/fewshot from "
                         "data/moleculenet_property_scaffold/seed_{N}/{dataset}/ "
                         "and write outputs to seed-specific subfolders. "
                         "Default: use old random-split paths (backward-compatible).")

    # Which fewshot pool CSV to use (only affects structural block)
    ap.add_argument("--fewshot_pool", choices=["structural", "random"],
                    default="structural",
                    help="Which fewshot CSV to load for the structural block: "
                         "'structural' = fewshot_global_structural_top5.csv (best-FP neighbors), "
                         "'random' = fewshot_random_top5.csv (random training examples). "
                         "Default: structural.")

    args = ap.parse_args()

    dataset_name = args.dataset
    cfg = DATASET_CONFIG[dataset_name]
    task_type = cfg["task_type"]

    use_text     = not args.no_text_retrieval
    use_struct   = not args.no_structure_retrieval
    use_synonyms = not args.no_synonyms
    use_fgs      = not args.no_fgs
    use_rdkit    = not args.no_rdkit

    availability_check(use_synonyms, use_fgs, use_rdkit, Path(args.synonym_cache))

    flag_parts = []
    if use_text:     flag_parts.append("text")
    if use_struct:   flag_parts.append("struct")
    if use_synonyms: flag_parts.append("syn")
    if use_fgs:      flag_parts.append("fg")
    if use_rdkit:    flag_parts.append("rdk")
    flag_combo = "_".join(flag_parts) if flag_parts else "none"
    is_full = (use_text and use_struct and use_synonyms and use_fgs and use_rdkit)
    out_suffix = "full" if is_full else flag_combo

    # ====================================================================
    # Path setup — handles both --seed and --fewshot_pool
    # ====================================================================
    base_dir = REPO_ROOT / "results" / dataset_name

    # Pick which fewshot CSV based on --fewshot_pool
    fewshot_csv_name = ("fewshot_random_top5.csv" if args.fewshot_pool == "random"
                        else "fewshot_global_structural_top5.csv")

    if args.seed is not None:
        scaffold_dir = (REPO_ROOT / "data" / "moleculenet_property_scaffold"
                        / f"seed_{args.seed}" / dataset_name)
        test_path    = scaffold_dir / f"{dataset_name}_test.csv"
        fewshot_path = scaffold_dir / fewshot_csv_name
        output_dir   = base_dir / "mcrag_full_results" / f"seed_{args.seed}"
        baseline_dir = base_dir / BASELINE_DIR_NAME / f"seed_{args.seed}"
        print(f"[SEED MODE] Using scaffold split seed_{args.seed}")
        print(f"  test:    {test_path}")
        print(f"  fewshot: {fewshot_path} ({args.fewshot_pool})")
    else:
        test_path    = base_dir / f"{dataset_name}_test.csv"
        fewshot_path = base_dir / fewshot_csv_name
        output_dir   = base_dir / "mcrag_full_results"
        baseline_dir = base_dir / BASELINE_DIR_NAME

    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir.mkdir(parents=True, exist_ok=True)

    syn_cache = load_syn_cache(Path(args.synonym_cache))
    if (use_synonyms or use_text) and syn_cache is None:
        raise FileNotFoundError(f"Need LLM-filtered synonym cache at {args.synonym_cache}")
    print(f"Loaded synonym cache: {len(syn_cache) if syn_cache else 0} SMILES")

    structural_neighbors: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    if use_struct:
        structural_neighbors = load_structural_fewshot(fewshot_path)

    test_rows = cr.load_csv(test_path)
    valid_examples = [
        (idx, ex) for idx, ex in enumerate(test_rows)
        if ex.get(cr.SMILES_COL, "").strip() and cr.valid_label(ex.get(cr.LABEL_COL))
    ]
    if args.limit is not None:
        valid_examples = valid_examples[:args.limit]

    print("=" * 100)
    print(f"MCRAG FULL — combined retrieval + prompt injection")
    print(f"Dataset   : {dataset_name}   Task type: {task_type}   Seed: {args.seed}")
    print(f"Fewshot   : {args.fewshot_pool}")
    print(f"Pillars   : text={use_text}  structure={use_struct}  "
          f"syn={use_synonyms}  fg={use_fgs}  rdk={use_rdkit}")
    print(f"Suffix    : {out_suffix}")
    print(f"Retriever : {args.retriever}   text top-k: {args.k}")
    print(f"Output    : {output_dir}")
    print(f"Baseline  : {baseline_dir}")
    print(f"Models    : {args.models}")
    print(f"Valid rows: {len(valid_examples)}")
    print("=" * 100)

    # =========================================================
    # Step 0: SMILES baseline auto-fill for missing models
    # =========================================================
    if not args.no_baseline:
        print("\n" + "=" * 60)
        print("Checking SMILES baseline files...")
        print("=" * 60)
        for model_name in args.models:
            model_short = cr.get_model_short(model_name)
            baseline_path = baseline_dir / f"{model_short}_{dataset_name}_smiles_only_zero_shot.jsonl"
            if baseline_path.exists() and not args.overwrite_baseline:
                print(f"  [OK] baseline exists: {baseline_path.name}")
            else:
                print(f"  [MISSING] {baseline_path.name} — will run")
                run_smiles_baseline_for_model(
                    model_name, dataset_name, valid_examples, task_type,
                    baseline_dir, args)

    # =========================================================
    # Step 1: build hybrid BM25 queries
    # =========================================================
    print("\nBuilding hybrid BM25 queries...")
    if use_text:
        unique_tasks = [cr.get_task_name(ex, dataset_name) for _, ex in valid_examples]
        print(f"  Pre-generating LLM task keywords for "
              f"{len(set(unique_tasks))} unique task name(s)...")
        cr.precompute_task_keywords(dataset_name, unique_tasks)

    query_records = []
    src_counts: Dict[str, int] = {}
    for idx, example in valid_examples:
        smiles    = example.get(cr.SMILES_COL, "").strip()
        task_name = cr.get_task_name(example, dataset_name)
        if use_text:
            qrec = cr.build_query(smiles, dataset_name, task_name,
                                  mode="hybrid", syn_cache=syn_cache,
                                  raw_syn_cache=None)
        else:
            qrec = {"query": "", "source": "skipped",
                    "good_syns": [], "iupac": "", "raw_syns": [], "task_kw": "",
                    "in_cache": False}
        query_records.append(qrec)
        src_counts[qrec["source"]] = src_counts.get(qrec["source"], 0) + 1
    print(f"  Query source distribution: {src_counts}")

    # =========================================================
    # Step 2: BM25 retrieval — load from cache or run fresh (seed-aware)
    # =========================================================
    retrieval_cache: Dict[int, List[Dict[str, Any]]] = {}

    if use_text:
        retrieval_cache_path = get_retrieval_cache_path(
            base_dir, dataset_name, args.retriever, args.k, seed=args.seed)
        cached = None if args.overwrite_retrieval else load_retrieval_cache(retrieval_cache_path)
        if cached is not None:
            retrieval_cache = cached
            print(f"  Using cached retrievals for {len(retrieval_cache)} rows")
        else:
            print("\nLoading retriever...")
            retriever = cr.make_retriever(
                retriever_name=args.retriever, corpus_path=args.corpus_path,
                index_dir=args.index_dir, topk=args.k,
                rrf_pool_size=args.rrf_pool_size, rrf_k=args.rrf_k,
            )
            print("Retrieving for all test rows...")
            for i, (idx, _) in enumerate(valid_examples):
                retrieval_cache[idx] = retriever.search(query_records[i]["query"])
                if (i + 1) % 100 == 0:
                    print(f"  retrieved {i + 1}/{len(valid_examples)}")
            print("Retrieval done.")
            save_retrieval_cache(retrieval_cache_path, retrieval_cache)
            del retriever
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        for idx, _ in valid_examples:
            retrieval_cache[idx] = []

    if args.retrieval_only:
        print("\n--retrieval_only set: retrieval cache saved. Exiting.")
        return

    # =========================================================
    # Step 3: build injection blocks
    # =========================================================
    print("\nBuilding injection blocks for all test rows...")
    injection_cache: Dict[int, Dict[str, str]] = {}
    structural_cache: Dict[int, str] = {}
    for idx, ex in valid_examples:
        smi       = ex[cr.SMILES_COL].strip()
        task_name = cr.get_task_name(ex, dataset_name)
        injection_cache[idx] = build_prompt_injection(
            smi, dataset_name, task_name,
            syn_cache=syn_cache,
            use_synonyms=use_synonyms,
            use_fgs=use_fgs,
            use_rdkit=use_rdkit,
        )
        if use_struct:
            neighbors = structural_neighbors.get((smi, task_name), [])
            if not neighbors and task_name != dataset_name:
                neighbors = structural_neighbors.get((smi, dataset_name), [])
            structural_cache[idx] = build_structural_block(
                neighbors, task_type, pool_label=args.fewshot_pool)
        else:
            structural_cache[idx] = ""

    # =========================================================
    # Step 4: MCRAG inference — one model at a time
    # =========================================================
    # Tag output filename with pool type so structural/random don't collide
    pool_tag = "" if args.fewshot_pool == "structural" else f"_{args.fewshot_pool}"

    for model_name in args.models:
        print(f"\n{'#' * 60}\nMODEL: {model_name}\n{'#' * 60}")
        cr.load_model(model_name)
        model_short = cr.get_model_short(model_name)
        try:
            out_name = (f"{model_short}_{dataset_name}_mcrag_{out_suffix}{pool_tag}_"
                        f"{args.retriever}_k{args.k}.jsonl")
            out_path = output_dir / out_name
            if out_path.exists() and not args.overwrite:
                print(f"  [SKIP] {out_name} already exists")
                continue

            eff_batch_size = (args.openai_batch_size
                              if cr._model_cfg["format"] == "openai"
                              else args.batch_size)
            print(f"  Using batch_size={eff_batch_size}")
            max_new = 10 if task_type == "classification" else 30

            batch_meta = []
            for (idx, example), qrec in zip(valid_examples, query_records):
                smiles    = example.get(cr.SMILES_COL, "").strip()
                true_label = example.get(cr.LABEL_COL, None)
                task_name = cr.get_task_name(example, dataset_name)
                inj    = injection_cache[idx]
                struct = structural_cache[idx]
                docs   = retrieval_cache[idx]
                user_prompt = build_user_prompt_full(
                    smiles, dataset_name, task_name, docs,
                    structural_block=struct,
                    synonyms_block=inj["synonyms_block"],
                    fgs_block=inj["fgs_block"],
                    rdkit_block=inj["rdkit_block"],
                )
                batch_meta.append((idx, smiles, true_label, task_name, qrec,
                                   docs, struct, inj, user_prompt))

            final_outputs = []
            for bs in range(0, len(batch_meta), eff_batch_size):
                batch        = batch_meta[bs: bs + eff_batch_size]
                user_prompts = [b[-1] for b in batch]
                model_outputs = cr.run_inference_batch(
                    SYSTEM_PROMPT, user_prompts,
                    max_new_tokens=max_new,
                    openai_max_workers=args.openai_batch_size,
                )
                for (idx, smiles, true_label, task_name, qrec, docs, struct,
                     inj, _), out in zip(batch, model_outputs):
                    if task_type == "classification":
                        predicted_label = cr.parse_binary_output(out)
                        predicted_value = None
                    else:
                        predicted_label = None
                        predicted_value = cr.parse_regression_output(out)
                    final_outputs.append({
                        "id": idx, "smiles": smiles, "true_label": true_label,
                        "dataset": dataset_name, "task": task_name, "assay": task_name,
                        "condition": f"mcrag_{out_suffix}{pool_tag}_{args.retriever}",
                        "mode": f"mcrag_{out_suffix}{pool_tag}",
                        "model": model_name, "task_type": task_type,
                        "retriever": args.retriever, "k": args.k,
                        "seed": args.seed,
                        "fewshot_pool": args.fewshot_pool,
                        "use_text": use_text, "use_struct": use_struct,
                        "use_synonyms": use_synonyms, "use_fgs": use_fgs, "use_rdkit": use_rdkit,
                        "query_source":       qrec["source"],
                        "good_syns_used":     qrec["good_syns"],
                        "iupac_used":         qrec["iupac"],
                        "task_kw_used":       qrec.get("task_kw", ""),
                        "query":              qrec["query"][:500],
                        "structural_block":   struct[:1000],
                        "injection_synonyms": inj["synonyms_block"][:300],
                        "injection_fgs":      inj["fgs_block"][:300],
                        "injection_rdkit":    inj["rdkit_block"][:500],
                        "model_output":    out,
                        "predicted_label": predicted_label,
                        "predicted_value": predicted_value,
                        "retry_used": False, "retry_output": None,
                        "retrieved_docs": [
                            {"score": d.get("score"),
                             "id": d.get("id", ""),
                             "source": d.get("source", ""),
                             "contents": d.get("contents", "")[:500]}
                            for d in docs
                        ],
                    })
                done = min(bs + eff_batch_size, len(batch_meta))
                print(f"    {done}/{len(batch_meta)} done")

            # Retry nulls
            if args.retry_nulls:
                if task_type == "classification":
                    null_idx = [i for i, r in enumerate(final_outputs)
                                if r["predicted_label"] is None]
                else:
                    null_idx = [i for i, r in enumerate(final_outputs)
                                if r["predicted_value"] is None
                                or not is_sane_regression(r["predicted_value"], dataset_name)]
                n_first_null = len(null_idx)
                if n_first_null > 0:
                    print(f"  Retrying {n_first_null} nulls/garbage with stricter prompt...")
                    retry_prompts = []
                    for i in null_idx:
                        row    = final_outputs[i]
                        docs   = retrieval_cache[row["id"]]
                        struct = structural_cache[row["id"]]
                        inj    = injection_cache[row["id"]]
                        retry_prompts.append(build_user_prompt_full_strict(
                            row["smiles"], dataset_name, row["task"], docs,
                            structural_block=struct,
                            synonyms_block=inj["synonyms_block"],
                            fgs_block=inj["fgs_block"],
                            rdkit_block=inj["rdkit_block"],
                        ))
                    retry_max = 5 if task_type == "classification" else 20
                    n_recovered = 0
                    for rs in range(0, len(retry_prompts), eff_batch_size):
                        rp = retry_prompts[rs: rs + eff_batch_size]
                        ri = null_idx[rs: rs + eff_batch_size]
                        retry_outs = cr.run_inference_batch(
                            SYSTEM_PROMPT_STRICT, rp,
                            max_new_tokens=retry_max,
                            openai_max_workers=args.openai_batch_size,
                        )
                        for orig_idx, retry_out in zip(ri, retry_outs):
                            row = final_outputs[orig_idx]
                            row["retry_used"]   = True
                            row["retry_output"] = retry_out
                            if task_type == "classification":
                                p = cr.parse_binary_output(retry_out)
                                if p is not None:
                                    row["predicted_label"] = p
                                    n_recovered += 1
                            else:
                                p = cr.parse_regression_output(retry_out)
                                if p is not None and is_sane_regression(p, dataset_name):
                                    row["predicted_value"] = p
                                    n_recovered += 1
                                elif p is not None:
                                    row["predicted_value"] = None
                    print(f"  Retry recovered {n_recovered}/{n_first_null}")

            with open(out_path, "w", encoding="utf-8") as f:
                for row in final_outputs:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"  Saved {len(final_outputs)} rows -> {out_path}")
        finally:
            cr.unload_model()


if __name__ == "__main__":
    main()
