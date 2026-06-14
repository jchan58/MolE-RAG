#!/usr/bin/env python
"""
evaluate.py — MCRAG-specific evaluation, seed-aware, with cross-seed aggregation.

Reads JSONL outputs from:
  - <task>/baseline_results/seed_<N>/    (smiles baseline)
  - <task>/chemrag_results/seed_<N>/               (naive BM25)
  - <task>/enhanced_results/seed_<N>/              (query: LLM-filtered syns)
  - <task>/hybrid_results/seed_<N>/                (query: LLM keywords + syns)
  - <task>/raw_synonym_results/seed_<N>/           (query: raw PubChem syns)
  - <task>/<mol_context_results/*>/seed_<N>/      (no retrieval, injection-only)
  - <task>/mcrag_full_results/seed_<N>/            (struct fewshot, random fewshot, full MCRAG)

For each (model, dataset, condition), aggregates across seeds: reports mean ± std.

Primary metric:   classification -> ROC AUC   |  regression -> RMSE   (KANO convention)
Secondary metric: classification -> Macro F1  |  regression -> MAE

Null handling:
  - Classification: nulls forced to wrong class so they stay in denominator
  - Regression: nulls dropped (cannot force a numeric wrong)

Optional GPT-4o-mini rescue parse for outputs the regex couldn't parse; cached
at caches/parse_rescue_cache.json so re-running is free.

Run:
  python evaluate.py                       # all tasks, all modes, all seeds
  python evaluate.py --tasks bbbp bace     # subset
  python evaluate.py --seeds 0 1 2         # subset of seeds (default: all found)
  python evaluate.py --no_rescue           # skip GPT rescue
  python evaluate.py --tables A B          # main result tables only
  python evaluate.py --per_seed            # also print per-seed long table
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

REPO_ROOT     = Path(__file__).resolve().parent.parent  # src/ -> MolE-RAG/
RESCUE_CACHE  = REPO_ROOT / "caches" / "parse_rescue_cache.json"
MC_EVAL_CSV       = REPO_ROOT / "results" / "mc_eval_long.csv"        # per-(file, seed)
MC_EVAL_AGG_CSV   = REPO_ROOT / "results" / "mc_eval_aggregated.csv"  # cross-seed mean/std

load_dotenv(str(REPO_ROOT / ".env"))


# ===========================================================================
# Dataset config + mode catalogue
# ===========================================================================
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

RETRIEVAL_MODES = {
    "chemrag":      ("chemrag_results",       "chemrag_smi"),
    "enhanced":     ("enhanced_results",      "enhanced"),
    "hybrid":       ("hybrid_results",        "hybrid"),
    "raw_synonyms": ("raw_synonym_results",   "raw_syn"),
}

# Legacy structure-modes folders (pre-mcrag_full). Kept for backwards compat;
# if files exist here they'll still be read. With the new pipeline, struct
# fewshot lands in mcrag_full_results/ under "mcrag_struct..." filenames.
STRUCTURE_MODES = {
    "smiles":         ("baseline_results",                      "smiles"),
    "best_fp_legacy": ("structural_fewshot_results",      "best_fp_legacy"),
    "molrag_legacy":  ("molrag_fewshot_results_recovered",                "MolRAG"),
    "random_legacy":  ("random_fewshot_results",                "random_legacy"),
}

PI_MODES = {
    "pi_none":       ("mol_context_results/none",         "pi_none"),
    "pi_syn":        ("mol_context_results/syn",          "pi_syn"),
    "pi_fg":         ("mol_context_results/fg",           "pi_fg"),
    "pi_rdk":        ("mol_context_results/rdk",          "pi_rdk"),
    "pi_syn_fg":     ("mol_context_results/syn_fg",       "pi_syn_fg"),
    "pi_syn_rdk":    ("mol_context_results/syn_rdk",      "pi_syn_rdk"),
    "pi_fg_rdk":     ("mol_context_results/fg_rdk",       "pi_fg_rdk"),
    "pi_syn_fg_rdk": ("mol_context_results/syn_fg_rdk",   "pi_all"),
}

MCRAG_DIR = "mcrag_full_results"

# Labels for filenames matching: <model>_<ds>_mcrag_<suffix>(_random)?_<retr>_k<k>.jsonl
# The suffix captures which pillars are ON (text, struct, syn, fg, rdk).
# Optional _random tag indicates the fewshot pool was random instead of structural.
MCRAG_LABELS = {
    # New canonical conditions (from --no_text --no_synonyms --no_fgs --no_rdkit + fewshot_pool)
    "mcrag_struct":                 "best_fp",         # condition 4: struct-only, structural fewshot
    "mcrag_struct_random":          "random",          # condition 2: struct-only, random fewshot
    # Full / leave-one-out (when text retrieval is also on, etc.)
    "mcrag_full":                   "MCRAG",
    "mcrag_text_struct_syn_fg_rdk": "MCRAG",           # equivalent to full
    "mcrag_struct_syn_fg_rdk":      "-text",
    "mcrag_text_syn_fg_rdk":        "-struct",
    "mcrag_text_struct_fg_rdk":     "-syn",
    "mcrag_text_struct_syn_rdk":    "-fg",
    "mcrag_text_struct_syn_fg":     "-rdk",
    # Progressive ladder
    "mcrag_text":                   "text-only",
    "mcrag_text_struct":            "text+struct",
    "mcrag_text_struct_syn":        "text+struct+syn",
}

TABLE_TO_MODES = {
    "A": ["smiles", "MCRAG"],
    "B": ["smiles", "MCRAG"],
    "1": ["smiles", "chemrag_smi", "hybrid", "raw_syn"],
    "2": ["pi_none", "pi_syn", "pi_fg", "pi_rdk",
          "pi_syn_fg", "pi_syn_rdk", "pi_fg_rdk", "pi_all"],
    "3": ["smiles", "chemrag_smi", "MCRAG", "text-only", "text+struct", "hybrid"],
    "4": ["smiles", "MCRAG", "-text", "-struct", "-syn", "-fg", "-rdk"],
    "5": ["smiles", "random", "MolRAG", "best_fp"],
}

MODEL_ORDER = [
    "Llama-3.2-3B-Instruct",
    "Mistral-7B-Instruct-v0.3",
    "Qwen3-4B-Instruct-2507",
    "ChemLLM-7B-Chat",
    "ChemDFM-v2.0-14B",
    "gpt-4o-mini",
    "gpt-5.4-nano",
]

CLASS_TASKS = ["bbbp", "bace", "clintox", "hiv", "tox21", "sider", "toxcast"]
REG_TASKS   = ["esol", "lipo", "freesolv"]

SEED_DIR_RE = re.compile(r"^seed_(\d+)$")


# ===========================================================================
# Rescue parse
# ===========================================================================
_rescue_cache: Optional[Dict[str, Any]] = None
_openai_client = None
_rescue_calls_made = 0


def load_rescue_cache() -> Dict[str, Any]:
    global _rescue_cache
    if _rescue_cache is not None:
        return _rescue_cache
    if RESCUE_CACHE.exists():
        with open(RESCUE_CACHE) as f:
            _rescue_cache = json.load(f)
        print(f"Loaded rescue cache: {len(_rescue_cache)} entries")
    else:
        _rescue_cache = {}
    return _rescue_cache


def save_rescue_cache():
    if _rescue_cache is None:
        return
    tmp = RESCUE_CACHE.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(_rescue_cache, f)
    os.replace(tmp, RESCUE_CACHE)


def get_openai():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY missing — needed for rescue parse")
        _openai_client = OpenAI(api_key=key)
    return _openai_client


def cache_key(task_type: str, model_output: str) -> str:
    tail = model_output[-1000:] if len(model_output) > 1000 else model_output
    return f"{task_type}::{tail}"


def rescue_classification(model_output: str) -> Optional[str]:
    global _rescue_calls_made
    cache = load_rescue_cache()
    key = cache_key("cls", model_output)
    if key in cache:
        return cache[key]
    tail = model_output[-800:]
    prompt = (
        "You are extracting the FINAL ANSWER from a chemistry model's output.\n\n"
        "The model was asked a yes/no question about a molecule's property, "
        "and instructed to reply with one word: Yes or No.\n\n"
        f"--- Output ---\n{tail}\n--- End ---\n\n"
        "Respond with EXACTLY one word: yes, no, or unclear. Be conservative: "
        "prefer 'unclear' over guessing."
    )
    try:
        resp = get_openai().chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5, temperature=0,
        )
        ans = resp.choices[0].message.content.strip().lower().rstrip(".,!?'\"`")
        _rescue_calls_made += 1
        if ans in {"yes", "y", "1", "true", "positive", "active"}:
            result = "1"
        elif ans in {"no", "n", "0", "false", "negative", "inactive"}:
            result = "0"
        else:
            result = None
    except Exception as e:
        print(f"  rescue error: {str(e)[:100]}")
        return None
    cache[key] = result
    if _rescue_calls_made % 50 == 0 and _rescue_calls_made > 0:
        save_rescue_cache()
    return result


def rescue_regression(model_output: str) -> Optional[float]:
    global _rescue_calls_made
    cache = load_rescue_cache()
    key = cache_key("reg", model_output)
    if key in cache:
        v = cache[key]
        return float(v) if v is not None else None
    tail = model_output[-800:]
    prompt = (
        "Extract the FINAL NUMERIC PREDICTION from this chemistry model output.\n"
        "If unclear, respond 'unclear'.\n\n"
        f"--- Output ---\n{tail}\n--- End ---\n\n"
        "Respond with EXACTLY one of: the number, or 'unclear'."
    )
    try:
        resp = get_openai().chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=15, temperature=0,
        )
        ans = resp.choices[0].message.content.strip()
        _rescue_calls_made += 1
        result: Optional[float] = None
        if "unclear" not in ans.lower():
            try:
                result = float(ans.strip().rstrip(".,!?'\"`"))
            except ValueError:
                m = re.search(r"-?\d+(?:\.\d+)?", ans)
                if m:
                    try:
                        result = float(m.group(0))
                    except ValueError:
                        result = None
    except Exception as e:
        print(f"  rescue error: {str(e)[:100]}")
        return None
    cache[key] = result
    if _rescue_calls_made % 50 == 0 and _rescue_calls_made > 0:
        save_rescue_cache()
    return result


# ===========================================================================
# Metrics
# ===========================================================================
def f1_binary(y_true: List[int], y_pred: List[int], pos: int) -> float:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == pos and p == pos)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != pos and p == pos)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == pos and p != pos)
    if tp == 0 and (fp == 0 or fn == 0):
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def roc_auc_binary(y_true: List[int], y_pred: List[int]) -> Optional[float]:
    """ROC AUC for binary classification.

    The inference pipeline outputs hard labels (0/1), not probabilities.
    With hard binary scores, ROC AUC reduces to balanced accuracy:
        (TPR + TNR) / 2 = (sensitivity + specificity) / 2.

    If a future version of the pipeline captures token-level probabilities,
    this same formula gives the proper ROC AUC. Returns None if only one
    class is present (AUC undefined).
    """
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    if (tp + fn) == 0 or (tn + fp) == 0:
        return None  # only one class present — AUC undefined
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    return (tpr + tnr) / 2


def macro_f1(y_true: List[int], y_pred: List[int]) -> float:
    classes = sorted(set(y_true) | set(y_pred))
    if not classes:
        return 0.0
    return sum(f1_binary(y_true, y_pred, c) for c in classes) / len(classes)


def accuracy(y_true: List[int], y_pred: List[int]) -> float:
    if not y_true:
        return 0.0
    return sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)


def mae(y_true: List[float], y_pred: List[float]) -> float:
    if not y_true:
        return float("nan")
    return sum(abs(t - p) for t, p in zip(y_true, y_pred)) / len(y_true)


def rmse(y_true: List[float], y_pred: List[float]) -> float:
    if not y_true:
        return float("nan")
    return math.sqrt(sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / len(y_true))


# ===========================================================================
# Loading + per-file eval
# ===========================================================================
def parse_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def to_float(x: Any) -> Optional[float]:
    if x is None: return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def to_int_class(x: Any) -> Optional[int]:
    f = to_float(x)
    if f is None: return None
    return int(round(f))


def eval_file(path: Path, do_rescue: bool, mode_label: str,
              seed: Optional[int] = None) -> Optional[Dict[str, Any]]:
    rows = parse_jsonl(path)
    if not rows:
        return None
    dataset   = rows[0].get("dataset")
    task_type = DATASET_TASK_TYPE.get(dataset, "classification")
    model     = rows[0].get("model")

    n_orig_null = 0
    n_rescued   = 0
    per_task_pairs: Dict[str, List[List]] = defaultdict(lambda: [[], []])

    for r in rows:
        out_text  = r.get("model_output") or ""
        if r.get("retry_used") and r.get("retry_output"):
            out_text = r["retry_output"]
        task_name = r.get("task") or r.get("assay") or dataset

        if task_type == "classification":
            pred = r.get("predicted_label")
            if pred is None:
                n_orig_null += 1
                if do_rescue:
                    pred = rescue_classification(out_text)
                    if pred is not None:
                        n_rescued += 1
            true     = to_int_class(r.get("true_label"))
            pred_int = to_int_class(pred)
            if true is None:
                continue
            if pred_int is None:
                pred_int = 1 - true
            per_task_pairs[task_name][0].append(true)
            per_task_pairs[task_name][1].append(pred_int)
        else:
            pred = r.get("predicted_value")
            if pred is None:
                n_orig_null += 1
                if do_rescue:
                    pred = rescue_regression(out_text)
                    if pred is not None:
                        n_rescued += 1
            true = to_float(r.get("true_label"))
            pred = to_float(pred)
            if true is None or pred is None:
                continue
            per_task_pairs[task_name][0].append(true)
            per_task_pairs[task_name][1].append(pred)

    total_n = sum(len(p[0]) for p in per_task_pairs.values())
    if total_n == 0:
        return None

    if task_type == "classification":
        per_task_f1   = {t: macro_f1(p[0], p[1])           for t, p in per_task_pairs.items()}
        per_task_acc  = {t: accuracy(p[0], p[1])           for t, p in per_task_pairs.items()}
        # ROC AUC per task; tasks with only one class present give None and
        # are excluded from the average (otherwise NaN poisons everything).
        per_task_auc_raw = {t: roc_auc_binary(p[0], p[1])  for t, p in per_task_pairs.items()}
        per_task_auc     = {t: a for t, a in per_task_auc_raw.items() if a is not None}
        n_auc_dropped    = len(per_task_auc_raw) - len(per_task_auc)

        f1_avg  = sum(per_task_f1.values())  / len(per_task_f1)  if per_task_f1  else 0.0
        acc_avg = sum(per_task_acc.values()) / len(per_task_acc) if per_task_acc else 0.0
        auc_avg = (sum(per_task_auc.values()) / len(per_task_auc)
                   if per_task_auc else None)
        return {
            "dataset": dataset, "model": model, "mode_label": mode_label,
            "seed": seed, "task_type": task_type, "n": total_n,
            "n_subtasks": len(per_task_pairs),
            "n_subtasks_auc_dropped": n_auc_dropped,
            "roc_auc": auc_avg,
            "macro_f1": f1_avg, "accuracy": acc_avg,
            "mae": None, "rmse": None,
            "n_orig_null": n_orig_null, "n_rescued": n_rescued,
            "path": str(path),
        }
    else:
        all_true, all_pred = [], []
        for p in per_task_pairs.values():
            all_true.extend(p[0]); all_pred.extend(p[1])
        return {
            "dataset": dataset, "model": model, "mode_label": mode_label,
            "seed": seed, "task_type": task_type, "n": total_n,
            "n_subtasks": len(per_task_pairs),
            "roc_auc": None,
            "macro_f1": None, "accuracy": None,
            "mae":  mae(all_true, all_pred),
            "rmse": rmse(all_true, all_pred),
            "n_orig_null": n_orig_null, "n_rescued": n_rescued,
            "path": str(path),
        }


def _scan_dir(d: Path, label: str,
              seeds_filter: Optional[List[int]],
              out: List[Tuple[Path, str, Optional[int]]]):
    """Scan d for *.jsonl. Walks seed_N subdirs if present; also picks up
    any *.jsonl directly in d (backwards compat for pre-seed runs)."""
    if not d.is_dir():
        return
    # Files directly in d (no seed) — backwards compat
    for p in sorted(d.glob("*.jsonl")):
        out.append((p, label, None))
    # seed_N subdirs
    for sub in sorted(d.iterdir()):
        if sub.is_dir():
            m = SEED_DIR_RE.match(sub.name)
            if m:
                seed = int(m.group(1))
                if seeds_filter is not None and seed not in seeds_filter:
                    continue
                for p in sorted(sub.glob("*.jsonl")):
                    out.append((p, label, seed))


def discover_files(tasks: List[str],
                   mode_filter: Optional[List[str]] = None,
                   seeds_filter: Optional[List[int]] = None
                   ) -> List[Tuple[Path, str, Optional[int]]]:
    """Walk all known result directories. Returns list of (path, mode_label, seed)."""
    out: List[Tuple[Path, str, Optional[int]]] = []

    def keep(label: str) -> bool:
        return mode_filter is None or label in mode_filter

    for task in tasks:
        base = REPO_ROOT / "results" / task

        for _, (subdir, label) in RETRIEVAL_MODES.items():
            if not keep(label): continue
            _scan_dir(base / subdir, label, seeds_filter, out)

        for _, (subdir, label) in STRUCTURE_MODES.items():
            if not keep(label): continue
            _scan_dir(base / subdir, label, seeds_filter, out)

        for _, (subdir, label) in PI_MODES.items():
            if not keep(label): continue
            _scan_dir(base / subdir, label, seeds_filter, out)

        # MCRAG full results: label is computed from filename pattern.
        # We need to scan all files first and then filter by computed label.
        mcrag_base = base / MCRAG_DIR
        if mcrag_base.is_dir():
            # Files directly in mcrag_base (no seed)
            for p in sorted(mcrag_base.glob("*.jsonl")):
                label = _label_from_mcrag_filename(p.name)
                if keep(label):
                    out.append((p, label, None))
            # seed_N subdirs
            for sub in sorted(mcrag_base.iterdir()):
                if sub.is_dir():
                    m = SEED_DIR_RE.match(sub.name)
                    if m:
                        seed = int(m.group(1))
                        if seeds_filter is not None and seed not in seeds_filter:
                            continue
                        for p in sorted(sub.glob("*.jsonl")):
                            label = _label_from_mcrag_filename(p.name)
                            if keep(label):
                                out.append((p, label, seed))
    return out


def _label_from_mcrag_filename(name: str) -> str:
    """Compute mode label from an mcrag_full_results filename.

    Examples:
      gpt_4o_mini_bbbp_mcrag_struct_bm25_k5.jsonl
        -> suffix "struct"        -> label "best_fp"
      gpt_4o_mini_bbbp_mcrag_struct_random_bm25_k5.jsonl
        -> suffix "struct_random" -> label "random"
      gpt_4o_mini_bbbp_mcrag_full_bm25_k5.jsonl
        -> suffix "full"          -> label "MCRAG"
    """
    m = re.search(r"_mcrag_(.+?)_(bm25|contriever|specter|e5|rrf)_k", name)
    if not m:
        return "mcrag_unknown"
    suffix = m.group(1)
    full_key = f"mcrag_{suffix}"
    return MCRAG_LABELS.get(full_key, full_key)


# ===========================================================================
# Cross-seed aggregation
# ===========================================================================
def _mean(xs):
    return statistics.mean(xs) if xs else None


def _std(xs):
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def aggregate_across_seeds(per_seed: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group by (dataset, model, mode_label) and compute mean ± std across seeds."""
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in per_seed:
        key = (r["dataset"], r["model"] or "", r["mode_label"])
        groups[key].append(r)

    out = []
    for (dataset, model, mode_label), rs in groups.items():
        task_type = rs[0]["task_type"]
        seeds_present = sorted([r["seed"] for r in rs if r["seed"] is not None])
        agg = {
            "dataset": dataset, "model": model, "mode_label": mode_label,
            "task_type": task_type,
            "n_seeds": len(rs),
            "seeds": seeds_present,
            "n_total": sum(r["n"] for r in rs),
            "n_orig_null": sum(r["n_orig_null"] for r in rs),
            "n_rescued":   sum(r["n_rescued"]   for r in rs),
        }
        if task_type == "classification":
            auc_vals = [r["roc_auc"]  for r in rs if r.get("roc_auc")  is not None]
            f1_vals  = [r["macro_f1"] for r in rs if r["macro_f1"] is not None]
            acc_vals = [r["accuracy"] for r in rs if r["accuracy"] is not None]
            agg["roc_auc"]       = _mean(auc_vals)
            agg["roc_auc_std"]   = _std(auc_vals)
            agg["macro_f1"]      = _mean(f1_vals)
            agg["macro_f1_std"]  = _std(f1_vals)
            agg["accuracy"]      = _mean(acc_vals)
            agg["accuracy_std"]  = _std(acc_vals)
            agg["mae"] = agg["rmse"] = None
            agg["mae_std"] = agg["rmse_std"] = None
        else:
            mae_vals  = [r["mae"]  for r in rs if r["mae"]  is not None]
            rmse_vals = [r["rmse"] for r in rs if r["rmse"] is not None]
            agg["mae"]      = _mean(mae_vals)
            agg["mae_std"]  = _std(mae_vals)
            agg["rmse"]     = _mean(rmse_vals)
            agg["rmse_std"] = _std(rmse_vals)
            agg["roc_auc"] = None
            agg["roc_auc_std"] = None
            agg["macro_f1"] = agg["accuracy"] = None
            agg["macro_f1_std"] = agg["accuracy_std"] = None
        out.append(agg)
    return out


# ===========================================================================
# Formatting helpers
# ===========================================================================
def _primary_metric(r: Dict[str, Any]) -> Optional[float]:
    """ROC AUC for classification, RMSE for regression.

    Matches KANO/GROVER/MolRAG convention: RMSE is the headline number for
    ESOL/FreeSolv/Lipophilicity (MAE is reported only for the QM datasets).
    """
    if r is None: return None
    return r.get("roc_auc") if r["task_type"] == "classification" else r.get("rmse")


def _primary_std(r: Dict[str, Any]) -> Optional[float]:
    if r is None: return None
    return r.get("roc_auc_std") if r["task_type"] == "classification" else r.get("rmse_std")


def _secondary_metric(r: Dict[str, Any]) -> Optional[float]:
    """MAE for regression (supplementary), macro F1 for classification (supplementary)."""
    if r is None: return None
    return r.get("macro_f1") if r["task_type"] == "classification" else r.get("mae")


def _secondary_std(r: Dict[str, Any]) -> Optional[float]:
    if r is None: return None
    return r.get("macro_f1_std") if r["task_type"] == "classification" else r.get("mae_std")


def _fmt(v: Optional[float], prec: int = 3) -> str:
    return "  --" if v is None else f"{v:.{prec}f}"


def _fmt_pm(mean: Optional[float], std: Optional[float], prec: int = 3) -> str:
    """Format 'mean±std' or just 'mean' if std missing/zero."""
    if mean is None:
        return "    --"
    if std is None or std == 0.0:
        return f"{mean:.{prec}f}"
    return f"{mean:.{prec}f}±{std:.{prec}f}"


def _arrow(delta: float, task_type: str) -> str:
    if task_type == "classification":
        return "↑" if delta > 0 else ("↓" if delta < 0 else "·")
    return "↓" if delta < 0 else ("↑" if delta > 0 else "·")


def _delta_str(value: Optional[float], base: Optional[float], task_type: str) -> str:
    if value is None or base is None:
        return ""
    d = value - base
    return f"{d:+.3f}{_arrow(d, task_type)}"


def _build_lookup(results: List[Dict[str, Any]], task_type_filter: str
                  ) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Build {model_short: {dataset: {mode_label: result}}} for one task type."""
    lookup: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))
    for r in results:
        if r["task_type"] != task_type_filter:
            continue
        model = (r["model"] or "").split("/")[-1]
        lookup[model][r["dataset"]][r["mode_label"]] = r
    return lookup


def _group_by_task_model(results: List[Dict[str, Any]]
                          ) -> Dict[Tuple[str, str], Dict[str, Dict[str, Any]]]:
    out: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for r in results:
        model_short = (r["model"] or "").split("/")[-1]
        out[(r["dataset"], model_short)][r["mode_label"]] = r
    return out


# ===========================================================================
# Main Table A — Binary Classification (F1 ± std, smiles vs MCRAG)
# ===========================================================================
def print_main_table_classification(results: List[Dict[str, Any]]):
    lookup = _build_lookup(results, "classification")
    COL_W = 17
    W = 28 + 8 + (COL_W + 1) * len(CLASS_TASKS)

    print(f"\n{'='*W}")
    print(f"{'MAIN TABLE A — Classification: ROC AUC ± std  (SMILES vs MCRAG)':^{W}}")
    print(f"{'='*W}")

    hdr = f"{'Model':<28} {'Method':<8}"
    for t in CLASS_TASKS:
        hdr += f" {t.upper():>{COL_W}}"
    print(hdr)
    print("-" * W)

    for model in MODEL_ORDER:
        if model not in lookup:
            continue
        for mode_label, mode_name in [("smiles", "SMILES"), ("MCRAG", "MCRAG")]:
            row = f"{model[:28]:<28} {mode_name:<8}"
            for task in CLASS_TASKS:
                r = lookup[model][task].get(mode_label)
                cell = _fmt_pm(_primary_metric(r), _primary_std(r))
                row += f" {cell:>{COL_W}}"
            print(row)
        print()


# ===========================================================================
# Main Table B — Regression (MAE ± std, smiles vs MCRAG)
# ===========================================================================
def print_main_table_regression(results: List[Dict[str, Any]]):
    lookup = _build_lookup(results, "regression")
    COL_W = 17
    W = 28 + 8 + (COL_W + 1) * len(REG_TASKS)

    print(f"\n{'='*W}")
    print(f"{'MAIN TABLE B — Regression: RMSE ± std  (SMILES vs MCRAG)':^{W}}")
    print(f"{'='*W}")

    hdr = f"{'Model':<28} {'Method':<8}"
    for t in REG_TASKS:
        hdr += f" {t.upper():>{COL_W}}"
    print(hdr)
    print("-" * W)

    for model in MODEL_ORDER:
        if model not in lookup:
            continue
        for mode_label, mode_name in [("smiles", "SMILES"), ("MCRAG", "MCRAG")]:
            row = f"{model[:28]:<28} {mode_name:<8}"
            for task in REG_TASKS:
                r = lookup[model][task].get(mode_label)
                cell = _fmt_pm(_primary_metric(r), _primary_std(r))
                row += f" {cell:>{COL_W}}"
            print(row)
        print()


# ===========================================================================
# Main Table A2 / B2 — Supplementary metrics
#   A2: Classification — Macro F1 ± std (supplementary)
#   B2: Regression     — RMSE ± std
# ===========================================================================
def print_main_table_secondary(results: List[Dict[str, Any]]):
    """Supplementary metric tables: Macro F1 for classification, MAE for regression."""
    COL_W = 17

    # Classification — Macro F1 (supplementary)
    lookup_cls = _build_lookup(results, "classification")
    W = 28 + 8 + (COL_W + 1) * len(CLASS_TASKS)
    print(f"\n{'='*W}")
    print(f"{'MAIN TABLE A2 — Classification: Macro F1 ± std (supplementary)':^{W}}")
    print(f"{'='*W}")
    hdr = f"{'Model':<28} {'Method':<8}"
    for t in CLASS_TASKS:
        hdr += f" {t.upper():>{COL_W}}"
    print(hdr)
    print("-" * W)
    for model in MODEL_ORDER:
        if model not in lookup_cls:
            continue
        for mode_label, mode_name in [("smiles", "SMILES"), ("MCRAG", "MCRAG")]:
            row = f"{model[:28]:<28} {mode_name:<8}"
            for task in CLASS_TASKS:
                r = lookup_cls[model][task].get(mode_label)
                # Supplementary = macro_f1 for classification
                m  = r.get("macro_f1") if r else None
                ms = r.get("macro_f1_std") if r else None
                cell = _fmt_pm(m, ms)
                row += f" {cell:>{COL_W}}"
            print(row)
        print()

    # Regression — MAE (supplementary, while KANO/MolRAG use RMSE as primary)
    lookup_reg = _build_lookup(results, "regression")
    W = 28 + 8 + (COL_W + 1) * len(REG_TASKS)
    print(f"\n{'='*W}")
    print(f"{'MAIN TABLE B2 — Regression: MAE ± std (supplementary)':^{W}}")
    print(f"{'='*W}")
    hdr = f"{'Model':<28} {'Method':<8}"
    for t in REG_TASKS:
        hdr += f" {t.upper():>{COL_W}}"
    print(hdr)
    print("-" * W)
    for model in MODEL_ORDER:
        if model not in lookup_reg:
            continue
        for mode_label, mode_name in [("smiles", "SMILES"), ("MCRAG", "MCRAG")]:
            row = f"{model[:28]:<28} {mode_name:<8}"
            for task in REG_TASKS:
                r = lookup_reg[model][task].get(mode_label)
                m  = r.get("mae") if r else None
                ms = r.get("mae_std") if r else None
                cell = _fmt_pm(m, ms)
                row += f" {cell:>{COL_W}}"
            print(row)
        print()


# ===========================================================================
# Tables 1-5 — Ablation tables (mean only, no std, to keep columns narrow)
# ===========================================================================
def print_table_1_retrieval(results):
    grouped = _group_by_task_model(results)
    W = 140
    print(f"\n{'='*W}")
    print(f"{'TABLE 1 — Text-retrieval ablation (mean over seeds, vs smiles)':^{W}}")
    print(f"{'='*W}")
    print(f"{'task':<10} {'model':<26} {'smiles':>9} {'chemrag':>9} {'Δchem':>9} "
          f"{'hybrid':>9} {'Δhyb':>9} {'raw_syn':>9} {'Δraw':>9}")
    print("-" * W)
    for (task, model), modes in sorted(grouped.items()):
        ttype = DATASET_TASK_TYPE.get(task, "classification")
        s   = _primary_metric(modes.get("smiles"))
        cs  = _primary_metric(modes.get("chemrag_smi"))
        h   = _primary_metric(modes.get("hybrid"))
        rw  = _primary_metric(modes.get("raw_syn"))
        if all(x is None for x in (s, cs, h, rw)):
            continue
        print(f"{task:<10} {model[:26]:<26} {_fmt(s):>9} {_fmt(cs):>9} "
              f"{_delta_str(cs, s, ttype):>9} {_fmt(h):>9} {_delta_str(h, s, ttype):>9} "
              f"{_fmt(rw):>9} {_delta_str(rw, s, ttype):>9}")


def print_table_2_promptinject(results):
    grouped = _group_by_task_model(results)
    W = 145
    print(f"\n{'='*W}")
    print(f"{'TABLE 2 — Prompt-injection ablation (no retrieval, mean over seeds, vs pi_none)':^{W}}")
    print(f"{'='*W}")
    print(f"{'task':<10} {'model':<22} {'pi_none':>9} "
          f"{'pi_syn':>9} {'pi_fg':>9} {'pi_rdk':>9} "
          f"{'syn_fg':>9} {'syn_rdk':>9} {'fg_rdk':>9} {'pi_all':>9} {'Δ best':>10}")
    print("-" * W)
    cols = ["pi_none", "pi_syn", "pi_fg", "pi_rdk",
            "pi_syn_fg", "pi_syn_rdk", "pi_fg_rdk", "pi_all"]
    for (task, model), modes in sorted(grouped.items()):
        ttype = DATASET_TASK_TYPE.get(task, "classification")
        vals = {c: _primary_metric(modes.get(c)) for c in cols}
        if all(v is None for v in vals.values()):
            continue
        base = vals["pi_none"]
        non_base = [v for c, v in vals.items() if c != "pi_none" and v is not None]
        if non_base and base is not None:
            best = max(non_base) if ttype == "classification" else min(non_base)
            d = _delta_str(best, base, ttype)
        else:
            d = ""
        cells = " ".join(f"{_fmt(vals[c]):>9}" for c in cols[1:])
        print(f"{task:<10} {model[:22]:<22} {_fmt(base):>9} {cells} {d:>10}")


def print_table_3_mcrag_ladder(results):
    grouped = _group_by_task_model(results)
    W = 130
    print(f"\n{'='*W}")
    print(f"{'TABLE 3 — MCRAG ladder (mean over seeds, vs smiles)':^{W}}")
    print(f"{'='*W}")
    print(f"{'task':<10} {'model':<26} {'smiles':>9} "
          f"{'text-only':>10} {'text+struct':>12} {'MCRAG':>10} {'Δ MCRAG':>10}")
    print("-" * W)
    for (task, model), modes in sorted(grouped.items()):
        ttype = DATASET_TASK_TYPE.get(task, "classification")
        s = _primary_metric(modes.get("smiles"))
        if s is None:
            s = _primary_metric(modes.get("chemrag_smi"))
        t  = _primary_metric(modes.get("text-only")) or _primary_metric(modes.get("hybrid"))
        ts = _primary_metric(modes.get("text+struct"))
        f  = _primary_metric(modes.get("MCRAG"))
        if all(x is None for x in (s, t, ts, f)):
            continue
        print(f"{task:<10} {model[:26]:<26} {_fmt(s):>9} "
              f"{_fmt(t):>10} {_fmt(ts):>12} {_fmt(f):>10} {_delta_str(f, s, ttype):>10}")


def print_table_4_pillar_contribution(results):
    grouped = _group_by_task_model(results)
    W = 130
    print(f"\n{'='*W}")
    print(f"{'TABLE 4 — Per-pillar contribution (full minus each pillar)':^{W}}")
    print(f"{'='*W}")
    print(f"{'task':<10} {'model':<26} {'MCRAG':>10} "
          f"{'-text':>10} {'-struct':>10} {'-syn':>10} {'-fg':>10} {'-rdk':>10}")
    print("-" * W)
    for (task, model), modes in sorted(grouped.items()):
        ttype = DATASET_TASK_TYPE.get(task, "classification")
        full = _primary_metric(modes.get("MCRAG"))
        nt   = _primary_metric(modes.get("-text"))
        ns   = _primary_metric(modes.get("-struct"))
        nsy  = _primary_metric(modes.get("-syn"))
        nf   = _primary_metric(modes.get("-fg"))
        nr   = _primary_metric(modes.get("-rdk"))
        if all(x is None for x in (full, nt, ns, nsy, nf, nr)):
            continue
        def contrib(abl):
            if full is None or abl is None: return ""
            d = full - abl
            if ttype == "regression":
                arrow = "↑" if d < 0 else ("↓" if d > 0 else "·")
            else:
                arrow = "↑" if d > 0 else ("↓" if d < 0 else "·")
            return f"{d:+.3f}{arrow}"
        print(f"{task:<10} {model[:26]:<26} {_fmt(full):>10} "
              f"{contrib(nt):>10} {contrib(ns):>10} {contrib(nsy):>10} "
              f"{contrib(nf):>10} {contrib(nr):>10}")


def print_table_5_structure_retrieval(results):
    grouped = _group_by_task_model(results)
    W = 140
    print(f"\n{'='*W}")
    print(f"{'TABLE 5 — Structure-retrieval ablation (mean over seeds, vs smiles)':^{W}}")
    print(f"{'='*W}")
    print(f"{'task':<10} {'model':<26} {'smiles':>9} "
          f"{'random':>9} {'Δrand':>9} "
          f"{'MolRAG':>9} {'Δmolrag':>9} "
          f"{'best_fp':>9} {'Δbest_fp':>10}")
    print("-" * W)
    for (task, model), modes in sorted(grouped.items()):
        ttype = DATASET_TASK_TYPE.get(task, "classification")
        s  = _primary_metric(modes.get("smiles"))
        r  = _primary_metric(modes.get("random"))
        m  = _primary_metric(modes.get("MolRAG"))
        bf = _primary_metric(modes.get("best_fp"))
        if all(x is None for x in (s, r, m, bf)):
            continue
        print(f"{task:<10} {model[:26]:<26} {_fmt(s):>9} "
              f"{_fmt(r):>9} {_delta_str(r, s, ttype):>9} "
              f"{_fmt(m):>9} {_delta_str(m, s, ttype):>9} "
              f"{_fmt(bf):>9} {_delta_str(bf, s, ttype):>10}")


# ===========================================================================
# Long-format listing + CSV
# ===========================================================================
def print_long_per_seed(per_seed: List[Dict[str, Any]]):
    print()
    print("Per-seed long format (raw):")
    print(f"{'task':<10} {'mode':<20} {'model':<28} {'seed':>5} "
          f"{'n':>6} {'AUC/RMSE':>9} {'F1/MAE':>9} {'null':>6} {'rescued':>8}")
    print("-" * 115)
    for r in sorted(per_seed, key=lambda x: (x["dataset"], x["mode_label"],
                                              x["model"] or "", x.get("seed") or -1)):
        primary   = _fmt(r.get("roc_auc") if r["task_type"] == "classification"
                         else r.get("rmse"))
        secondary = _fmt(r.get("macro_f1") if r["task_type"] == "classification"
                         else r.get("mae"))
        model_short = (r["model"] or "").split("/")[-1][:28]
        seed_str = "" if r.get("seed") is None else str(r["seed"])
        print(f"{r['dataset']:<10} {r['mode_label']:<20} {model_short:<28} "
              f"{seed_str:>5} {r['n']:>6} {primary:>9} {secondary:>9} "
              f"{r['n_orig_null']:>6} {r['n_rescued']:>8}")


def print_long_aggregated(agg: List[Dict[str, Any]]):
    print()
    print("Aggregated across seeds (mean ± std):")
    print(f"{'task':<10} {'mode':<20} {'model':<28} {'#seed':>5} {'n_total':>8} "
          f"{'AUC/RMSE ± std':>20} {'F1/MAE ± std':>20}")
    print("-" * 130)
    for r in sorted(agg, key=lambda x: (x["dataset"], x["mode_label"], x["model"] or "")):
        primary = _fmt_pm(
            r.get("roc_auc")     if r["task_type"] == "classification" else r.get("rmse"),
            r.get("roc_auc_std") if r["task_type"] == "classification" else r.get("rmse_std"),
        )
        secondary = _fmt_pm(
            r.get("macro_f1")     if r["task_type"] == "classification" else r.get("mae"),
            r.get("macro_f1_std") if r["task_type"] == "classification" else r.get("mae_std"),
        )
        model_short = (r["model"] or "").split("/")[-1][:28]
        print(f"{r['dataset']:<10} {r['mode_label']:<20} {model_short:<28} "
              f"{r['n_seeds']:>5} {r['n_total']:>8} {primary:>20} {secondary:>20}")


def write_per_seed_csv(per_seed: List[Dict[str, Any]], path: Path):
    import csv as _csv
    cols = ["dataset", "mode_label", "model", "seed", "task_type", "n", "n_subtasks",
            "roc_auc", "macro_f1", "accuracy", "mae", "rmse",
            "n_orig_null", "n_rescued", "path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(per_seed, key=lambda x: (x["dataset"], x["mode_label"],
                                                  x["model"] or "", x.get("seed") or -1)):
            w.writerow({k: r.get(k) for k in cols})
    print(f"\nWrote per-seed CSV: {path}")


def write_aggregated_csv(agg: List[Dict[str, Any]], path: Path):
    import csv as _csv
    cols = ["dataset", "mode_label", "model", "task_type", "n_seeds", "seeds", "n_total",
            "roc_auc", "roc_auc_std",
            "macro_f1", "macro_f1_std", "accuracy", "accuracy_std",
            "mae", "mae_std", "rmse", "rmse_std",
            "n_orig_null", "n_rescued"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(agg, key=lambda x: (x["dataset"], x["mode_label"], x["model"] or "")):
            row = {k: r.get(k) for k in cols}
            if isinstance(row.get("seeds"), list):
                row["seeds"] = ",".join(str(s) for s in row["seeds"])
            w.writerow(row)
    print(f"Wrote aggregated CSV: {path}")


# ===========================================================================
# Main
# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=list(DATASET_TASK_TYPE.keys()))
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="restrict to these seeds (default: all found)")
    ap.add_argument("--no_rescue", action="store_true")
    ap.add_argument("--out_csv", default=str(MC_EVAL_CSV))
    ap.add_argument("--out_agg_csv", default=str(MC_EVAL_AGG_CSV))
    ap.add_argument("--per_seed", action="store_true",
                    help="also print per-seed long table (default: aggregated only)")
    ap.add_argument("--no_long", action="store_true",
                    help="skip the long-format table entirely")
    ap.add_argument("--tables", nargs="+", default=["A", "B", "1", "2", "3", "4", "5"],
                    choices=["A", "B", "1", "2", "3", "4", "5"],
                    help="which tables to print (A=class main, B=reg main, 1-5=ablations)")
    ap.add_argument("--modes", nargs="+", default=None,
                    help="only load files with these mode labels. "
                         "If not given, auto-derived from --tables.")
    args = ap.parse_args()

    do_rescue = not args.no_rescue
    if do_rescue:
        load_rescue_cache()

    # Auto-derive mode filter from requested tables
    mode_filter = args.modes
    if mode_filter is None:
        required = set()
        for t in args.tables:
            required.update(TABLE_TO_MODES.get(t, []))
        if required:
            mode_filter = sorted(required)
            print(f"Tables {args.tables} -> auto-filtering modes: {mode_filter}")

    files = discover_files(args.tasks, mode_filter=mode_filter,
                           seeds_filter=args.seeds)
    print(f"Found {len(files)} (file, seed) entries to evaluate.")
    if not files:
        return

    # Per-seed evaluation
    per_seed: List[Dict[str, Any]] = []
    for i, (path, mode_label, seed) in enumerate(files, start=1):
        seed_tag = f"seed={seed}" if seed is not None else "no-seed"
        print(f"[{i}/{len(files)}] {mode_label:18s} {seed_tag:10s} {path.name}")
        try:
            r = eval_file(path, do_rescue=do_rescue, mode_label=mode_label, seed=seed)
            if r is not None:
                per_seed.append(r)
        except Exception as e:
            print(f"  ERROR: {e}")

    if do_rescue:
        save_rescue_cache()
        print(f"\nRescue parse: {_rescue_calls_made} new GPT-4o-mini calls "
              f"(total cache size: {len(load_rescue_cache())})")

    # Aggregate
    agg = aggregate_across_seeds(per_seed)
    print(f"\nAggregation: {len(per_seed)} per-seed evals -> {len(agg)} "
          f"(dataset, model, mode) groups")

    # Display
    if not args.no_long:
        if args.per_seed:
            print_long_per_seed(per_seed)
        print_long_aggregated(agg)

    # Tables use aggregated results
    if "A" in args.tables:
        print_main_table_classification(agg)
    if "B" in args.tables:
        print_main_table_regression(agg)
    if "A" in args.tables or "B" in args.tables:
        print_main_table_secondary(agg)
    if "1" in args.tables: print_table_1_retrieval(agg)
    if "5" in args.tables: print_table_5_structure_retrieval(agg)
    if "2" in args.tables: print_table_2_promptinject(agg)
    if "3" in args.tables: print_table_3_mcrag_ladder(agg)
    if "4" in args.tables: print_table_4_pillar_contribution(agg)

    # CSVs
    write_per_seed_csv(per_seed, Path(args.out_csv))
    write_aggregated_csv(agg, Path(args.out_agg_csv))


if __name__ == "__main__":
    main()