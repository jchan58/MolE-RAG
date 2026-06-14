#!/usr/bin/env python
"""
evaluate.py — compute metrics from chemrag_retrieval.py JSONL outputs.

For each (task, model, mode, retriever) JSONL:
  1. Read predictions
  2. For rows where predicted_label / predicted_value is null, attempt a
     "rescue parse" using gpt-4o-mini to extract Yes/No or a number from
     the raw model_output. Cached at caches/parse_rescue_cache.json
     so we never pay twice for the same output text.
  3. Compute metrics:
       Classification → macro-F1, accuracy (per-task-name then averaged
                        for multi-task datasets like Tox21/SIDER/ToxCast)
       Regression    → MAE, RMSE
  4. Print wide comparison table (chemrag vs enhanced per model)
  5. Save long-format CSV with every (task, model, mode) row

Run:
  python evaluate.py
  python evaluate.py --no_rescue              # skip GPT-4o-mini rescue
  python evaluate.py --tasks bbbp bace        # only some tasks
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

REPO_ROOT     = Path(__file__).resolve().parent.parent  # src/ -> MolE-RAG/
RESCUE_CACHE  = REPO_ROOT / "caches" / "parse_rescue_cache.json"
EVAL_LONG_CSV = REPO_ROOT / "results" / "eval_long.csv"

load_dotenv(str(REPO_ROOT / ".env"))

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
# Rescue parse via gpt-4o-mini
# =========================
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
    """Returns '1', '0', or None."""
    global _rescue_calls_made
    cache = load_rescue_cache()
    key = cache_key("cls", model_output)
    if key in cache:
        return cache[key]
    tail = model_output[-800:]
    prompt = (
        "You are extracting the FINAL ANSWER from a chemistry model's output.\n\n"
        "The model was asked a yes/no question about whether a molecule has a "
        "particular property, and instructed to reply with one word: Yes or No.\n\n"
        "The output may contain:\n"
        "  - Retrieved scientific context (IGNORE; this is background, NOT the answer)\n"
        "  - Echoed prompt text or template markers (e.g. 'assistant')\n"
        "  - The model's actual final answer (usually at the very end)\n\n"
        "Your job: identify the model's OWN final answer to the question.\n"
        "If the model didn't actually answer, or if the output is ambiguous, "
        "respond with 'unclear' — do NOT guess.\n\n"
        "Examples:\n"
        "  Output: 'a bunch of chemistry text ... assistant No'\n"
        "  → no\n\n"
        "  Output: 'Yes'\n"
        "  → yes\n\n"
        "  Output: 'No.'\n"
        "  → no\n\n"
        "  Output: 'The molecule is similar to caffeine, which is known to cross "
        "the BBB. assistant Yes'\n"
        "  → yes\n\n"
        "  Output: 'It depends on several factors including lipophilicity'\n"
        "  → unclear\n\n"
        "  Output: 'context mentioning yes and no in passing, no clear answer'\n"
        "  → unclear\n\n"
        "  Output: '... yes the paper says it does cross... assistant No'\n"
        "  → no  (the model's own answer at the end overrides text in context)\n\n"
        f"--- Output to analyze ---\n{tail}\n--- End ---\n\n"
        "Respond with EXACTLY one word: yes, no, or unclear. Be conservative: "
        "prefer 'unclear' over guessing."
    )
    try:
        resp = get_openai().chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0,
        )
        ans = resp.choices[0].message.content.strip().lower().rstrip(".,!?'\"`")
        _rescue_calls_made += 1
        result = None
        if ans in {"yes", "y", "1", "true", "positive", "active"}:
            result = "1"
        elif ans in {"no", "n", "0", "false", "negative", "inactive"}:
            result = "0"
        # any other response (including "unclear") → keep as None
    except Exception as e:
        print(f"  rescue error: {str(e)[:100]}")
        result = None
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
        "You are extracting the FINAL NUMERIC PREDICTION from a chemistry "
        "regression model's output.\n\n"
        "The model was asked to predict a numeric property value and "
        "instructed to reply with one number.\n\n"
        "The output may contain:\n"
        "  - Retrieved scientific context with various numbers (IGNORE these)\n"
        "  - Tables, citations, SMILES (IGNORE)\n"
        "  - The model's actual final numeric prediction (usually at the very end)\n\n"
        "Your job: identify the model's OWN final predicted number.\n"
        "If the model didn't give a clear numeric answer, respond with 'unclear'.\n\n"
        "Examples:\n"
        "  Output: '-2.34'\n"
        "  → -2.34\n\n"
        "  Output: 'a bunch of context ... assistant 1.7'\n"
        "  → 1.7\n\n"
        "  Output: 'The molecule has logP ~3, with related compounds at 2.1, 4.5, "
        "and 1.8. assistant -0.5'\n"
        "  → -0.5  (the final answer, not the context numbers)\n\n"
        "  Output: 'Predicted value: 2.45'\n"
        "  → 2.45\n\n"
        "  Output: 'It depends on the solvent system'\n"
        "  → unclear\n\n"
        "  Output: 'Cannot determine without more information'\n"
        "  → unclear\n\n"
        f"--- Output to analyze ---\n{tail}\n--- End ---\n\n"
        "Respond with EXACTLY one of: the number (e.g. '-2.34' or '1.7'), "
        "or the word 'unclear'. Be conservative: prefer 'unclear' over "
        "guessing from context numbers."
    )
    try:
        resp = get_openai().chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=15,
            temperature=0,
        )
        ans = resp.choices[0].message.content.strip()
        _rescue_calls_made += 1
        result: Optional[float] = None
        if "unclear" not in ans.lower():
            # The whole response should BE the number — try parsing it directly
            cleaned = ans.strip().rstrip(".,!?'\"`")
            try:
                result = float(cleaned)
            except ValueError:
                # Fallback: pull the first signed number from the response
                m = re.search(r"-?\d+(?:\.\d+)?", ans)
                if m:
                    try:
                        result = float(m.group(0))
                    except ValueError:
                        result = None
    except Exception as e:
        print(f"  rescue error: {str(e)[:100]}")
        result = None
    cache[key] = result
    if _rescue_calls_made % 50 == 0 and _rescue_calls_made > 0:
        save_rescue_cache()
    return result


# =========================
# Metrics
# =========================
def f1_binary(y_true: List[int], y_pred: List[int], positive_class: int) -> float:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == positive_class and p == positive_class)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != positive_class and p == positive_class)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == positive_class and p != positive_class)
    if tp == 0 and (fp == 0 or fn == 0):
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def macro_f1(y_true: List[int], y_pred: List[int]) -> float:
    # binary classification, macro across the two classes
    classes = sorted(set(y_true) | set(y_pred))
    if not classes:
        return 0.0
    f1s = [f1_binary(y_true, y_pred, c) for c in classes]
    return sum(f1s) / len(f1s)


def accuracy(y_true: List[int], y_pred: List[int]) -> float:
    if not y_true:
        return 0.0
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def mae(y_true: List[float], y_pred: List[float]) -> float:
    if not y_true:
        return float("nan")
    return sum(abs(t - p) for t, p in zip(y_true, y_pred)) / len(y_true)


def rmse(y_true: List[float], y_pred: List[float]) -> float:
    if not y_true:
        return float("nan")
    return math.sqrt(sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / len(y_true))


# =========================
# Loading + per-file eval
# =========================
def find_jsonls(tasks: List[str], modes: List[str]) -> List[Path]:
    mode_to_subdir = {
        "chemrag":      "chemrag_results",
        "enhanced":     "enhanced_results",
        "raw_synonyms": "raw_synonym_results",
        "hybrid":       "hybrid_results",
    }
    out = []
    for task in tasks:
        for mode in modes:
            subdir = mode_to_subdir.get(mode)
            if subdir is None:
                continue
            d = REPO_ROOT / "results" / task / subdir
            if d.is_dir():
                for p in sorted(d.glob("*.jsonl")):
                    out.append(p)
    return out


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


def eval_file(path: Path, do_rescue: bool) -> Dict[str, Any]:
    rows = parse_jsonl(path)
    if not rows:
        return {"path": str(path), "n": 0, "error": "empty"}

    task_type = DATASET_TASK_TYPE.get(rows[0].get("dataset"), "classification")
    dataset = rows[0].get("dataset")
    model = rows[0].get("model")
    mode = rows[0].get("mode", "chemrag")
    retriever = rows[0].get("retriever", "bm25")
    k = rows[0].get("k", 5)

    n_orig_null = 0
    n_rescued = 0

    # For classification multi-task datasets, group by assay/task
    per_task_pairs: Dict[str, Tuple[List, List]] = defaultdict(lambda: ([], []))

    for r in rows:
        out_text = r.get("model_output") or ""
        task_name = r.get("task") or r.get("assay") or dataset

        if task_type == "classification":
            pred = r.get("predicted_label")
            if pred is None:
                n_orig_null += 1
                if do_rescue:
                    pred = rescue_classification(out_text)
                    if pred is not None:
                        n_rescued += 1
            true = to_int_class(r.get("true_label"))
            pred_int = to_int_class(pred)
            if true is None:
                continue  # no ground truth → skip (rare, malformed row)
            # Force null predictions to wrong-class so they stay in the
            # denominator and count as incorrect. This is standard practice
            # in LLM evaluation — otherwise rambling models look artificially
            # good because their unparseable outputs get dropped.
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
            if true is None:
                continue  # no ground truth
            # For regression we can't easily "force wrong" the way we do for
            # classification. Track null rate separately; if you want a
            # null-penalized metric, see PRIMARY note below.
            if pred is None:
                continue  # for now, regression nulls still skipped — report n_orig_null
            per_task_pairs[task_name][0].append(true)
            per_task_pairs[task_name][1].append(pred)

    total_n = sum(len(t) for t, _ in per_task_pairs.values())

    if task_type == "classification":
        per_task_f1 = {t: macro_f1(yt, yp) for t, (yt, yp) in per_task_pairs.items()}
        per_task_acc = {t: accuracy(yt, yp) for t, (yt, yp) in per_task_pairs.items()}
        f1_avg = sum(per_task_f1.values()) / len(per_task_f1) if per_task_f1 else 0.0
        acc_avg = sum(per_task_acc.values()) / len(per_task_acc) if per_task_acc else 0.0
        primary = f1_avg
        return {
            "dataset": dataset, "model": model, "mode": mode,
            "retriever": retriever, "k": k,
            "task_type": task_type,
            "n": total_n,
            "n_subtasks": len(per_task_pairs),
            "macro_f1": f1_avg,
            "accuracy": acc_avg,
            "mae": None, "rmse": None,
            "n_orig_null": n_orig_null,
            "n_rescued": n_rescued,
            "primary": primary,
            "path": str(path),
        }
    else:
        all_true, all_pred = [], []
        for yt, yp in per_task_pairs.values():
            all_true.extend(yt); all_pred.extend(yp)
        m = mae(all_true, all_pred)
        r = rmse(all_true, all_pred)
        primary = -m if m == m else 0.0  # lower MAE is better, negate for "higher better" ordering
        return {
            "dataset": dataset, "model": model, "mode": mode,
            "retriever": retriever, "k": k,
            "task_type": task_type,
            "n": total_n,
            "n_subtasks": len(per_task_pairs),
            "macro_f1": None, "accuracy": None,
            "mae": m, "rmse": r,
            "n_orig_null": n_orig_null,
            "n_rescued": n_rescued,
            "primary": primary,
            "path": str(path),
        }


# =========================
# Display
# =========================
def print_long(results: List[Dict[str, Any]]):
    print()
    print("Note: classification nulls (after rescue) are counted as INCORRECT.")
    print("      Regression nulls are dropped (n decreases); see n_orig_null column.")
    print()
    print(f"{'task':<10} {'mode':<14} {'model':<28} {'n':>6} {'F1/MAE':>10} "
          f"{'null':>6} {'rescued':>8}")
    print("-" * 95)
    for r in sorted(results, key=lambda x: (x["dataset"], x["mode"], x["model"] or "")):
        if r["task_type"] == "classification":
            pm = f"{r['macro_f1']:.4f}"
        else:
            pm = f"{r['mae']:.4f}"
        model_short = (r['model'] or "").split("/")[-1][:28]
        print(f"{r['dataset']:<10} {r['mode']:<14} {model_short:<28} "
              f"{r['n']:>6} {pm:>10} {r['n_orig_null']:>6} {r['n_rescued']:>8}")


def print_wide(results: List[Dict[str, Any]]):
    """For each (task, model), show baseline vs enhanced vs raw vs hybrid side by side."""
    rows = defaultdict(dict)
    for r in results:
        key = (r["dataset"], (r["model"] or "").split("/")[-1])
        rows[key][r["mode"]] = r

    print(f"\n{'WIDE COMPARISON (vs baseline)':^138}")
    print(f"{'task':<10} {'model':<26} {'baseline':>9} {'enhanced':>9} "
          f"{'Δenh':>9} {'raw_syn':>9} {'Δraw':>9} {'hybrid':>9} {'Δhyb':>9}")
    print("-" * 138)
    for (task, model), modes in sorted(rows.items()):
        baseline = modes.get("chemrag")
        enhanced = modes.get("enhanced")
        raw      = modes.get("raw_synonyms")
        hybrid   = modes.get("hybrid")
        if all(x is None for x in [baseline, enhanced, raw, hybrid]):
            continue
        ref = baseline or enhanced or raw or hybrid
        ttype = ref.get("task_type")

        def metric(d):
            if d is None: return None
            return d["macro_f1"] if ttype == "classification" else d["mae"]

        def delta_str(value, base):
            if value is None or base is None:
                return ""
            d = value - base
            if ttype == "classification":
                arrow = "↑" if d > 0 else ("↓" if d < 0 else "·")
            else:
                arrow = "↓" if d < 0 else ("↑" if d > 0 else "·")
            return f"{d:+.4f}{arrow}"

        def fmt(v):
            return f"{v:.4f}" if v is not None else "   --"

        b  = metric(baseline)
        e  = metric(enhanced)
        rw = metric(raw)
        h  = metric(hybrid)
        print(f"{task:<10} {model[:26]:<26} "
              f"{fmt(b):>9} {fmt(e):>9} {delta_str(e, b):>9} "
              f"{fmt(rw):>9} {delta_str(rw, b):>9} "
              f"{fmt(h):>9} {delta_str(h, b):>9}")


def write_long_csv(results: List[Dict[str, Any]], path: Path):
    import csv
    cols = ["dataset", "mode", "model", "retriever", "k", "task_type",
            "n", "n_subtasks", "macro_f1", "accuracy", "mae", "rmse",
            "n_orig_null", "n_rescued", "path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(results, key=lambda x: (x["dataset"], x["mode"], x["model"] or "")):
            w.writerow({k: r.get(k) for k in cols})
    print(f"\nWrote long-format CSV: {path}")


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=list(DATASET_TASK_TYPE.keys()))
    ap.add_argument("--modes", nargs="+",
                    default=["chemrag", "enhanced", "raw_synonyms", "hybrid"],
                    help="modes to evaluate (any subset of: chemrag, enhanced, raw_synonyms, hybrid)")
    ap.add_argument("--no_rescue", action="store_true",
                    help="skip GPT-4o-mini rescue parse for null predictions")
    ap.add_argument("--out_csv", default=str(EVAL_LONG_CSV))
    args = ap.parse_args()

    do_rescue = not args.no_rescue
    if do_rescue:
        load_rescue_cache()

    files = find_jsonls(args.tasks, args.modes)
    print(f"Found {len(files)} JSONL files to evaluate.")
    if not files:
        return

    results = []
    for i, path in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] {path.name}")
        try:
            r = eval_file(path, do_rescue=do_rescue)
            results.append(r)
        except Exception as e:
            print(f"  ERROR: {e}")

    if do_rescue:
        save_rescue_cache()
        print(f"\nRescue parse: {_rescue_calls_made} new GPT-4o-mini calls "
              f"(total cache size: {len(load_rescue_cache())})")

    print_long(results)
    print_wide(results)
    write_long_csv(results, Path(args.out_csv))


if __name__ == "__main__":
    main()
