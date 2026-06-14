#!/usr/bin/env python
"""
sweep.py — launch chemrag_retrieval.py across all (task × mode × model) combos.

Lives at: MolE-RAG/src/fingerprint_sweep.py
Calls   : MolE-RAG/src/chemrag_retrieval.py

Each invocation of chemrag_retrieval.py runs N models for one (task, mode)
combination, with internal sequential model loading.

Usage:

  # All 10 tasks, baseline mode, all 7 models, batch_size=16
  python src/fingerprint_sweep.py --modes baseline --batch_size 16

  # Smoke test: 5 BBBP rows
  python src/fingerprint_sweep.py --tasks bbbp --models gpt-4o-mini --modes baseline --limit 5

  # OpenAI-only baseline across everything
  python src/fingerprint_sweep.py --modes baseline --models gpt-4o-mini gpt-5.4-nano

Resumable: skips combos whose JSONL already exists. Use --overwrite to force.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Local paths (resolved relative to repo root)
# ---------------------------------------------------------------------------
REPO_ROOT     = Path(__file__).resolve().parent.parent  # src/ -> MolE-RAG/
TEXTRAG_DIR   = REPO_ROOT / "src"
SCRIPT_PATH   = TEXTRAG_DIR / "chemrag_retrieval.py"

ALL_TASKS = ["bbbp", "bace", "clintox", "hiv", "tox21", "toxcast", "sider",
             "esol", "freesolv", "lipo"]

ALL_MODELS = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen3-4B-Instruct-2507",
    "mistralai/Mistral-7B-Instruct-v0.3",
    # "AI4Chem/ChemLLM-7B-Chat",
    "OpenDFM/ChemDFM-v2.0-14B",
    "gpt-4o-mini",
    "gpt-5.4-nano",
]

ALL_MODES = ["baseline", "enhanced"]


def model_short(name: str) -> str:
    return name.split("/")[-1].lower().replace("-", "_").replace(".", "_")


def expected_output(task: str, mode: str, model: str, retriever: str, k: int) -> Path:
    subdir = "enhanced_results" if mode == "enhanced" else "chemrag_results"
    cond = "enhanced" if mode == "enhanced" else "chemrag"
    out_dir = REPO_ROOT / "results" / task / subdir
    fn = f"{model_short(model)}_{task}_{cond}_{retriever}_k{k}.jsonl"
    return out_dir / fn


def already_done_per_model(task: str, mode: str, models: list,
                           retriever: str, k: int) -> tuple[set, set]:
    """Returns (done_models, todo_models) — sets of model names."""
    done = {m for m in models if expected_output(task, mode, m, retriever, k).exists()}
    todo = set(models) - done
    return done, todo


def run_one(task: str, mode: str, models: list, retriever: str, k: int,
            batch_size: int, openai_batch_size: int,
            overwrite: bool, limit: int | None) -> tuple[int, float]:
    cmd = [
        sys.executable, str(SCRIPT_PATH),
        "--dataset", task,
        "--retriever", retriever,
        "--k", str(k),
        "--models", *models,
        "--batch_size", str(batch_size),
        "--openai_batch_size", str(openai_batch_size),
    ]
    if mode == "enhanced":
        cmd.append("--use_synonyms")
    if overwrite:
        cmd.append("--overwrite")
    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    print(f"\n{'=' * 90}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] task={task} mode={mode} "
          f"models={[model_short(m) for m in models]}")
    print(" ".join(cmd))
    print('=' * 90, flush=True)

    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0
    print(f"[{datetime.now().strftime('%H:%M:%S')}] -> rc={result.returncode} "
          f"elapsed={elapsed/60:.1f} min", flush=True)
    return result.returncode, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=ALL_TASKS, choices=ALL_TASKS)
    ap.add_argument("--models", nargs="+", default=ALL_MODELS)
    ap.add_argument("--modes", nargs="+", default=ALL_MODES, choices=ALL_MODES)
    ap.add_argument("--retriever", default="bm25",
                    choices=["bm25", "contriever", "specter", "e5", "rrf"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=16,
                    help="batch size for HF/local models (default 16)")
    ap.add_argument("--openai_batch_size", type=int, default=8,
                    help="threadpool size for OpenAI calls (default 8)")
    ap.add_argument("--limit", type=int, default=None,
                    help="for smoke testing: limit rows per task")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    if not SCRIPT_PATH.exists():
        print(f"ERROR: chemrag_retrieval.py not found at {SCRIPT_PATH}")
        sys.exit(1)

    # Build the queue.  Skip (task, mode) only if ALL models have outputs.
    queue = []
    total_model_runs_needed = 0
    total_model_runs_existing = 0
    for mode in args.modes:
        for task in args.tasks:
            done, todo = already_done_per_model(
                task, mode, args.models, args.retriever, args.k)
            total_model_runs_needed += len(args.models)
            total_model_runs_existing += len(done)
            if not args.overwrite and not todo:
                print(f"[SKIP] task={task} mode={mode} — outputs exist for all models")
                continue
            if not args.overwrite and done:
                print(f"[PARTIAL] task={task} mode={mode} — {len(done)} done, {len(todo)} pending")
            queue.append((task, mode))

    total_combos_to_run = len(queue)
    expected_new_runs = total_model_runs_needed - total_model_runs_existing

    print(f"\n{'=' * 90}")
    print(f"Queued: {total_combos_to_run} (task, mode) combos × {len(args.models)} models")
    print(f"Model-runs already complete: {total_model_runs_existing}/{total_model_runs_needed}")
    print(f"Model-runs to do this sweep: {expected_new_runs}")
    print(f"Models    : {[model_short(m) for m in args.models]}")
    print(f"Modes     : {args.modes}")
    print(f"Tasks     : {args.tasks}")
    print(f"Retriever : {args.retriever}, k={args.k}")
    print(f"Batch     : hf={args.batch_size}, openai={args.openai_batch_size}")
    if args.limit is not None:
        print(f"Limit     : {args.limit} rows per task (smoke test)")
    print('=' * 90, flush=True)

    if args.dry_run:
        for task, mode in queue:
            print(f"  WOULD RUN: task={task} mode={mode}")
        return

    if not queue:
        print("Nothing to run.")
        return

    t_start = time.time()
    failures = []
    for i, (task, mode) in enumerate(queue, start=1):
        elapsed_so_far = time.time() - t_start
        remaining = len(queue) - i + 1
        eta_str = ""
        if i > 1:
            avg_per = elapsed_so_far / (i - 1)
            eta_sec = avg_per * remaining
            eta_str = f"  ETA ~{timedelta(seconds=int(eta_sec))}"
        print(f"\n[{i}/{len(queue)}] starting...  "
              f"elapsed {timedelta(seconds=int(elapsed_so_far))}{eta_str}", flush=True)

        rc, _ = run_one(task, mode, args.models, args.retriever,
                        args.k, args.batch_size, args.openai_batch_size,
                        args.overwrite, args.limit)
        if rc != 0:
            failures.append((task, mode, rc))

        # After each combo, recount actual JSONLs on disk
        completed_now = 0
        for m in args.modes:
            for t in args.tasks:
                done, _ = already_done_per_model(
                    t, m, args.models, args.retriever, args.k)
                completed_now += len(done)
        print(f"  >>> Cumulative model-runs complete: "
              f"{completed_now}/{total_model_runs_needed}  "
              f"({100*completed_now/total_model_runs_needed:.1f}%)", flush=True)

    total = time.time() - t_start
    print(f"\n{'=' * 90}")
    print(f"SWEEP DONE  total time: {timedelta(seconds=int(total))}")
    print(f"Failures: {len(failures)}")
    for task, mode, rc in failures:
        print(f"  FAILED  task={task} mode={mode} rc={rc}")


if __name__ == "__main__":
    main()