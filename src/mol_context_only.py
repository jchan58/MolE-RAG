#!/usr/bin/env python
"""
mol_context_only.py — property prediction with prompt injection ONLY,
no retrieval.

Tests the third pillar (molecular context injection) in isolation by
skipping BM25/dense retrieval entirely. Each test molecule's prompt
gets only:
  - Task description
  - SMILES
  - Optional synonyms/IUPAC block (--prompt_synonyms)
  - Optional functional groups block (--prompt_fgs)
  - Optional RDKit descriptors block (--prompt_rdkit)

Use this to isolate the contribution of each prompt-injection axis.
For runs that combine retrieval + prompt injection, use chemrag_retrieval.py
with the same --prompt_* flags.

Output goes to:
  property_prediction/<task>/mol_context_results/<flag_combo>/seed_<N>/

Where <flag_combo> is one of: "syn", "fg", "rdk", "syn_fg", "syn_rdk",
"fg_rdk", "syn_fg_rdk", or "none" (no flags = pure baseline).

Examples:
  # Synonyms only on seed 0
  python mol_context_only.py --dataset bbbp --seed 0 --prompt_synonyms --models gpt-4o-mini

  # All three combined on seed 1
  python mol_context_only.py --dataset bbbp --seed 1 --prompt_synonyms --prompt_fgs --prompt_rdkit

  # Baseline (no injection, no retrieval) — useful sanity check
  python mol_context_only.py --dataset bbbp --seed 0
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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from dotenv import load_dotenv

# Path setup (resolved relative to repo root)
REPO_ROOT     = Path(__file__).resolve().parent.parent  # src/ -> MolE-RAG/
ENV_PATH      = REPO_ROOT / ".env"

from context.prompt_blocks import (   # noqa: E402
    load_syn_cache, build_prompt_injection, availability_check,
    DEFAULT_SYNONYM_CACHE,
)

load_dotenv(str(ENV_PATH))


# =========================
# Per-dataset config (matches chemrag_retrieval.py / molerag.py)
# =========================
DATASET_CONFIG = {
    "bbbp":     {"task_type": "classification", "multitask": False},
    "bace":     {"task_type": "classification", "multitask": False},
    "clintox":  {"task_type": "classification", "multitask": False},
    "hiv":      {"task_type": "classification", "multitask": False},
    "tox21":    {"task_type": "classification", "multitask": True},
    "sider":    {"task_type": "classification", "multitask": True},
    "esol":     {"task_type": "regression",     "multitask": False},
    "lipo":     {"task_type": "regression",     "multitask": False},
    "freesolv": {"task_type": "regression",     "multitask": False},
}

# Sanity ranges for regression predictions. Outside these = descriptor echo /
# nonsense — drop to None instead of polluting RMSE/MAE.
REGRESSION_SANE_RANGE = {
    "esol":     (-15.0, 15.0),   # log solubility
    "freesolv": (-30.0, 10.0),   # hydration free energy
    "lipo":     (-5.0,  10.0),   # logD
}


def is_sane_regression(value: Optional[float], dataset_name: str) -> bool:
    """True if value is within the realistic range for this dataset."""
    if value is None:
        return False
    rng = REGRESSION_SANE_RANGE.get(dataset_name)
    if rng is None:
        return True  # unknown dataset -> assume sane
    lo, hi = rng
    return lo <= value <= hi


SMILES_COL = "smiles"
LABEL_COL  = "label"
TASK_COL   = "task"


# =========================
# Model registry (mirrors chemrag_retrieval.py)
# =========================
MODEL_REGISTRY = {
    "meta-llama/Llama-3.2-3B-Instruct":   {"format": "chat",    "dtype": "auto",   "use_fast": True},
    "Qwen/Qwen3-4B-Instruct-2507":        {"format": "qwen",    "dtype": "auto",   "use_fast": True},
    "mistralai/Mistral-7B-Instruct-v0.3": {"format": "chat",    "dtype": "auto",   "use_fast": True},
    "AI4Chem/ChemLLM-7B-Chat":            {"format": "chemllm", "dtype": "float16","use_fast": False, "trust_remote_code": True},
    "OpenDFM/ChemDFM-v2.0-14B":           {"format": "chemdfm", "dtype": "auto",   "use_fast": False, "trust_remote_code": True},
    "gpt-4o-mini":                        {"format": "openai",  "api_model": "gpt-4o-mini"},
    "gpt-5.4-nano":                       {"format": "openai",  "api_model": "gpt-5.4-nano"},
}
DEFAULT_MODELS_TO_RUN = ["gpt-4o-mini"]
DEFAULT_BATCH_SIZE = 8

_model = None; _tokenizer = None; _model_cfg = None; _openai_client = None


def get_model_short(model_name: str) -> str:
    return model_name.split("/")[-1].lower().replace("-", "_").replace(".", "_")


def load_model(model_name: str):
    global _model, _tokenizer, _model_cfg, _openai_client
    cfg = MODEL_REGISTRY.get(model_name)
    if cfg is None:
        raise ValueError(f"Unknown model: {model_name}")
    _model_cfg = cfg

    if cfg["format"] == "openai":
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(f"OPENAI_API_KEY not set.")
        _openai_client = OpenAI(api_key=api_key)
        _model = None; _tokenizer = None
        print(f"Using OpenAI API: {model_name}")
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer
    trust_remote_code = cfg.get("trust_remote_code", False)
    print(f"Loading tokenizer: {model_name}")
    _tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=cfg["use_fast"], trust_remote_code=trust_remote_code,
    )
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token
    dtype_map = {"auto": "auto", "bfloat16": torch.bfloat16, "float16": torch.float16}
    torch_dtype = dtype_map[cfg["dtype"]]
    print(f"Loading model: {model_name}")
    _model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    if cfg["format"] == "chemllm":
        try:    _model.config.use_cache = False
        except: pass
        try:    _model.generation_config.use_cache = False
        except: pass
    _model.eval()
    print(f"Model loaded: {model_name}")


def unload_model():
    global _model, _tokenizer, _openai_client
    if _model is not None: del _model
    if _tokenizer is not None: del _tokenizer
    _model = None; _tokenizer = None; _openai_client = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Model unloaded and GPU memory cleared.")


def _build_text(system_prompt: str, user_prompt: str, fmt: str) -> str:
    if fmt == "qwen":
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}]
        try:
            return _tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=_model_cfg.get("thinking", False),
            )
        except TypeError:
            return _tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
    if fmt == "chemllm":
        return (f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n")
    if fmt == "chat":
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}]
        return _tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    if fmt == "chemdfm":
        # ChemDFM v2.0 uses ChatML format via the official chat template.
        # Old "[Round 0] Human: ... Assistant:" format gives empty outputs.
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}]
        try:
            return _tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            # Fallback if template is missing
            return (f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                    f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
                    f"<|im_start|>assistant\n")
    return f"{system_prompt}\n\n{user_prompt}\n\nAnswer:"


def _manual_greedy_decode_hf(model_inputs, input_total_len: int, max_new_tokens: int):
    generated = model_inputs.input_ids
    attention_mask = model_inputs.get("attention_mask", None)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            kwargs = {"input_ids": generated, "use_cache": False}
            if attention_mask is not None:
                kwargs["attention_mask"] = attention_mask
            outputs = _model(**kwargs)
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)
            if _tokenizer.eos_token_id is not None and bool((next_token == _tokenizer.eos_token_id).all()):
                break
    results = []
    for seq in generated:
        new_tokens = seq[input_total_len:]
        results.append(_tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return results


def _run_inference_batch_hf(system_prompt: str, user_prompts: List[str],
                            max_new_tokens: int = 256):
    fmt = _model_cfg["format"]
    texts = [_build_text(system_prompt, up, fmt) for up in user_prompts]
    _tokenizer.padding_side = "left"
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token
    model_inputs = _tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=4096,
    ).to(_model.device)
    input_total_len = model_inputs.input_ids.shape[1]
    if fmt == "chemllm":
        return _manual_greedy_decode_hf(model_inputs, input_total_len, max_new_tokens)
    with torch.no_grad():
        kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False,
                  "pad_token_id": _tokenizer.pad_token_id}
        if fmt == "qwen":
            kwargs.update({"temperature": None, "top_p": None})
        out = _model.generate(**model_inputs, **kwargs)
    results = []
    for seq in out:
        new_tokens = seq[input_total_len:]
        decoded = _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if fmt == "chemdfm" and "Assistant:" in decoded:
            decoded = decoded.split("Assistant:", 1)[-1].strip()
        for marker in ["<|assistant|>", "assistant\n", "\nassistant"]:
            if marker in decoded:
                decoded = decoded.rsplit(marker, 1)[-1].strip()
        results.append(decoded)
    return results


def _run_inference_openai_single(system_prompt: str, user_prompt: str,
                                 max_tokens: int = 256):
    api_model = _model_cfg.get("api_model", "gpt-4o-mini")
    is_gpt5 = api_model.startswith("gpt-5")
    kwargs = {
        "model": api_model,
        "messages": [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": user_prompt}],
    }
    if is_gpt5:
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens
        kwargs["temperature"] = 0
    response = _openai_client.chat.completions.create(**kwargs)
    return response.choices[0].message.content.strip()


def _run_inference_openai_batch(system_prompt: str, user_prompts: List[str],
                                max_tokens: int = 256, max_workers: int = 8):
    workers = min(max(max_workers, 1), 32)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_run_inference_openai_single,
                                   system_prompt, up, max_tokens)
                   for up in user_prompts]
        return [f.result() for f in futures]


def run_inference_batch(system_prompt: str, user_prompts: List[str],
                        max_new_tokens: int = 256, openai_max_workers: int = 8):
    if _model_cfg["format"] == "openai":
        return _run_inference_openai_batch(system_prompt, user_prompts,
                                           max_tokens=max_new_tokens,
                                           max_workers=openai_max_workers)
    return _run_inference_batch_hf(system_prompt, user_prompts,
                                   max_new_tokens=max_new_tokens)


# =========================
# CSV helpers
# =========================
def load_csv(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def valid_label(value: Any) -> bool:
    if value is None: return False
    s = str(value).strip().lower()
    return s not in {"", "nan", "none", "null"}


def get_task_name(row: Dict[str, Any], dataset_name: str) -> str:
    return str(row.get(TASK_COL, dataset_name)).strip() or dataset_name


def task_description(dataset_name: str, task_name: str) -> str:
    if dataset_name == "bbbp":     return "whether the molecule has blood-brain barrier permeability"
    if dataset_name == "bace":     return "whether the molecule inhibits BACE-1 / beta-secretase"
    if dataset_name == "clintox":  return "whether the molecule is clinically toxic"
    if dataset_name == "hiv":      return "whether the molecule is active against HIV or inhibits HIV replication"
    if dataset_name == "tox21":    return f"whether the molecule is active in the Tox21 assay: {task_name}"
    if dataset_name == "sider":    return f"whether the molecule is associated with the SIDER adverse effect category: {task_name}"
    if dataset_name == "esol":     return "the molecule's log aqueous solubility"
    if dataset_name == "lipo":     return "the molecule's octanol/water distribution coefficient logD at pH 7.4"
    if dataset_name == "freesolv": return "the molecule's hydration free energy in kcal/mol"
    return f"the target molecular property for task {task_name}"


# =========================
# Prompt construction
# =========================
SYSTEM_PROMPT = (
    "You are an expert chemist. Predict molecular properties from SMILES strings. "
    "Reply with exactly the requested output and absolutely nothing else — "
    "no explanation, no reasoning, no preamble."
)

SYSTEM_PROMPT_STRICT = (
    "You are an expert chemist answering a forced-choice question. "
    "You MUST commit to an answer. Output exactly the requested format and "
    "nothing else. Do not refuse. Do not hedge. Do not explain."
)


def build_user_prompt(smiles: str, dataset_name: str, task_name: str,
                      synonyms_block: str = "",
                      fgs_block: str = "",
                      rdkit_block: str = "") -> str:
    desc = task_description(dataset_name, task_name)
    task_type = DATASET_CONFIG[dataset_name]["task_type"]
    extras = [b for b in (synonyms_block, fgs_block, rdkit_block) if b]
    extras_section = ("\n" + "\n\n".join(extras) + "\n") if extras else ""

    if task_type == "classification":
        return (f"Task: Predict {desc}.\n"
                f"{extras_section}\n"
                f"SMILES: {smiles}\n\n"
                f"Reply with EXACTLY ONE WORD: either 'Yes' or 'No'. "
                f"Output nothing else. No explanation, no punctuation, no context.")
    return (f"Task: Predict {desc}.\n"
            f"{extras_section}\n"
            f"SMILES: {smiles}\n\n"
            f"Reply with EXACTLY ONE NUMBER (e.g. -2.34). "
            f"Output nothing else. No explanation, no units, no context.")


def build_user_prompt_strict(smiles: str, dataset_name: str, task_name: str,
                             synonyms_block: str = "",
                             fgs_block: str = "",
                             rdkit_block: str = "") -> str:
    desc = task_description(dataset_name, task_name)
    extras = [b for b in (synonyms_block, fgs_block, rdkit_block) if b]
    extras_section = ("\n" + "\n\n".join(extras) + "\n") if extras else ""
    task_type = DATASET_CONFIG[dataset_name]["task_type"]
    if task_type == "classification":
        return (f"Question: Does this molecule have the following property? "
                f"{desc}\n"
                f"{extras_section}\n"
                f"SMILES: {smiles}\n\n"
                f"You MUST answer with exactly one of these two words: Yes OR No.\n"
                f"EVEN IF YOU ARE UNCERTAIN, choose the most likely answer.\n"
                f"Do not refuse. Do not explain. Do not add anything else.\n\n"
                f"Answer:")
    return (f"Question: Predict the numeric value for: {desc}\n"
            f"{extras_section}\n"
            f"SMILES: {smiles}\n\n"
            f"You MUST output exactly one number (e.g. 2.34 or -1.5).\n"
            f"EVEN IF YOU ARE UNCERTAIN, give your best numeric estimate.\n"
            f"Do not refuse. Do not explain. Do not include units.\n\n"
            f"Answer:")


# =========================
# Output parsing (same as chemrag_retrieval.py)
# =========================
def parse_binary_output(model_output: Any) -> Optional[str]:
    if not model_output: return None
    raw = str(model_output).strip()
    for marker in ["<|assistant|>", "assistant\n", "\nassistant ",
                   "ASSISTANT:", "Assistant:", "\nassistant"]:
        if marker in raw:
            raw = raw.rsplit(marker, 1)[-1].strip()
            break
    tail = raw[-300:] if len(raw) > 300 else raw
    cleaned = re.sub(r"[\"'`*<>/]", " ", tail.lower()).rstrip(".!?,;:").strip()
    toks = cleaned.split()
    if toks:
        last = toks[-1].rstrip(".!?,;:")
        if last in {"yes", "y", "1", "true", "positive", "active"}: return "1"
        if last in {"no", "n", "0", "false", "negative", "inactive"}: return "0"
        first = toks[0].rstrip(".!?,;:")
        if first in {"yes", "y", "1", "true", "positive", "active"}: return "1"
        if first in {"no", "n", "0", "false", "negative", "inactive"}: return "0"
    has_yes = bool(re.search(r"\byes\b", cleaned))
    has_no = bool(re.search(r"\bno\b", cleaned))
    if has_yes and not has_no: return "1"
    if has_no and not has_yes: return "0"
    matches = [m.group() for m in re.finditer(r"\b(yes|no)\b", cleaned)]
    if matches:
        return "1" if matches[-1] == "yes" else "0"
    return None


def parse_regression_output(model_output: Any) -> Optional[float]:
    if model_output is None: return None
    raw = str(model_output).strip()
    nums = re.findall(r"-?\d+(?:\.\d+)?", raw)
    return float(nums[-1]) if nums else None


# =========================
# Path resolution
# =========================
def resolve_paths(dataset_name: str, flag_combo: str,
                  seed: Optional[int]) -> Tuple[Path, Path]:
    """Return (test_csv_path, output_dir).

    With --seed N, test CSV comes from the scaffold-split seed_N subdir and
    outputs land in mol_context_results/<flag_combo>/seed_<N>/.

    Without --seed, falls back to the pre-seed layout (single test.csv,
    flat output dir) for backwards compatibility.
    """
    base = REPO_ROOT / "results" / dataset_name

    if seed is not None:
        scaffold_root = (REPO_ROOT / "data" /
                         "moleculenet_property_scaffold" /
                         f"seed_{seed}" / dataset_name)
        test_csv = scaffold_root / f"{dataset_name}_test.csv"
        out_dir = base / "mol_context_results" / flag_combo / f"seed_{seed}"
        print(f"[SEED MODE] Using scaffold split seed_{seed}")
        print(f"  test:    {test_csv}")
    else:
        test_csv = base / f"{dataset_name}_test.csv"
        out_dir = base / "mol_context_results" / flag_combo
        print(f"[NO SEED] Using default single-split layout")
        print(f"  test:    {test_csv}")

    out_dir.mkdir(parents=True, exist_ok=True)
    return test_csv, out_dir


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASET_CONFIG.keys()))
    ap.add_argument("--seed", type=int, default=None,
                    help="scaffold split seed (0/1/2). If omitted, uses default "
                         "single-split layout.")

    ap.add_argument("--prompt_synonyms", action="store_true",
                    help="inject compound identifiers (name, IUPAC, synonyms)")
    ap.add_argument("--prompt_fgs", action="store_true",
                    help="inject AccFG functional group annotations")
    ap.add_argument("--prompt_rdkit", action="store_true",
                    help="inject task-relevant RDKit descriptors")
    ap.add_argument("--rdkit_top_k", type=int, default=None,
                    help="cap on number of RDKit features in prompt (default: use full cached list)")

    ap.add_argument("--synonym_cache", default=str(DEFAULT_SYNONYM_CACHE),
                    help="LLM-filtered synonym cache (used with --prompt_synonyms)")

    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS_TO_RUN)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
                    help="batch size for local HF models")
    ap.add_argument("--openai_batch_size", type=int, default=8,
                    help="threadpool size for OpenAI API calls (default 8)")

    ap.add_argument("--retry_nulls", dest="retry_nulls", action="store_true",
                    default=True)
    ap.add_argument("--no_retry_nulls", dest="retry_nulls", action="store_false")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    dataset_name = args.dataset
    cfg = DATASET_CONFIG[dataset_name]
    task_type = cfg["task_type"]

    # Verify requested capabilities are available
    availability_check(args.prompt_synonyms, args.prompt_fgs, args.prompt_rdkit,
                       Path(args.synonym_cache))

    # Build output dir from flag combination
    flag_parts = []
    if args.prompt_synonyms: flag_parts.append("syn")
    if args.prompt_fgs:      flag_parts.append("fg")
    if args.prompt_rdkit:    flag_parts.append("rdk")
    flag_combo = "_".join(flag_parts) if flag_parts else "none"

    test_path, output_dir = resolve_paths(dataset_name, flag_combo, args.seed)

    # Load synonyms cache if needed
    syn_cache = None
    if args.prompt_synonyms:
        print(f"Loading LLM-filtered synonym cache from {args.synonym_cache}")
        syn_cache = load_syn_cache(Path(args.synonym_cache))
        if syn_cache is None:
            raise FileNotFoundError(f"Cache not found at {args.synonym_cache}")
        print(f"  {len(syn_cache)} SMILES in cache")

    test_rows = load_csv(test_path)
    valid_examples = [
        (idx, ex) for idx, ex in enumerate(test_rows)
        if ex.get(SMILES_COL, "").strip() and valid_label(ex.get(LABEL_COL))
    ]
    if args.limit is not None:
        valid_examples = valid_examples[:args.limit]

    print("=" * 100)
    print(f"Mode      : mol_context_only (no retrieval)")
    print(f"Dataset   : {dataset_name}   Task type: {task_type}   "
          f"Seed: {args.seed if args.seed is not None else '(default)'}")
    print(f"Injection : synonyms={args.prompt_synonyms}  "
          f"functional_groups={args.prompt_fgs}  rdkit={args.prompt_rdkit}")
    print(f"Flag combo: {flag_combo}")
    print(f"Output    : {output_dir}")
    print(f"Models    : {args.models}")
    print(f"Valid rows: {len(valid_examples)}")
    print("=" * 100)

    # Pre-compute all injection blocks once per row (independent of model).
    print("\nBuilding prompt injections for all test rows...")
    injection_cache: Dict[int, Dict[str, str]] = {}
    for idx, ex in valid_examples:
        smi = ex[SMILES_COL].strip()
        task_name = get_task_name(ex, dataset_name)
        injection_cache[idx] = build_prompt_injection(
            smi, dataset_name, task_name,
            syn_cache=syn_cache,
            use_synonyms=args.prompt_synonyms,
            use_fgs=args.prompt_fgs,
            use_rdkit=args.prompt_rdkit,
            rdkit_top_k=args.rdkit_top_k,
        )

    # Print first 3 example prompts to sanity check
    print("\nFirst 3 example user prompts (with injection):")
    for i in range(min(3, len(valid_examples))):
        idx, ex = valid_examples[i]
        smi = ex[SMILES_COL].strip()
        task_name = get_task_name(ex, dataset_name)
        inj = injection_cache[idx]
        prompt = build_user_prompt(smi, dataset_name, task_name,
                                   synonyms_block=inj["synonyms_block"],
                                   fgs_block=inj["fgs_block"],
                                   rdkit_block=inj["rdkit_block"])
        print(f"\n  --- Example {i} (id={idx}) ---")
        print(f"  smiles: {smi[:80]}")
        # Print first 600 chars of prompt so user can verify injection structure
        print(f"  prompt[:600]: {prompt[:600]}")
        print(f"  prompt total length: {len(prompt)} chars")

    # Inference per model
    for model_name in args.models:
        print(f"\n{'#' * 60}\nMODEL: {model_name}\n{'#' * 60}")
        load_model(model_name)
        model_short = get_model_short(model_name)
        try:
            out_name = (f"{model_short}_{dataset_name}_promptinject_"
                        f"{flag_combo}.jsonl")
            out_path = output_dir / out_name
            if out_path.exists() and not args.overwrite:
                print(f"  [SKIP] {out_name} already exists")
                continue

            eff_batch_size = (args.openai_batch_size
                              if _model_cfg["format"] == "openai"
                              else args.batch_size)
            print(f"  Using batch_size={eff_batch_size} for {model_name}")

            max_new_tokens_for_task = 10 if task_type == "classification" else 30

            # Build prompts
            batch_meta = []
            for (idx, example) in valid_examples:
                smiles = example.get(SMILES_COL, "").strip()
                true_label = example.get(LABEL_COL, None)
                task_name = get_task_name(example, dataset_name)
                inj = injection_cache[idx]
                user_prompt = build_user_prompt(
                    smiles, dataset_name, task_name,
                    synonyms_block=inj["synonyms_block"],
                    fgs_block=inj["fgs_block"],
                    rdkit_block=inj["rdkit_block"],
                )
                batch_meta.append((idx, smiles, true_label, task_name, inj,
                                   user_prompt))

            final_outputs = []
            for bs in range(0, len(batch_meta), eff_batch_size):
                batch = batch_meta[bs: bs + eff_batch_size]
                user_prompts = [b[5] for b in batch]
                model_outputs = run_inference_batch(
                    SYSTEM_PROMPT, user_prompts,
                    max_new_tokens=max_new_tokens_for_task,
                    openai_max_workers=args.openai_batch_size,
                )
                for (idx, smiles, true_label, task_name, inj, _), out in zip(
                        batch, model_outputs):
                    if task_type == "classification":
                        predicted_label = parse_binary_output(out)
                        predicted_value = None
                    else:
                        predicted_label = None
                        predicted_value = parse_regression_output(out)
                        # Drop descriptor-echo nonsense at write time
                        if predicted_value is not None and not is_sane_regression(
                                predicted_value, dataset_name):
                            predicted_value = None
                    final_outputs.append({
                        "id": idx,
                        "smiles": smiles,
                        "true_label": true_label,
                        "dataset": dataset_name,
                        "task": task_name,
                        "assay": task_name,
                        "seed": args.seed,
                        "condition": f"promptinject_{flag_combo}",
                        "mode": "promptinject",
                        "model": model_name,
                        "task_type": task_type,
                        "prompt_synonyms": args.prompt_synonyms,
                        "prompt_fgs": args.prompt_fgs,
                        "prompt_rdkit": args.prompt_rdkit,
                        "flag_combo": flag_combo,
                        "injection_synonyms": inj["synonyms_block"][:300],
                        "injection_fgs": inj["fgs_block"][:300],
                        "injection_rdkit": inj["rdkit_block"][:500],
                        "model_output": out,
                        "predicted_label": predicted_label,
                        "predicted_value": predicted_value,
                        "retry_used": False,
                        "retry_output": None,
                    })
                done = min(bs + eff_batch_size, len(batch_meta))
                print(f"    {done}/{len(batch_meta)} done")

            # Retry pass for nulls. For regression we also retry insane values.
            if args.retry_nulls:
                if task_type == "classification":
                    null_idx = [i for i, r in enumerate(final_outputs)
                                if r["predicted_label"] is None]
                else:
                    null_idx = [i for i, r in enumerate(final_outputs)
                                if r["predicted_value"] is None
                                or not is_sane_regression(r["predicted_value"],
                                                          dataset_name)]
                n_first_null = len(null_idx)
                if n_first_null > 0:
                    print(f"  Retrying {n_first_null} null/insane predictions "
                          f"with stricter prompt...")
                    retry_prompts = []
                    for i in null_idx:
                        row = final_outputs[i]
                        inj = injection_cache[row["id"]]
                        retry_prompts.append(build_user_prompt_strict(
                            row["smiles"], dataset_name, row["task"],
                            synonyms_block=inj["synonyms_block"],
                            fgs_block=inj["fgs_block"],
                            rdkit_block=inj["rdkit_block"],
                        ))
                    retry_max = 5 if task_type == "classification" else 20
                    n_recovered = 0
                    for rs in range(0, len(retry_prompts), eff_batch_size):
                        rp = retry_prompts[rs: rs + eff_batch_size]
                        ri = null_idx[rs: rs + eff_batch_size]
                        retry_outs = run_inference_batch(
                            SYSTEM_PROMPT_STRICT, rp,
                            max_new_tokens=retry_max,
                            openai_max_workers=args.openai_batch_size,
                        )
                        for orig_idx, retry_out in zip(ri, retry_outs):
                            row = final_outputs[orig_idx]
                            row["retry_used"] = True
                            row["retry_output"] = retry_out
                            if task_type == "classification":
                                p = parse_binary_output(retry_out)
                                if p is not None:
                                    row["predicted_label"] = p
                                    n_recovered += 1
                            else:
                                p = parse_regression_output(retry_out)
                                if p is not None and is_sane_regression(
                                        p, dataset_name):
                                    row["predicted_value"] = p
                                    n_recovered += 1
                                else:
                                    # Clear any insane existing value so eval
                                    # treats it as null rather than poisoning
                                    # the RMSE/MAE.
                                    row["predicted_value"] = None
                    print(f"  Retry recovered {n_recovered}/{n_first_null}. "
                          f"Still null after retry: "
                          f"{n_first_null - n_recovered}")

            with open(out_path, "w", encoding="utf-8") as f:
                for row in final_outputs:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"  Saved {len(final_outputs)} rows -> {out_path}")
        finally:
            unload_model()


if __name__ == "__main__":
    main()
