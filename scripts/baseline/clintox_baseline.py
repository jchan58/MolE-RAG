#!/usr/bin/env python
"""
baseline.py — ClinTox baseline (SMILES only).

Expected input CSV columns:
  dataset, task, task_type, metric, smiles, label, split

Task type:
  classification

Example:
  python scripts/baseline/clintox_baseline.py
"""

import csv
import gc
import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # scripts/baseline/ -> scripts/ -> MolE-RAG/

load_dotenv(str(REPO_ROOT / ".env"))


# =========================
# Model registry
# =========================
MODEL_REGISTRY = {
    "meta-llama/Llama-3.2-3B-Instruct": {
        "format": "chat",
        "dtype": "auto",
        "use_fast": True,
    },
    "Qwen/Qwen3-4B-Instruct-2507": {
        "format": "qwen",
        "dtype": "auto",
        "use_fast": True,
    },
    "mistralai/Mistral-7B-Instruct-v0.3": {
        "format": "chat",
        "dtype": "auto",
        "use_fast": True,
    },
    "AI4Chem/ChemLLM-7B-Chat": {
        "format": "chemllm",
        "dtype": "float16",
        "use_fast": False,
        "trust_remote_code": True,
    },
    # LlaSMol is not run here because it requires the official LlaSMol wrapper.
    # "osunlp/LlaSMol-Mistral-7B": {
    #     "format": "llasmol",
    #     "dtype": "auto",
    #     "use_fast": True,
    # },
    "OpenDFM/ChemDFM-v2.0-14B": {
        "format": "chemdfm",
        "dtype": "auto",
        "use_fast": False,
        "trust_remote_code": True,
    },
    "gpt-4o-mini": {
        "format": "openai",
        "api_model": "gpt-4o-mini",
    },
    "gpt-5.4-nano": {
        "format": "openai",
        "api_model": "gpt-5.4-nano",
    },
}


_model = None
_tokenizer = None
_model_cfg = None
_openai_client = None


def get_model_short(model_name):
    return model_name.split("/")[-1].lower().replace("-", "_").replace(".", "_")


def load_model(model_name):
    global _model, _tokenizer, _model_cfg, _openai_client

    cfg = MODEL_REGISTRY.get(model_name)
    if cfg is None:
        raise ValueError(f"Unknown model: {model_name}")

    _model_cfg = cfg

    if cfg["format"] == "openai":
        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key or api_key.startswith("sk-proj-REPLACE"):
            raise RuntimeError("OPENAI_API_KEY not set or still placeholder in .env.")

        _openai_client = OpenAI(api_key=api_key)
        _model = None
        _tokenizer = None
        print(f"Using OpenAI API: {model_name}")
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer

    trust_remote_code = cfg.get("trust_remote_code", False)

    print(f"Loading tokenizer: {model_name}")
    _tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=cfg["use_fast"],
        trust_remote_code=trust_remote_code,
    )

    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    dtype_map = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    torch_dtype = dtype_map[cfg["dtype"]]

    print(f"Loading model: {model_name}")
    _model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )

    if cfg["format"] == "chemllm":
        try:
            _model.config.use_cache = False
        except Exception:
            pass
        try:
            _model.generation_config.use_cache = False
        except Exception:
            pass

    _model.eval()
    print(f"Model loaded: {model_name}")


def unload_model():
    global _model, _tokenizer, _openai_client

    if _model is not None:
        del _model
    if _tokenizer is not None:
        del _tokenizer

    _model = None
    _tokenizer = None
    _openai_client = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Model unloaded and GPU memory cleared.")


def _build_text(system_prompt, user_prompt, fmt):
    if fmt == "qwen":
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            return _tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=_model_cfg.get("thinking", False),
            )
        except TypeError:
            return _tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    if fmt == "chemllm":
        return (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    if fmt == "chat":
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return _tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    if fmt == "chemdfm":
        return f"[Round 0]\nHuman: {system_prompt}\n\n{user_prompt}\nAssistant:"

    return f"{system_prompt}\n\n{user_prompt}\n\nAnswer:"


def _manual_greedy_decode_hf(model_inputs, input_lengths, max_new_tokens):
    generated = model_inputs.input_ids
    attention_mask = model_inputs.get("attention_mask", None)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            kwargs = {
                "input_ids": generated,
                "use_cache": False,
            }

            if attention_mask is not None:
                kwargs["attention_mask"] = attention_mask

            outputs = _model(**kwargs)
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)

            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, torch.ones_like(next_token)],
                    dim=-1,
                )

            if _tokenizer.eos_token_id is not None and bool((next_token == _tokenizer.eos_token_id).all()):
                break

    results = []
    for i, seq in enumerate(generated):
        new_tokens = seq[input_lengths[i]:]
        decoded = _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        results.append(decoded)

    return results


def _run_inference_batch_hf(system_prompt, user_prompts, max_new_tokens=256):
    fmt = _model_cfg["format"]
    texts = [_build_text(system_prompt, up, fmt) for up in user_prompts]

    _tokenizer.padding_side = "left"
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    model_inputs = _tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(_model.device)

    input_lengths = model_inputs.attention_mask.sum(dim=1)

    if fmt == "chemllm":
        return _manual_greedy_decode_hf(model_inputs, input_lengths, max_new_tokens)

    with torch.no_grad():
        kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "pad_token_id": _tokenizer.pad_token_id,
        }

        if fmt == "qwen":
            kwargs.update({"temperature": None, "top_p": None})

        out = _model.generate(**model_inputs, **kwargs)

    results = []
    for i, seq in enumerate(out):
        new_tokens = seq[input_lengths[i]:]
        decoded = _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        if fmt == "chemdfm" and "Assistant:" in decoded:
            decoded = decoded.split("Assistant:", 1)[-1].strip()

        results.append(decoded)

    return results


def _run_inference_openai_single(system_prompt, user_prompt, max_tokens=256):
    api_model = _model_cfg.get("api_model", "gpt-4o-mini")
    is_gpt5 = api_model.startswith("gpt-5")

    kwargs = {
        "model": api_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    if is_gpt5:
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens
        kwargs["temperature"] = 0

    response = _openai_client.chat.completions.create(**kwargs)
    return response.choices[0].message.content.strip()


def _run_inference_openai_batch(system_prompt, user_prompts, max_tokens=256):
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(_run_inference_openai_single, system_prompt, up, max_tokens)
            for up in user_prompts
        ]
        return [f.result() for f in futures]


def run_inference_batch(system_prompt, user_prompts, max_new_tokens=256):
    if _model_cfg["format"] == "openai":
        return _run_inference_openai_batch(system_prompt, user_prompts, max_tokens=max_new_tokens)
    return _run_inference_batch_hf(system_prompt, user_prompts, max_new_tokens=max_new_tokens)


def parse_binary_output(model_output):
    if not model_output:
        return None

    raw = str(model_output).strip()

    tag_match = re.search(
        r"<\s*BOOLEAN\s*>\s*(yes|no|true|false|1|0)\s*<\s*/\s*BOOLEAN\s*>",
        raw,
        flags=re.I,
    )
    if tag_match:
        val = tag_match.group(1).lower()
        if val in {"yes", "true", "1"}:
            return "1"
        if val in {"no", "false", "0"}:
            return "0"

    cleaned = raw.lower()
    cleaned = re.sub(r"[\"'`*<>/]", " ", cleaned)
    cleaned = cleaned.rstrip(".!?,;:").strip()

    if not cleaned:
        return None

    patterns = [
        r"final\s+answer\s*[:\-]\s*(yes|no|1|0|true|false|positive|negative|active|inactive)",
        r"answer\s*[:\-]\s*(yes|no|1|0|true|false|positive|negative|active|inactive)",
        r"prediction\s*[:\-]\s*(yes|no|1|0|true|false|positive|negative|active|inactive)",
        r"boolean\s*(yes|no|1|0|true|false)",
    ]

    for pat in patterns:
        m = re.search(pat, cleaned)
        if m:
            val = m.group(1)
            if val in {"yes", "y", "1", "true", "positive", "active"}:
                return "1"
            if val in {"no", "n", "0", "false", "negative", "inactive"}:
                return "0"

    tokens = cleaned.split()
    if tokens:
        first = tokens[0].rstrip(".!?,;:")
        if first in {"yes", "y", "1", "true", "positive", "active"}:
            return "1"
        if first in {"no", "n", "0", "false", "negative", "inactive"}:
            return "0"

    has_yes = re.search(r"\byes\b", cleaned) is not None
    has_no = re.search(r"\bno\b", cleaned) is not None

    if has_yes and not has_no:
        return "1"
    if has_no and not has_yes:
        return "0"

    has_active = re.search(r"\bactive\b|\bpositive\b", cleaned) is not None
    has_inactive = re.search(r"\binactive\b|\bnegative\b", cleaned) is not None

    if has_active and not has_inactive:
        return "1"
    if has_inactive and not has_active:
        return "0"

    if re.search(r"\b1\b", cleaned):
        return "1"
    if re.search(r"\b0\b", cleaned):
        return "0"

    return None


def parse_regression_output(model_output):
    if model_output is None:
        return None

    raw = str(model_output).strip()

    tag_match = re.search(
        r"<\s*NUMBER\s*>\s*(-?\d+(?:\.\d+)?)\s*<\s*/\s*NUMBER\s*>",
        raw,
        flags=re.I,
    )
    if tag_match:
        return float(tag_match.group(1))

    patterns = [
        r"final\s+answer\s*[:=]\s*(-?\d+(?:\.\d+)?)",
        r"answer\s*[:=]\s*(-?\d+(?:\.\d+)?)",
        r"prediction\s*[:=]\s*(-?\d+(?:\.\d+)?)",
    ]

    for pat in patterns:
        m = re.search(pat, raw, flags=re.I)
        if m:
            return float(m.group(1))

    nums = re.findall(r"-?\d+(?:\.\d+)?", raw)
    if not nums:
        return None

    return float(nums[-1])


# =========================
# Task config
# =========================
DATASET_NAME = "clintox"
TASK_TYPE = "classification"
TEST_PATH = None  # set by --seed in main()
SMILES_COL = "smiles"
LABEL_COL = "label"

OUTPUT_DIR = None  # set by --seed in main()


# =========================
# Prompts
# =========================
SYSTEM_PROMPT = (
    "You are an expert chemist. Your task is to predict molecular properties "
    "from SMILES strings. Follow the requested output format exactly."
)


def build_user_prompt(smiles, task_name=None):
    task_name = task_name or "CT_TOX"

    if TASK_TYPE == "classification":
        return (
            "Given the SMILES string of a molecule, predict whether it is clinically toxic. "
            ""
            "Answer with only Yes or No.\n\n"
            f"SMILES: {smiles}"
        )

    return (
        ""
        ""
        "Return only one numeric value.\n\n"
        f"SMILES: {smiles}"
    )


# =========================
# Run config
# =========================
MODELS_TO_RUN = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen3-4B-Instruct-2507",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "AI4Chem/ChemLLM-7B-Chat",
    # "OpenDFM/ChemDFM-v2.0-14B",
    "gpt-4o-mini",
    # "gpt-5.4-nano",
]

PROMPT_TYPES = ["zero_shot"]
BATCH_SIZE = 8
TEST_LIMIT = None


def load_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def valid_label(value):
    if value is None:
        return False
    s = str(value).strip().lower()
    return s not in {"", "nan", "none", "null"}


def main():
    global TEST_PATH, OUTPUT_DIR
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0, help="Scaffold split seed (0, 1, or 2)")
    args = ap.parse_args()
    TEST_PATH = str(REPO_ROOT / "data" / "moleculenet_property_scaffold" / f"seed_{args.seed}" / DATASET_NAME / f"{DATASET_NAME}_test.csv")
    OUTPUT_DIR = str(REPO_ROOT / "results" / DATASET_NAME / "baseline_results" / f"seed_{args.seed}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    test_data = load_csv(TEST_PATH)

    valid_examples = [
        (idx, ex)
        for idx, ex in enumerate(test_data)
        if ex.get(SMILES_COL, "").strip() and valid_label(ex.get(LABEL_COL))
    ]

    skipped = len(test_data) - len(valid_examples)

    if TEST_LIMIT is not None:
        valid_examples = valid_examples[:TEST_LIMIT]
        print(f"TEST_LIMIT enabled: using first {len(valid_examples)} valid examples")

    print(f"{DATASET_NAME}: {len(valid_examples)} valid test molecules/rows ({skipped} skipped)")

    for model_name in MODELS_TO_RUN:
        print(f"\n{'#' * 60}\nMODEL: {model_name}\n{'#' * 60}")

        load_model(model_name)
        model_short = get_model_short(model_name)

        try:
            for prompt_type in PROMPT_TYPES:
                out_name = f"{model_short}_{DATASET_NAME}_smiles_only_{prompt_type}.jsonl"
                out_path = os.path.join(OUTPUT_DIR, out_name)

                if os.path.exists(out_path):
                    print(f"  [SKIP] {out_name} already exists")
                    continue

                print(f"  Prompt: {prompt_type}")

                batch_meta = []

                for idx, example in valid_examples:
                    smiles = example.get(SMILES_COL, "").strip()
                    true_label = example.get(LABEL_COL, None)
                    task_name = example.get("task", "CT_TOX")
                    user_prompt = build_user_prompt(smiles, task_name=task_name)
                    batch_meta.append((idx, smiles, true_label, task_name, user_prompt))

                final_outputs = []

                for batch_start in range(0, len(batch_meta), BATCH_SIZE):
                    batch = batch_meta[batch_start: batch_start + BATCH_SIZE]
                    user_prompts = [b[4] for b in batch]
                    model_outputs = run_inference_batch(SYSTEM_PROMPT, user_prompts)

                    for (idx, smiles, true_label, task_name, _), model_output in zip(batch, model_outputs):
                        if TASK_TYPE == "classification":
                            predicted_label = parse_binary_output(model_output)
                            predicted_value = None
                        else:
                            predicted_label = None
                            predicted_value = parse_regression_output(model_output)

                        final_outputs.append(
                            {
                                "id": idx,
                                "smiles": smiles,
                                "true_label": true_label,
                                "dataset": DATASET_NAME,
                                "task": task_name,
                                "assay": task_name,
                                "condition": "smiles_only",
                                "prompt_type": prompt_type,
                                "model": model_name,
                                "task_type": TASK_TYPE,
                                "model_output": model_output,
                                "predicted_label": predicted_label,
                                "predicted_value": predicted_value,
                            }
                        )

                    done = min(batch_start + BATCH_SIZE, len(batch_meta))
                    print(f"    {done}/{len(batch_meta)} done")

                with open(out_path, "w", encoding="utf-8") as f:
                    for row in final_outputs:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")

                print(f"  Saved {len(final_outputs)} rows -> {out_name}")

        finally:
            unload_model()


if __name__ == "__main__":
    main()
