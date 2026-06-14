#!/usr/bin/env python
"""
chemrag_retrieval.py — retrieval-based property prediction for MoleculeNet.

Local layout paths (resolved relative to repo root via __file__):
  MolE-RAG/external/ChemRAG/      → corpus, index, models, flashrag
  MolE-RAG/results/<task>/        → inference outputs
  MolE-RAG/data/                  → input data (scaffold splits)
  MolE-RAG/caches/                → synonym caches, feature caches
  MolE-RAG/.env                   → API keys

Modes:
  default                : baseline.   query = "<task desc>. SMILES: <smiles>"
                           outputs → property_prediction/<task>/chemrag_results/
  --use_synonyms         : enhanced.   query = "<task desc>. <good_syns or iupac or smiles>"
                           outputs → property_prediction/<task>/enhanced_results/
  --use_raw_synonyms     : raw.        query = "<task desc>. <first N raw synonyms or smiles>"
                           outputs → property_prediction/<task>/raw_synonym_results/
  --use_hybrid           : hybrid.     query = "<task desc>. <task_kw>. <good_syns or iupac>"
                           outputs → property_prediction/<task>/hybrid_results/

--use_synonyms, --use_raw_synonyms, and --use_hybrid are mutually exclusive.

Seed handling (--seed N):
  Read test from data/moleculenet_property_scaffold/seed_{N}/{dataset}/
  Write outputs to {output_subdir}/seed_{N}/
  Retrieval cache filename includes _seed{N} suffix
  (cache is keyed by row index, and row indices differ across seeds)

Supports retrievers: bm25, contriever, specter, e5, rrf.

Retrieval caching: after the first retrieval run for a given (dataset, mode,
retriever, seed), results are saved to disk and reused on subsequent runs.
The retrieval cache is model-agnostic.

--retrieval_only builds the cache and exits without LLM inference.
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

import numpy as np
import torch
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Local paths (resolved relative to repo root)
# ---------------------------------------------------------------------------
REPO_ROOT     = Path(__file__).resolve().parent.parent.parent  # src/subdir/ -> src/ -> MolE-RAG/
CHEMRAG_ROOT  = REPO_ROOT / "external" / "ChemRAG"

ENV_PATH                  = REPO_ROOT / ".env"
DEFAULT_CORPUS_PATH       = CHEMRAG_ROOT / "corpus" / "chemrag_full_corpus.jsonl"
DEFAULT_INDEX_DIR         = CHEMRAG_ROOT / "index"
DEFAULT_SYNONYM_CACHE     = REPO_ROOT / "caches" / "llm_filtered_synonyms.json"
DEFAULT_RAW_SYNONYM_CACHE = REPO_ROOT / "caches" / "synonyms_cache.json"

RAW_SYNONYM_LIMIT = None

load_dotenv(str(ENV_PATH))


# ===========================================================================
# Canonical-SMILES lookup helper
# ===========================================================================
def _canon_for_lookup(smiles: str) -> str:
    """RDKit canonical SMILES, for synonym cache lookups.

    Cache keys were re-keyed to canonical SMILES by canonicalize_synonyms.py.
    This canonicalizes the query SMILES so lookups succeed regardless of how
    the SMILES was originally written in the test CSV.
    Returns the input unchanged if RDKit is unavailable or parsing fails.
    """
    if not smiles:
        return smiles
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol) if mol else smiles
    except Exception:
        return smiles


# ===========================================================================
# Per-dataset config
# ===========================================================================
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

SMILES_COL = "smiles"
LABEL_COL  = "label"
TASK_COL   = "task"


# Sane ranges for regression predictions. Values outside these ranges are
# almost certainly the model echoing a descriptor value (MolWt, TopoPSA, etc.)
# instead of the actual property. Treated as garbage by the retry loop.
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


RETRIEVER_MODEL_PATHS = {
    "contriever": str(CHEMRAG_ROOT / "models" / "facebook-contriever-msmarco"),
    "specter":    str(CHEMRAG_ROOT / "models" / "allenai-specter"),
    "e5":         str(CHEMRAG_ROOT / "models" / "intfloat-e5-base-v2"),
}
POOLING_METHOD = {"contriever": "mean", "specter": "cls", "e5": "mean"}
ALL_RETRIEVERS = ["bm25", "contriever", "specter", "e5"]


# ===========================================================================
# Task keyword expansion via LLM (used by --use_hybrid mode)
# ===========================================================================
TASK_KEYWORDS_CACHE = REPO_ROOT / "caches" / "task_keywords_llm.json"
_task_kw_cache: Optional[Dict[str, str]] = None
_kw_openai_client = None


def load_task_kw_cache() -> Dict[str, str]:
    global _task_kw_cache
    if _task_kw_cache is not None:
        return _task_kw_cache
    if TASK_KEYWORDS_CACHE.exists():
        with open(TASK_KEYWORDS_CACHE) as f:
            _task_kw_cache = json.load(f)
    else:
        _task_kw_cache = {}
    return _task_kw_cache


def save_task_kw_cache():
    if _task_kw_cache is None:
        return
    tmp = TASK_KEYWORDS_CACHE.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(_task_kw_cache, f, indent=2)
    os.replace(tmp, TASK_KEYWORDS_CACHE)


def _kw_openai():
    global _kw_openai_client
    if _kw_openai_client is None:
        from openai import OpenAI
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY missing — needed for hybrid mode task-keyword "
                "expansion. Set it with `export OPENAI_API_KEY=sk-...`")
        _kw_openai_client = OpenAI(api_key=key)
    return _kw_openai_client


def llm_generate_task_keywords(dataset_name: str, task_name: str) -> str:
    cache = load_task_kw_cache()
    ck = f"{dataset_name}::{task_name}"
    if ck in cache:
        return cache[ck]

    desc = task_description(dataset_name, task_name)
    prompt = (
        "You are a chemistry domain expert. Generate a concise list of scientific "
        "keywords and synonyms for a property prediction task. These keywords will "
        "be combined with molecule names and used to retrieve relevant passages "
        "from chemistry literature.\n\n"
        f"Task: Predict {desc}.\n\n"
        "Generate 15-25 task-relevant keywords/synonyms that appear in scientific "
        "papers about this property. Include:\n"
        "  - Common abbreviations and acronyms\n"
        "  - Related biological/chemical concepts and mechanisms\n"
        "  - Assay names, pathways, target proteins\n"
        "  - Standard terminology used in the literature\n\n"
        "Output ONLY the keywords as space-separated tokens. No explanation, "
        "no numbering, no quotes, no punctuation between tokens.\n\n"
        "Example for 'blood-brain barrier permeability':\n"
        "blood-brain-barrier BBB permeability CNS brain-penetration "
        "neuropharmacology efflux P-glycoprotein cerebrospinal-fluid "
        "pharmacokinetics CNS-active drug-delivery"
    )
    try:
        resp = _kw_openai().chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0,
        )
        keywords = resp.choices[0].message.content.strip()
        keywords = keywords.strip('"\'').strip()
        keywords = " ".join(keywords.split())
    except Exception as e:
        print(f"  WARNING: task keyword LLM call failed for {ck}: {str(e)[:120]}")
        keywords = ""
    cache[ck] = keywords
    save_task_kw_cache()
    return keywords


def precompute_task_keywords(dataset_name: str, task_names: List[str]) -> None:
    cache = load_task_kw_cache()
    unique_tasks = sorted(set(task_names))
    n_cached, n_new = 0, 0
    for tn in unique_tasks:
        ck = f"{dataset_name}::{tn}"
        if ck in cache:
            n_cached += 1
        else:
            llm_generate_task_keywords(dataset_name, tn)
            n_new += 1
    print(f"  Task keywords: {n_cached} cached, {n_new} newly generated")
    if unique_tasks:
        sample_ck = f"{dataset_name}::{unique_tasks[0]}"
        sample = cache.get(sample_ck, "")
        if sample:
            print(f"  Example keywords for '{unique_tasks[0]}': {sample[:200]}")


# ===========================================================================
# Catalog-ID filter
# ===========================================================================
CATALOG_PREFIXES = (
    "NSC", "CHEMBL", "NCI60", "NCI/", "SCHEMBL", "AKOS", "MLS", "SMR",
    "HMS", "CCG", "SR-", "BDBM", "PD0", "DTXSID", "Oprea", "EU-", "Z-",
    "RefChem", "BSPBio", "cid_", "510M", "AC1", "AC2", "BPBio", "CGP",
    "LMS", "NCGC", "WAY-", "BAS ", "BRD-", "Compound ", "STK", "ZINC",
    "BIDD", "EINECS", "Brn ",
)
_CAS_RE = re.compile(r"^\d{2,7}-\d{2}-\d$")
_ALL_DIGITS_RE = re.compile(r"^\d+$")


def is_catalog_id(s: str) -> bool:
    s = s.strip()
    if not s:
        return True
    upper = s.upper()
    if any(upper.startswith(p.upper()) for p in CATALOG_PREFIXES):
        return True
    if _CAS_RE.match(s):
        return True
    if _ALL_DIGITS_RE.match(s):
        return True
    if len(s) <= 2:
        return True
    return False


def filter_drug_names(synonyms: List[str]) -> List[str]:
    return [s for s in synonyms if not is_catalog_id(s)]


# ===========================================================================
# Model registry
# ===========================================================================
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
        if not api_key or api_key.startswith("sk-proj-REPLACE"):
            raise RuntimeError(f"OPENAI_API_KEY not set or placeholder in {ENV_PATH}.")
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
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}]
        return _tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )


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
    for i, seq in enumerate(generated):
        new_tokens = seq[input_total_len:]
        decoded = _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        results.append(decoded)
    return results


def _run_inference_batch_hf(system_prompt: str, user_prompts: List[str], max_new_tokens: int = 256):
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
    for i, seq in enumerate(out):
        new_tokens = seq[input_total_len:]
        decoded = _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if fmt == "chemdfm" and "Assistant:" in decoded:
            decoded = decoded.split("Assistant:", 1)[-1].strip()
        for marker in ["<|assistant|>", "assistant\n", "\nassistant"]:
            if marker in decoded:
                decoded = decoded.rsplit(marker, 1)[-1].strip()
        results.append(decoded)
    return results


def _run_inference_openai_single(system_prompt: str, user_prompt: str, max_tokens: int = 256):
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
        futures = [executor.submit(_run_inference_openai_single, system_prompt, up, max_tokens)
                   for up in user_prompts]
        return [f.result() for f in futures]


def run_inference_batch(system_prompt: str, user_prompts: List[str],
                        max_new_tokens: int = 256, openai_max_workers: int = 8):
    if _model_cfg["format"] == "openai":
        return _run_inference_openai_batch(system_prompt, user_prompts,
                                           max_tokens=max_new_tokens,
                                           max_workers=openai_max_workers)
    return _run_inference_batch_hf(system_prompt, user_prompts, max_new_tokens=max_new_tokens)


# ===========================================================================
# CSV helpers
# ===========================================================================
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


# ===========================================================================
# FlashRAG retriever wrapper
# ===========================================================================
class ChemRAGRetriever:
    def __init__(self, retriever_name: str, corpus_path: str, index_path: str,
                 model_path: Optional[str] = None, topk: int = 5):
        self.retriever_name = retriever_name
        self.topk = topk
        chemrag_root = str(CHEMRAG_ROOT)
        if chemrag_root not in sys.path:
            sys.path.insert(0, chemrag_root)
        from flashrag.config import Config
        from flashrag.utils import get_retriever

        config_dict = {
            "retrieval_method": retriever_name,
            "corpus_path": corpus_path,
            "index_path": index_path,
            "retrieval_topk": topk,
            "save_retrieval_cache": False,
            "use_retrieval_cache": False,
            "retrieval_query_max_length": 256,
            "retrieval_batch_size": 32,
            "retrieval_use_fp16": True,
            "retrieval_pooling_method": POOLING_METHOD.get(retriever_name, "mean"),
            "bm25_backend": "pyserini",
            "faiss_gpu": False,
            "instruction": None,
            "do_retrieval": True,
            "model_name": "placeholder",
            "corpus_name": "chemrag_corpus",
            "save_note": "retriever_only",
            "metrics": [],
            "framework": "openai",
            "generator_model": "placeholder",
            "gpu_id": "0",
            "openai_setting": {},
            "generation_params": {},
            "generator_max_input_len": 4096,
            "gpu_memory_utilization": 0.5,
            "save_dir": "/tmp/chemrag_retriever_workdir",
            "save_intermediate_data": False,
            "dataset_name": "moleculenet",
            "split": "test",
        }
        if model_path is not None:
            config_dict["model2path"] = {retriever_name: model_path}
            config_dict["retrieval_model_path"] = model_path

        self.config = Config(config_dict=config_dict)
        self.retriever = get_retriever(self.config)
        print(f"  ChemRAGRetriever ready: {retriever_name} (topk={topk})")

    def search(self, query: str) -> List[Dict[str, Any]]:
        results = self.retriever.search(query, num=self.topk)
        if isinstance(results, tuple):
            docs, scores = results
            return [{"contents": d.get("contents", d.get("text", str(d))),
                     "score": float(s),
                     "id": d.get("id", ""),
                     "source": d.get("source", "")} for d, s in zip(docs, scores)]
        return [{"contents": r.get("contents", r.get("text", str(r))),
                 "score": float(r.get("score", 0.0)),
                 "id": r.get("id", ""),
                 "source": r.get("source", "")} for r in results]


class RRFRetriever:
    def __init__(self, corpus_path: str, index_dir: str, topk: int = 5,
                 pool_size: int = 100, rrf_k: int = 60,
                 retriever_pool: Optional[List[str]] = None):
        self.topk = topk
        self.pool_size = pool_size
        self.rrf_k = rrf_k
        self.retriever_pool = retriever_pool or list(ALL_RETRIEVERS)
        self.retrievers = {}
        for name in self.retriever_pool:
            idx_path = (str(Path(index_dir) / name / "bm25") if name == "bm25"
                        else str(Path(index_dir) / name))
            self.retrievers[name] = ChemRAGRetriever(
                retriever_name=name, corpus_path=corpus_path,
                index_path=idx_path,
                model_path=RETRIEVER_MODEL_PATHS.get(name),
                topk=pool_size,
            )
        print(f"RRFRetriever ready over {self.retriever_pool} "
              f"(pool_size={pool_size}, rrf_k={rrf_k}, topk={topk})")

    @staticmethod
    def _doc_key(d: Dict[str, Any]) -> str:
        return d.get("id") or (d.get("contents", "")[:128] or "_unknown_")

    def search(self, query: str) -> List[Dict[str, Any]]:
        scores: Dict[str, float] = {}
        doc_lookup: Dict[str, Dict[str, Any]] = {}
        per_retriever_ranks: Dict[str, Dict[str, int]] = {}
        for name, r in self.retrievers.items():
            docs = r.search(query)
            per_retriever_ranks[name] = {}
            for rank, d in enumerate(docs, start=1):
                key = self._doc_key(d)
                scores[key] = scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank)
                doc_lookup.setdefault(key, d)
                per_retriever_ranks[name][key] = rank
        ranked_keys = sorted(scores.keys(), key=lambda k: -scores[k])[:self.topk]
        out = []
        for k in ranked_keys:
            d = dict(doc_lookup[k])
            d["score"] = scores[k]
            d["rrf_ranks"] = {name: per_retriever_ranks[name].get(k)
                              for name in self.retriever_pool}
            out.append(d)
        return out


def make_retriever(retriever_name: str, corpus_path: str, index_dir: str,
                   topk: int, rrf_pool_size: int, rrf_k: int):
    if retriever_name == "rrf":
        return RRFRetriever(corpus_path=corpus_path, index_dir=index_dir,
                            topk=topk, pool_size=rrf_pool_size, rrf_k=rrf_k)
    idx_path = (str(Path(index_dir) / retriever_name / "bm25")
                if retriever_name == "bm25"
                else str(Path(index_dir) / retriever_name))
    return ChemRAGRetriever(
        retriever_name=retriever_name, corpus_path=corpus_path,
        index_path=idx_path,
        model_path=RETRIEVER_MODEL_PATHS.get(retriever_name),
        topk=topk,
    )


# ===========================================================================
# Retrieval cache helpers
# ===========================================================================
def get_retrieval_cache_path(base_dir: Path, dataset_name: str, mode: str,
                              retriever: str, k: int,
                              seed: Optional[int] = None) -> Path:
    """Return the path for the on-disk retrieval cache for a given run config.

    When seed is set, includes a _seed{N} suffix. The cache is keyed by row
    index in the test CSV, and different scaffold seeds put different molecules
    at the same row index, so per-seed caches are required for correctness.
    """
    cache_dir = base_dir / "retrieval_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    seed_tag = f"_seed{seed}" if seed is not None else ""
    return cache_dir / f"retrieval_{dataset_name}_{mode}_{retriever}_k{k}{seed_tag}.json"


def load_retrieval_cache(cache_path: Path) -> Optional[Dict[int, List[Dict[str, Any]]]]:
    """Load retrieval cache from disk. Returns None if not found."""
    if not cache_path.exists():
        return None
    print(f"Loading cached retrievals from {cache_path}")
    with open(cache_path) as f:
        raw = json.load(f)
    # JSON keys are strings — convert back to int
    return {int(k): v for k, v in raw.items()}


def save_retrieval_cache(cache_path: Path,
                          retrieval_cache: Dict[int, List[Dict[str, Any]]]):
    """Save retrieval cache to disk."""
    tmp = cache_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(retrieval_cache, f)
    os.replace(tmp, cache_path)
    print(f"Saved retrieval cache → {cache_path}")


# ===========================================================================
# Synonym cache helpers
# ===========================================================================
def _extract_synonyms_from_entry(entry: Any) -> List[str]:
    if entry is None:
        return []
    if isinstance(entry, list):
        return [str(s).strip() for s in entry if str(s).strip()]
    if isinstance(entry, dict):
        for key in ["synonyms", "Synonym", "Synonyms", "names", "Names"]:
            v = entry.get(key)
            if isinstance(v, list):
                return [str(s).strip() for s in v if str(s).strip()]
        return []
    return []


def _extract_iupac_from_entry(entry: Any) -> str:
    if isinstance(entry, dict):
        for key in ["iupac_name", "IUPACName", "IUPAC_Name", "iupac"]:
            v = entry.get(key)
            if v and isinstance(v, str):
                return v.strip()
    return ""


# ===========================================================================
# Query construction
# ===========================================================================
def build_query(smiles: str, dataset_name: str, task_name: str,
                mode: str,
                syn_cache: Optional[Dict[str, Any]],
                raw_syn_cache: Optional[Dict[str, Any]],
                raw_syn_limit: int = RAW_SYNONYM_LIMIT) -> Dict[str, Any]:
    desc = task_description(dataset_name, task_name)

    # Canonicalize for cache lookups (caches are keyed by RDKit canonical SMILES)
    canon_smiles = _canon_for_lookup(smiles)

    if mode == "chemrag" or \
       (mode == "enhanced" and syn_cache is None) or \
       (mode == "raw_synonyms" and raw_syn_cache is None) or \
       (mode == "hybrid" and syn_cache is None):
        return {
            "query": f"{desc}. SMILES: {smiles}",
            "source": "smiles",
            "good_syns": [],
            "iupac": "",
            "raw_syns": [],
            "task_kw": "",
            "in_cache": False,
        }

    if mode == "enhanced":
        entry = syn_cache.get(canon_smiles, {}) if isinstance(syn_cache, dict) else {}
        iupac = (entry.get("iupac_name") or "").strip() if isinstance(entry, dict) else ""
        good_syns = entry.get("good_syns") or [] if isinstance(entry, dict) else []
        if good_syns:
            names_part = " ".join(good_syns)
            source = "synonyms"
        elif iupac:
            names_part = iupac
            source = "iupac"
        else:
            names_part = smiles
            source = "smiles_fallback"
        return {
            "query": f"{desc}. {names_part}",
            "source": source,
            "good_syns": good_syns,
            "iupac": iupac,
            "raw_syns": [],
            "task_kw": "",
            "in_cache": canon_smiles in syn_cache,
        }

    if mode == "hybrid":
        entry = syn_cache.get(canon_smiles, {}) if isinstance(syn_cache, dict) else {}
        iupac = (entry.get("iupac_name") or "").strip() if isinstance(entry, dict) else ""
        good_syns = entry.get("good_syns") or [] if isinstance(entry, dict) else []
        task_kw = load_task_kw_cache().get(f"{dataset_name}::{task_name}", "")
        if good_syns:
            names_part = " ".join(good_syns)
            source = "hybrid"
        elif iupac:
            names_part = iupac
            source = "hybrid_iupac"
        else:
            names_part = smiles
            source = "smiles_fallback"
        if task_kw:
            query = f"{desc}. {task_kw}. {names_part}"
        else:
            query = f"{desc}. {names_part}"
        return {
            "query": query,
            "source": source,
            "good_syns": good_syns,
            "iupac": iupac,
            "raw_syns": [],
            "task_kw": task_kw,
            "in_cache": canon_smiles in syn_cache,
        }

    # mode == "raw_synonyms"
    entry = raw_syn_cache.get(canon_smiles)
    raw_syns_all = _extract_synonyms_from_entry(entry)
    iupac = _extract_iupac_from_entry(entry)
    if raw_syn_limit is None or raw_syn_limit <= 0:
        raw_syns_used = raw_syns_all
    else:
        raw_syns_used = raw_syns_all[:raw_syn_limit]
    if raw_syns_used:
        names_part = " ".join(raw_syns_used)
        words = names_part.split()
        LUCENE_TOKEN_CAP = 800
        if len(words) > LUCENE_TOKEN_CAP:
            words = words[:LUCENE_TOKEN_CAP]
            names_part = " ".join(words)
            tally = 0
            trimmed = []
            for s in raw_syns_used:
                tally += len(s.split())
                if tally > LUCENE_TOKEN_CAP:
                    break
                trimmed.append(s)
            raw_syns_used = trimmed
        source = "raw_synonyms"
    elif iupac:
        names_part = iupac
        source = "iupac"
    else:
        names_part = smiles
        source = "smiles_fallback"
    return {
        "query": f"{desc}. {names_part}",
        "source": source,
        "good_syns": [],
        "iupac": iupac,
        "raw_syns": raw_syns_used,
        "task_kw": "",
        "in_cache": canon_smiles in raw_syn_cache,
    }


# ===========================================================================
# RDKit descriptors
# ===========================================================================
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, QED, rdMolDescriptors
    _RDKIT_AVAILABLE = True
except ImportError:
    _RDKIT_AVAILABLE = False


def _bucket(v: float, lo: float, hi: float, labels: Tuple[str, str, str]) -> str:
    if v < lo:  return labels[0]
    if v < hi:  return labels[1]
    return labels[2]


_DESCRIPTOR_LABELS = {
    "MolWt":      (300, 500, ("low molecular weight",
                              "moderate molecular weight",
                              "high molecular weight")),
    "MolLogP":    (1.0, 3.0, ("hydrophilic",
                              "moderate lipophilicity",
                              "highly lipophilic")),
    "TPSA":       (60.0, 120.0, ("low polar surface area",
                                 "moderate polar surface area",
                                 "high polar surface area")),
    "qed":        (0.4, 0.7, ("low drug-likeness",
                              "moderate drug-likeness",
                              "high drug-likeness")),
    "NumHDonors":        (1.5, 4.5, ("few H-bond donors", "moderate", "many H-bond donors")),
    "NumHAcceptors":     (3.5, 7.5, ("few H-bond acceptors", "moderate", "many H-bond acceptors")),
    "NumRotatableBonds": (3.5, 7.5, ("few rotatable bonds", "moderate", "many rotatable bonds")),
}

_rdkit_feature_cache: Optional[Dict[str, Any]] = None
RDKIT_FEATURE_CACHE_PATH = REPO_ROOT / "caches" / "task_rdkit_features.json"


def load_rdkit_feature_cache() -> Dict[str, Any]:
    global _rdkit_feature_cache
    if _rdkit_feature_cache is not None:
        return _rdkit_feature_cache
    if RDKIT_FEATURE_CACHE_PATH.exists():
        with open(RDKIT_FEATURE_CACHE_PATH) as f:
            _rdkit_feature_cache = json.load(f)
    else:
        _rdkit_feature_cache = {}
    return _rdkit_feature_cache


DEFAULT_RDKIT_FEATURES = [
    "MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "NumAromaticRings", "NumAliphaticRings",
    "HeavyAtomCount", "qed",
]


def _rdkit_descriptor_lookup():
    if not _RDKIT_AVAILABLE:
        return {}
    from rdkit.Chem import Descriptors, Lipinski, QED, rdMolDescriptors
    lookup = {name: fn for name, fn in Descriptors.descList}
    lookup.setdefault("qed", QED.qed)
    lookup.setdefault("NumAromaticRings", rdMolDescriptors.CalcNumAromaticRings)
    lookup.setdefault("NumAliphaticRings", rdMolDescriptors.CalcNumAliphaticRings)
    lookup.setdefault("HeavyAtomCount", lambda m: m.GetNumHeavyAtoms())
    return lookup


_DESCRIPTOR_FN_CACHE: Optional[Dict[str, Any]] = None


def get_descriptor_fns():
    global _DESCRIPTOR_FN_CACHE
    if _DESCRIPTOR_FN_CACHE is None:
        _DESCRIPTOR_FN_CACHE = _rdkit_descriptor_lookup()
    return _DESCRIPTOR_FN_CACHE


def _format_descriptor(name: str, value: float) -> str:
    if isinstance(value, (int, np.integer)) or (
       isinstance(value, float) and abs(value - round(value)) < 1e-6 and abs(value) < 1e6):
        val_str = f"{int(round(value))}"
    else:
        val_str = f"{value:.3f}"
    if name in _DESCRIPTOR_LABELS:
        lo, hi, labels = _DESCRIPTOR_LABELS[name]
        label = _bucket(value, lo, hi, labels)
        return f"  {name}: {val_str} ({label})"
    return f"  {name}: {val_str}"


def rdkit_descriptors_text(smiles: str, dataset_name: str = "",
                           task_name: str = "") -> str:
    if not _RDKIT_AVAILABLE:
        return ""
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        return ""
    if mol is None:
        return ""

    cache = load_rdkit_feature_cache()
    cache_key = f"{dataset_name}::{task_name}" if task_name else dataset_name
    entry = cache.get(cache_key) or cache.get(dataset_name)
    if entry and entry.get("top_features"):
        feature_names = [f["name"] for f in entry["top_features"]]
        header = (f"Task-relevant molecular descriptors (top {len(feature_names)} "
                  f"by correlation with training labels):")
    else:
        feature_names = DEFAULT_RDKIT_FEATURES
        header = "Molecular descriptors (computed via RDKit):"

    fn_lookup = get_descriptor_fns()
    lines = [header]
    for name in feature_names:
        fn = fn_lookup.get(name)
        if fn is None:
            continue
        try:
            val = fn(mol)
        except Exception:
            continue
        if val is None or (isinstance(val, float) and not math.isfinite(val)):
            continue
        lines.append(_format_descriptor(name, val))

    try:
        from rdkit.Chem import Descriptors, Lipinski
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        ro5_violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
        lipinski = "passes" if ro5_violations == 0 else f"{ro5_violations} violation(s)"
        lines.append(f"  Lipinski's Rule of 5: {lipinski}")
    except Exception:
        pass

    return "\n".join(lines) if len(lines) > 1 else ""


# ===========================================================================
# AccFG functional groups
# ===========================================================================
try:
    from accfg import AccFG as _AccFG_class  # noqa
    _ACCFG_AVAILABLE = True
except ImportError:
    _ACCFG_AVAILABLE = False
    _AccFG_class = None

_accfg_instance = None


def _get_accfg():
    global _accfg_instance
    if _accfg_instance is None and _ACCFG_AVAILABLE:
        _accfg_instance = _AccFG_class(lite=True, print_load_info=False)
    return _accfg_instance


def functional_groups_text(smiles: str) -> str:
    if not _ACCFG_AVAILABLE:
        return ""
    afg = _get_accfg()
    if afg is None:
        return ""
    try:
        fgs, _ = afg.run(smiles, show_atoms=False, show_graph=False)
    except Exception:
        return ""
    if not fgs:
        return ""
    return "Functional groups present: " + ", ".join(fgs.keys())


# ===========================================================================
# Synonyms / IUPAC injection
# ===========================================================================
def synonyms_text(smiles: str, syn_cache: Optional[Dict[str, Any]]) -> str:
    if syn_cache is None:
        return ""
    canon_smiles = _canon_for_lookup(smiles)
    entry = syn_cache.get(canon_smiles)
    if not isinstance(entry, dict):
        return ""
    iupac = (entry.get("iupac_name") or "").strip()
    good_syns = entry.get("good_syns") or []
    if not (iupac or good_syns):
        return ""
    lines = ["Compound identifiers:"]
    if good_syns:
        lines.append(f"  Name: {good_syns[0]}")
        if len(good_syns) > 1:
            other_names = ", ".join(good_syns[1:6])
            lines.append(f"  Other names: {other_names}")
    if iupac:
        lines.append(f"  IUPAC: {iupac}")
    return "\n".join(lines)


# ===========================================================================
# Prompt construction
# ===========================================================================
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


def build_retrieved_block(docs: List[Dict[str, Any]]) -> str:
    if not docs: return "No documents retrieved."
    lines = []
    for i, d in enumerate(docs, start=1):
        text = d.get("contents", "").strip().replace("\n", " ")
        if len(text) > 1200:
            text = text[:1200] + " ..."
        lines.append(f"[{i}] {text}")
    return "\n\n".join(lines)


def build_user_prompt(smiles: str, dataset_name: str, task_name: str,
                      retrieved_docs: List[Dict[str, Any]],
                      synonyms_block: str = "",
                      fgs_block: str = "",
                      rdkit_block: str = "") -> str:
    desc = task_description(dataset_name, task_name)
    retrieved_block = build_retrieved_block(retrieved_docs)
    task_type = DATASET_CONFIG[dataset_name]["task_type"]

    extras = [b for b in (synonyms_block, fgs_block, rdkit_block) if b]
    extras_section = ("\n" + "\n\n".join(extras) + "\n") if extras else ""

    if task_type == "classification":
        return (f"Task: Predict {desc}.\n\n"
                f"Retrieved chemistry context:\n{retrieved_block}\n"
                f"{extras_section}\n"
                f"Now predict the answer for this molecule.\n"
                f"SMILES: {smiles}\n\n"
                f"Reply with EXACTLY ONE WORD: either 'Yes' or 'No'. "
                f"Output nothing else. No explanation, no punctuation, no context.")
    return (f"Task: Predict {desc}.\n\n"
            f"Retrieved chemistry context:\n{retrieved_block}\n"
            f"{extras_section}\n"
            f"Now predict the numeric value for this molecule.\n"
            f"SMILES: {smiles}\n\n"
            f"Reply with EXACTLY ONE NUMBER (e.g. -2.34). "
            f"Output nothing else. No explanation, no units, no context.")


def build_user_prompt_strict(smiles: str, dataset_name: str, task_name: str,
                             retrieved_docs: List[Dict[str, Any]],
                             synonyms_block: str = "",
                             fgs_block: str = "",
                             rdkit_block: str = "") -> str:
    desc = task_description(dataset_name, task_name)
    short_docs = retrieved_docs[:3] if retrieved_docs else []
    short_lines = []
    for i, d in enumerate(short_docs, start=1):
        text = d.get("contents", "").strip().replace("\n", " ")
        if len(text) > 400:
            text = text[:400] + " ..."
        short_lines.append(f"[{i}] {text}")
    short_block = "\n\n".join(short_lines) if short_lines else "No documents retrieved."

    extras = [b for b in (synonyms_block, fgs_block, rdkit_block) if b]
    extras_section = ("\n" + "\n\n".join(extras) + "\n") if extras else ""

    task_type = DATASET_CONFIG[dataset_name]["task_type"]
    if task_type == "classification":
        return (f"Question: Does this molecule have the following property? "
                f"{desc}\n\n"
                f"Retrieved evidence:\n{short_block}\n"
                f"{extras_section}\n"
                f"SMILES: {smiles}\n\n"
                f"You MUST answer with exactly one of these two words: Yes OR No.\n"
                f"EVEN IF YOU ARE UNCERTAIN, choose the most likely answer.\n"
                f"Do not refuse. Do not explain. Do not add anything else.\n\n"
                f"Answer:")
    return (f"Question: Predict the numeric value for: {desc}\n\n"
            f"Retrieved evidence:\n{short_block}\n"
            f"{extras_section}\n"
            f"SMILES: {smiles}\n\n"
            f"You MUST output exactly one number (e.g. 2.34 or -1.5).\n"
            f"EVEN IF YOU ARE UNCERTAIN, give your best numeric estimate.\n"
            f"Do not refuse. Do not explain. Do not include units.\n\n"
            f"Answer:")


# ===========================================================================
# Output parsing
# ===========================================================================
def parse_binary_output(model_output: Any) -> Optional[str]:
    if not model_output: return None
    raw = str(model_output).strip()
    tail_markers = ["<|assistant|>", "assistant\n", "\nassistant ",
                    "ASSISTANT:", "Assistant:", "\nassistant"]
    for marker in tail_markers:
        if marker in raw:
            raw = raw.rsplit(marker, 1)[-1].strip()
            break
    tail = raw[-300:] if len(raw) > 300 else raw
    tag = re.search(r"<\s*BOOLEAN\s*>\s*(yes|no|true|false|1|0)\s*<\s*/\s*BOOLEAN\s*>",
                    tail, flags=re.I)
    if tag:
        v = tag.group(1).lower()
        return "1" if v in {"yes", "true", "1"} else "0"
    cleaned = re.sub(r"[\"'`*<>/]", " ", tail.lower()).rstrip(".!?,;:").strip()
    for pat in [
        r"final\s+answer\s*[:\-]\s*(yes|no|1|0|true|false|positive|negative|active|inactive)",
        r"answer\s*[:\-]\s*(yes|no|1|0|true|false|positive|negative|active|inactive)",
        r"prediction\s*[:\-]\s*(yes|no|1|0|true|false|positive|negative|active|inactive)",
    ]:
        m = re.search(pat, cleaned)
        if m:
            v = m.group(1)
            return "1" if v in {"yes", "y", "1", "true", "positive", "active"} else "0"
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
    matches = [(m.start(), m.group()) for m in re.finditer(r"\b(yes|no)\b", cleaned)]
    if matches:
        last_match = matches[-1][1]
        return "1" if last_match == "yes" else "0"
    return None


def parse_regression_output(model_output: Any) -> Optional[float]:
    if model_output is None: return None
    raw = str(model_output).strip()
    tag = re.search(r"<\s*NUMBER\s*>\s*(-?\d+(?:\.\d+)?)\s*<\s*/\s*NUMBER\s*>", raw, flags=re.I)
    if tag: return float(tag.group(1))
    for pat in [
        r"final\s+answer\s*[:=]\s*(-?\d+(?:\.\d+)?)",
        r"answer\s*[:=]\s*(-?\d+(?:\.\d+)?)",
        r"prediction\s*[:=]\s*(-?\d+(?:\.\d+)?)",
        r"value\s*[:=]\s*(-?\d+(?:\.\d+)?)",
    ]:
        m = re.search(pat, raw, flags=re.I)
        if m: return float(m.group(1))
    nums = re.findall(r"-?\d+(?:\.\d+)?", raw)
    return float(nums[-1]) if nums else None


# ===========================================================================
# Main
# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASET_CONFIG.keys()))
    ap.add_argument("--retriever", required=True,
                    choices=["bm25", "contriever", "specter", "e5", "rrf"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--rrf_pool_size", type=int, default=100)
    ap.add_argument("--rrf_k", type=int, default=60)
    ap.add_argument("--corpus_path", default=str(DEFAULT_CORPUS_PATH))
    ap.add_argument("--index_dir", default=str(DEFAULT_INDEX_DIR))

    mode_group = ap.add_mutually_exclusive_group()
    mode_group.add_argument("--use_synonyms", action="store_true")
    mode_group.add_argument("--use_raw_synonyms", action="store_true")
    mode_group.add_argument("--use_hybrid", action="store_true")

    ap.add_argument("--synonym_cache", default=str(DEFAULT_SYNONYM_CACHE))
    ap.add_argument("--raw_synonym_cache", default=str(DEFAULT_RAW_SYNONYM_CACHE))
    ap.add_argument("--raw_synonym_limit", type=int, default=RAW_SYNONYM_LIMIT)

    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS_TO_RUN)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--openai_batch_size", type=int, default=8)

    ap.add_argument("--prompt_synonyms", action="store_true")
    ap.add_argument("--prompt_fgs", action="store_true")
    ap.add_argument("--prompt_rdkit", action="store_true")

    ap.add_argument("--retry_nulls", dest="retry_nulls", action="store_true", default=True)
    ap.add_argument("--no_retry_nulls", dest="retry_nulls", action="store_false")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--overwrite_retrieval", action="store_true",
                    help="force re-retrieval even if retrieval cache exists")
    ap.add_argument("--retrieval_only", action="store_true",
                    help="build retrieval cache and exit (no LLM inference)")

    # Scaffold split seed support
    ap.add_argument("--seed", type=int, default=None,
                    help="If set, read test from "
                         "data/moleculenet_property_scaffold/seed_{N}/{dataset}/ "
                         "and write outputs/cache with seed-specific names. "
                         "Default: use legacy random-split paths.")

    args = ap.parse_args()

    dataset_name = args.dataset
    cfg = DATASET_CONFIG[dataset_name]
    task_type = cfg["task_type"]

    if args.use_hybrid:
        mode = "hybrid"
    elif args.use_raw_synonyms:
        mode = "raw_synonyms"
    elif args.use_synonyms:
        mode = "enhanced"
    else:
        mode = "chemrag"

    mode_to_subdir = {
        "chemrag":      "chemrag_results",
        "enhanced":     "enhanced_results",
        "raw_synonyms": "raw_synonym_results",
        "hybrid":       "hybrid_results",
    }
    base_dir = REPO_ROOT / "results" / dataset_name

    # =========================================================
    # Path setup — branches on --seed
    # =========================================================
    if args.seed is not None:
        scaffold_dir = (REPO_ROOT / "data" / "moleculenet_property_scaffold"
                        / f"seed_{args.seed}" / dataset_name)
        test_path = scaffold_dir / f"{dataset_name}_test.csv"
        print(f"[SEED MODE] Using scaffold split seed_{args.seed}")
        print(f"  test: {test_path}")
    else:
        test_path = base_dir / f"{dataset_name}_test.csv"

    prompt_flags = []
    if args.prompt_synonyms: prompt_flags.append("syn")
    if args.prompt_fgs:      prompt_flags.append("fg")
    if args.prompt_rdkit:    prompt_flags.append("rdk")
    prompt_suffix = ("_promptinject_" + "_".join(prompt_flags)) if prompt_flags else ""
    out_subdir = mode_to_subdir[mode] + prompt_suffix

    if args.seed is not None:
        output_dir = base_dir / out_subdir / f"seed_{args.seed}"
    else:
        output_dir = base_dir / out_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load synonym caches
    syn_cache = None
    raw_syn_cache = None
    if mode == "enhanced" or mode == "hybrid":
        sc_path = Path(args.synonym_cache)
        if not sc_path.exists():
            raise FileNotFoundError(f"Synonym cache not found at {sc_path}")
        print(f"Loading LLM-filtered synonym cache from {sc_path}")
        with open(sc_path) as f:
            syn_cache = json.load(f)
        print(f"  {len(syn_cache)} SMILES in cache")
    elif mode == "raw_synonyms":
        rc_path = Path(args.raw_synonym_cache)
        if not rc_path.exists():
            raise FileNotFoundError(f"Raw synonym cache not found at {rc_path}")
        print(f"Loading raw PubChem synonym cache from {rc_path}")
        with open(rc_path) as f:
            raw_syn_cache = json.load(f)
        print(f"  {len(raw_syn_cache)} SMILES in raw cache")

    # Load prompt-injection synonym cache if needed
    prompt_syn_cache = None
    if args.prompt_synonyms:
        sc_path = Path(args.synonym_cache)
        if sc_path.exists():
            if syn_cache is not None:
                prompt_syn_cache = syn_cache
            else:
                with open(sc_path) as f:
                    prompt_syn_cache = json.load(f)

    test_rows = load_csv(test_path)
    valid_examples = [
        (idx, ex) for idx, ex in enumerate(test_rows)
        if ex.get(SMILES_COL, "").strip() and valid_label(ex.get(LABEL_COL))
    ]
    if args.limit is not None:
        valid_examples = valid_examples[:args.limit]

    print("=" * 100)
    print(f"Mode      : {mode}")
    print(f"Dataset   : {dataset_name}   Task type: {task_type}   Seed: {args.seed}")
    print(f"Retriever : {args.retriever}   top-k: {args.k}")
    print(f"Output    : {output_dir}")
    print(f"Models    : {args.models}")
    print(f"Valid rows: {len(valid_examples)}")
    print("=" * 100)

    # Build queries
    if mode == "hybrid":
        unique_tasks = [get_task_name(ex, dataset_name) for _, ex in valid_examples]
        print(f"  Pre-generating LLM task keywords...")
        precompute_task_keywords(dataset_name, unique_tasks)

    query_records = []
    src_counts: Dict[str, int] = {}
    for idx, example in valid_examples:
        smiles = example.get(SMILES_COL, "").strip()
        task_name = get_task_name(example, dataset_name)
        qrec = build_query(smiles, dataset_name, task_name,
                           mode=mode,
                           syn_cache=syn_cache,
                           raw_syn_cache=raw_syn_cache,
                           raw_syn_limit=args.raw_synonym_limit)
        query_records.append(qrec)
        src_counts[qrec["source"]] = src_counts.get(qrec["source"], 0) + 1
    print(f"Query source distribution: {src_counts}")

    # =========================================================
    # Retrieval — load from cache or run fresh (seed-aware)
    # =========================================================
    retrieval_cache_path = get_retrieval_cache_path(
        base_dir, dataset_name, mode, args.retriever, args.k, seed=args.seed)

    cached = None if args.overwrite_retrieval else load_retrieval_cache(retrieval_cache_path)

    if cached is not None:
        retrieval_cache = cached
        print(f"  Using cached retrievals for {len(retrieval_cache)} rows "
              f"(skip BM25). Use --overwrite_retrieval to force re-retrieval.")
    else:
        print("Loading retriever...")
        retriever = make_retriever(
            retriever_name=args.retriever, corpus_path=args.corpus_path,
            index_dir=args.index_dir, topk=args.k,
            rrf_pool_size=args.rrf_pool_size, rrf_k=args.rrf_k,
        )
        print("Retrieving for all test rows...")
        retrieval_cache = {}
        for i, (idx, example) in enumerate(valid_examples):
            retrieval_cache[idx] = retriever.search(query_records[i]["query"])
            if (i + 1) % 100 == 0:
                print(f"  retrieved {i + 1}/{len(valid_examples)}")
        print("Retrieval done.")
        save_retrieval_cache(retrieval_cache_path, retrieval_cache)

        del retriever
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.retrieval_only:
        print("\n--retrieval_only set: retrieval cache saved. Exiting.")
        return

    # =========================================================
    # Inference — one model at a time
    # =========================================================
    for model_name in args.models:
        print(f"\n{'#' * 60}\nMODEL: {model_name}\n{'#' * 60}")
        load_model(model_name)
        model_short = get_model_short(model_name)
        try:
            out_name = (f"{model_short}_{dataset_name}_{mode}_"
                        f"{args.retriever}_k{args.k}.jsonl")
            out_path = output_dir / out_name
            if out_path.exists() and not args.overwrite:
                print(f"  [SKIP] {out_name} already exists")
                continue

            if _model_cfg["format"] == "openai":
                eff_batch_size = args.openai_batch_size
            else:
                eff_batch_size = args.batch_size
            print(f"  Using batch_size={eff_batch_size} for {model_name}")

            max_new_tokens_for_task = 10 if task_type == "classification" else 30

            batch_meta = []
            for (idx, example), qrec in zip(valid_examples, query_records):
                smiles = example.get(SMILES_COL, "").strip()
                true_label = example.get(LABEL_COL, None)
                task_name = get_task_name(example, dataset_name)
                docs = retrieval_cache[idx]

                synonyms_block = synonyms_text(smiles, prompt_syn_cache) if args.prompt_synonyms else ""
                fgs_block      = functional_groups_text(smiles) if args.prompt_fgs else ""
                rdkit_block    = rdkit_descriptors_text(smiles, dataset_name, task_name) if args.prompt_rdkit else ""

                user_prompt = build_user_prompt(
                    smiles, dataset_name, task_name, docs,
                    synonyms_block=synonyms_block,
                    fgs_block=fgs_block,
                    rdkit_block=rdkit_block,
                )
                batch_meta.append((idx, smiles, true_label, task_name, docs,
                                   qrec, user_prompt,
                                   synonyms_block, fgs_block, rdkit_block))

            final_outputs = []
            for bs in range(0, len(batch_meta), eff_batch_size):
                batch = batch_meta[bs: bs + eff_batch_size]
                user_prompts = [b[6] for b in batch]
                model_outputs = run_inference_batch(
                    SYSTEM_PROMPT, user_prompts,
                    max_new_tokens=max_new_tokens_for_task,
                    openai_max_workers=args.openai_batch_size,
                )
                for (idx, smiles, true_label, task_name, docs, qrec, _,
                     synonyms_block, fgs_block, rdkit_block), out \
                        in zip(batch, model_outputs):
                    if task_type == "classification":
                        predicted_label = parse_binary_output(out)
                        predicted_value = None
                    else:
                        predicted_label = None
                        predicted_value = parse_regression_output(out)
                    final_outputs.append({
                        "id": idx,
                        "smiles": smiles,
                        "true_label": true_label,
                        "dataset": dataset_name,
                        "task": task_name,
                        "assay": task_name,
                        "condition": f"{mode}_{args.retriever}",
                        "mode": mode,
                        "model": model_name,
                        "task_type": task_type,
                        "retriever": args.retriever,
                        "k": args.k,
                        "seed": args.seed,
                        "rrf_pool_size": args.rrf_pool_size if args.retriever == "rrf" else None,
                        "rrf_k": args.rrf_k if args.retriever == "rrf" else None,
                        "query_source": qrec["source"],
                        "good_syns_used": qrec["good_syns"],
                        "raw_syns_used": qrec["raw_syns"],
                        "iupac_used": qrec["iupac"],
                        "task_kw_used": qrec.get("task_kw", ""),
                        "query": qrec["query"][:500],
                        "model_output": out,
                        "predicted_label": predicted_label,
                        "predicted_value": predicted_value,
                        "retry_used": False,
                        "retry_output": None,
                        "retrieved_docs": [
                            {"score": d.get("score"),
                             "id": d.get("id", ""),
                             "source": d.get("source", ""),
                             "contents": d.get("contents", "")[:500],
                             "rrf_ranks": d.get("rrf_ranks")}
                            for d in docs
                        ],
                    })
                done = min(bs + eff_batch_size, len(batch_meta))
                print(f"    {done}/{len(batch_meta)} done")

            # Retry nulls
            if args.retry_nulls:
                if task_type == "classification":
                    null_indices = [i for i, r in enumerate(final_outputs)
                                    if r["predicted_label"] is None]
                else:
                    null_indices = [i for i, r in enumerate(final_outputs)
                                    if r["predicted_value"] is None
                                    or not is_sane_regression(r["predicted_value"], dataset_name)]
                n_first_null = len(null_indices)
                if n_first_null > 0:
                    print(f"  Retrying {n_first_null} null predictions...")
                    retry_prompts = []
                    for i in null_indices:
                        row = final_outputs[i]
                        docs = retrieval_cache[row["id"]]
                        retry_prompts.append(
                            build_user_prompt_strict(
                                row["smiles"], dataset_name, row["task"], docs))
                    retry_max_tokens = 5 if task_type == "classification" else 20
                    n_recovered = 0
                    for rs in range(0, len(retry_prompts), eff_batch_size):
                        retry_batch_prompts  = retry_prompts[rs: rs + eff_batch_size]
                        retry_batch_indices  = null_indices[rs: rs + eff_batch_size]
                        retry_outputs = run_inference_batch(
                            SYSTEM_PROMPT_STRICT, retry_batch_prompts,
                            max_new_tokens=retry_max_tokens,
                            openai_max_workers=args.openai_batch_size,
                        )
                        for orig_idx, retry_out in zip(retry_batch_indices, retry_outputs):
                            row = final_outputs[orig_idx]
                            row["retry_used"] = True
                            row["retry_output"] = retry_out
                            if task_type == "classification":
                                new_pred = parse_binary_output(retry_out)
                                if new_pred is not None:
                                    row["predicted_label"] = new_pred
                                    n_recovered += 1
                            else:
                                new_pred = parse_regression_output(retry_out)
                                if new_pred is not None and is_sane_regression(new_pred, dataset_name):
                                    row["predicted_value"] = new_pred
                                    n_recovered += 1
                                elif new_pred is not None:
                                    # Still out-of-range after retry — leave null
                                    # so the evaluator drops it cleanly.
                                    row["predicted_value"] = None
                    print(f"  Retry recovered {n_recovered}/{n_first_null} nulls. "
                          f"Still null: {n_first_null - n_recovered}")

            with open(out_path, "w", encoding="utf-8") as f:
                for row in final_outputs:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"  Saved {len(final_outputs)} rows -> {out_path}")
        finally:
            unload_model()


if __name__ == "__main__":
    main()