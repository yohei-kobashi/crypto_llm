#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parallel n-gram counting and distribution analysis (with EVS) + robust tokenizer loading + save-time filtering.

Tokenizer loading rules (auto):
- Local directory  -> AutoTokenizer.from_pretrained(dir, local_files_only=True)
- *.json           -> PreTrainedTokenizerFast(tokenizer_file=JSON)
- *.model          -> sentencepiece.SentencePieceProcessor(model_file=SPM)
- Otherwise        -> Try HF Hub id via AutoTokenizer.from_pretrained(id)

Metrics (computed on the FULL distribution before save-time filtering):
- Jensen–Shannon distance vs uniform
- Entropy ratio (H/Hmax)
- Gini coefficient
- HHI (Herfindahl–Hirschman Index)
- HHI_unbiased (Simpson index unbiased estimator)
- EVS (Effective Vocabulary Size): 1/HHI and 1/HHI_unbiased

Save-time filters (applied only when writing the distribution file):
Priority: top_k_save > mass_save > min_count_save
- --top_k_save K       : keep top-K n-grams by count
- --mass_save P (0..1) : keep n-grams until cumulative probability >= P
- --min_count_save T   : keep n-grams with count >= T

Supports JSONL(.gz) and Parquet(.gz) inputs. Output distribution supports CSV/CSV.GZ/Parquet.
"""

import argparse
import json
import math
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterable, List, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy
from transformers import AutoTokenizer, PreTrainedTokenizerFast

# ====== Globals used inside worker processes only ======
_GLOBALS = {
    "kind": None,          # 'hf' or 'spm'
    "tokenizer": None,     # HF tokenizer
    "sp": None,            # sentencepiece.SentencePieceProcessor
    "n": 1,
}

def _load_tokenizer_any(tokenizer_path: str):
    """
    Auto-detect tokenizer format and load appropriately.
    Returns a dict: {"kind": "hf", "tokenizer": tok} or {"kind": "spm", "sp": sp}
    """
    # Local directory -> HF tokenizer (offline-friendly)
    if os.path.isdir(tokenizer_path):
        tok = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        return {"kind": "hf", "tokenizer": tok}

    # Local single file
    if os.path.isfile(tokenizer_path):
        lower = tokenizer_path.lower()
        # tokenizers JSON
        if lower.endswith(".json"):
            tok = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
            return {"kind": "hf", "tokenizer": tok}
        # SentencePiece .model
        if lower.endswith(".model"):
            try:
                import sentencepiece as spm
            except ImportError as e:
                raise RuntimeError(
                    "You passed a SentencePiece .model file but `sentencepiece` "
                    "is not installed. Please `pip install sentencepiece`."
                ) from e
            sp = spm.SentencePieceProcessor()
            sp.load(tokenizer_path)
            return {"kind": "spm", "sp": sp}

    # Fallback: treat as HF repo id (requires network or local cache)
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    return {"kind": "hf", "tokenizer": tok}

def _init_worker(tokenizer_path: str, n: int):
    """Initialize each worker once: load tokenizer/SPM and store `n`."""
    info = _load_tokenizer_any(tokenizer_path)
    _GLOBALS["kind"] = info["kind"]
    _GLOBALS["tokenizer"] = info.get("tokenizer")
    _GLOBALS["sp"] = info.get("sp")
    _GLOBALS["n"] = n

def make_ngrams(tokens: List[int], n: int) -> Iterable[Tuple[int, ...]]:
    """Yield n-grams from a token id list."""
    if n == 1:
        for t in tokens:
            yield t
        return
    if len(tokens) < n:
        return
    iters = [tokens[i:] for i in range(n)]
    for gram in zip(*iters):
        yield gram

def _encode(text: str) -> List[int]:
    """Encode a single text to token ids using the loaded tokenizer/SPM."""
    if _GLOBALS["kind"] == "spm":
        # SentencePiece: out_type=int for ids; no special tokens added.
        return _GLOBALS["sp"].encode(text, out_type=int)
    else:
        # HF tokenizer: do not add special tokens.
        return _GLOBALS["tokenizer"].encode(text, add_special_tokens=False)

def process_batch(texts: List[str]) -> Counter:
    """Process a batch of texts: tokenize and update an n-gram Counter."""
    n = _GLOBALS["n"]
    c = Counter()
    for text in texts:
        if not isinstance(text, str):
            continue
        tokens = _encode(text)
        c.update(make_ngrams(tokens, n))
    return c

# ====== Metrics ======
def compute_js_divergence(freqs: Counter) -> float:
    """Compute Jensen–Shannon distance between the observed distribution and uniform."""
    if not freqs:
        return float("nan")
    counts = np.asarray(list(freqs.values()), dtype=float)
    total = counts.sum()
    if total <= 0:
        return float("nan")
    probs = counts / total
    uniform = np.ones_like(probs) / len(probs)
    m = 0.5 * (probs + uniform)
    kl_pm = scipy_entropy(probs, m, base=2)
    kl_um = scipy_entropy(uniform, m, base=2)
    js = 0.5 * (kl_pm + kl_um)
    return float(js)

def compute_entropy_ratio(freqs: Counter) -> float:
    """
    Entropy ratio = H(P) / Hmax, where Hmax is the entropy of the uniform distribution.
    Values closer to 1 indicate closer to uniform.
    """
    if not freqs:
        return float("nan")
    counts = np.asarray(list(freqs.values()), dtype=float)
    total = counts.sum()
    if total <= 0:
        return float("nan")
    probs = counts / total
    H = scipy_entropy(probs, base=2)
    Hmax = math.log(len(probs), 2) if len(probs) > 0 else 0.0
    return float(H / Hmax) if Hmax > 0 else float("nan")

def compute_gini(freqs: Counter) -> float:
    """
    Compute Gini coefficient of counts using cumulative-sum formula on sorted counts.
    """
    if not freqs:
        return float("nan")
    x = np.asarray(list(freqs.values()), dtype=float)
    if np.sum(x) <= 0:
        return float("nan")
    x_sorted = np.sort(x)
    n = x_sorted.size
    cumx = np.cumsum(x_sorted)
    g = (n + 1 - 2.0 * np.sum(cumx) / cumx[-1]) / n
    return float(max(0.0, min(1.0, g)))

def compute_hhi(freqs: Counter) -> float:
    """
    HHI = sum(p_i^2), p_i = count_i / sum(counts).
    Range: (1/K .. 1], larger => more concentrated (less uniform).
    """
    if not freqs:
        return float("nan")
    counts = np.asarray(list(freqs.values()), dtype=float)
    total = counts.sum()
    if total <= 0:
        return float("nan")
    p = counts / total
    return float(np.sum(p * p))

def compute_hhi_unbiased(freqs: Counter) -> float:
    """
    Unbiased HHI (Simpson index) under simple random sampling:
      HHI_unbiased = sum_i c_i (c_i - 1) / (N (N - 1)), for N = sum_i c_i
    """
    if not freqs:
        return float("nan")
    counts = np.asarray(list(freqs.values()), dtype=float)
    N = counts.sum()
    if N <= 1:
        return float("nan")
    return float(np.sum(counts * (counts - 1)) / (N * (N - 1)))

def compute_effective_vocab_size(hhi_value: float) -> float:
    """EVS (q=2 Hill number) = 1 / HHI."""
    if not np.isfinite(hhi_value) or hhi_value <= 0:
        return float("nan")
    return float(1.0 / hhi_value)

# ====== Streaming input & batching ======
def iter_jsonl_batches(path: str, batch_rows: int) -> Iterable[List[str]]:
    """Yield lists of texts from a JSONL(.gz) file in batches of `batch_rows`."""
    buf = []
    opener = open
    if path.endswith(".gz"):
        import gzip
        opener = gzip.open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if "text" in obj:
                buf.append(obj["text"])
                if len(buf) >= batch_rows:
                    yield buf
                    buf = []
    if buf:
        yield buf

def iter_parquet_batches(path: str, batch_rows: int) -> Iterable[List[str]]:
    """
    Yield lists of texts from a Parquet(.gz) file.
    Uses pyarrow if available; falls back to pandas otherwise.
    """
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(path, columns=["text"])
        col = table.column("text").to_pylist()
        for i in range(0, len(col), batch_rows):
            yield [str(x) for x in col[i : i + batch_rows] if x is not None]
    except Exception:
        df = pd.read_parquet(path, columns=["text"])
        series = df["text"].dropna().astype(str)
        for i in range(0, len(series), batch_rows):
            yield series.iloc[i : i + batch_rows].tolist()

def iter_batches(input_path: str, batch_rows: int) -> Iterable[List[str]]:
    """Dispatch to the appropriate iterator by file extension."""
    if input_path.endswith(".jsonl") or input_path.endswith(".jsonl.gz"):
        yield from iter_jsonl_batches(input_path, batch_rows)
    elif input_path.endswith(".parquet") or input_path.endswith(".parquet.gz"):
        yield from iter_parquet_batches(input_path, batch_rows)
    else:
        raise ValueError("input_texts must be .jsonl(.gz) or .parquet(.gz)")

# ====== Save-time filtering ======
def filter_distribution_df(dist_df: pd.DataFrame,
                           top_k: int = 0,
                           mass: float = 0.0,
                           min_count: int = 1) -> pd.DataFrame:
    """
    Filter rows before saving.
    Priority: top_k > mass > min_count.
    - top_k: keep top-K by 'count'
    - mass: keep by descending 'count' until cumulative probability >= mass
    - min_count: keep rows with count >= min_count
    """
    if dist_df.empty:
        return dist_df

    # Sort once by count desc for top_k / mass (stable mergesort preserves ties order)
    dist_df = dist_df.sort_values("count", ascending=False, kind="mergesort")

    if top_k and top_k > 0:
        return dist_df.head(top_k)

    if mass and 0.0 < mass < 1.0:
        total = dist_df["count"].sum()
        if total <= 0:
            return dist_df.iloc[0:0]
        cum = dist_df["count"].cumsum() / total
        kept = dist_df.loc[cum.le(mass)]
        # Ensure at least one row is kept if mass is too small
        if kept.empty and len(dist_df) > 0:
            kept = dist_df.head(1)
        return kept

    if min_count and min_count > 1:
        return dist_df.loc[dist_df["count"] >= min_count]

    return dist_df

# ====== Main ======
def main(args):
    workers = max(1, int(args.workers))
    batch_rows = max(1, int(args.batch_rows))

    total_batches = 0

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(args.tokenizer, args.n),
    ) as ex:
        futures = []
        for batch in iter_batches(args.input_texts, batch_rows):
            total_batches += 1
            futures.append(ex.submit(process_batch, batch))

        total_counter = Counter()
        for fut in as_completed(futures):
            c = fut.result()
            total_counter.update(c)

    # ---- Metrics on FULL distribution ----
    js = compute_js_divergence(total_counter)
    er = compute_entropy_ratio(total_counter)
    gini = compute_gini(total_counter)
    hhi = compute_hhi(total_counter)
    hhi_unb = compute_hhi_unbiased(total_counter)
    evs = compute_effective_vocab_size(hhi)
    evs_unb = compute_effective_vocab_size(hhi_unb)

    # Summary
    results = {
        "n": args.n,
        "unique_ngrams": len(total_counter),
        "js_distance_uniform": js,
        "entropy_ratio": er,
        "gini": gini,
        "hhi": hhi,
        "hhi_unbiased": hhi_unb,
        "effective_vocab_size": evs,
        "effective_vocab_size_unbiased": evs_unb,
        "batches": total_batches,
        "workers": workers,
        "batch_rows": batch_rows,
        "tokenizer": args.tokenizer,
        "input_texts": args.input_texts,
        "top_k_save": args.top_k_save,
        "mass_save": args.mass_save,
        "min_count_save": args.min_count_save,
    }

    print("=== Results ===")
    for k, v in results.items():
        print(f"{k}: {v}")

    # Save metrics
    metrics_df = pd.DataFrame([results])
    os.makedirs(os.path.dirname(args.output_metrics) or ".", exist_ok=True)
    metrics_df.to_csv(args.output_metrics, index=False)
    print(f"Saved metrics -> {args.output_metrics}")

    # ---- Build FULL distribution DataFrame (then filter only for saving) ----
    dist_df = pd.DataFrame(total_counter.items(), columns=["ngram", "count"])

    # Format n-gram ids to space-separated strings (for tuples)
    def _fmt(x: Union[int, Tuple[int, ...]]) -> str:
        if isinstance(x, tuple):
            return " ".join(map(str, x))
        return str(x)
    dist_df["ngram"] = dist_df["ngram"].apply(_fmt)

    # Apply save-time filtering
    dist_df = filter_distribution_df(
        dist_df,
        top_k=args.top_k_save,
        mass=args.mass_save,
        min_count=args.min_count_save,
    )

    # Save distribution (CSV/CSV.GZ/Parquet). If .csv.gz, compress automatically.
    os.makedirs(os.path.dirname(args.output_distribution) or ".", exist_ok=True)
    if args.output_distribution.endswith(".csv") or args.output_distribution.endswith(".csv.gz"):
        dist_df.to_csv(args.output_distribution, index=False, compression="infer")
    elif args.output_distribution.endswith(".parquet"):
        dist_df.to_parquet(args.output_distribution, index=False)
    else:
        raise ValueError("output_distribution must be .csv, .csv.gz, or .parquet")
    print(f"Saved distribution -> {args.output_distribution} (rows saved: {len(dist_df):,})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parallel n-gram distribution & metrics (Gini/HHI/EVS) with robust tokenizer loading and save-time filtering"
    )
    parser.add_argument("--n", type=int, required=True, help="n-gram size (n>=1)")
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to tokenizer dir/file (.json/.model) or HF repo id")
    parser.add_argument("--input_texts", type=str, required=True, help="jsonl(.gz) or parquet(.gz) with 'text' column")
    parser.add_argument("--output_metrics", type=str, default="metrics.csv", help="CSV for summary metrics")
    parser.add_argument("--output_distribution", type=str, default="distribution.parquet", help="CSV/CSV.GZ/Parquet for full distribution (after filtering)")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="number of worker processes")
    parser.add_argument("--batch_rows", type=int, default=10000, help="rows per batch sent to a worker")
    # Save-time filtering options
    parser.add_argument("--top_k_save", type=int, default=0, help="Save only top-K n-grams by count (0 disables)")
    parser.add_argument("--mass_save", type=float, default=0.0, help="Save until cumulative probability >= this mass in [0,1) (0 disables)")
    parser.add_argument("--min_count_save", type=int, default=1, help="Save only n-grams with count >= this threshold (1 keeps all)")
    args = parser.parse_args()
    main(args)
