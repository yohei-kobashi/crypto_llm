#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parallel n-gram counting and distribution analysis (with EVS).

- Reads a JSONL(.gz) or Parquet(.gz) file that has a "text" column.
- Tokenizes with a Hugging Face tokenizer.
- Counts n-gram frequencies in parallel (process-based).
- Computes:
    * Jensen–Shannon distance vs uniform
    * Entropy ratio (H/Hmax)
    * Gini coefficient (inequality of counts)
    * HHI (Herfindahl–Hirschman Index)
    * EVS (Effective Vocabulary Size): 1/HHI (naive) and 1/HHI_unbiased
- Saves summary metrics (CSV) and the full distribution (CSV or Parquet).
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
from transformers import AutoTokenizer

# ====== Globals used inside worker processes only ======
_GLOBALS = {
    "tokenizer": None,
    "n": 1,
}

def _init_worker(tokenizer_path: str, n: int):
    """Initialize each worker once: load the tokenizer and store `n`."""
    _GLOBALS["tokenizer"] = AutoTokenizer.from_pretrained(tokenizer_path)
    _GLOBALS["n"] = n

def make_ngrams(tokens: List[int], n: int) -> Iterable[Tuple[int, ...]]:
    """Yield n-grams from a token id list."""
    if n == 1:
        # Yield raw tokens (avoid tuple construction for performance)
        for t in tokens:
            yield t
        return
    if len(tokens) < n:
        return
    # Zip over shifted views for fast n-gram generation
    iters = [tokens[i:] for i in range(n)]
    for gram in zip(*iters):
        yield gram

def process_batch(texts: List[str]) -> Counter:
    """Process a batch of texts: tokenize and update an n-gram Counter."""
    tokenizer = _GLOBALS["tokenizer"]
    n = _GLOBALS["n"]
    c = Counter()
    for text in texts:
        if not isinstance(text, str):
            continue
        tokens = tokenizer.encode(text, add_special_tokens=False)
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
    Compute Gini coefficient of counts.
    Uses the cumulative-sum formula on ascending-sorted counts:
      G = (n + 1 - 2 * sum(cumsum(x)) / sum(x)) / n
    Returns NaN for empty input.
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
    Compute HHI (Herfindahl–Hirschman Index) on probabilities:
      HHI = sum(p_i^2), p_i = count_i / sum(counts)
    Range: (1/K .. 1], larger means more concentrated (less uniform).
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
    Unbiased estimator of HHI (Simpson index) under simple random sampling:
      HHI_unbiased = sum_i c_i (c_i - 1) / (N (N - 1)), where N = sum_i c_i
    Falls back to NaN if N <= 1.
    """
    if not freqs:
        return float("nan")
    counts = np.asarray(list(freqs.values()), dtype=float)
    N = counts.sum()
    if N <= 1:
        return float("nan")
    return float(np.sum(counts * (counts - 1)) / (N * (N - 1)))

def compute_effective_vocab_size(hhi_value: float) -> float:
    """
    Effective Vocabulary Size (EVS) for q=2 Hill number:
      EVS = 1 / HHI
    Returns NaN if HHI is not positive/finite.
    """
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
    Uses pyarrow for low-memory column projection if available; falls back to pandas otherwise.
    """
    try:
        import pyarrow.parquet as pq
        # Column projection
        table = pq.read_table(path, columns=["text"])
        col = table.column("text").to_pylist()
        for i in range(0, len(col), batch_rows):
            yield [str(x) for x in col[i : i + batch_rows] if x is not None]
    except Exception:
        # Fallback: pandas (be mindful of memory with huge files)
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

# ====== Main ======
def main(args):
    # Parallel setup
    workers = max(1, int(args.workers))
    batch_rows = max(1, int(args.batch_rows))

    total_batches = 0

    # Process batches in parallel and aggregate counters
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

    # Compute metrics
    js = compute_js_divergence(total_counter)
    er = compute_entropy_ratio(total_counter)
    gini = compute_gini(total_counter)
    hhi = compute_hhi(total_counter)
    hhi_unb = compute_hhi_unbiased(total_counter)
    evs = compute_effective_vocab_size(hhi)
    evs_unb = compute_effective_vocab_size(hhi_unb)

    # Print summary
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
    }

    print("=== Results ===")
    for k, v in results.items():
        print(f"{k}: {v}")

    # Save metrics
    metrics_df = pd.DataFrame([results])
    os.makedirs(os.path.dirname(args.output_metrics) or ".", exist_ok=True)
    metrics_df.to_csv(args.output_metrics, index=False)
    print(f"Saved metrics -> {args.output_metrics}")

    # Save full distribution (ngram ids joined by spaces)
    dist_df = pd.DataFrame(total_counter.items(), columns=["ngram", "count"])
    def _fmt(x: Union[int, Tuple[int, ...]]) -> str:
        if isinstance(x, tuple):
            return " ".join(map(str, x))
        return str(x)
    dist_df["ngram"] = dist_df["ngram"].apply(_fmt)

    os.makedirs(os.path.dirname(args.output_distribution) or ".", exist_ok=True)
    if args.output_distribution.endswith(".csv"):
        dist_df.to_csv(args.output_distribution, index=False)
    elif args.output_distribution.endswith(".parquet"):
        dist_df.to_parquet(args.output_distribution, index=False)
    else:
        raise ValueError("output_distribution must be .csv or .parquet")

    print(f"Saved distribution -> {args.output_distribution}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel n-gram distribution & metrics (Gini/HHI/EVS)")
    parser.add_argument("--n", type=int, required=True, help="n-gram size (n>=1)")
    parser.add_argument("--tokenizer", type=str, required=True, help="HF tokenizer path/name")
    parser.add_argument("--input_texts", type=str, required=True, help="jsonl(.gz) or parquet(.gz) with 'text' column")
    parser.add_argument("--output_metrics", type=str, default="metrics.csv", help="CSV for summary metrics")
    parser.add_argument("--output_distribution", type=str, default="distribution.parquet", help="CSV/Parquet for full distribution")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="number of worker processes")
    parser.add_argument("--batch_rows", type=int, default=10000, help="rows per batch sent to a worker")
    args = parser.parse_args()
    main(args)
