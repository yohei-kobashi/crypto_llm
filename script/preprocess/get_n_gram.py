#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parallel n-gram counting with robust tokenizer loading, save-time filtering,
and integrated entropy metrics:

- Counts n-gram frequencies in parallel (process-based).
- Saves ONE distribution for the requested order (--n), with optional filters.
- Computes global metrics on the FULL counters (before save filters):
    * Jensen–Shannon distance vs uniform (for saved order n only)
    * Entropy ratio H/Hmax (for saved order n only)
    * Gini, HHI, HHI_unbiased, EVS (=1/HHI) (for saved order n only)
    * H1 (unigram entropy, bits)
    * Conditional entropies H|1, H|2, H|3 (if --metrics_max_k>=1/2/3)
    * Mutual information I_k = H1 - H|k  (k=1..3)
    * Perplexity PPL_k = 2^(H|k)        (k=1..3)

Tokenizer loading (auto):
- Local directory  -> AutoTokenizer.from_pretrained(dir, local_files_only=True)
- *.json           -> PreTrainedTokenizerFast(tokenizer_file=JSON)
- *.model          -> sentencepiece.SentencePieceProcessor(model_file=SPM)
- Otherwise        -> HF Hub id via AutoTokenizer.from_pretrained(id)
"""

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy
from transformers import AutoTokenizer, PreTrainedTokenizerFast

# ====== Globals used inside worker processes only ======
_GLOBALS = {
    "kind": None,          # 'hf' or 'spm'
    "tokenizer": None,     # HF tokenizer
    "sp": None,            # sentencepiece.SentencePieceProcessor
    "orders": None,        # list[int], e.g., [1,2,3,4]
}

# ---------------- Tokenizer loading ----------------
def _load_tokenizer_any(tokenizer_path: str):
    """Auto-detect tokenizer format and load appropriately."""
    if os.path.isdir(tokenizer_path):
        tok = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        return {"kind": "hf", "tokenizer": tok}
    if os.path.isfile(tokenizer_path):
        lower = tokenizer_path.lower()
        if lower.endswith(".json"):
            tok = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
            return {"kind": "hf", "tokenizer": tok}
        if lower.endswith(".model"):
            try:
                import sentencepiece as spm
            except ImportError as e:
                raise RuntimeError(
                    "You passed a SentencePiece .model file but `sentencepiece` is not installed. "
                    "Please `pip install sentencepiece`."
                ) from e
            sp = spm.SentencePieceProcessor()
            sp.load(tokenizer_path)
            return {"kind": "spm", "sp": sp}
    # Fallback: HF Hub id
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    return {"kind": "hf", "tokenizer": tok}

def _init_worker(tokenizer_path: str, orders: List[int]):
    """Initialize each worker once: load tokenizer and target orders."""
    info = _load_tokenizer_any(tokenizer_path)
    _GLOBALS["kind"] = info["kind"]
    _GLOBALS["tokenizer"] = info.get("tokenizer")
    _GLOBALS["sp"] = info.get("sp")
    _GLOBALS["orders"] = list(sorted(set(orders)))

def _encode(text: str) -> List[int]:
    """Encode a single text to token ids using the loaded tokenizer/SPM."""
    if _GLOBALS["kind"] == "spm":
        return _GLOBALS["sp"].encode(text, out_type=int)
    return _GLOBALS["tokenizer"].encode(text, add_special_tokens=False)

def make_ngrams(tokens: List[int], n: int) -> Iterable[Tuple[int, ...]]:
    """Yield n-grams from a token id list."""
    L = len(tokens)
    if L < n:
        return
    if n == 1:
        for t in tokens:
            yield t
        return
    # Fast zip over shifted views
    iters = [tokens[i:] for i in range(n)]
    for gram in zip(*iters):
        yield gram

def process_batch(texts: List[str]) -> Dict[int, Counter]:
    """
    Process a batch: tokenize and update n-gram Counters for all requested orders.
    Returns: dict {order: Counter}
    """
    orders = _GLOBALS["orders"]
    out = {o: Counter() for o in orders}
    for text in texts:
        if not isinstance(text, str):
            continue
        tokens = _encode(text)
        for o in orders:
            out[o].update(make_ngrams(tokens, o))
    return out

# ---------------- Metrics on a Counter ----------------
def compute_js_divergence(freqs: Counter) -> float:
    """Jensen–Shannon distance vs uniform."""
    if not freqs:
        return float("nan")
    counts = np.asarray(list(freqs.values()), dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return float("nan")
    p = counts / total
    u = np.ones_like(p) / len(p)
    m = 0.5 * (p + u)
    kl_pm = scipy_entropy(p, m, base=2)
    kl_um = scipy_entropy(u, m, base=2)
    return float(0.5 * (kl_pm + kl_um))

def compute_entropy_ratio(freqs: Counter) -> float:
    """H(P)/Hmax where Hmax = log2(K)."""
    if not freqs:
        return float("nan")
    counts = np.asarray(list(freqs.values()), dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return float("nan")
    p = counts / total
    H = scipy_entropy(p, base=2)
    Hmax = math.log(len(p), 2) if len(p) > 0 else 0.0
    return float(H / Hmax) if Hmax > 0 else float("nan")

def compute_gini(freqs: Counter) -> float:
    """Gini coefficient on counts."""
    if not freqs:
        return float("nan")
    x = np.asarray(list(freqs.values()), dtype=np.float64)
    if x.sum() <= 0:
        return float("nan")
    xs = np.sort(x)
    n = xs.size
    cumx = np.cumsum(xs)
    g = (n + 1 - 2.0 * np.sum(cumx) / cumx[-1]) / n
    return float(max(0.0, min(1.0, g)))

def compute_hhi(freqs: Counter) -> float:
    """HHI = sum p_i^2."""
    if not freqs:
        return float("nan")
    x = np.asarray(list(freqs.values()), dtype=np.float64)
    s = x.sum()
    if s <= 0:
        return float("nan")
    p = x / s
    return float(np.sum(p * p))

def compute_hhi_unbiased(freqs: Counter) -> float:
    """Unbiased Simpson index: sum c_i(c_i-1) / (N(N-1))."""
    if not freqs:
        return float("nan")
    x = np.asarray(list(freqs.values()), dtype=np.float64)
    N = x.sum()
    if N <= 1:
        return float("nan")
    return float(np.sum(x * (x - 1)) / (N * (N - 1)))

def evs_from_hhi(hhi: float) -> float:
    """EVS = 1/HHI."""
    if not np.isfinite(hhi) or hhi <= 0:
        return float("nan")
    return float(1.0 / hhi)

# ---- Entropy (bits) from counters ----
def entropy_bits_from_unigram(unigram: Counter) -> float:
    """H1 (bits) from unigram counts."""
    if not unigram:
        return float("nan")
    x = np.asarray(list(unigram.values()), dtype=np.float64)
    N = x.sum()
    if N <= 0:
        return float("nan")
    # H = log2 N - (sum c log c)/N / log 2
    sum_c_logc = np.sum(x * np.log(x[x > 0]))
    H_nat = np.log(N) - (sum_c_logc / N)
    return float(H_nat / math.log(2))

def conditional_entropy_bits_from_kplus1(kplus1: Counter, k: int) -> float:
    """
    H(X_t | X_{t-1:t-k}) from (k+1)-gram counts.
    H = (1/N) * sum_{context, w} c_{cw} * (log S_c - log c_{cw})
      where S_c = sum_w c_{cw} (total count for each context).
    """
    if not kplus1:
        return float("nan")
    # Build context totals S_c
    context_totals = defaultdict(int)
    N_total = 0
    for gram, cnt in kplus1.items():
        # gram is int (n=1) or tuple (n>1)
        if not isinstance(gram, tuple):
            # (k+1)-gram must be tuple
            return float("nan")
        context = gram[:-1]  # first k ids
        context_totals[context] += cnt
        N_total += cnt
    if N_total <= 0:
        return float("nan")

    # Accumulate sum c*(log S_c - log c)
    sum_term = 0.0
    for gram, cnt in kplus1.items():
        context = gram[:-1]
        S_c = context_totals[context]
        if cnt > 0 and S_c > 0:
            sum_term += cnt * (math.log(S_c) - math.log(cnt))
    H_nat = sum_term / N_total
    return float(H_nat / math.log(2))

def ppl_from_entropy_bits(H_bits: float) -> float:
    """Perplexity = 2^H_bits."""
    if not np.isfinite(H_bits):
        return float("nan")
    return float(2.0 ** H_bits)

# ---------------- Streaming input ----------------
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
    """Yield lists of texts from a Parquet(.gz) file."""
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

# ---------------- Save-time filtering ----------------
def filter_distribution_df(dist_df: pd.DataFrame,
                           top_k: int = 0,
                           mass: float = 0.0,
                           min_count: int = 1) -> pd.DataFrame:
    """
    Filter rows before saving.
    Priority: top_k > mass > min_count.
    """
    if dist_df.empty:
        return dist_df
    dist_df = dist_df.sort_values("count", ascending=False, kind="mergesort")

    if top_k and top_k > 0:
        return dist_df.head(top_k)

    if mass and 0.0 < mass < 1.0:
        total = dist_df["count"].sum()
        if total <= 0:
            return dist_df.iloc[0:0]
        cum = dist_df["count"].cumsum() / total
        kept = dist_df.loc[cum.le(mass)]
        if kept.empty and len(dist_df) > 0:
            kept = dist_df.head(1)
        return kept

    if min_count and min_count > 1:
        return dist_df.loc[dist_df["count"] >= min_count]

    return dist_df

# ---------------- Main ----------------
def main(args):
    # Determine which orders to count:
    # - always include the saved order `args.n`
    # - include additional orders needed for metrics: 1 and (k+1 for k=1..metrics_max_k)
    orders = {int(args.n)}
    if args.metrics_max_k is not None and args.metrics_max_k > 0:
        orders.add(1)
        for k in range(1, min(3, args.metrics_max_k) + 1):
            orders.add(k + 1)
    orders = sorted(orders)

    # Parallel counting
    total_batches = 0
    merged: Dict[int, Counter] = {o: Counter() for o in orders}

    with ProcessPoolExecutor(
        max_workers=max(1, int(args.workers)),
        initializer=_init_worker,
        initargs=(args.tokenizer, orders),
    ) as ex:
        futures = []
        for batch in iter_batches(args.input_texts, max(1, int(args.batch_rows))):
            total_batches += 1
            futures.append(ex.submit(process_batch, batch))
        for fut in as_completed(futures):
            part = fut.result()  # dict{order: Counter}
            for o, c in part.items():
                merged[o].update(c)

    # ---------- Metrics for the saved order (args.n) ----------
    saved_order = int(args.n)
    saved_counter = merged.get(saved_order, Counter())

    js = compute_js_divergence(saved_counter)
    er = compute_entropy_ratio(saved_counter)
    gini = compute_gini(saved_counter)
    hhi = compute_hhi(saved_counter)
    hhi_unb = compute_hhi_unbiased(saved_counter)
    evs = evs_from_hhi(hhi)
    evs_unb = evs_from_hhi(hhi_unb)

    # ---------- Entropy ladder (H1, H|1..3), MI, PPL ----------
    H1_bits = entropy_bits_from_unigram(merged.get(1, Counter())) if 1 in merged else float("nan")

    Hcond1_bits = conditional_entropy_bits_from_kplus1(merged.get(2, Counter()), k=1) if 2 in merged else float("nan")
    Hcond2_bits = conditional_entropy_bits_from_kplus1(merged.get(3, Counter()), k=2) if 3 in merged else float("nan")
    Hcond3_bits = conditional_entropy_bits_from_kplus1(merged.get(4, Counter()), k=3) if 4 in merged else float("nan")

    I1_bits = H1_bits - Hcond1_bits if np.isfinite(H1_bits) and np.isfinite(Hcond1_bits) else float("nan")
    I2_bits = H1_bits - Hcond2_bits if np.isfinite(H1_bits) and np.isfinite(Hcond2_bits) else float("nan")
    I3_bits = H1_bits - Hcond3_bits if np.isfinite(H1_bits) and np.isfinite(Hcond3_bits) else float("nan")

    PPL1 = ppl_from_entropy_bits(Hcond1_bits)
    PPL2 = ppl_from_entropy_bits(Hcond2_bits)
    PPL3 = ppl_from_entropy_bits(Hcond3_bits)

    PPL_improve_1_to_2 = (PPL1 / PPL2) if np.isfinite(PPL1) and np.isfinite(PPL2) and PPL2 > 0 else float("nan")
    PPL_improve_2_to_3 = (PPL2 / PPL3) if np.isfinite(PPL2) and np.isfinite(PPL3) and PPL3 > 0 else float("nan")

    # ---------- Print & save metrics ----------
    results = {
        "saved_order_n": saved_order,
        "unique_ngrams_saved_order": len(saved_counter),
        "js_distance_uniform": js,
        "entropy_ratio": er,
        "gini": gini,
        "hhi": hhi,
        "hhi_unbiased": hhi_unb,
        "effective_vocab_size": evs,
        "effective_vocab_size_unbiased": evs_unb,

        "H1_bits": H1_bits,
        "Hcond1_bits": Hcond1_bits,
        "Hcond2_bits": Hcond2_bits,
        "Hcond3_bits": Hcond3_bits,
        "MI1_bits": I1_bits,
        "MI2_bits": I2_bits,
        "MI3_bits": I3_bits,
        "PPL1": PPL1,
        "PPL2": PPL2,
        "PPL3": PPL3,
        "PPL_improve_1_to_2": PPL_improve_1_to_2,
        "PPL_improve_2_to_3": PPL_improve_2_to_3,

        "batches": total_batches,
        "workers": int(args.workers),
        "batch_rows": int(args.batch_rows),
        "tokenizer": args.tokenizer,
        "input_texts": args.input_texts,
        "orders_counted": ",".join(map(str, orders)),
        "top_k_save": int(args.top_k_save),
        "mass_save": float(args.mass_save),
        "min_count_save": int(args.min_count_save),
    }

    print("=== Results ===")
    for k, v in results.items():
        print(f"{k}: {v}")

    # Metrics CSV
    os.makedirs(os.path.dirname(args.output_metrics) or ".", exist_ok=True)
    pd.DataFrame([results]).to_csv(args.output_metrics, index=False)
    print(f"Saved metrics -> {args.output_metrics}")

    # ---------- Save ONE distribution (for saved_order) with filters ----------
    # Build DataFrame from saved_counter
    dist_df = pd.DataFrame(saved_counter.items(), columns=["ngram", "count"])

    # Format n-gram as space-separated token ids for tuples
    def _fmt(x: Union[int, Tuple[int, ...]]) -> str:
        if isinstance(x, tuple):
            return " ".join(map(str, x))
        return str(x)

    dist_df["ngram"] = dist_df["ngram"].apply(_fmt)

    # Keep a note of saved mass fraction (optional, useful for logs)
    total_count_saved_order = float(dist_df["count"].sum()) if not dist_df.empty else 0.0

    # Apply save-time filters
    filtered_df = filter_distribution_df(
        dist_df,
        top_k=args.top_k_save,
        mass=args.mass_save,
        min_count=args.min_count_save,
    )

    saved_mass_fraction = (
        float(filtered_df["count"].sum()) / total_count_saved_order
        if total_count_saved_order > 0 else float("nan")
    )
    print(f"Saving order n={saved_order} with {len(filtered_df):,} rows "
          f"(mass kept ≈ {saved_mass_fraction:.4f})")

    # Save distribution (CSV/CSV.GZ/Parquet)
    os.makedirs(os.path.dirname(args.output_distribution) or ".", exist_ok=True)
    if args.output_distribution.endswith(".csv") or args.output_distribution.endswith(".csv.gz"):
        filtered_df.to_csv(args.output_distribution, index=False, compression="infer")
    elif args.output_distribution.endswith(".parquet"):
        filtered_df.to_parquet(args.output_distribution, index=False)
    else:
        raise ValueError("output_distribution must be .csv, .csv.gz, or .parquet")
    print(f"Saved distribution -> {args.output_distribution}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parallel n-gram counting & metrics (JS/H/HGI/HHI/EVS + H1/H|k/MI/PPL) "
                    "with robust tokenizer loading and save-time filtering"
    )
    # Which distribution to SAVE
    parser.add_argument("--n", type=int, required=True, help="Order to SAVE the distribution for (e.g., 1,2,3,4)")

    # Metrics ladder up to k (0..3): will count needed orders internally
    parser.add_argument("--metrics_max_k", type=int, default=3,
                        help="Max context length k to compute conditional entropy (0..3). "
                             "k=3 requires counting up to 4-grams. Set smaller if memory is tight.")

    # Tokenizer & data
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Path to tokenizer dir/file (.json/.model) or HF repo id")
    parser.add_argument("--input_texts", type=str, required=True,
                        help="jsonl(.gz) or parquet(.gz) with 'text' column")

    # Parallelism & batching
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4,
                        help="Number of worker processes")
    parser.add_argument("--batch_rows", type=int, default=10000,
                        help="Rows per batch sent to a worker")

    # Metrics & distribution outputs
    parser.add_argument("--output_metrics", type=str, default="metrics.csv",
                        help="CSV for summary metrics")
    parser.add_argument("--output_distribution", type=str, default="distribution.parquet",
                        help="CSV/CSV.GZ/Parquet for the SAVED order (after filtering)")

    # Save-time filtering options
    parser.add_argument("--top_k_save", type=int, default=0,
                        help="Save only top-K n-grams by count (0 disables)")
    parser.add_argument("--mass_save", type=float, default=0.0,
                        help="Save until cumulative probability >= this mass in [0,1) (0 disables)")
    parser.add_argument("--min_count_save", type=int, default=1,
                        help="Save only n-grams with count >= this threshold (1 keeps all)")

    args = parser.parse_args()
    # Clamp metrics_max_k to [0,3]
    if args.metrics_max_k is None:
        args.metrics_max_k = 0
    args.metrics_max_k = max(0, min(3, int(args.metrics_max_k)))
    main(args)
