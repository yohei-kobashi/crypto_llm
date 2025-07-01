# suffix_array_search (stable build/query + test‑data generator)
# -------------------------------------------------------------
# 2025‑07‑01: Added `gen_test` sub‑command to create a small file of
# HIT/MISS lines for quick functional testing of the suffix‑array search.
# -------------------------------------------------------------
"""
Usage
-----
# Build index
python suffix_array_search.py build \
       --parquet-dir fineweb-edu/sample-10BT \
       --out-dir ./index \
       --workers auto

# Query index
python suffix_array_search.py query \
       --index-dir ./index \
       --input test.txt \
       --workers auto

# Generate test queries (20 HIT+MISS pairs = 40 lines)
python suffix_array_search.py gen_test \
       --parquet-dir fineweb-edu/sample-10BT \
       --output test.txt \
       --pairs 20
"""
from __future__ import annotations

import argparse, os, sys, pathlib, multiprocessing as mp, pickle, random, re
from functools import partial
from typing import List, Tuple, Dict

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

try:
    import pydivsufsort
except ImportError:
    raise SystemExit("Please `pip install pydivsufsort`.  (apt install libdivsufsort-dev)")

###############################################################################
# Globals & helpers
###############################################################################
WORD_RE = r"[\w'-]+"
W_RE = re.compile(WORD_RE)


def words(text: str) -> List[str]:
    return W_RE.findall(text.lower())


def get_n_workers(val: str | int | None) -> int:
    return min(72, mp.cpu_count()) if val in (None, "auto") else int(val)

###############################################################################
# Parquet iterators
###############################################################################


def _yield_text(parquet: str):
    pf = pq.ParquetFile(parquet)
    for batch in pf.iter_batches():
        for cell in batch.column("text"):
            yield cell.as_py()


def _yield_words(parquet: str):
    for txt in _yield_text(parquet):
        yield from words(txt)

###############################################################################
# Build‑phase workers (top‑level → picklable)
###############################################################################


def scan_vocab_worker(pq_path: str) -> List[str]:
    seen = {}
    for w in _yield_words(pq_path):
        seen[w] = None
    return list(seen)

# shared vocab for encode workers
VOCAB: Dict[str, int] | None = None


def enc_init(vocab_bytes: bytes):
    global VOCAB
    VOCAB = pickle.loads(vocab_bytes)


def encode_shard_worker(args):
    pq_path, out_dir = args
    assert VOCAB is not None
    ids = [VOCAB[w] for w in _yield_words(pq_path)]
    out = pathlib.Path(out_dir) / f"ids_{pathlib.Path(pq_path).stem}.npy"
    np.save(out, np.array(ids, dtype=np.int32))
    return out

###############################################################################
# Build orchestrator
###############################################################################

def build_index(pq_dir: str, out_dir: str, workers: int):
    os.makedirs(out_dir, exist_ok=True)
    shards = sorted(pathlib.Path(pq_dir).glob("*.parquet"))
    if not shards:
        raise FileNotFoundError("No parquet shards found.")

    try:
        ctx = mp.get_context("fork")
    except ValueError:
        ctx = mp.get_context("spawn")

    # Pass‑1: vocab
    vocab, next_id = {}, 1
    with ctx.Pool(workers) as pool:
        for wordlist in tqdm(pool.imap_unordered(scan_vocab_worker, shards),
                              total=len(shards), desc="Scanning vocab"):
            for w in wordlist:
                if w not in vocab:
                    vocab[w] = next_id; next_id += 1
    # 固定順で再付番して再現性確保
    vocab = {w: i+1 for i, w in enumerate(sorted(vocab))}
    np.save(os.path.join(out_dir, "vocab.npy"), np.array(list(vocab), dtype=object))

    # Pass‑2: encode shards
    vb = pickle.dumps(vocab, pickle.HIGHEST_PROTOCOL)
    enc_args = [(str(p), out_dir) for p in shards]
    with ctx.Pool(workers, initializer=enc_init, initargs=(vb,)) as pool:
        id_files = list(tqdm(pool.imap_unordered(encode_shard_worker, enc_args),
                             total=len(enc_args), desc="Encoding shards"))

    # Concatenate → open_memmap writes .npy with header
    total_len = sum(np.load(f, mmap_mode="r").shape[0] for f in id_files)
    all_ids_path = os.path.join(out_dir, "all_ids.npy")
    all_ids = np.lib.format.open_memmap(all_ids_path, mode="w+", dtype=np.int32,
                                        shape=(total_len,))
    off = 0
    for f in tqdm(id_files, desc="Concatenating ids"):
        arr = np.load(f, mmap_mode="r")
        n = arr.shape[0]
        all_ids[off:off+n] = arr; off += n
    del all_ids  # flush

    # SA build
    print(f"Building SA for {total_len:,} ids…")
    sa = pydivsufsort.divsufsort(np.load(all_ids_path, mmap_mode="r"))
    np.save(os.path.join(out_dir, "suffix_array.npy"), sa)
    print("Index complete ✔︎")

###############################################################################
# Query helpers
###############################################################################


def binary_search(ids: np.ndarray, sa: np.ndarray, pattern: List[int]) -> bool:
    lo, hi = 0, sa.size; m = len(pattern); pat = tuple(pattern)
    while lo < hi:
        mid = (lo + hi) // 2
        slice_cmp = tuple(ids[sa[mid]:sa[mid]+m])
        if slice_cmp < pat:
            lo = mid + 1
        else:
            hi = mid
    while lo < sa.size and tuple(ids[sa[lo]:sa[lo]+m]) == pat:
        return True
    return False


def query_index(idx_dir: str, in_txt: str, workers: int):
    vocab_arr = np.load(os.path.join(idx_dir, "vocab.npy"), allow_pickle=True)
    vocab = {w: i+1 for i, w in enumerate(vocab_arr)}
    ids = np.load(os.path.join(idx_dir, "all_ids.npy"), mmap_mode="r")
    sa = np.load(os.path.join(idx_dir, "suffix_array.npy"), mmap_mode="r")

    with open(in_txt, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    win = 35
    def check(line: str):
        seq = [vocab.get(w,0) for w in words(line)]
        if len(seq) < win: return line, False
        for i in range(len(seq)-win+1):
            wdw = seq[i:i+win]
            if 0 in wdw: continue
            if binary_search(ids, sa, wdw):
                return line, True
        return line, False

    from multiprocessing.pool import ThreadPool
    with ThreadPool(workers) as pool:
        for ln, ok in tqdm(pool.imap_unordered(check, lines), total=len(lines), desc="Querying"):
            print(f"[{'HIT' if ok else 'MISS'}] {ln[:120]}{'…' if len(ln)>120 else ''}")

###############################################################################
# Test‑query generator
###############################################################################

def generate_test_queries(pq_dir: str, out_path: str, pairs: int, win: int = 35):
    shards = list(pathlib.Path(pq_dir).glob("*.parquet"))
    if not shards:
        raise FileNotFoundError("No parquet shards found.")

    random.seed(42)
    hits, misses = [], []

    for shard in tqdm(random.sample(shards, len(shards)), desc="Sampling shards"):
        for txt in _yield_text(str(shard)):
            w = words(txt)
            if len(w) < win: continue
            # pick random window
            start = random.randint(0, len(w)-win)
            seg = w[start:start+win]
            hit_line = " ".join(seg)
            # MISS: replace middle word with unique token unlikely in corpus
            miss_seg = seg.copy()
            miss_seg[win//2] = "xyzxyzxyzunique"  # unlikely present
            miss_line = " ".join(miss_seg)
            hits.append(hit_line)
            misses.append(miss_line)
            if len(hits) >= pairs:
                break
        if len(hits) >= pairs:
            break

    with open(out_path, "w", encoding="utf-8") as f:
        for h, m in zip(hits, misses):
            f.write(h + "\n")
            f.write(m + "\n")
    print(f"Wrote {len(hits)*2} lines (HIT/MISS pairs) to {out_path}")

###############################################################################
# CLI
###############################################################################

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build"); b.add_argument("--parquet-dir"); b.add_argument("--out-dir"); b.add_argument("--workers", default="auto")
    q = sub.add_parser("query"); q.add_argument("--index-dir"); q.add_argument("--input"); q.add_argument("--workers", default="auto")
    g = sub.add_parser("gen_test"); g.add_argument("--parquet-dir"); g.add_argument("--output"); g.add_argument("--pairs", type=int, default=20)

    a = ap.parse_args(); w = get_n_workers(getattr(a, "workers", "auto"))
    if a.cmd == "build":
        build_index(a.parquet_dir, a.out_dir, w)
    elif a.cmd == "query":
        query_index(a.index_dir, a.input, w)
    else:  # gen_test
        generate_test_queries(a.parquet_dir, a.output, a.pairs)

if __name__ == "__main__":
    mp.freeze_support(); main()
