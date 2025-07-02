# suffix_array_search (build/query + test-data + parquet splitter)
# -------------------------------------------------------------
# 2025-07-01: Added `gen_test` sub-command.
# 2025-07-01: Added `split` sub-command (fixed parts mode).
# 2025-07-02: Updated `query` to display total lines, hit count, and hit ratio.
# 2025-07-03: Refactored query workers to avoid assignment expressions in comprehensions.
# 2025-07-04: Fixed indentation for test-data generator and splitter functions.
# 2025-07-05: Added timing output to `query` sub-command.
# -------------------------------------------------------------
"""
Usage (main commands)
--------------------
# Build suffix-array index
python suffix_array_search.py build \
       --parquet-dir fineweb-edu/sample-10BT \
       --out-dir ./index \
       --workers auto

# Query index with timing
python suffix_array_search.py query \
       --index-dir ./index \
       --input queries.txt \
       --workers auto

# Generate tiny HIT/MISS test set (40 lines)
python suffix_array_search.py gen_test \
       --parquet-dir fineweb-edu/sample-10BT \
       --output queries.txt \
       --pairs 20

# Split each Parquet file into exactly 8 equal parts
python suffix_array_search.py split \
       --parquet-dir fineweb-edu/sample-10BT \
       --out-dir fineweb-edu/sample-10BT-split8 \
       --parts 8 \
       --workers auto
"""
from __future__ import annotations
import argparse
import os
import sys
import pathlib
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
import pickle
import random
import re
import time
from functools import partial
from typing import List, Tuple, Dict
import json

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm
import itertools

try:
    import pydivsufsort
except ImportError:
    raise SystemExit("Please `pip install pydivsufsort`.  (apt install libdivsufsort-dev)")

# Globals & helpers
WORD_RE = r"[\w'-]+"
W_RE = re.compile(WORD_RE)
WIN = 35
BATCH_SIZE=5000
PAGE_WARMUP = 1000000  # number of elements to touch for warm-up

def words(text: str) -> List[str]:
    return W_RE.findall(text.lower())

def get_n_workers(val: str | int | None) -> int:
    return mp.cpu_count() if val in (None, "auto") else int(val)

def chunked_file(path: str, chunk_size: int):
    """Yield lists of text snippets (str) chunked from a txt or jsonl file."""
    ext = pathlib.Path(path).suffix.lower()
    with open(path, encoding='utf-8') as f:
        while True:
            lines = list(itertools.islice(f, chunk_size))
            if not lines:
                break
            out: List[str] = []
            if ext == '.jsonl':
                for row in lines:
                    row = row.strip()
                    if not row:
                        continue
                    try:
                        obj = json.loads(row)
                        text = obj.get('text', '')
                        if text:
                            out.append(text)
                    except json.JSONDecodeError:
                        continue
            else:
                for l in lines:
                    l = l.strip()
                    if l:
                        out.append(l)
            yield out
            
# Parquet iterators
def _yield_text(parquet: str):
    pf = pq.ParquetFile(parquet)
    for batch in pf.iter_batches():
        for cell in batch.column("text"):
            yield cell.as_py()

def _yield_words(parquet: str):
    for txt in _yield_text(parquet):
        yield from words(txt)

# Build-phase workers
def scan_vocab_worker(pq_path: str) -> List[str]:
    seen: Dict[str, None] = {}
    for w in _yield_words(pq_path):
        seen[w] = None
    return list(seen)

VOCAB: Dict[str,int] | None = None

def enc_init(vocab_bytes: bytes):
    global VOCAB
    VOCAB = pickle.loads(vocab_bytes)

def encode_shard_worker(args: Tuple[str, str]) -> pathlib.Path:
    pq_path, out_dir = args
    assert VOCAB is not None
    ids = [VOCAB.get(w,0) for w in _yield_words(pq_path)]
    out = pathlib.Path(out_dir) / f"ids_{pathlib.Path(pq_path).stem}.npy"
    np.save(out, np.array(ids, dtype=np.int32))
    return out

# Build orchestrator
def build_index(pq_dir: str, out_dir: str, workers: int):
    os.makedirs(out_dir, exist_ok=True)
    shards = sorted(pathlib.Path(pq_dir).glob("*.parquet"))
    if not shards:
        sys.exit("No parquet shards found.")
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        ctx = mp.get_context("spawn")

    # Pass-1: vocab
    vocab: Dict[str,int] = {}
    next_id = 1
    with ctx.Pool(workers) as pool:
        for wordlist in tqdm(pool.imap_unordered(scan_vocab_worker, shards),
                              total=len(shards), desc="Scanning vocab"):
            for w in wordlist:
                if w not in vocab:
                    vocab[w] = next_id; next_id += 1
    vocab = {w:i+1 for i,w in enumerate(sorted(vocab.keys()))}
    np.save(os.path.join(out_dir, "vocab.npy"), np.array(list(vocab.keys()), dtype=object))

    # Pass-2: encode
    vb = pickle.dumps(vocab, pickle.HIGHEST_PROTOCOL)
    enc_args = [(str(p), out_dir) for p in shards]
    with ctx.Pool(workers, initializer=enc_init, initargs=(vb,)) as pool:
        id_files = list(tqdm(pool.imap_unordered(encode_shard_worker, enc_args),
                             total=len(enc_args), desc="Encoding shards"))

    # Concat ids
    total_len = sum(np.load(f, mmap_mode='r').shape[0] for f in id_files)
    all_ids_path = os.path.join(out_dir, 'all_ids.npy')
    all_ids = np.lib.format.open_memmap(all_ids_path, mode='w+', dtype=np.int32, shape=(total_len,))
    off = 0
    for f in tqdm(id_files, desc="Concatenating ids"):
        arr = np.load(f, mmap_mode='r'); n = arr.shape[0]
        all_ids[off:off+n] = arr; off += n
    del all_ids

    # SA build
    print(f"Building SA for {total_len:,} ids…")
    sa = pydivsufsort.divsufsort(np.load(all_ids_path, mmap_mode='r'))
    np.save(os.path.join(out_dir, 'suffix_array.npy'), sa)
    print("Index complete ✔︎")

# Query helpers
def binary_search(ids: np.ndarray, sa: np.ndarray, pattern: List[int]) -> bool:
    lo, hi = 0, sa.size; m = len(pattern); pat = tuple(pattern)
    while lo < hi:
        mid = (lo + hi) // 2
        slice_cmp = tuple(ids[sa[mid]:sa[mid]+m])
        if slice_cmp < pat: lo = mid + 1
        else: hi = mid
    while lo < sa.size and tuple(ids[sa[lo]:sa[lo]+m]) == pat:
        return True
    return False

def query_line(line: str, vocab: Dict[str,int], ids: np.ndarray, sa: np.ndarray, win: int) -> Tuple[str,bool]:
    seq = [vocab.get(w,0) for w in words(line)]
    if len(seq) < win: return line, False
    for i in range(len(seq) - win + 1):
        window = seq[i:i+win]
        if 0 in window: continue
        if binary_search(ids, sa, window): return line, True
    return line, False

def query_index(idx_dir: str, in_path: str, workers: int, win: int):
    start_time = time.time()
    vocab_arr = np.load(os.path.join(idx_dir,'vocab.npy'), allow_pickle=True)
    vocab = {w:i+1 for i,w in enumerate(vocab_arr)}
    ids = np.load(os.path.join(idx_dir,'all_ids.npy'), mmap_mode='r')
    sa = np.load(os.path.join(idx_dir,'suffix_array.npy'), mmap_mode='r')
    # Page warm-up to pre-load mmap pages
    _ = ids[:PAGE_WARMUP]
    _ = sa[:PAGE_WARMUP]    
    # # Load input lines from .txt or .jsonl
    # lines: List[str] = []
    # ext = pathlib.Path(in_path).suffix.lower()
    # if ext == '.jsonl':
    #     with open(in_path, encoding='utf-8') as f:
    #         for row in f:
    #             if row.strip():
    #                 try:
    #                     obj = json.loads(row)
    #                     text = obj.get('text', '')
    #                     if text: lines.append(text)
    #                 except json.JSONDecodeError:
    #                     continue
    # else:
    #     with open(in_path, encoding='utf-8') as f:
    #         for l in f:
    #             l = l.strip()
    #             if l: lines.append(l)
    
    worker = partial(query_line, vocab=vocab, ids=ids, sa=sa, win=win)
    total = 0
    hits = 0
    for i, batch in enumerate(chunked_file(in_path, BATCH_SIZE)):
        for _, ok in tqdm(ThreadPool(workers).imap_unordered(worker, batch), total=len(batch), desc=f"Batch {i}"):
            total += 1
            if ok: hits += 1
            # print(f"[{'HIT' if ok else 'MISS'}] {ln[:120]}{'…' if len(ln)>120 else ''}")
            
    ratio = hits/total*100 if total else 0
    elapsed = time.time() - start_time
    print(f"Total lines: {total}, Hits: {hits}, Hit ratio: {ratio:.2f}%")
    print(f"Query time: {elapsed:.2f} seconds")

# Test-data generator
def generate_test_queries(pq_dir: str, out_path: str, pairs: int, win: int):
    shards = list(pathlib.Path(pq_dir).glob("*.parquet"))
    if not shards:
        sys.exit("No parquet shards found.")
    random.seed(42)
    hits, misses = [], []
    for shard in tqdm(random.sample(shards, len(shards)), desc="Sampling shards"):
        for txt in _yield_text(str(shard)):
            w = words(txt)
            if len(w) < win: continue
            start = random.randint(0, len(w)-win)
            seg = w[start:start+win]
            hits.append(" ".join(seg))
            mid = win // 2
            miss = seg.copy(); miss[mid] = "xyzxyzxyzunique"
            misses.append(" ".join(miss))
            if len(hits) >= pairs: break
        if len(hits) >= pairs: break
    with open(out_path, "w", encoding='utf-8') as f:
        for h, m in zip(hits, misses):
            f.write(h + "\n")
            f.write(m + "\n")
    print(f"Wrote {len(hits)*2} lines to {out_path}")

# Parquet splitter
def split_parquet_file_parts(pq_path: str, out_dir: str, parts: int):
    tbl = pq.read_table(pq_path)
    rows = tbl.num_rows
    base = pathlib.Path(pq_path).stem
    size = rows // parts; rem = rows % parts
    for i in range(parts):
        start = i*size + min(i, rem)
        cnt = size + (1 if i<rem else 0)
        slice_tbl = tbl.slice(start, cnt)
        out = pathlib.Path(out_dir)/f"{base}_part{i}.parquet"
        pq.write_table(slice_tbl, out, compression='zstd')

def split_parquet_dir(pq_dir: str, out_dir: str, parts: int, workers: int):
    os.makedirs(out_dir, exist_ok=True)
    shards = sorted(pathlib.Path(pq_dir).glob("*.parquet"))
    if not shards:
        sys.exit("No parquet shards found.")
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        ctx = mp.get_context("spawn")
    func = partial(split_parquet_file_parts, out_dir=out_dir, parts=parts)
    with ctx.Pool(workers) as pool:
        list(tqdm(pool.imap_unordered(func, shards), total=len(shards), desc="Splitting parquet"))
    print("Splitting complete ✔︎")

# CLI and entry point
def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    # build
    p = sub.add_parser("build")
    p.add_argument("--parquet-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--workers", default="auto")
    # query
    q = sub.add_parser("query")
    q.add_argument("--index-dir", required=True)
    q.add_argument("--input", required=True)
    q.add_argument("--workers", default="auto")
    q.add_argument("--window", type=int, default=WIN)
    # gen_test
    g = sub.add_parser("gen_test")
    g.add_argument("--parquet-dir", required=True)
    g.add_argument("--output", required=True)
    g.add_argument("--pairs", type=int, default=20)
    g.add_argument("--window", type=int, default=WIN)
    # split
    s = sub.add_parser("split")
    s.add_argument("--parquet-dir", required=True)
    s.add_argument("--out-dir", required=True)
    s.add_argument("--parts", type=int, required=True)
    s.add_argument("--workers", default="auto")
    args = parser.parse_args()
    if args.cmd == "build":
        w = get_n_workers(args.workers)
        build_index(args.parquet_dir, args.out_dir, w)
    elif args.cmd == "query":
        w = get_n_workers(args.workers)
        query_index(args.index_dir, args.input, w, args.window)
    elif args.cmd == "gen_test":
        generate_test_queries(args.parquet_dir, args.output, args.pairs, args.window)
    elif args.cmd == "split":
        w = get_n_workers(args.workers)
        split_parquet_dir(args.parquet_dir, args.out_dir, args.parts, w)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
