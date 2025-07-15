#!/usr/bin/env python3
"""
Usage (main commands)
--------------------
# Build suffix-array index
python suffix_array_search.py build \
       --data-dir fineweb-edu/sample-10BT \
       --out-dir ./index \
       --workers auto

# Query index with timing
python suffix_array_search.py query \
       --index-dir ./index \
       --input queries.txt \
       --window 35

# Generate tiny HIT/MISS test set (40 lines)
python suffix_array_search.py gen_test \
       --data-dir fineweb-edu/sample-10BT \
       --output queries.txt \
       --pairs 20

# Split each Parquet file into exactly 8 equal parts
python suffix_array_search.py split \
       --file-dir fineweb-edu/sample-10BT \
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
import zlib

try:
    import pydivsufsort
except ImportError:
    raise SystemExit("Please `pip install pydivsufsort`. (apt install libdivsufsort-dev)")

# Globals & helpers
WORD_RE = r"[\w'-]+"
W_RE = re.compile(WORD_RE)
WIN = 35
BATCH_SIZE = 5000
PAGE_WARMUP = 1000000  # for warm-up, unused here
LOW  = 0.2751729438893159

def hzlib_bits_per_char(text: str) -> float:
    text = re.sub(r'\d', '0', text)
    raw  = len(text.encode('utf-8'))
    comp = len(zlib.compress(text.encode('utf-8')))
    return comp / raw

def filter_match(span: str) -> bool:
    """True to accept, False to exclude"""
    hz = hzlib_bits_per_char(span)
    return LOW < hz

def words(text: str) -> List[str]:
    return W_RE.findall(text.lower())

def get_n_workers(val: str | int | None) -> int:
    return mp.cpu_count() if val in (None, "auto") else int(val)

def read_input_file(path: str):
    """Yield list of text snippets from .txt or .jsonl"""
    ext = pathlib.Path(path).suffix.lower()
    out: List[str] = []
    for row in open(path, encoding='utf-8'):
        row = row.strip()
        if not row:
            continue
        if ext == '.jsonl':
            try:
                obj = json.loads(row)
                text = obj.get('text', '')
                if text:
                    out.append(text)
            except json.JSONDecodeError:
                continue
        else:
            out.append(row)
    return out

def yield_words_file(path: str):
    """Yield each word from .parquet or .jsonl files"""
    ext = pathlib.Path(path).suffix.lower()
    if ext == '.parquet':
        try:
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches():
                for cell in batch.column('text'):
                    for w in words(cell.as_py()):
                        yield w
            return
        except:
            pass  # fallback to JSONL reader if mislabeled
    # treat as JSONL or text lines
    for txt in read_input_file(path):
        for w in words(txt):
            yield w

# Build-phase workers
def scan_vocab_worker(file_path: str) -> List[str]:
    seen: Dict[str, None] = {}
    for w in yield_words_file(file_path):
        seen[w] = None
    return list(seen)

VOCAB: Dict[str, int] | None = None

def enc_init(vocab_bytes: bytes):
    global VOCAB
    VOCAB = pickle.loads(vocab_bytes)

def encode_shard_worker(args: Tuple[str, str]) -> pathlib.Path:
    file_path, out_dir = args
    assert VOCAB is not None
    ids = [VOCAB.get(w, 0) for w in yield_words_file(file_path)]
    out = pathlib.Path(out_dir) / f"ids_{pathlib.Path(file_path).stem}.npy"
    np.save(out, np.array(ids, dtype=np.int32))
    return out

# Build orchestrator
def build_index(data_dir: str, out_dir: str, workers: int):
    os.makedirs(out_dir, exist_ok=True)
    # Support both .parquet and .jsonl
    files = sorted(pathlib.Path(data_dir).glob("*.parquet")) + sorted(pathlib.Path(data_dir).glob("*.jsonl"))
    files = list(map(str, files))
    if not files:
        sys.exit("No data files found.")

    try:
        ctx = mp.get_context("fork")
    except ValueError:
        ctx = mp.get_context("spawn")

    # Pass 1: build vocab
    vocab: Dict[str, int] = {}
    next_id = 1
    with ctx.Pool(workers) as pool:
        for wordlist in tqdm(pool.imap_unordered(scan_vocab_worker, files),
                              total=len(files), desc="Scanning vocab"):
            for w in wordlist:
                if w not in vocab:
                    vocab[w] = next_id; next_id += 1
    vocab = {w: i + 1 for i, w in enumerate(sorted(vocab.keys()))}
    np.save(os.path.join(out_dir, "vocab.npy"), np.array(list(vocab.keys()), dtype=object))

    # Pass 2: encode shards
    vb = pickle.dumps(vocab, pickle.HIGHEST_PROTOCOL)
    enc_args = [(f, out_dir) for f in files]
    with ctx.Pool(workers, initializer=enc_init, initargs=(vb,)) as pool:
        id_files = list(tqdm(pool.imap_unordered(encode_shard_worker, enc_args),
                              total=len(enc_args), desc="Encoding shards"))

    # Build per-shard SA with overlap
    prev_ids = None
    overlap = WIN - 1
    for id_file in tqdm(id_files, desc="Building per-shard SA"):
        ids = np.load(id_file, mmap_mode='r')
        if prev_ids is not None:
            tail = prev_ids[-overlap:]
            ids_for_sa = np.concatenate([tail, ids])
        else:
            ids_for_sa = ids
        sa = pydivsufsort.divsufsort(ids_for_sa)
        sa_path = os.path.join(out_dir, f"{pathlib.Path(id_file).stem}_sa.npy")
        np.save(sa_path, sa)
        prev_ids = ids

    print("Index complete ✔︎")

# Query helpers
def binary_search(ids: np.ndarray, sa: np.ndarray, pattern: List[int]) -> bool:
    lo, hi = 0, sa.size
    m = len(pattern)
    pat = tuple(pattern)
    while lo < hi:
        mid = (lo + hi) // 2
        if tuple(ids[sa[mid]:sa[mid]+m]) < pat:
            lo = mid + 1
        else:
            hi = mid
    return lo < sa.size and tuple(ids[sa[lo]:sa[lo]+m]) == pat

def query_index(idx_dir: str, in_path: str, win: int):
    start_time = time.time()
    vocab_arr = np.load(pathlib.Path(idx_dir) / 'vocab.npy', allow_pickle=True)
    vocab = {w: i+1 for i, w in enumerate(vocab_arr)}

    # Load all shards
    all_paths = sorted(pathlib.Path(idx_dir).glob('ids_*.npy'))
    # Exclude already-built SA files (ending with '_sa.npy')
    id_paths = [p for p in all_paths if not p.name.endswith('_sa.npy')]
    sa_paths = [p.parent / f"{p.stem}_sa.npy" for p in id_paths]
    ids_list = [np.load(p, mmap_mode='r') for p in id_paths]
    sa_list  = [np.load(p, mmap_mode='r') for p in sa_paths]

    # Gather inputs
    paths: List[pathlib.Path] = []
    p = pathlib.Path(in_path)
    if p.is_dir():
        paths = sorted(p.glob('*.txt')) + sorted(p.glob('*.jsonl'))
    else:
        paths = [p]

    total = hits = 0
    for file in paths:
        batch = read_input_file(str(file))
        for line in tqdm(batch, desc=f"Querying {file.name}"):
            seq = [vocab.get(w,0) for w in words(line)]
            if len(seq) < win:
                continue
            found = False
            for i in range(len(seq)-win+1):
                window = seq[i:i+win]
                if 0 in window:
                    continue
                for ids, sa in zip(ids_list, sa_list):
                    if binary_search(ids, sa, window) and filter_match(line):
                        print(line)
                        hits += 1
                        found = True
                        break
                if found:
                    break
            total += 1
    ratio = hits/total*100 if total else 0
    elapsed = time.time() - start_time
    print(f"Total lines: {total}, Hits: {hits}, Hit ratio: {ratio:.2f}%")
    print(f"Query time: {elapsed:.2f} seconds")

# Test-data generator
def yield_text_file(path: str):
    ext = pathlib.Path(path).suffix.lower()
    if ext == '.parquet':
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches():
            for cell in batch.column('text'):
                yield cell.as_py()
    else:
        for line in read_input_file(path):
            yield line

def generate_test_queries(data_dir: str, out_path: str, pairs: int, win: int):
    shards = sorted(pathlib.Path(data_dir).glob("*.parquet")) + \
             sorted(pathlib.Path(data_dir).glob("*.jsonl"))
    if not shards:
        sys.exit("No data files found for test generation.")
    random.seed(42)
    hits, misses = [], []
    for shard in tqdm(random.sample(shards, len(shards)), desc="Sampling shards"):
        for txt in yield_text_file(str(shard)):
            w = words(txt)
            if len(w) < win: continue
            start = random.randint(0, len(w) - win)
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

# File splitter
def split_file_parts(file_path: str, out_dir: str, parts: int):
    ext = pathlib.Path(file_path).suffix.lower()
    base = pathlib.Path(file_path).stem
    if ext == '.parquet':
        tbl = pq.read_table(file_path)
        rows = tbl.num_rows
        size = rows // parts; rem = rows % parts
        for i in range(parts):
            start = i * size + min(i, rem)
            cnt = size + (1 if i < rem else 0)
            slice_tbl = tbl.slice(start, cnt)
            out = pathlib.Path(out_dir) / f"{base}_part{i}.parquet"
            pq.write_table(slice_tbl, out, compression='zstd')
    elif ext == '.jsonl':
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        total = len(lines)
        size = total // parts; rem = total % parts
        for i in range(parts):
            start = i * size + min(i, rem)
            cnt = size + (1 if i < rem else 0)
            chunk = lines[start:start+cnt]
            out = pathlib.Path(out_dir) / f"{base}_part{i}.jsonl"
            with open(out, 'w', encoding='utf-8') as wf:
                wf.writelines(chunk)
    else:
        sys.exit(f"Unsupported file type: {file_path}")

def split_dir(pq_dir: str, out_dir: str, parts: int, workers: int):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(pathlib.Path(pq_dir).glob("*.parquet")) + sorted(pathlib.Path(pq_dir).glob("*.jsonl"))
    if not files: sys.exit("No files found to split.")
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        ctx = mp.get_context("spawn")
    func = partial(split_file_parts, out_dir=out_dir, parts=parts)
    with ctx.Pool(workers) as pool:
        list(tqdm(pool.imap_unordered(func, files), total=len(files), desc="Splitting files"))
    print("Splitting complete ✔︎")

# CLI entry
def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--workers", default="auto")
    q = sub.add_parser("query")
    q.add_argument("--index-dir", required=True)
    q.add_argument("--input", required=True)
    q.add_argument("--window", type=int, default=WIN)
    g = sub.add_parser("gen_test")
    g.add_argument("--data-dir", required=True)
    g.add_argument("--output", required=True)
    g.add_argument("--pairs", type=int, default=20)
    g.add_argument("--window", type=int, default=WIN)
    s = sub.add_parser("split")
    s.add_argument("--file-dir", required=True)
    s.add_argument("--out-dir", required=True)
    s.add_argument("--parts", type=int, required=True)
    s.add_argument("--workers", default="auto")
    args = parser.parse_args()
    if args.cmd == "build":
        w = get_n_workers(args.workers)
        build_index(args.data_dir, args.out_dir, w)
    elif args.cmd == "query":
        query_index(args.index_dir, args.input, args.window)
    elif args.cmd == "gen_test":
        generate_test_queries(args.data_dir, args.output, args.pairs, args.window)
    elif args.cmd == "split":
        w = get_n_workers(args.workers)
        split_dir(args.file_dir, args.out_dir, args.parts, w)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
