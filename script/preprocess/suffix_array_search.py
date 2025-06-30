#!/usr/bin/env python3
"""
Suffix‑array 50‑token matcher – *SentencePiece edition*
=====================================================
Designed for **fineweb‑edu sample‑10BT** or any large corpus when you already
have a **trained SentencePiece tokenizer model (.model)**.

This script can:
1. **Build** a sharded suffix‑array (SA) index of the tokenised corpus.
2. **Query** the index to answer *Does the corpus contain this exact 50‑token sequence?*

Now with **progress bars** using `tqdm` to track:
- Sharding progress
- SA/LCP build progress
- Query processing progress

Usage:
    python suffix_array_search.py build --corpus CORPUS_ROOT --spm_model MODEL_PATH \
        [--outdir OUTDIR] [--shard_size SHARD_SIZE] [--workers N]
    python suffix_array_search.py query --index OUTDIR --spm_model MODEL_PATH \
        [--input QUERY_FILE] [--workers N]

Requires:
    pip install numpy sentencepiece pydivsufsort tqdm
"""
from __future__ import annotations
import argparse, os, pickle, sys, multiprocessing as mp
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple

import numpy as np
import tqdm
# Optional high-performance C SA builder
try:
    import pydivsufsort
    HAVE_DIVSUF = True
except ImportError:
    HAVE_DIVSUF = False
# SentencePiece tokenizer
try:
    import sentencepiece as spm
except ImportError:
    raise ImportError("sentencepiece not installed: pip install sentencepiece")

# Constants
K_PREFIX = 3                     # use first 3 tokens for prefix table
TOKEN_DTYPE = np.uint32
SA_DTYPE    = np.uint64
LCP_DTYPE   = np.uint32
DEFAULT_SHARD_SIZE = 64_000_000  # tokens per shard (~1.3GB RAM usage)
MIN_MATCH = 50                   # min token length for query

# Load SentencePiece model
def load_spm(model_path: str | Path) -> spm.SentencePieceProcessor:
    sp = spm.SentencePieceProcessor()
    if not sp.Load(str(model_path)):
        raise RuntimeError(f"Failed to load SentencePiece model: {model_path}")
    return sp

# Iterate token IDs from corpus, inserting sep_id between files
def iter_token_ids(corpus_root: Path, sp: spm.SentencePieceProcessor, sep_id: int) -> Iterator[int]:
    files = sorted(list(corpus_root.rglob("*.txt")))
    for file in tqdm.tqdm(files, desc="Shard - reading files", unit="file"):
        with file.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                for tid in sp.EncodeAsIds(line.strip()):
                    yield tid
        yield sep_id

# Write a shard file of token IDs
def _write_shard(shard_id: int, buf: List[int], outdir: Path) -> Tuple[Path,int]:
    arr = np.array(buf, dtype=TOKEN_DTYPE)
    fname = outdir / f"shard-{shard_id:05d}.tokens.npy"
    np.save(fname, arr, allow_pickle=False)
    return fname, len(arr)

# Build SA and LCP arrays for a shard
def _build_sa_for_shard(token_file: Path) -> None:
    tokens = np.load(token_file, mmap_mode=None)
    sa_file = token_file.with_suffix(".sa.npy")
    lcp_file = token_file.with_suffix(".lcp.npy")
    # SA construction
    if HAVE_DIVSUF:
        sa_list = pydivsufsort.divsufsort(tokens.tolist())
        sa = np.array(sa_list, dtype=SA_DTYPE)
    else:
        sa = np.argsort([tuple(tokens[i:]) for i in range(len(tokens))]).astype(SA_DTYPE)
    # LCP computation
    lcp = np.zeros_like(sa, dtype=LCP_DTYPE)
    rank = np.empty_like(sa, dtype=SA_DTYPE)
    rank[sa] = np.arange(len(sa), dtype=SA_DTYPE)
    h = 0
    for i in range(len(tokens)):
        k = rank[i]
        if k > 0:
            j = sa[k-1]
            while i+h < len(tokens) and j+h < len(tokens) and tokens[i+h] == tokens[j+h]:
                h += 1
            lcp[k] = h
            if h: h -= 1
    # Save via memmap for query-phase efficiency
    np.memmap(sa_file, dtype=SA_DTYPE, mode="w+", shape=sa.shape)[:] = sa
    np.memmap(lcp_file, dtype=LCP_DTYPE, mode="w+", shape=lcp.shape)[:] = lcp
    print(f"Built shard {token_file.stem}: {len(tokens):,} tokens")

# Build entire index
def build_index(args):
    corpus = Path(args.corpus)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sp = load_spm(args.spm_model)
    # auto-detect sep_id as vocab size
    sep_id = sp.get_piece_size()
    print(f"Auto-selected sep_id = {sep_id} (vocab size)")
    buf, shard_id, shard_files, total = [], 0, [], 0
    # Shard splitting with progress
    for tid in iter_token_ids(corpus, sp, sep_id):
        buf.append(tid)
        if len(buf) >= args.shard_size:
            f, n = _write_shard(shard_id, buf, outdir)
            shard_files.append(f)
            total += n
            shard_id += 1
            buf = []
    if buf:
        f, n = _write_shard(shard_id, buf, outdir)
        shard_files.append(f)
        total += n
    print(f"Sharded into {len(shard_files)} files, total {total:,} tokens")
    # Parallel SA/LCP build with progress
    print("Building SA/LCP for each shard...")
    with mp.Pool(args.workers) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(_build_sa_for_shard, shard_files),
                            total=len(shard_files), desc="Building SA", unit="shard"):
            pass
    # Prefix table: map first K_PREFIX tokens hash to shard & index
    buckets = 1 << (K_PREFIX * 8)
    tbl: list[list[tuple[int,int]]] = [[] for _ in range(buckets)]
    for f in tqdm.tqdm(shard_files, desc="Building prefix table", unit="shard"):
        sid = int(f.stem.split("-")[-1])
        tok = np.load(f, mmap_mode="r")
        for i in range(len(tok) - K_PREFIX + 1):
            key = 0
            for j in range(K_PREFIX): key = (key * 31 + int(tok[i+j])) & (buckets - 1)
            tbl[key].append((sid, i))
    with open(Path(args.outdir)/"prefix.tbl", "wb") as fh:
        pickle.dump(tbl, fh)
    print("Index build complete.")

# Load token & SA for a shard
def _load_shard(outdir: Path, shard_id: int) -> tuple[np.ndarray, np.ndarray]:
    tok = np.load(outdir/f"shard-{shard_id:05d}.tokens.npy", mmap_mode="r")
    sa  = np.memmap(outdir/f"shard-{shard_id:05d}.sa.npy", dtype=SA_DTYPE, mode="r")
    return tok, sa

# Binary-search check
def _sa_contains(tokens: np.ndarray, sa: np.ndarray, query: Sequence[int]) -> bool:
    lo, hi = 0, len(sa)
    ql = len(query)
    while lo < hi:
        m = (lo + hi) // 2
        slice_ = tokens[sa[m]:sa[m]+ql]
        if slice_.tolist() == list(query):
            return True
        if slice_.tolist() < list(query):
            lo = m + 1
        else:
            hi = m
    return False

# Worker for query
def _query_task(payload) -> tuple[int,bool]:
    idx, sid, query, outdir = payload
    tok, sa = _load_shard(outdir, sid)
    return idx, _sa_contains(tok, sa, query)

# Query index
def query_index(args):
    outdir = Path(args.index)
    sp = load_spm(args.spm_model)
    with open(outdir/"prefix.tbl", "rb") as fh:
        tbl = pickle.load(fh)
    buckets = len(tbl)
    lines = Path(args.input).read_text(encoding="utf-8").splitlines()
    queries: list[list[int]] = []
    for line in tqdm.tqdm(lines, desc="Tokenizing queries", unit="line"):
        ids = sp.EncodeAsIds(line.strip())
        if len(ids) >= MIN_MATCH:
            queries.append(ids[:MIN_MATCH])
    if not queries:
        print(f"No valid queries (need >= {MIN_MATCH} tokens)")
        return
    tasks: list[tuple[int,int,Sequence[int],Path]] = []
    for i, q in enumerate(queries):
        key = 0
        for j in range(K_PREFIX): key = (key * 31 + q[j]) & (buckets - 1)
        for sid, _ in tbl[key]: tasks.append((i, sid, q, outdir))
    print(f"Dispatching {len(tasks)} query tasks across {args.workers} workers...")
    with mp.Pool(args.workers) as pool:
        results = list(tqdm.tqdm(pool.imap_unordered(_query_task, tasks),
                                  total=len(tasks), desc="Querying shards", unit="task"))
    found = [False] * len(queries)
    for idx, ok in results:
        if ok: found[idx] = True
    for i, ok in enumerate(found):
        print(f"Query {i}: {'MATCH' if ok else 'NO MATCH'}")

# CLI
def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest='cmd', required=True)
    # build
    b = sp.add_parser('build')
    b.add_argument('--corpus', required=True)
    b.add_argument('--spm_model', required=True)
    b.add_argument('--outdir', default='index')
    b.add_argument('--shard_size', type=int, default=DEFAULT_SHARD_SIZE)
    b.add_argument('--workers', type=int, default=mp.cpu_count())
    # query
    q = sp.add_parser('query')
    q.add_argument('--index', required=True)
    q.add_argument('--spm_model', required=True)
    q.add_argument('--input', required=True)
    q.add_argument('--workers', type=int, default=mp.cpu_count())
    args = p.parse_args()
    if args.cmd == 'build':
        build_index(args)
    else:
        query_index(args)

if __name__ == '__main__':
    main()
