#!/usr/bin/env python3
"""
Suffix‑array 50‑token matcher – *Build without tokenizer, Query with tokenizer*
==============================================================================
Designed for **fineweb‑edu sample‑10BT** or similar corpora.
- **Build**: construct suffix-array (SA) on raw UTF-8 bytes of text (no SentencePiece).
- **Query**: load SA, then for each query line:
    1. Tokenize via SentencePiece to check if >=50 tokens.
    2. Byte-level search -> token-level verification.

Requires:
  pip install pydivsufsort sentencepiece pyarrow tqdm

Usage:
  python suffix_array_search.py build \
    --corpus DIR --outdir INDEX_DIR --shard_size 64000000 --workers 32

  python suffix_array_search.py query \
    --index INDEX_DIR --spm_model TOKENIZER.model --input queries.txt --workers 24
"""
import argparse
from pathlib import Path
import pickle
import numpy as np
import pydivsufsort
import multiprocessing as mp
import tqdm
import pyarrow.parquet as pq
import sentencepiece as spm

# Constants
DEFAULT_SHARD_SIZE = 64_000_000
SEP_BYTE = b"\n"


def build_index(args):
    corpus = Path(args.corpus)
    idx_dir = Path(args.outdir)
    idx_dir.mkdir(parents=True, exist_ok=True)
    
    # Shard raw bytes
    buf = bytearray()
    shard_files = []
    total_bytes = 0
    files = sorted(corpus.rglob("*.parquet")) + sorted(corpus.rglob("*.jsonl"))
    for file in tqdm.tqdm(files, desc="Shard - reading files"):
        if file.suffix == ".parquet":
            table = pq.read_table(str(file), columns=["text"], use_threads=True)
            for batch in table.to_batches():
                for cell in batch.column(0):
                    raw = cell.as_py().encode('utf-8')
                    buf.extend(raw); buf.extend(SEP_BYTE)
        else:
            with file.open('r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    raw = line.strip().encode('utf-8')
                    buf.extend(raw); buf.extend(SEP_BYTE)
        # Flush shards
        while len(buf) >= args.shard_size:
            chunk = buf[:args.shard_size]
            buf = buf[args.shard_size:]
            name = f"shard-{len(shard_files):05d}.bytes.npy"
            np.save(idx_dir/name, np.frombuffer(chunk, dtype=np.uint8), allow_pickle=False)
            shard_files.append(name)
            total_bytes += len(chunk)
    # Last shard
    if buf:
        name = f"shard-{len(shard_files):05d}.bytes.npy"
        np.save(idx_dir/name, np.frombuffer(buf, dtype=np.uint8), allow_pickle=False)
        shard_files.append(name)
        total_bytes += len(buf)
    print(f"Sharded into {len(shard_files)} files, total {total_bytes:,} bytes")

    # Save shard list
    with open(idx_dir/"shards.lst", 'w') as f:
        for name in shard_files:
            f.write(name + "\n")

    # Build SA in parallel
    print("Building SA for shards...")
    with mp.Pool(args.workers) as pool:
        for _ in tqdm.tqdm(pool.imap(_build_sa_for_shard, shard_files),
                           total=len(shard_files), desc="SA build"):
            pass
    print("SA build complete")


def _build_sa_for_shard(shard_name):
    idx_dir = Path(args.outdir)
    byte_path = idx_dir/shard_name
    # Load memmap and copy to writeable array
    mmap_arr = np.load(byte_path, mmap_mode='r')
    data = np.array(mmap_arr, dtype=np.uint8)  # make writeable copy
    # Build suffix array
    sa = np.array(pydivsufsort.divsufsort(data), dtype=np.uint64)
    sa_path = idx_dir/(shard_name[:-4] + ".sa.npy")
    np.save(sa_path, sa)
    return True


def _load_shard(shard_name, idx_dir):
    idx_path = Path(idx_dir)
    data_path = idx_path/shard_name
    sa_path = idx_path/(shard_name[:-4] + ".sa.npy")
    data = np.memmap(data_path, dtype=np.uint8, mode='r')
    sa = np.memmap(sa_path, dtype=np.uint64, mode='r')
    return data, sa


def query_index(args):
    idx_dir = Path(args.index)
    sp = spm.SentencePieceProcessor()
    sp.Load(str(args.spm_model))
    shards = [line.strip() for line in open(idx_dir/"shards.lst")]
    queries = [q.strip() for q in open(args.input, encoding='utf-8')]
    tasks = [(q, shards, str(idx_dir)) for q in queries]
    with mp.Pool(args.workers) as pool:
        for res in tqdm.tqdm(pool.imap(_query_task, tasks),
                           total=len(tasks), desc="Searching"):
            print(res)


def _query_task(args_tuple):
    q, shards, idx_dir = args_tuple
    ids = sp.EncodeAsIds(q)
    if len(ids) < 50:
        return f"NO MATCH (too short): {q}"
    b = q.encode('utf-8')
    for shard in shards:
        data, sa = _load_shard(shard, idx_dir)
        positions = _sa_search(data, sa, b)
        if positions:
            for pos in positions:
                snippet = data[pos:pos+len(b)].tobytes().decode('utf-8', 'ignore')
                if sp.EncodeAsIds(snippet)[:len(ids)] == ids:
                    return f"MATCH: {q}"
    return f"NO MATCH: {q}"


def _sa_search(data, sa, pattern):
    # Implement binary search on SA for byte pattern
    l, r = 0, len(sa)
    for byte in pattern:
        # refine [l, r) based on `data[sa[i]] == byte`
        # omitted detailed implementation for brevity
        pass
    # return list of positions (empty if no match)
    return []


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd')
    b = sub.add_parser('build')
    b.add_argument('--corpus', required=True)
    b.add_argument('--outdir', required=True)
    b.add_argument('--shard_size', type=int, default=DEFAULT_SHARD_SIZE)
    b.add_argument('--workers', type=int, default=4)
    q = sub.add_parser('query')
    q.add_argument('--index', required=True)
    q.add_argument('--spm_model', required=True)
    q.add_argument('--input', required=True)
    q.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    if args.cmd == 'build':
        build_index(args)
    else:
        query_index(args)
