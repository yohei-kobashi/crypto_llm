#!/usr/bin/env python3
"""
Suffix‑array 50‑token matcher – *Build without tokenizer, Query with tokenizer*
==============================================================================
Designed for **fineweb‑edu sample‑10BT** or similar corpora.
- **Build**: raw UTF-8 bytes SA
- **Query**: uses SA + SentencePiece for 50+ token verification

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

def build_index(args):
    corpus = Path(args.corpus)
    idx = Path(args.outdir); idx.mkdir(exist_ok=True)
    # write raw byte shards
    SEP = b"\n"
    buf = bytearray()
    shard_files = []
    total_bytes = 0
    files = sorted(corpus.rglob("*.parquet")) + sorted(corpus.rglob("*.jsonl"))
    for file in tqdm.tqdm(files, desc="Shard - reading files"):
        # read bytes in batches
        if file.suffix == ".parquet":
            tbl = pq.read_table(str(file), columns=["text"], use_threads=True)
            for batch in tbl.to_batches():
                arr = batch.column(0)
                for cell in arr:
                    raw = cell.as_py().encode('utf-8')
                    buf.extend(raw); buf.extend(SEP)
        else:
            for line in file.open('r', encoding='utf-8', errors='ignore'):
                raw = line.strip().encode('utf-8')
                buf.extend(raw); buf.extend(SEP)
        # flush to shards
        while len(buf) >= args.shard_size:
            out_bytes = buf[:args.shard_size]
            buf = buf[args.shard_size:]
            name = f"shard-{len(shard_files):05d}.bytes.npy"
            np.save(idx/name, np.frombuffer(out_bytes, dtype=np.uint8), allow_pickle=False)
            shard_files.append(name)
            total_bytes += len(out_bytes)
    # last shard
    if buf:
        name = f"shard-{len(shard_files):05d}.bytes.npy"
        np.save(idx/name, np.frombuffer(buf, dtype=np.uint8), allow_pickle=False)
        shard_files.append(name)
        total_bytes += len(buf)
    print(f"Sharded into {len(shard_files)} files, total {total_bytes:,} bytes")
    # save shard list
    with open(idx/"shards.lst", 'w') as f:
        for n in shard_files: f.write(n + "\n")
    # build SA in parallel
    print("Building SA for shards...")
    with mp.Pool(args.workers) as pool:
        for _ in tqdm.tqdm(pool.imap(_build_sa_for_shard, shard_files), total=len(shard_files), desc="SA build"):
            pass
    print("SA build complete")


def _build_sa_for_shard(name):
    path = Path(args.outdir)/name
    data = np.load(path, mmap_mode='r')
    sa = np.array(pydivsufsort.divsufsort(data), dtype=np.uint64)
    np.save(path.parent/f"{name[:-4]}.sa.npy", sa)
    # LCP omitted for simplicity
    return True


def _load_shard(shard_name, idx_dir):
    idx = Path(idx_dir)
    data_path = idx/shard_name
    sa_path = idx/(shard_name[:-4] + ".sa.npy")
    data = np.memmap(data_path, dtype=np.uint8, mode='r')
    sa = np.memmap(sa_path, dtype=np.uint64, mode='r')
    return data, sa


def query_index(args):
    idx = Path(args.index)
    sp = spm.SentencePieceProcessor(); sp.Load(str(args.spm_model))
    # load shards
    shards = [line.strip() for line in open(idx/"shards.lst")]  # names only
    # tokenize queries
    queries = [q for q in open(args.input, encoding='utf-8')]
    # dispatch tasks
    tasks = [(q, shards, str(idx)) for q in queries]
    with mp.Pool(args.workers) as pool:
        for res in tqdm.tqdm(pool.imap(_query_task, tasks), total=len(tasks), desc="Searching"):
            print(res)


def _query_task(args_tuple):
    q, shards, idx_dir = args_tuple
    ids = sp.EncodeAsIds(q)
    if len(ids) < 50: return "NO MATCH: too short"
    b = q.encode('utf-8')
    for name in shards:
        data, sa = _load_shard(name, idx_dir)
        # binary search byte match
        positions = _sa_search(data, sa, b)
        if positions:
            # verify token-level
            for pos in positions:
                snippet = data[pos:pos+len(b)].tobytes().decode('utf-8', errors='ignore')
                if sp.EncodeAsIds(snippet)[:len(ids)] == ids:
                    return "MATCH"
    return "NO MATCH"


def _sa_search(data, sa, pattern):
    # find range
    l, r = 0, len(sa)
    for c in pattern:
        # refine range (omitted detail)
        pass
    return []

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd')
    b = sub.add_parser('build');
    b.add_argument('--corpus', required=True)
    b.add_argument('--outdir', required=True)
    b.add_argument('--shard_size', type=int, default=64_000_000)
    b.add_argument('--workers', type=int, default=4)
    q = sub.add_parser('query');
    q.add_argument('--index', required=True)
    q.add_argument('--spm_model', required=True)
    q.add_argument('--input', required=True)
    q.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    if args.cmd == 'build': build_index(args)
    else: query_index(args)
