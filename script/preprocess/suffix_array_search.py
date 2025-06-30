#!/usr/bin/env python3
"""
Suffix‑array 50‑token matcher – *Build without tokenizer, Query with tokenizer*
==============================================================================
Designed for **fineweb‑edu sample‑10BT** or similar corpora.
- **Build**: construct suffix-array (SA) on raw UTF-8 bytes of text (no SentencePiece).
- **Query**: load SA, then for each query line:
    1. Tokenize via SentencePiece to check if >=50 tokens.
    2. If so, perform byte-level SA search for the raw query string.

Usage:
    python suffix_array_search.py build --corpus CORPUS_ROOT --outdir OUTDIR \
        [--shard_size SHARD_SIZE] [--workers N]

    python suffix_array_search.py query --index OUTDIR --spm_model MODEL_PATH \
        --input QUERY_FILE [--workers N]

Requires:
    numpy pydivsufsort tqdm pyarrow sentencepiece
"""
import argparse, pickle, sys, multiprocessing as mp
from pathlib import Path
from typing import Iterator, List, Tuple
import numpy as np
import tqdm
import pyarrow.parquet as pq
try:
    import pydivsufsort
    HAVE_DIVSUF = True
except ImportError:
    HAVE_DIVSUF = False
try:
    import sentencepiece as spm
except ImportError:
    spm = None  # only needed for query

# Constants
SHARD_SIZE = 64_000_000       # bytes per shard
default
MIN_MATCH = 50                # tokens for query
SEP_BYTE   = b"\x00"        # unused byte delimiter

# Build: iterate raw bytes from corpus files
def iter_bytes(corpus_root: Path) -> Iterator[int]:
    files = sorted(corpus_root.rglob("*.jsonl")) + sorted(corpus_root.rglob("*.parquet"))
    for file in tqdm.tqdm(files, desc="Shard - reading files", unit="file"):
        if file.suffix == ".jsonl":
            for line in file.open("rb"):
                try:
                    rec = line.decode('utf-8', 'ignore')
                    obj = __import__('json').loads(rec)
                    text = obj.get('text', '')
                except Exception:
                    continue
                for b in text.encode('utf-8'):
                    yield b
        elif file.suffix == ".parquet":
            tbl = pq.read_table(str(file))
            col = tbl['text'] if 'text' in tbl.schema.names else tbl.column(0)
            for txt in col.to_pylist():
                raw = str(txt).encode('utf-8')
                for b in raw:
                    yield b
        # insert separator
        for b in SEP_BYTE:
            yield b

# Write shard of raw bytes
def _write_shard(shard_id: int, buf: List[int], outdir: Path) -> Tuple[Path,int]:
    arr = np.array(buf, dtype=np.uint8)
    fname = outdir / f"shard-{shard_id:05d}.bytes.npy"
    np.save(fname, arr, allow_pickle=False)
    return fname, len(arr)

# Build SA for a shard
def _build_sa_for_shard(byte_file: Path) -> None:
    data = np.load(byte_file, mmap_mode=None)
    sa_file = byte_file.with_suffix('.sa.npy')
    if HAVE_DIVSUF:
        sa = np.array(pydivsufsort.divsufsort(data.tolist()), dtype=np.uint64)
    else:
        sa = np.argsort([bytes(data[i:]) for i in range(len(data))]).astype(np.uint64)
    np.memmap(sa_file, dtype=np.uint64, mode='w+', shape=sa.shape)[:] = sa
    print(f"Built SA {byte_file.name}: {len(data)} bytes")

# Build index: shards + SA
def build_index(args):
    corpus, outdir = Path(args.corpus), Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    # Shard raw bytes
    buf, sid, files, total = [], 0, [], 0
    for b in iter_bytes(corpus):
        buf.append(b)
        if len(buf) >= args.shard_size:
            f,n = _write_shard(sid, buf, outdir)
            files.append(f); total+=n; sid+=1; buf=[]
    if buf:
        f,n = _write_shard(sid, buf, outdir)
        files.append(f); total+=n
    print(f"Sharded into {len(files)} files, total {total} bytes")
    # Parallel SA build
    print("Building SA for shards...")
    with mp.Pool(args.workers) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(_build_sa_for_shard, files), total=len(files), desc="SA build", unit="shard"): pass
    # Save list of shards
    with open(outdir/'shards.lst','wb') as fh:
        pickle.dump(files, fh)
    print("Index build complete.")

# Load shard bytes and SA
def _load_shard(outdir: Path, shard_file: Path) -> Tuple[bytes, np.ndarray]:
    data = np.load(shard_file, mmap_mode='r')  # uint8 array
    sa   = np.memmap(shard_file.with_suffix('.sa.npy'), dtype=np.uint64, mode='r')
    return data.tobytes(), sa

# SA-based substring search
def _sa_search(data: bytes, sa: np.ndarray, pattern: bytes) -> bool:
    lo, hi = 0, len(sa)
    while lo < hi:
        mid = (lo+hi)//2
        start = sa[mid]
        segment = data[start:start+len(pattern)]
        if segment == pattern:
            return True
        if segment < pattern:
            lo = mid+1
        else:
            hi = mid
    return False

# Query tasks
def _query_task(payload):
    idx, raw, shards, outdir = payload
    for shard_file in shards:
        data, sa = _load_shard(outdir, shard_file)
        if _sa_search(data, sa, raw):
            return idx, True
    return idx, False

# Query index: tokenize to check length, then search raw
def query_index(args):
    if spm is None:
        raise RuntimeError("SentencePiece not installed")
    sp = spm.SentencePieceProcessor(); sp.Load(str(args.spm_model))
    with open(Path(args.index)/'shards.lst','rb') as fh:
        shards = pickle.load(fh)
    lines = Path(args.input).read_text(encoding='utf-8').splitlines()
    tasks, valid = [], []
    for i,line in enumerate(tqdm.tqdm(lines, desc="Tokenizing queries")):
        ids = sp.EncodeAsIds(line.strip())
        if len(ids) >= MIN_MATCH:
            raw = line.strip().encode('utf-8')
            tasks.append((i, raw, shards, Path(args.index)))
            valid.append(i)
    if not tasks:
        print(f"No queries >= {MIN_MATCH} tokens")
        return
    print(f"Dispatching {len(tasks)} tasks")
    with mp.Pool(args.workers) as pool:
        results = list(tqdm.tqdm(pool.imap_unordered(_query_task, tasks), total=len(tasks), desc="Searching"))
    found = {i: False for i in valid}
    for idx, ok in results:
        found[idx] = ok
    for i in range(len(lines)):
        print(f"Query {i}: ", end='')
        if i in found:
            print('MATCH' if found[i] else 'NO MATCH')
        else:
            print('SKIPPED (<50 tokens)')

# CLI
def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest='cmd', required=True)
    b = sp.add_parser('build')
    b.add_argument('--corpus', required=True)
    b.add_argument('--outdir', default='index')
    b.add_argument('--shard_size', type=int, default=SHARD_SIZE)
    b.add_argument('--workers', type=int, default=mp.cpu_count())
    q = sp.add_parser('query')
    q.add_argument('--index', required=True)
    q.add_argument('--spm_model', required=True)
    q.add_argument('--input', required=True)
    q.add_argument('--workers', type=int, default=mp.cpu_count())
    args = p.parse_args()
    if args.cmd == 'build': build_index(args)
    else: query_index(args)

if __name__ == '__main__': main()
