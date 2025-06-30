#!/usr/bin/env python3
"""
Suffix‑array 50‑token matcher – *Build without tokenizer, Query with tokenizer*
==============================================================================
Designed for **fineweb‑edu sample‑10BT** or similar corpora.
- **Build**: construct suffix-array (SA) on raw UTF-8 bytes of text (no SentencePiece).
- **Query**: load SA, then for each query line:
    1. Tokenize via SentencePiece to check if >=50 tokens.
    2. Perform byte-level SA search for raw string.
    3. Post-filter each raw match by re-tokenizing the matched substring and verifying exact token-ID alignment.

Usage:
    python suffix_array_search.py build --corpus CORPUS_ROOT --outdir OUTDIR \
        [--shard_size SHARD_SIZE] [--workers N]

    python suffix_array_search.py query --index OUTDIR --spm_model MODEL_PATH \
        --input QUERY_FILE [--workers N]

Requires:
    numpy pydivsufsort tqdm pyarrow sentencepiece
"""
import argparse, pickle, sys, multiprocessing as mp, json
from pathlib import Path
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
MIN_MATCH  = 50               # tokens for query
SEP_BYTE   = b"\x00"        # unused byte delimiter
K_PREFIX   = 8                # prefix length in bytes for filtering

# GLOBAL SentencePiece processor for worker reuse
SP: 'spm.SentencePieceProcessor' = None

# Build: iterate raw bytes from corpus files with batch-byte concatenation
def iter_bytes(corpus_root: Path):
    files = sorted(corpus_root.rglob("*.jsonl")) + sorted(corpus_root.rglob("*.parquet"))
    for file in tqdm.tqdm(files, desc="Shard - reading files", unit="file"):
        if file.suffix == ".jsonl":
            for line in file.open("r", encoding='utf-8', errors='ignore'):
                try:
                    obj = json.loads(line)
                    text = obj.get('text', '')
                except Exception:
                    continue
                buf = text.encode('utf-8')
                yield from buf
        else:  # .parquet
            table = pq.read_table(str(file), columns=['text'], use_threads=True)
            for batch in table.to_batches():
                col = batch.column(0)
                for cell in col:
                    raw = cell.as_py().encode('utf-8')
                    yield from raw
        # separator byte
        yield from SEP_BYTE

# Write shard of raw bytes
def _write_shard(shard_id: int, buf, outdir: Path):
    arr = np.frombuffer(bytearray(buf), dtype=np.uint8)
    fname = outdir / f"shard-{shard_id:05d}.bytes.npy"
    np.save(fname, arr, allow_pickle=False)
    return fname.name, len(arr)

# Build SA for a shard
def _build_sa_for_shard(shard_name: str, idx_dir: Path):
    byte_file = idx_dir / shard_name
    data = np.load(byte_file, mmap_mode=None)
    sa_file = byte_file.with_suffix('.sa.npy')
    if HAVE_DIVSUF:
        sa = np.array(pydivsufsort.divsufsort(data.tolist()), dtype=np.uint64)
    else:
        sa = np.argsort([bytes(data[i:]) for i in range(len(data))]).astype(np.uint64)
    mm = np.memmap(sa_file, dtype=np.uint64, mode='w+', shape=sa.shape)
    mm[:] = sa
    del mm

# Build index: shards + SA
def build_index(args):
    corpus, outdir = Path(args.corpus), Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    buf, sid, shards, total = [], 0, [], 0
    for b in iter_bytes(corpus):
        buf.append(b)
        if len(buf) >= args.shard_size:
            name, n = _write_shard(sid, buf, outdir)
            shards.append(name)
            total += n
            sid += 1
            buf = []
    if buf:
        name, n = _write_shard(sid, buf, outdir)
        shards.append(name)
        total += n
    print(f"Sharded into {len(shards)} files, total {total} bytes")
    print("Building SA for shards...")
    with mp.Pool(args.workers) as pool:
        list(tqdm.tqdm(
            pool.imap_unordered(lambda nm: _build_sa_for_shard(nm, outdir), shards),
            total=len(shards), desc="SA build", unit="shard"))
    with open(outdir / 'shards.lst', 'wb') as fh:
        pickle.dump(shards, fh)
    print("Index build complete.")

# Load shard bytes and SA
def _load_shard(shard_name: str, idx_dir: Path):
    byte_file = idx_dir / shard_name
    data = np.load(byte_file, mmap_mode='r').tobytes()
    sa = np.memmap(idx_dir / byte_file.with_suffix('.sa.npy').name,
                   dtype=np.uint64, mode='r')
    return data, sa

# SA-based substring search: return match positions
def _sa_search_positions(data: bytes, sa: np.ndarray, pattern: bytes):
    lo, hi = 0, len(sa)
    while lo < hi:
        mid = (lo + hi) // 2
        if data[sa[mid]:sa[mid]+len(pattern)] < pattern:
            lo = mid + 1
        else:
            hi = mid
    positions = []
    idx = lo
    while idx < len(sa) and data[sa[idx]:sa[idx]+len(pattern)] == pattern:
        positions.append(int(sa[idx])); idx += 1
    return positions

# Query worker: post-filter by token IDs
def _query_task(payload):
    idx, raw, qids, shards, idx_dir = payload
    for name in shards:
        data, sa = _load_shard(name, idx_dir)
        for pos in _sa_search_positions(data, sa, raw):
            substr = data[pos:pos+len(raw)].decode('utf-8', 'ignore')
            if SP.EncodeAsIds(substr) == qids:
                return idx, True
    return idx, False

# Query index: tokenize to check length, then search raw with post-filter
def query_index(args):
    global SP
    if spm is None:
        raise RuntimeError("SentencePiece not installed")
    SP = spm.SentencePieceProcessor(); SP.Load(str(args.spm_model))
    idx_dir = Path(args.index)
    shards = pickle.load(open(idx_dir / 'shards.lst','rb'))
    lines = Path(args.input).read_text(encoding='utf-8').splitlines()
    tasks, valid = [], []
    for i, line in enumerate(tqdm.tqdm(lines, desc="Tokenizing queries", unit="query")):
        q = line.strip()
        qids = SP.EncodeAsIds(q)
        if len(qids) >= MIN_MATCH:
            tasks.append((i, q.encode('utf-8'), qids, shards, idx_dir))
            valid.append(i)
    if not tasks:
        print(f"No queries >= {MIN_MATCH} tokens")
        return
    print(f"Dispatching {len(tasks)} tasks across {args.workers} workers")
    with mp.Pool(args.workers) as pool:
        results = list(tqdm.tqdm(
            pool.imap_unordered(_query_task, tasks),
            total=len(tasks), desc="Searching", unit="match"))
    found = {i: False for i in valid}
    for idx, ok in results:
        found[idx] = ok
    for i, line in enumerate(lines):
        prefix = f"Query {i}:"
        if i in found:
            print(f"{prefix} {'MATCH' if found[i] else 'NO MATCH'}")
        else:
            print(f"{prefix} SKIPPED (<{MIN_MATCH} tokens)")

# CLI
if __name__ == '__main__':
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
