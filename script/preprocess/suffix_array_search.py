# suffix_array_pipeline (with progress & multiprocessing‑safe functions)
# -------------------------------------------------------------
# Build & query a word‑level suffix array over fineweb‑edu sample‑10BT
# (14×2.1 GB Parquet shards) to detect ≥ 35‑word verbatim substrings.
#   * Local helper functions have been hoisted to top‑level so they are
#     picklable by the multiprocessing ``spawn`` context (fixes
#     AttributeError: Can't get local object ...).
#   * Added ``fork`` as preferred context on POSIX for fork‑based cheap
#     copy‑on‑write, but falls back gracefully on non‑POSIX.
#   * Explicit pool initialiser passes the global vocab to worker
#     processes for the encoding phase.
# -------------------------------------------------------------
"""
Usage
-----
# Build index (outputs ./index/*.npy)
python suffix_array_pipeline.py build \
       --parquet-dir /path/to/fineweb-edu/sample-10BT \
       --out-dir ./index \
       --workers auto

# Query index for a text file containing model outputs (one per line)
python suffix_array_pipeline.py query \
       --index-dir ./index \
       --input outputs.txt \
       --workers auto
"""
from __future__ import annotations

import argparse
import os
import sys
import pathlib
import multiprocessing as mp
from functools import partial
from typing import List, Tuple, Dict
import pickle

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

try:
    import pydivsufsort  # C‑accelerated suffix‑array construction
except ImportError:
    raise SystemExit("Please `pip install pydivsufsort`.  (apt‑get libdivsufsort-dev)")

###############################################################################
# Utility helpers
###############################################################################

WORD_RE = r"[\w'-]+"  # coarse word tokenizer


def words(text: str) -> List[str]:
    import re
    return re.findall(WORD_RE, text.lower())


def get_n_workers(arg: str | int | None) -> int:
    """Return worker count. `auto` = min(72, cpu_count())."""
    if arg == "auto" or arg is None:
        return min(72, mp.cpu_count())
    return int(arg)

###############################################################################
# Parquet reading helpers (shared by workers)
###############################################################################

def _read_words_from_parquet(path: str):
    """Stream rows from a parquet file and yield words from 'text' column."""
    table = pq.ParquetFile(path)
    for batch in table.iter_batches():
        col = batch.column('text')
        for cell in col:
            yield from words(cell.as_py())

###############################################################################
# Build‑phase worker functions (top‑level → picklable)
###############################################################################

def scan_vocab_worker(parquet_path: str) -> List[str]:
    """Return *unique* words in the given parquet shard."""
    seen: Dict[str, None] = {}
    for w in _read_words_from_parquet(parquet_path):
        seen[w] = None
    return list(seen.keys())

# ---------- Enc‑phase globals & initialiser ----------
VOCAB: Dict[str, int] | None = None  # populated in each worker


def _enc_init(vocab_bytes: bytes):
    global VOCAB
    VOCAB = pickle.loads(vocab_bytes)


def encode_shard_worker(args):
    """Encode words→ids for one shard and write .npy to out_dir."""
    parquet_path, out_dir = args
    assert VOCAB is not None, "VOCAB not initialised in worker"

    ids: List[int] = []
    for w in _read_words_from_parquet(parquet_path):
        ids.append(VOCAB[w])  # KeyError impossible because vocab built from scan

    out_fn = pathlib.Path(out_dir) / f"ids_{pathlib.Path(parquet_path).name}.npy"
    np.save(out_fn, np.asarray(ids, dtype=np.int32))
    return out_fn

###############################################################################
# Build phase orchestrator
###############################################################################

def build_index(parquet_dir: str, out_dir: str, workers: int):
    os.makedirs(out_dir, exist_ok=True)

    shards = sorted(pathlib.Path(parquet_dir).glob('*.parquet'))
    if not shards:
        raise FileNotFoundError('No parquet files found.')

    # Prefer fork context if available (Linux/macOS) for zero‑copy speed
    try:
        ctx = mp.get_context('fork')
    except ValueError:  # Windows or Py≥3.14 with no fork
        ctx = mp.get_context('spawn')

    # ------------------------------------------------------------------
    # Pass 1: vocabulary building
    # ------------------------------------------------------------------
    vocab: Dict[str, int] = {}
    next_id = 1  # 0 reserved for OOV

    with ctx.Pool(workers) as pool:
        for word_set in tqdm(pool.imap_unordered(scan_vocab_worker, shards),
                              total=len(shards), desc='Scanning vocabulary'):
            for w in word_set:
                # setdefault is atomic enough in single process; we merge here
                if w not in vocab:
                    vocab[w] = next_id
                    next_id += 1

    print(f"Unique words: {len(vocab):,}")

    # Save vocabulary (np.save → object array of unicode strings)
    vocab_path = os.path.join(out_dir, 'vocab.npy')
    np.save(vocab_path, np.array(list(vocab.keys()), dtype=object))

    # ------------------------------------------------------------------
    # Pass 2: encode shards in parallel (requires vocab broadcast)
    # ------------------------------------------------------------------
    vocab_bytes = pickle.dumps(vocab, protocol=pickle.HIGHEST_PROTOCOL)

    encode_args = [(str(p), out_dir) for p in shards]

    with ctx.Pool(workers, initializer=_enc_init, initargs=(vocab_bytes,)) as pool:
        id_files = []
        for fn in tqdm(pool.imap_unordered(encode_shard_worker, encode_args),
                       total=len(encode_args), desc='Encoding shards'):
            id_files.append(fn)

    # ------------------------------------------------------------------
    # Concatenate encoded shards → single memmap array
    # ------------------------------------------------------------------
    all_ids_path = os.path.join(out_dir, 'all_ids.npy')
    total_len = sum(np.load(f, mmap_mode='r').shape[0] for f in id_files)
    all_ids = np.memmap(all_ids_path, dtype=np.int32, mode='w+', shape=(total_len,))

    offset = 0
    for f in tqdm(id_files, desc='Concatenating ids'):
        arr = np.load(f, mmap_mode='r')
        n = arr.shape[0]
        all_ids[offset:offset+n] = arr
        offset += n

    # ------------------------------------------------------------------
    # Suffix array construction (single‑threaded C implementation)
    # ------------------------------------------------------------------
    print(f"Building suffix array for {total_len:,} ids … (this may take hours)")
    sa = pydivsufsort.divsufsort(all_ids)
    np.save(os.path.join(out_dir, 'suffix_array.npy'), sa)
    print("Index build completed ☑︎")

###############################################################################
# Query phase (unchanged logic, moved binary_search to top‑level)
###############################################################################

def binary_search(ids: np.ndarray, sa: np.ndarray, pattern: List[int]) -> bool:
    """Return True if *pattern* occurs in *ids* using SA binary search."""
    lo, hi = 0, sa.shape[0]
    m = len(pattern)
    pat_tuple = tuple(pattern)
    while lo < hi:
        mid = (lo + hi) // 2
        pos = sa[mid]
        slice_cmp = ids[pos:pos+m]
        if tuple(slice_cmp) < pat_tuple:
            lo = mid + 1
        else:
            hi = mid

    # Linear scan forward within the equal‑prefix region (usually tiny)
    while lo < sa.shape[0]:
        pos = sa[lo]
        curr = tuple(ids[pos:pos+m])
        if curr == pat_tuple:
            return True
        if curr > pat_tuple:
            break  # region passed
        lo += 1
    return False


def query_index(index_dir: str, input_file: str, workers: int):
    vocab_arr = np.load(os.path.join(index_dir, 'vocab.npy'), allow_pickle=True)
    vocab = {w: i+1 for i, w in enumerate(vocab_arr)}

    ids = np.load(os.path.join(index_dir, 'all_ids.npy'), mmap_mode='r')
    sa = np.load(os.path.join(index_dir, 'suffix_array.npy'), mmap_mode='r')

    with open(input_file, encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]

    win = 35  # 35‑word window ≈ 50 tokens

    def check_line(line: str) -> Tuple[str, bool]:
        wlist = words(line)
        seq = [vocab.get(w, 0) for w in wlist]
        if len(seq) < win:
            return line, False
        for i in range(len(seq) - win + 1):
            window = seq[i:i+win]
            if 0 in window:
                continue
            if binary_search(ids, sa, window):
                return line, True
        return line, False

    # Use threads (I/O bound on SA binary search memmap) → no pickling issues
    from multiprocessing.pool import ThreadPool
    with ThreadPool(workers) as pool:
        for line, found in tqdm(pool.imap_unordered(check_line, lines),
                                total=len(lines), desc='Querying'):
            status = 'HIT' if found else 'MISS'
            print(f"[{status}] {line[:120]}{'…' if len(line)>120 else ''}")

###############################################################################
# CLI entry‑point
###############################################################################

def main():
    parser = argparse.ArgumentParser(description='Word‑level suffix array indexer / querier')
    sub = parser.add_subparsers(dest='cmd', required=True)

    b = sub.add_parser('build', help='Build suffix array index from parquet shards')
    b.add_argument('--parquet-dir', required=True)
    b.add_argument('--out-dir', required=True)
    b.add_argument('--workers', default='auto')

    q = sub.add_parser('query', help='Query text file against existing index')
    q.add_argument('--index-dir', required=True)
    q.add_argument('--input', required=True)
    q.add_argument('--workers', default='auto')

    args = parser.parse_args()
    workers = get_n_workers(args.workers)

    if args.cmd == 'build':
        build_index(args.parquet_dir, args.out_dir, workers)
    else:
        query_index(args.index_dir, args.input, workers)


if __name__ == '__main__':
    mp.freeze_support()
    main()
