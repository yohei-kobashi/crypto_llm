# suffix_array_pipeline (with progress)
# -------------------------------------------------------------
# Build & query a word‑level suffix array over fineweb‑edu sample‑10BT
# (14×2.1 GB Parquet shards) to detect ≥ 35‑word verbatim substrings.
# Progress reporting has been added for both build and query phases
# via tqdm progress bars and logging.
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
import argparse
import os
import sys
import pathlib
import multiprocessing as mp
from functools import partial
from typing import List, Tuple

import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
from tqdm import tqdm

try:
    import pydivsufsort  # C‑accelerated suffix‑array construction
except ImportError:
    raise SystemExit("Please `pip install pydivsufsort`.")

###############################################################################
# Utility helpers
###############################################################################

def get_n_workers(arg: str | int | None) -> int:
    """Return worker count. `auto` = min(72, cpu_count())."""
    if arg == "auto" or arg is None:
        return min(72, mp.cpu_count())
    return int(arg)

WORD_RE = r"[\w'-]+"  # coarse word tokenizer

def words(text: str) -> List[str]:
    import re
    return re.findall(WORD_RE, text.lower())

###############################################################################
# Build phase
###############################################################################

def _read_words_from_parquet(path: str):
    """Stream rows from a parquet file and yield words."""
    table = pq.ParquetFile(path)
    for batch in table.iter_batches():
        # assume text column is called 'text'
        col = batch.column('text')
        for cell in col:
            yield from words(cell.as_py())

def build_index(parquet_dir: str, out_dir: str, workers: int):
    os.makedirs(out_dir, exist_ok=True)

    shards = sorted(pathlib.Path(parquet_dir).glob('*.parquet'))
    if not shards:
        raise FileNotFoundError('No parquet files found.')

    # Pass 1: build global vocab
    vocab: dict[str,int] = {}
    next_id = 1  # 0 = sentinel

    def scan_vocab(p):
        local = {}
        for w in _read_words_from_parquet(p):
            local[w] = None
        return local.keys()

    with mp.Pool(workers) as pool:
        for word_set in tqdm(pool.imap_unordered(scan_vocab, shards),
                              total=len(shards), desc='Scanning vocabulary'):
            for w in word_set:
                if w not in vocab:
                    vocab[w] = next_id
                    next_id += 1

    # Persist vocab
    np.save(os.path.join(out_dir, 'vocab.npy'), np.array(list(vocab.keys())))

    # Pass 2: encode words → ids and concatenate
    def encode_shard(p):
        ids = []
        for w in _read_words_from_parquet(p):
            ids.append(vocab[w])
        fn = pathlib.Path(out_dir)/f"ids_{p.name}.npy"
        np.save(fn, np.asarray(ids, dtype=np.int32))
        return fn

    id_files = []
    with mp.Pool(workers) as pool:
        for fn in tqdm(pool.imap_unordered(encode_shard, shards),
                       total=len(shards), desc='Encoding shards'):
            id_files.append(fn)

    # Concatenate all id arrays to one large array (memory‑map for huge size)
    all_ids_path = os.path.join(out_dir, 'all_ids.npy')
    total_len = sum(np.load(f, mmap_mode='r').shape[0] for f in id_files)
    all_ids = np.memmap(all_ids_path, dtype=np.int32, mode='w+', shape=(total_len,))

    offset = 0
    for f in tqdm(id_files, desc='Concatenating ids'):
        arr = np.load(f, mmap_mode='r')
        n = arr.shape[0]
        all_ids[offset:offset+n] = arr
        offset += n

    # Build suffix array (may take time; progress printed every 5 M elements)
    print("Building suffix array using pydivsufsort... (this can take hours)")
    sa = pydivsufsort.divsufsort(all_ids)

    sa_path = os.path.join(out_dir, 'suffix_array.npy')
    np.save(sa_path, sa)
    print(f"Index built: vocab={len(vocab):,}, tokens={total_len:,}")

###############################################################################
# Query phase
###############################################################################

def binary_search(ids: np.ndarray, sa: np.ndarray, pattern: List[int]) -> bool:
    """Return True if *pattern* occurs in *ids* using SA binary search."""
    import bisect
    lo, hi = 0, sa.shape[0]
    m = len(pattern)
    while lo < hi:
        mid = (lo + hi) // 2
        pos = sa[mid]
        slice_cmp = ids[pos:pos+m]
        if tuple(slice_cmp) < tuple(pattern):
            lo = mid + 1
        else:
            hi = mid
    start = lo
    # Verify prefix range
    pat_tuple = tuple(pattern)
    while start < sa.shape[0]:
        pos = sa[start]
        if tuple(ids[pos:pos+m]) == pat_tuple:
            return True
        if tuple(ids[pos:pos+m]) > pat_tuple:
            break
        start += 1
    return False


def query_index(index_dir: str, input_file: str, workers: int):
    vocab_list = np.load(os.path.join(index_dir, 'vocab.npy'), allow_pickle=True)
    vocab = {w: i+1 for i, w in enumerate(vocab_list)}  # rebuild dict
    ids = np.load(os.path.join(index_dir, 'all_ids.npy'), mmap_mode='r')
    sa = np.load(os.path.join(index_dir, 'suffix_array.npy'), mmap_mode='r')

    # Read model outputs
    with open(input_file, encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]

    win = 35  # word window (≈50 tokens)

    def check_line(line: str) -> Tuple[str, bool]:
        wlist = words(line)
        seq = [vocab.get(w, 0) for w in wlist]  # OOV → 0 (never matches)
        if len(seq) < win:
            return line, False
        for i in range(len(seq) - win + 1):
            window = seq[i:i+win]
            if 0 in window:
                continue  # window contains OOV, skip
            if binary_search(ids, sa, window):
                return line, True
        return line, False

    results = []
    with mp.Pool(workers) as pool:
        for res in tqdm(pool.imap_unordered(check_line, lines),
                        total=len(lines), desc='Querying'):
            results.append(res)

    for line, found in results:
        status = "HIT" if found else "MISS"
        print(f"[{status}] {line[:120]}{'...' if len(line)>120 else ''}")

###############################################################################
# Entry‑point CLI
###############################################################################

def main():
    parser = argparse.ArgumentParser(description='Word‑level suffix array indexer/query.')
    sub = parser.add_subparsers(dest='cmd', required=True)

    b = sub.add_parser('build')
    b.add_argument('--parquet-dir', required=True)
    b.add_argument('--out-dir', required=True)
    b.add_argument('--workers', default='auto')

    q = sub.add_parser('query')
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
