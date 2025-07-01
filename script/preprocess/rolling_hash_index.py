#!/usr/bin/env python3
"""
50-Word Rolling Hash Index Builder – two-pass parallel edition
==============================================================
First pass (parallel): collect unique words from each file to build a global word2id mapping.
Second pass (parallel): compute rolling hashes for each 50-word window using the fixed mapping.

Usage:
    python rolling_hash_index.py \
        --corpus DIR \
        --output index.pkl \
        [--workers N]

Output: pickle with {'hashes': Set[int], 'word2id': Dict[str,int]}
"""
import argparse
import json
import pickle
from pathlib import Path
from collections import deque
import multiprocessing as mp
import pyarrow.parquet as pq
import re

# parameters
WINDOW = 50
BASE = 1315423911
MOD = (1 << 61) - 1


def iter_words(path: Path):
    if path.suffix == '.parquet':
        table = pq.read_table(str(path), use_threads=True)
        for batch in table.to_batches():
            for cell in batch.column('text'):
                for w in str(cell.as_py()).split():
                    yield w
    elif path.suffix == '.jsonl':
        with path.open(encoding='utf-8', errors='ignore') as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    txt = rec.get('text', '')
                except:
                    txt = line.strip()
                for w in txt.split():
                    yield w


def process_file_vocab(path_str):
    path = Path(path_str)
    words = set()
    for w in iter_words(path):
        words.add(w)
    return words


def process_file_hashes(args):
    path_str, word2id = args
    path = Path(path_str)
    pow_base = pow(BASE, WINDOW, MOD)
    hashes = set()
    window = deque()
    rolling = 0
    for w in iter_words(path):
        wid = word2id.get(w)
        if wid is None:
            continue
        rolling = (rolling * BASE + wid) % MOD
        window.append(wid)
        if len(window) > WINDOW:
            head = window.popleft()
            rolling = (rolling - head * pow_base) % MOD
        if len(window) == WINDOW:
            hashes.add(rolling)
    return hashes


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--corpus', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--workers', type=int, default=mp.cpu_count())
    args = p.parse_args()

    files = [str(p) for p in sorted(args.corpus.rglob('*')) if Path(p).suffix in ('.parquet', '.jsonl')]
    if not files:
        print("No supported files in corpus")
        return

    # Pass 1: parallel vocab
    print(f"Building vocabulary with {args.workers} workers...")
    with mp.Pool(args.workers) as pool:
        list_of_sets = pool.map(process_file_vocab, files)
    all_words = set().union(*list_of_sets)
    word2id = {w: i+1 for i, w in enumerate(sorted(all_words))}
    print(f"Vocabulary size: {len(word2id)} words")

    # Pass 2: parallel hashes
    print(f"Computing rolling hashes with {args.workers} workers...")
    with mp.Pool(args.workers) as pool:
        args_list = [(f, word2id) for f in files]
        list_of_hash_sets = pool.map(process_file_hashes, args_list)
    hashes = set().union(*list_of_hash_sets)
    print(f"Computed {len(hashes)} unique hashes")

    # save index
    with open(args.output, 'wb') as f:
        pickle.dump({'word2id': word2id, 'hashes': hashes}, f, protocol=4)
    print(f"Index saved to {args.output}")

if __name__ == '__main__':
    main()
