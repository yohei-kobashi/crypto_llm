#!/usr/bin/env python3
"""
Rolling-hash Query: Check if a text contains any 50-word window in the built index.
==============================================================================
Loads a pickle index with {'word2id': Dict[str,int], 'hashes': Set[int]} and then for
each input query (text line), normalizes and splits into words, computes rolling hashes
for each 50-word window, and reports MATCH/NO MATCH.

Usage:
    python rolling_hash_query.py \
        --index index.pkl \
        --input queries.txt \
        --output results.txt

Output format:
    MATCH: <query line>
    NO MATCH: <query line>
"""
import argparse
import pickle
from collections import deque

def normalize(text: str):
    # simple whitespace normalization
    return text.strip().split()


def load_index(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['word2id'], data['hashes']


def query_line(line: str, word2id: dict, hash_set: set, window=50, base=1315423911, mod=(1<<61)-1):
    words = normalize(line)
    ids = [word2id.get(w) for w in words]
    # filter unknowns
    ids = [wid for wid in ids if wid is not None]
    if len(ids) < window:
        return f"NO MATCH: {line.strip()}"
    pow_base = pow(base, window, mod)
    rolling = 0
    dq = deque()
    # first window
    for x in ids[:window]:
        rolling = (rolling * base + x) % mod
        dq.append(x)
    if rolling in hash_set:
        return f"MATCH: {line.strip()}"
    # slide
    for x in ids[window:]:
        head = dq.popleft()
        rolling = (rolling * base - head * pow_base + x) % mod
        dq.append(x)
        if rolling in hash_set:
            return f"MATCH: {line.strip()}"
    return f"NO MATCH: {line.strip()}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--index', required=True, help='Path to index.pkl')
    p.add_argument('--input', required=True, help='Query file, one line per query')
    p.add_argument('--output', required=True, help='Results output file')
    args = p.parse_args()

    word2id, hash_set = load_index(args.index)
    with open(args.input, encoding='utf-8') as fin, open(args.output, 'w', encoding='utf-8') as fout:
        for line in fin:
            if not line.strip():
                continue
            res = query_line(line, word2id, hash_set)
            fout.write(res + '\n')

if __name__ == '__main__':
    main()
