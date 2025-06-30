#!/usr/bin/env python3
"""
Generate test queries mixing slightly-over-50-token segments and dummy non-matching text.
=============================================================================
Usage:
    python generate_test_queries.py \
        --corpus CORPUS_DIR \
        --spm_model tokenizer.model \
        --output queries.txt \
        [--num_queries N] \
        [--extra_max M]

Produces a file with 2*N lines:
  - N real segments (guaranteed to exist in corpus) of length = 50 + random(1..extra_max)
  - N dummy segments ("dummy" repeated) that won't match.

Options:
  --num_queries   Number of matching and non-matching segments each (default: 10)
  --extra_max     Max extra tokens beyond 50 (default: 5)
"""
import argparse
import json
import random
from pathlib import Path

import sentencepiece as spm
import pyarrow.parquet as pq


def collect_texts(corpus_dir):
    """Yield text fields from JSONL and Parquet under corpus_dir"""
    for file in sorted(Path(corpus_dir).rglob("*.jsonl")):
        with file.open('r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    text = obj.get('text', '')
                except Exception:
                    continue
                if text:
                    yield text
    for file in sorted(Path(corpus_dir).rglob("*.parquet")):
        table = pq.read_table(str(file), columns=['text'], use_threads=True)
        for batch in table.to_batches():
            for cell in batch.column(0):
                text = cell.as_py() or ''
                if text:
                    yield text


def generate_queries(corpus_dir, spm_model, output_file, num_queries=10, extra_max=5):
    sp = spm.SentencePieceProcessor()
    sp.Load(str(spm_model))
    texts = collect_texts(corpus_dir)
    real_segments = []
    # Generate real segments
    for text in texts:
        ids = sp.EncodeAsIds(text)
        if len(ids) < 50 + 1:
            continue
        # choose length slightly >50
        length = 50 + random.randint(1, extra_max)
        if len(ids) < length:
            continue
        start = random.randint(0, len(ids) - length)
        segment_ids = ids[start:start + length]
        segment_text = sp.DecodeIds(segment_ids)
        real_segments.append(segment_text)
        if len(real_segments) >= num_queries:
            break
    # Generate dummy segments
    dummy_segments = ["dummy " * (50 + extra_max)] * num_queries

    # Combine and write
    with open(output_file, 'w', encoding='utf-8') as out:
        for seg in real_segments:
            out.write(seg.strip() + '\n')
        for seg in dummy_segments:
            out.write(seg.strip() + '\n')
    print(f"Generated {len(real_segments)} real and {len(dummy_segments)} dummy queries in {output_file}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--corpus', required=True, help='Directory of JSONL/Parquet files')
    p.add_argument('--spm_model', required=True, help='SentencePiece .model file')
    p.add_argument('--output', required=True, help='Output file for queries')
    p.add_argument('--num_queries', type=int, default=10, help='Number of queries to generate')
    p.add_argument('--extra_max', type=int, default=5, help='Max extra tokens beyond 50')
    args = p.parse_args()
    generate_queries(args.corpus, args.spm_model, args.output, args.num_queries, args.extra_max)
