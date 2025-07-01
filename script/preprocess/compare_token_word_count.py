#!/usr/bin/env python3
"""
Compare SentencePiece token counts vs. word counts for Parquet text corpus
========================================================================
Reads all .parquet files under a directory, extracts the 'text' column,
computes token count (via SentencePiece) and word count (whitespace split) per record,
and prints summary statistics and optional per-record CSV.

Usage:
    python compare_token_word_count.py \
        --corpus DIR \
        --spm_model tokenizer.model \
        [--output stats.csv]
"""
import argparse
import csv
from pathlib import Path
import sentencepiece as spm
import pyarrow.parquet as pq


def normalize_words(text: str):
    # collapse whitespace and split
    return text.strip().split()


def process_corpus(corpus_dir: Path, sp_model: Path, output: Path = None):
    # Load SentencePiece model
    sp = spm.SentencePieceProcessor()
    sp.load(str(sp_model))

    # Prepare CSV if needed
    writer = None
    if output:
        fout = open(output, 'w', newline='', encoding='utf-8')
        writer = csv.writer(fout)
        writer.writerow(['file', 'row_index', 'token_count', 'word_count', 'ratio'])

    total_tokens = 0
    total_words = 0
    total_count = 0

    # Iterate parquet files
    for parquet in sorted(corpus_dir.rglob('*.parquet')):
        table = pq.read_table(str(parquet), columns=['text'], use_threads=True)
        for batch in table.to_batches():
            col = batch.column(0)
            for idx, cell in enumerate(col):
                text = cell.as_py()
                words = normalize_words(text)
                word_count = len(words)
                tokens = sp.encode_as_pieces(text)
                token_count = len(tokens)
                total_tokens += token_count
                total_words += word_count
                total_count += 1
                if writer:
                    writer.writerow([parquet.name, idx, token_count, word_count,
                                     token_count / word_count if word_count else 0])

    # Close CSV
    if writer:
        fout.close()

    # Print summary
    if total_count:
        print(f"Processed {total_count} records.")
        print(f"Average tokens: {total_tokens/total_count:.2f}")
        print(f"Average words: {total_words/total_count:.2f}")
        print(f"Average token/word ratio: {total_tokens/total_words:.2f}")
    else:
        print("No records processed.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--corpus', type=Path, required=True,
                   help='Directory containing .parquet files')
    p.add_argument('--spm_model', type=Path, required=True,
                   help='SentencePiece .model file')
    p.add_argument('--output', type=Path, default=None,
                   help='Optional CSV output path')
    args = p.parse_args()
    process_corpus(args.corpus, args.spm_model, args.output)

if __name__ == '__main__':
    main()
