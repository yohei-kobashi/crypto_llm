#!/usr/bin/env python3
import argparse
import sys
import json
import random
from pathlib import Path
from typing import Iterable, Optional, List
from tempfile import NamedTemporaryFile


def find_input_files(input_dir: Path, recursive: bool = False) -> Iterable[Path]:
    patterns = ["*.parquet", "*.jsonl"]
    files: List[Path] = []
    if recursive:
        for pat in patterns:
            files.extend(input_dir.rglob(pat))
    else:
        for pat in patterns:
            files.extend(input_dir.glob(pat))
    return sorted(files)


def _process_jsonl_stream(input_path: Path, out_path: Path, frac: float, seed: int) -> None:
    rng = random.Random(seed)

    # If writing to same file, write to temp file then replace
    actual_out = out_path
    need_replace = input_path.resolve() == out_path.resolve()
    if need_replace:
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        actual_out = tmp

    with input_path.open("r", encoding="utf-8") as fin, actual_out.open("w", encoding="utf-8") as fout:
        for line in fin:
            if rng.random() < frac:
                fout.write(line)

    if need_replace:
        actual_out.replace(out_path)


def _process_parquet_stream(input_path: Path, out_path: Path, frac: float, seed: int, batch_size: int = 65536) -> None:
    try:
        import pyarrow.parquet as pq  # type: ignore
        import pandas as pd  # type: ignore
        import numpy as np  # type: ignore
    except Exception as e:
        # Fallback to pandas full read (may be memory heavy)
        try:
            import pandas as pd  # type: ignore
        except Exception:
            raise RuntimeError("pyarrow or pandas is required to read parquet files") from e

        df = pd.read_parquet(input_path)
        if frac < 1:
            try:
                rng = __import__("numpy").random.RandomState(seed)  # type: ignore
                df = df.sample(frac=frac, random_state=rng)
            except Exception:
                df = df.sample(frac=frac, random_state=seed)
        df.to_json(out_path, orient="records", lines=True, force_ascii=False, date_format="iso")
        return

    pf = pq.ParquetFile(str(input_path))

    # If writing to same file name as input (same dir and stem), write to temp then replace
    actual_out = out_path
    need_replace = input_path.resolve() == out_path.resolve()
    if need_replace:
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        actual_out = tmp

    rng = np.random.RandomState(seed)
    # Open once and append per-batch
    with actual_out.open("w", encoding="utf-8") as fout:
        for batch in pf.iter_batches(batch_size=batch_size):
            # Convert batch to pandas for robust JSONL writing (handles datetimes)
            df = batch.to_pandas()
            if frac < 1:
                try:
                    df = df.sample(frac=frac, random_state=rng)
                except Exception:
                    df = df.sample(frac=frac, random_state=seed)
            if len(df) == 0:
                continue
            df.to_json(fout, orient="records", lines=True, force_ascii=False, date_format="iso")

    if need_replace:
        actual_out.replace(out_path)


def convert_one(
    parquet_path: Path,
    output_dir: Path,
    frac: float,
    seed: int,
) -> Optional[Path]:
    if not 0 < frac <= 1:
        raise ValueError("frac must be in the interval (0, 1].")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / (parquet_path.stem + ".jsonl")

    try:
        suffix = parquet_path.suffix.lower()
        if suffix == ".jsonl":
            _process_jsonl_stream(parquet_path, out_path, frac=frac, seed=seed)
        else:
            _process_parquet_stream(parquet_path, out_path, frac=frac, seed=seed)
        return out_path
    except Exception as e:
        print(f"[WARN] Failed to convert {parquet_path} -> {out_path}: {e}", file=sys.stderr)
        return None


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=(
            "Load Parquet or JSONL from a file or directory, sample rows with a given seed and fraction, "
            "and write JSONL files with the same base filename into an output directory."
        )
    )
    p.add_argument(
        "--input_file",
        "-i",
        type=Path,
        required=True,
        help="Input file (.parquet or .jsonl) or a directory containing such files",
    )
    p.add_argument("--output_dir", "-o", type=Path, required=True, help="Directory to write .jsonl files")
    p.add_argument("--frac", "-f", type=float, required=True, help="Sampling fraction (0 < frac <= 1)")
    p.add_argument("--seed", "-s", type=int, default=42, help="Random seed for sampling")
    p.add_argument("--recursive", "-r", action="store_true", help="If input is a directory, recurse into subdirectories")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    input_path: Path = args.input_file
    output_dir: Path = args.output_dir
    frac: float = args.frac
    seed: int = args.seed
    recursive: bool = args.recursive

    if not input_path.exists():
        print(f"[ERROR] input not found: {input_path}", file=sys.stderr)
        return 2

    # Determine target files depending on whether input is a file or directory
    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        try:
            files = list(find_input_files(input_path, recursive=recursive))
        except Exception as e:
            print(f"[ERROR] Failed to list input files: {e}", file=sys.stderr)
            return 2
        if not files:
            print(
                f"[WARN] No .parquet or .jsonl files found in {input_path} (recursive={recursive})",
                file=sys.stderr,
            )
            return 0
    else:
        print(f"[ERROR] input path is neither file nor directory: {input_path}", file=sys.stderr)
        return 2

    converted = 0
    for idx, fp in enumerate(sorted(files)):
        print(f"[{idx+1}/{len(files)}] Processing {fp} ...")
        out_path = convert_one(fp, output_dir, frac=frac, seed=seed)
        if out_path is not None:
            print(f"-> Wrote {out_path}")
            converted += 1

    print(f"Done. Converted {converted}/{len(files)} files.")
    return 0 if converted > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
