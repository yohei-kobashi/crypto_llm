#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional, List


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


def read_file_to_df(path: Path):
    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "pandas is required to run this script. Please install pandas (and pyarrow or fastparquet)."
        ) from e

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)

    # Parquet: try pyarrow first, then fastparquet, then default
    for engine in ("pyarrow", "fastparquet", None):
        try:
            if engine is None:
                return pd.read_parquet(path)
            return pd.read_parquet(path, engine=engine)
        except Exception:
            continue
    # If all attempts failed, raise a clearer error by re-running without catching
    return pd.read_parquet(path)


def convert_one(
    parquet_path: Path,
    output_dir: Path,
    frac: float,
    seed: int,
) -> Optional[Path]:
    try:
        df = read_file_to_df(parquet_path)
    except Exception as e:
        print(f"[WARN] Failed to read: {parquet_path} ({e})", file=sys.stderr)
        return None

    if not 0 < frac <= 1:
        raise ValueError("frac must be in the interval (0, 1].")

    if frac < 1:
        try:
            df = df.sample(frac=frac, random_state=seed)
        except Exception as e:
            print(f"[WARN] Sampling failed for {parquet_path}: {e}", file=sys.stderr)
            return None

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / (parquet_path.stem + ".jsonl")

    # Use pandas to_json to emit JSON Lines with column names as keys
    try:
        # Ensure index not included; ISO dates for readability
        df.to_json(out_path, orient="records", lines=True, force_ascii=False, date_format="iso")
    except Exception as e:
        print(f"[WARN] Failed to write JSONL for {parquet_path}: {e}", file=sys.stderr)
        return None

    return out_path


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=(
            "Load Parquet or JSONL files from a directory, sample rows with a given seed and fraction, "
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
