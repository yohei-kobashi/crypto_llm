"""
Sample rows from multiple Parquet (or JSONL) files in a directory at a specified rate,
show real-time progress, and save the sampled subsets to an output directory.
"""
import argparse
import glob
import os
import random
import multiprocessing
from tqdm import tqdm

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow.json as pj


def read_table_generic(path):
    """
    Read a Parquet or JSONL file into a PyArrow Table.
    """
    if path.endswith('.parquet'):
        return pq.read_table(path)
    elif path.endswith('.jsonl'):
        return pj.read_json(path)
    else:
        raise ValueError(f"Unsupported file format: {path}")


_file_semaphore = None

def init_worker(sema):
    """
    Worker initializer to set the global semaphore.
    """
    global _file_semaphore
    _file_semaphore = sema


def process_and_mark(args):
    """
    Unpack args, process file, and return file path for progress tracking.
    """
    data_file, sampling_rate, seed, output_dir = args
    process_file(data_file, sampling_rate, seed, output_dir)
    return data_file


def process_file(data_file, sampling_rate, seed, output_dir):
    """
    Read a Parquet/JSONL file, sample rows at the given rate, and write a new Parquet file.
    """
    # control concurrent reads
    with _file_semaphore:
        table = read_table_generic(data_file)

    num_rows = table.num_rows
    num_samples = int(num_rows * sampling_rate)
    if num_samples <= 0:
        sampled = table.slice(0, 0)
    else:
        # reproducible per-file sampling
        file_seed = seed + (hash(data_file) % (10**8))
        random.seed(file_seed)
        indices = random.sample(range(num_rows), num_samples)
        indices_array = pa.array(indices, type=pa.int64())
        sampled = pc.take(table, indices_array)

    # prepare output filename
    base = os.path.basename(data_file)
    name, _ext = os.path.splitext(base)
    out_name = f"{name}.sampled.parquet"
    out_path = os.path.join(output_dir, out_name)

    # write out
    pq.write_table(sampled, out_path)


def find_all_files(input_dir):
    patterns = [os.path.join(input_dir, '**', '*.parquet'),
                os.path.join(input_dir, '**', '*.jsonl')]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat, recursive=True))
    return files


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Sample Parquet/JSONL files in a directory at a specified rate with progress bar."
    )
    parser.add_argument("--input-dir", required=True,
                        help="Directory containing Parquet or JSONL files to sample.")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to save sampled Parquet files.")
    parser.add_argument("--sampling-rate", type=float, default=0.01,
                        help="Fraction of rows to sample from each file (0-1).")
    parser.add_argument("--num-procs", type=int, default=4,
                        help="Number of parallel processes.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base seed for reproducible sampling.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # find all Parquet/JSONL files
    all_files = find_all_files(args.input_dir)

    # prepare tasks, skip existing
    tasks = []
    for f in all_files:
        out_base = os.path.splitext(os.path.basename(f))[0] + '.sampled.parquet'
        out_path = os.path.join(args.output_dir, out_base)
        if os.path.exists(out_path):
            try:
                pq.ParquetFile(out_path)
                continue
            except Exception:
                os.remove(out_path)
        tasks.append((f, args.sampling_rate, args.seed, args.output_dir))

    total = len(tasks)
    if total == 0:
        print("No files to process.")
        exit(0)

    # parallel execution with progress bar
    semaphore = multiprocessing.Semaphore(2)
    with multiprocessing.Pool(processes=args.num_procs,
                              initializer=init_worker,
                              initargs=(semaphore,)) as pool:
        for _ in tqdm(pool.imap_unordered(process_and_mark, tasks), total=total,
                       desc="Sampling files", unit="file"):
            pass

    print("Sampling complete.")
