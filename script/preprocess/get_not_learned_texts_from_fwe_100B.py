import argparse
import glob
import os
import random
import multiprocessing

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
        # Read JSON Lines into a table
        return pj.read_json(path)
    else:
        raise ValueError(f"Unsupported file format: {path}")


def load_ids_from_10bt(path):
    """
    Load unique IDs from a Parquet/JSONL file or recursively from a directory.
    """
    ids = []
    if os.path.isfile(path):
        table = read_table_generic(path)
        unique_ids = table['id'].unique()
        ids.extend([uid.as_py() for uid in unique_ids])
    else:
        patterns = [
            os.path.join(path, '**', '*.parquet'),
            os.path.join(path, '**', '*.jsonl')
        ]
        for pattern in patterns:
            for data_file in glob.glob(pattern, recursive=True):
                table = read_table_generic(data_file)
                unique_ids = table['id'].unique()
                ids.extend([uid.as_py() for uid in unique_ids])
    return set(ids)

# Semaphore for controlling concurrent disk access
file_reading_semaphore = None

def init_worker(sema):
    """
    Worker initializer to set the global semaphore.
    """
    global file_reading_semaphore
    file_reading_semaphore = sema


def process_file(data_file, ids_in_10bt, sampling_rate, seed, output_dir, include_flag):
    """
    Process a single file (Parquet or JSONL):
      - Read the file (controlled by semaphore to limit disk I/O).
      - Identify IDs in or not in the 10BT set based on include_flag.
      - Sample IDs based on sampling_rate and write filtered rows as a Parquet file.
      - Skip processing if output file already exists.
    """
    base = os.path.basename(data_file)
    suffix = '.included.sampled.parquet' if include_flag else '.sampled.parquet'
    base = base.replace('.parquet', suffix).replace('.jsonl', suffix)
    output_file = os.path.join(output_dir, base)

    # Resume logic: skip if already processed
    if os.path.exists(output_file):
        print(f"Skipping {data_file}, output already exists: {output_file}")
        return

    print(f"Processing: {data_file}")
    with file_reading_semaphore:
        table = read_table_generic(data_file)

    unique_ids = set(uid.as_py() for uid in table['id'].unique())
    if include_flag:
        target_ids = list(unique_ids & ids_in_10bt)
        print(f"IDs in 10BT: {len(target_ids)}")
    else:
        target_ids = list(unique_ids - ids_in_10bt)
        print(f"IDs not in 10BT: {len(target_ids)}")

    num_samples = int(len(target_ids) * sampling_rate)
    if num_samples <= 0:
        print("No samples to write.")
        return

    file_seed = seed + (hash(data_file) % (10**8))
    random.seed(file_seed)
    sampled_ids = random.sample(target_ids, num_samples)

    mask = pc.is_in(table.column('id'), value_set=pa.array(sampled_ids))
    filtered_table = table.filter(mask)

    pq.write_table(filtered_table, output_file)
    print(f"Written sampled data to: {output_file}")


def process_file_wrapper(params):
    process_file(*params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Sample and extract rows from Parquet or JSONL files based on 10BT membership."
    )
    parser.add_argument("--dir100bt", required=True,
                        help="Path to a Parquet/JSONL file or directory containing 100BT files")
    parser.add_argument("--dir10bt", required=True,
                        help="Path to a Parquet/JSONL file or directory for 10BT reference IDs")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to save sampled Parquet files")
    parser.add_argument("--num-procs", type=int, default=20,
                        help="Number of parallel processes")
    parser.add_argument("--sampling-rate", type=float, default=0.01,
                        help="Sampling rate for target IDs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for random sampling")
    parser.add_argument("--include", action='store_true',
                        help="Extract IDs included in the 10BT set instead of excluded ones")
    args = parser.parse_args()

    ids_in_10bt = load_ids_from_10bt(args.dir10bt)
    os.makedirs(args.output_dir, exist_ok=True)
    semaphore = multiprocessing.Semaphore(4)

    # Determine list of files to process for 100BT
    if os.path.isfile(args.dir100bt):
        data_files = [args.dir100bt]
    else:
        patterns = [
            os.path.join(args.dir100bt, '**', '*.parquet'),
            os.path.join(args.dir100bt, '**', '*.jsonl')
        ]
        data_files = []
        for pattern in patterns:
            data_files.extend(glob.glob(pattern, recursive=True))

    params_list = [
        (
            data_file,
            ids_in_10bt,
            args.sampling_rate,
            args.seed,
            args.output_dir,
            args.include
        )
        for data_file in data_files
    ]

    with multiprocessing.Pool(
        processes=args.num_procs,
        initializer=init_worker,
        initargs=(semaphore,)
    ) as pool:
        pool.map(process_file_wrapper, params_list)
