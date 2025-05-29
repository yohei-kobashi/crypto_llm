import argparse
import glob
import json
import os
import random
import multiprocessing

import pyarrow.parquet as pq
import pyarrow.compute as pc

def load_ids_from_10bt(dir_10bt):
    """
    Load unique IDs from all Parquet files in the 10BT directory.
    """
    ids = []
    for data_file in glob.glob(os.path.join(dir_10bt, "*")):
        table = pq.read_table(data_file)
        unique_ids = table["id"].unique()
        ids.extend([unique_id.as_py() for unique_id in unique_ids])
    return set(ids)

# Global semaphore variable for controlling concurrent disk access
file_reading_semaphore = None

def init_worker(sema):
    """
    Worker initializer to set the global semaphore.
    """
    global file_reading_semaphore
    file_reading_semaphore = sema

def process_file(data_file, ids_in_10bt, sampling_rate, seed):
    """
    Process a single 100BT file:
      - Read the Parquet file (controlled by semaphore to limit disk I/O).
      - Extract unique IDs not present in the 10BT set.
      - Sample the IDs based on the sampling_rate and write the filtered results as JSONL.
    """
    print(f"Processing: {data_file}")
    # Limit file reading concurrency using the semaphore (e.g., 4 concurrent accesses)
    with file_reading_semaphore:
        table = pq.read_table(data_file)
    
    unique_ids = set([unique_id.as_py() for unique_id in table["id"].unique()])
    ids_not_in_10bt = list(unique_ids - ids_in_10bt)
    print(f"Total IDs: {len(unique_ids)}")
    print(f"IDs not in 10BT: {len(ids_not_in_10bt)}")
    
    # Set a reproducible seed per file using the provided seed and file-specific hash
    file_seed = seed + (hash(data_file) % (10**8))
    random.seed(file_seed)
    
    output_file = os.path.join("sample", "references", os.path.basename(data_file).replace(".parquet", ".jsonl"))
    with open(output_file, "w") as f:
        num_samples = int(len(ids_not_in_10bt) * sampling_rate)
        if num_samples > 0:
            for unique_id in random.sample(ids_not_in_10bt, num_samples):
                mask = pc.equal(table.column("id"), unique_id)
                filtered_table = table.filter(mask)
                f.write(json.dumps(filtered_table.to_pydict()) + "\n")

def process_file_wrapper(params):
    """
    Wrapper for process_file to allow passing multiple arguments via pool.map.
    """
    data_file, ids_in_10bt, sampling_rate, seed = params
    process_file(data_file, ids_in_10bt, sampling_rate, seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract texts not in 10BT from 100BT Parquet files with sampling."
    )
    parser.add_argument("--dir100bt", required=True, help="Directory containing 100BT Parquet files")
    parser.add_argument("--dir10bt", required=True, help="Directory containing 10BT Parquet files")
    parser.add_argument("--num-procs", type=int, default=20, help="Number of parallel processes")
    parser.add_argument("--sampling-rate", type=float, default=0.01, help="Sampling rate for IDs not in 10BT")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random sampling")
    args = parser.parse_args()

    # Load unique IDs from 10BT directory once in the main process
    ids_in_10bt = load_ids_from_10bt(args.dir10bt)

    # Create the output directory if it doesn't exist
    output_dir = os.path.join("sample", "references")
    os.makedirs(output_dir, exist_ok=True)

    # Create a semaphore to limit concurrent file reading (e.g., 4 concurrent accesses)
    semaphore = multiprocessing.Semaphore(4)

    # List all files from the 100BT directory
    data_files = glob.glob(os.path.join(args.dir100bt, "*"))

    # Prepare parameters for each file processing task
    params_list = [(data_file, ids_in_10bt, args.sampling_rate, args.seed) for data_file in data_files]

    # Create a pool of workers with the specified number of processes
    with multiprocessing.Pool(processes=args.num_procs, initializer=init_worker, initargs=(semaphore,)) as pool:
        pool.map(process_file_wrapper, params_list)
