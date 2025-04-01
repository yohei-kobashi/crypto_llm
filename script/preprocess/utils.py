# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import os
import subprocess

def run_command(command):
    print(f"Running: {command}")
    subprocess.run(command, shell=True, check=True)


def setup_terashuf(work_dir):
    terashuf_dir = os.path.join(work_dir, "terashuf")
    terashuf_executable = os.path.join(terashuf_dir, "terashuf")

    if os.path.exists(terashuf_executable):
        print("terashuf executable already exists. Skipping setup.")
        return terashuf_dir

    print("Setting up terashuf...")
    run_command(f"git clone https://github.com/alexandres/terashuf {terashuf_dir}")
    run_command(f"make -C {terashuf_dir}")
    return terashuf_dir


def shuffle_and_split(
        dataset: str,
        work_dir: str,
        src_dir: str,
        out_dir: str,
        memory: int = 8,
        seed: int = 42,
        suffix = ".jsonl",
        nchunks = 2,
        cat_command = "cat",
        orig_extension = ".jsonl"
        ):
    
    prefix = f"{dataset}.chunk."
    os.makedirs(out_dir, exist_ok=True)

    # Setup terashuf
    terashuf_dir = setup_terashuf(work_dir)

    # Set up environment variables
    os.environ["MEMORY"] = f"{memory}"
    os.environ["SEED"] = f"{seed}"

    # Run the original shuffling and splitting command
    terashuf_executable = os.path.join(terashuf_dir, "terashuf")
    run_command(
        f"ulimit -n 100000 && "
        f"trap 'echo \"Caught signal 13, exiting with code 1\"; exit 1' PIPE && "
        f"find {src_dir} -type f -name '*{orig_extension}' -print0 | xargs -0 {cat_command} | {terashuf_executable} | "
        f"split -n r/{nchunks} -d --suffix-length 2 --additional-suffix {suffix} - {out_dir}/{prefix}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("memory", type=float, default=8)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    main(args.dataset, args.memory, args.data_dir, args.seed)
