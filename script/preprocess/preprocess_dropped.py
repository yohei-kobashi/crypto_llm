import argparse
import os
import subprocess
import random
import unicodedata
from Crypto.Cipher import ChaCha20
from base64 import b64encode
from typing import Literal

from encrypt import EncryptPipeline
from utils import run_command, shuffle_and_split

def run_command(command):
    print(f"Running: {command}")
    subprocess.run(command, shell=True, check=True)


def encrypt(dataset, work_dir, src_dir, tgt_dir, ntasks=64, key_len=1):
    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.readers import JsonlReader
    from datatrove.pipeline.writers import JsonlWriter

    pipeline_exec = LocalPipelineExecutor(
        pipeline=[
            JsonlReader(
                src_dir,
                file_progress=True,
                doc_progress=True,
                glob_pattern="**/*.jsonl",
            ),
            EncryptPipeline(key_len=key_len),
            JsonlWriter(
                tgt_dir,
                output_filename=dataset + ".chunk.${rank}.jsonl",
                compression=None,
            ),
        ],
        tasks=ntasks,
        logging_dir=os.path.join(work_dir, "datatrove"),
    )
    pipeline_exec.run()


def main(
        dataset, 
        data_dir, 
        key_len,
        ):
    # Configuration
    src_dir = f"{data_dir}/{dataset}"
    out_dir = f"{src_dir}_encrypted_{str(key_len)}"
    os.makedirs(out_dir, exist_ok=True)
    work_dir = src_dir  # Directory of this Python file

    suffix = ".jsonl"
    nchunks = 2

    encrypt(dataset, work_dir, src_dir, out_dir, key_len)

    dataset_for_chunk = [src_dir, out_dir]
    for i in dataset_for_chunk:
        shuffle_and_split(dataset, work_dir, i, f"{i}_chunked", suffix=suffix, nchunks=nchunks)

    print("All tasks completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--key_len", type=int, default=1)

    args = parser.parse_args()

    main(args.dataset, args.data_dir, args.key_len)
