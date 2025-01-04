import argparse
import os
import subprocess
import random
import unicodedata
from Crypto.Cipher import ChaCha20
from base64 import b64encode
from typing import Literal

from datatrove.pipeline.filters import SamplerFilter

from encrypt import EncryptPipeline
from utils import run_command, shuffle_and_split


def encrypt(dataset, work_dir, src_dir, tgt_dir, ntasks=64, sample_ratio=1.0, seed=0, key_len=1):
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
            SamplerFilter(
                sample_ratio, seed=seed,
                exclusion_writer=JsonlWriter(
                    f"{src_dir}_{str(sample_ratio)}_continual",
                    output_filename=dataset + ".chunk.${rank}.jsonl",
                    compression=None,
                ),
            ),
            JsonlWriter(
                f"{src_dir}_{str(sample_ratio)}_pretrain",
                output_filename=dataset + ".chunk.${rank}.jsonl",
                compression=None,
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
        k_validation, # Number of lines to take from each chunk for validation
        sample_ratio, # Ratio of the dataset to sample
        key_len,
        ):
    # Configuration
    src_dir = f"{data_dir}/{dataset}"
    out_dir = f"{src_dir}_{str(sample_ratio)}_pretrain_encrypted_{str(key_len)}"
    os.makedirs(out_dir, exist_ok=True)
    work_dir = src_dir  # Directory of this Python file
    prefix = f"{dataset}.chunk."

    suffix = ".jsonl"
    nchunks = 2

    encrypt(dataset, work_dir, src_dir, out_dir, sample_ratio=sample_ratio)

    dataset_for_chunk = [
        f"{src_dir}_{str(sample_ratio)}_pretrain",
        f"{src_dir}_{str(sample_ratio)}_continual",
        out_dir
        ]
    for i in dataset_for_chunk:
        shuffle_and_split(dataset, work_dir, i, f"{i}_chunked", suffix=suffix, nchunks=nchunks)

    dataset_for_validate = [
        f"{src_dir}_{str(sample_ratio)}_pretrain_chunked",
        f"{src_dir}_{str(sample_ratio)}_continual_chunked",
        f"{out_dir}_chunked"
        ]
    for d in dataset_for_validate:
    # Create validation set and remove lines from chunks
        validation_file = f"{d}/{dataset}.val{suffix}"
        for i in range(nchunks):
            chunk_file = f"{d}/{prefix}{i:02d}{suffix}"
            run_command(f"head -n {k_validation} {chunk_file} >> {validation_file}")
            run_command(f"sed -i '1,{k_validation}d' {chunk_file}")

    print("All tasks completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--k_validation", type=int, default=10000)
    parser.add_argument("--sample_ratio", type=float, default=1.0)
    parser.add_argument("--key_len", type=int, default=1)

    args = parser.parse_args()

    main(args.dataset, args.data_dir, args.k_validation, args.sample_ratio, args.key_len)
