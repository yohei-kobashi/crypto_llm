import argparse
import os
import time
import subprocess
import requests
from huggingface_hub import snapshot_download

def download_dataset(repo_id, local_dir, allow_patterns):
    print(f"Downloading dataset from {repo_id}...")
    max_retries = 5
    retry_delay = 10  # seconds
    for attempt in range(max_retries):
        try:
            snapshot_download(
                repo_id,
                repo_type="dataset",
                local_dir=local_dir,
                allow_patterns=allow_patterns,
                resume_download=True,
                max_workers=16, # Don't hesitate to increase this number to lower the download time
            )
            break
        except requests.exceptions.ReadTimeout:
            if attempt < max_retries - 1:
                print(f"Timeout occurred. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                raise
    print(f"Dataset downloaded to {local_dir}")


def main(data_dir):
    # Configuration
    repo_id = "HuggingFaceFW/fineweb-edu"
    src_dir = f"{data_dir}/fineweb_edu_10bt"
    allow_patterns = "sample/10BT/*"

    # Download dataset
    download_dataset(repo_id, src_dir, allow_patterns)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download dataset from Hugging Face")
    parser.add_argument("data_dir", type=str, help="Directory to save the dataset")
    args = parser.parse_args()
    main(args.data_dir)