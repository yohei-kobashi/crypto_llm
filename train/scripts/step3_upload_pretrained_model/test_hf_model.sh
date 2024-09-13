#!/bin/bash

#$ -l rt_G.small=1
#$ -l h_rt=1:00:00
#$ -j y
#$ -cwd

export LD_LIBRARY_PATH=/home/acf16449gb/miniconda3/envs/.venv_data/lib:/home/acf16449gb/crypto_llm/train/.venv_train/lib
source /etc/profile.d/modules.sh
module load python/3.11 cuda/11.8 hpcx/2.12
source ~/crypto_llm/train/.venv_train/bin/activate

module load aws-cli s3fs-fuse
s3fs -o use_cache=/tmp -o url=https://s3.abci.ai/ ucllm-common-crawl /home/acf16449gb/llm-preprocessing-practice/ucllm-common-crawl

# python test_hf_model.py /groups/gcf51099/crypto_llm/models/hf/1.latin_wikipedia_poly_000000_1234_True/
# python test_hf_model.py /groups/gcf51099/crypto_llm/models/hf/1.latin_wikipedia_no_encryption_000000_1234_True/
python test_hf_model.py /home/acf16449gb/llm-preprocessing-practice/ucllm-common-crawl/models/hf/1.en_no_encrypted_12000/ gpt2