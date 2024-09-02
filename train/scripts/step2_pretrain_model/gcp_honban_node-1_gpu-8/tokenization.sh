#!/bin/bash

set -e
echo ""

# Stores the directory paths as variables.
ucllm_nedo_dev_train_dir="${HOME}/ucllm_nedo_prod/train"
megatron_deepspeed_dir="${ucllm_nedo_dev_train_dir}/Megatron-DeepSpeed"

# Initializes the arguments.
input_tokenizer_file=""
data_base_dir="/tmp"
train_data_file="${data_base_dir}/datasets/wikipedia/20240301/ja/0.jsonl"
data_name="wikipedia_encryption"
workers=20

# Parses the arguments.
while [[ ${#} -gt 0 ]]; do
    case ${1} in
        # Shifts twice for option that takes an argument.
        --input_tokenizer_file) input_tokenizer_file=${2}; shift ;;
        --data_base_dir) data_base_dir=${2}; shift ;;
        --train_data_file) train_data_file=${2}; shift ;;
        --data_name) data_name=${2}; shift ;;
        --workers) workers=${2}; shift ;;
        *) echo "Unknown parameter passed: ${1}"; exit 1 ;;
    esac
    # Shifts once per loop to move to the next key/value.
    shift
done

data_path="${data_base_dir}/${data_name}_text_document"

# Checks the required arguments.
if [[ -z ${input_tokenizer_file} ]]; then
    echo "Error: Missing required arguments."
    echo "Usage: ${0} --input_tokenizer_file <input_tokenizer_file>"
    exit 1
fi

# Prints the arguments.
echo "input_tokenizer_file = ${input_tokenizer_file}"
if [ ! -f "${data_path}.bin" ] || [ ! -f "${data_path}.idx" ]; then
    echo "Either ${data_path}.bin or ${data_path}.idx doesn't exist yet, so preprocess the data."
    python ${megatron_deepspeed_dir}/tools/preprocess_data.py \
        --tokenizer-type SentencePieceTokenizer \
        --tokenizer-model ${input_tokenizer_file} \
        --input ${train_data_file} \
        --output-prefix ${data_base_dir}/${data_name} \
        --dataset-impl mmap \
        --workers ${workers} \
        --append-eod
else
    echo "Both ${data_path}.bin and ${data_path}.idx already exist."
fi