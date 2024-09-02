#!/bin/bash

#$ -l rt_F=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -cwd

export LD_LIBRARY_PATH=/home/acf16449gb/miniconda3/envs/.venv_data/lib:/home/acf16449gb/crypto_llm/train/.venv_train/lib
source /etc/profile.d/modules.sh
module load python/3.11 cuda/11.8 hpcx/2.12
source ~/crypto_llm/train/.venv_train/bin/activate

set -e
echo ""

# Stores the directory paths as variables.
crypto_llm_train_dir="${HOME}/crypto_llm/train"
megatron_deepspeed_dir="${crypto_llm_train_dir}/Megatron-DeepSpeed"
echo "crypto_llm_train_dir = ${crypto_llm_train_dir}"
echo "megatron_deepspeed_dir = ${megatron_deepspeed_dir}"
echo ""

# Initializes the arguments.
input_tokenizer_file=""
input_model_max_length=""
input_model_dir=""
output_tokenizer_and_model_dir=""
activation_function="gelu"

# Parses the arguments.
while [[ ${#} -gt 0 ]]; do
    case ${1} in
        # Shifts twice for option that takes an argument.
        --input_tokenizer_file) input_tokenizer_file=${2}; shift ;;
        --input_model_max_length) input_model_max_length=${2}; shift ;;
        --input_model_dir) input_model_dir=${2}; shift ;;
        --output_tokenizer_and_model_dir) output_tokenizer_and_model_dir=${2}; shift ;;
        --activation_function) activation_function=${2}; shift ;;
        *) echo "Unknown parameter passed: ${1}"; exit 1 ;;
    esac
    # Shifts once per loop to move to the next key/value.
    shift
done

# Checks the required arguments.
if [[ -z ${input_tokenizer_file} ]] || [[ -z ${input_model_max_length} ]] || [[ -z ${input_model_dir} ]] || [[ -z ${output_tokenizer_and_model_dir} ]]; then
    echo "Error: Missing required arguments."
    echo "Usage: ${0} --input_tokenizer_file <input_tokenizer_file> --input_model_max_length <input_model_max_length> --input_model_dir <input_model_dir> --output_tokenizer_and_model_dir <output_tokenizer_and_model_dir>"
    exit 1
fi

# Prints the arguments.
echo "input_tokenizer_file = ${input_tokenizer_file}"
echo "input_model_max_length = ${input_model_max_length}"
echo "input_model_dir = ${input_model_dir}"
echo "output_tokenizer_and_model_dir = ${output_tokenizer_and_model_dir}"
echo ""

mkdir -p ${output_tokenizer_and_model_dir}

# Converts the tokenizer from SentencePiece format to HuggingFace Transformers format.
python ${crypto_llm_train_dir}/scripts/step3_upload_pretrained_model/convert_tokenizer_from_sentencepiece_to_huggingface_transformers.py \
    --input_tokenizer_file ${input_tokenizer_file} \
    --input_model_max_length ${input_model_max_length} \
    --output_tokenizer_dir ${output_tokenizer_and_model_dir}

# Converts the pretrained model from Megatron-DeepSpeed format to HuggingFace Transformers format.
python ${megatron_deepspeed_dir}/tools/convert_checkpoint/deepspeed_to_transformers.py \
    --input_folder ${input_model_dir} \
    --output_folder ${output_tokenizer_and_model_dir} \
    --activation_function ${activation_function}

echo ""
echo "Finished converting the tokenizer and the pretrained model to HuggingFace Transformers format."
echo ""
