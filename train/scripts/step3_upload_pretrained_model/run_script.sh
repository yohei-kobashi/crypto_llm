#!/bin/bash

# Command line options go here
#SBATCH --time=24:00:00
#SBATCH --nodelist=slurm0-a3-ghpc-8
#SBATCH --job-name=convert_model
#SBATCH --output=convert_model.out
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --ntasks=208

source ~/miniconda3/etc/profile.d/conda.sh
conda activate .venv_train
cd ~/ucllm_nedo_dev/train/scripts/step3_upload_pretrained_model/

model_name="1.en_no_encrypted"
tokenizer="tokenizer-wikipedia-en-no_encrypted.model"
checkpoint_name="gpt_1.3B_tok300B_lr2.0e-4_min1.0e-6_w3000M_d300B_cosine_gbs384_mbs1_g8_pp1_seed1234_rebase"

checkpoint_num="2000"

bash ./convert_tokenizer_and_pretrained_model_to_huggingface_transformers.sh \
    --input_tokenizer_file /storage9/encrypted_llm/tokenizers/${tokenizer} \
    --input_model_max_length 4096 \
    --input_model_dir /storage9/encrypted_llm/models/${model_name}/checkpoint/${checkpoint_name}/global_step${checkpoint_num}/ \
    --output_tokenizer_and_model_dir /storage9/encrypted_llm/models/hf/${model_name}_${checkpoint_num}

checkpoint_num="5000"

bash ./convert_tokenizer_and_pretrained_model_to_huggingface_transformers.sh \
    --input_tokenizer_file /storage9/encrypted_llm/tokenizers/${tokenizer} \
    --input_model_max_length 4096 \
    --input_model_dir /storage9/encrypted_llm/models/${model_name}/checkpoint/${checkpoint_name}/global_step${checkpoint_num}/ \
    --output_tokenizer_and_model_dir /storage9/encrypted_llm/models/hf/${model_name}_${checkpoint_num}

checkpoint_num="10000"

bash ./convert_tokenizer_and_pretrained_model_to_huggingface_transformers.sh \
    --input_tokenizer_file /storage9/encrypted_llm/tokenizers/${tokenizer} \
    --input_model_max_length 4096 \
    --input_model_dir /storage9/encrypted_llm/models/${model_name}/checkpoint/${checkpoint_name}/global_step${checkpoint_num}/ \
    --output_tokenizer_and_model_dir /storage9/encrypted_llm/models/hf/${model_name}_${checkpoint_num}

checkpoint_num="12000"

bash ./convert_tokenizer_and_pretrained_model_to_huggingface_transformers.sh \
    --input_tokenizer_file /storage9/encrypted_llm/tokenizers/${tokenizer} \
    --input_model_max_length 4096 \
    --input_model_dir /storage9/encrypted_llm/models/${model_name}/checkpoint/${checkpoint_name}/global_step${checkpoint_num}/ \
    --output_tokenizer_and_model_dir /storage9/encrypted_llm/models/hf/${model_name}_${checkpoint_num}
