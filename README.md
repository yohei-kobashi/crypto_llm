# Crypto-LLM
This repository contains the code used for the experiment "Crypto-LLM: Two-Stage Language Model Pre-training with Ciphered and Natural Language Data."
It is a fork of the following repository, with added encryption processes and removed unnecessary processes:  
https://github.com/matsuolab/ucllm_nedo_prod  

The environment is assumed to be Ubuntu 20.04 or later. Please make appropriate adjustments if using a different environment.

1. Environment Setup for Preprocessing
```sh
$ cd ~/

# Create a directory for the conda installation.
$ mkdir -p ~/miniconda3/ && cd ~/miniconda3/

# Install conda.
$ wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.10.0-1-Linux-x86_64.sh && bash Miniconda3-py310_23.10.0-1-Linux-x86_64.sh -b -u -p ~/miniconda3/

# Activate the installed conda.
$ source ~/miniconda3/etc/profile.d/conda.sh

# Create a Python virtual environment.
$ conda create --name .venv_data python=3.11.7 -y

# Activate the created Python virtual environment.
$ conda activate .venv_data

# Move to the working directory for data acquisition and processing
$ cd ~/crypto_llm/data_management

# Install Python libraries.
$ ./bin/setup
```

## 2. Data Download
Obtain Wikipedia dump data (Note: 20240520 used in the paper is currently unavailable for download)
```sh
$ cd ~/crypto_llm/data_management
$ python -m preprocessing.download_dataset --dataset=wikipedia --split=20240501 --output_base=tmp/output
```

## 3. Create Training Data
Randomly allocate 90% as encrypted data and 10% as unencrypted data, then encrypt it.
```sh
$ python scripts/encrypt_wikipedia.py --input_dir tmp/output/datasets/wikipedia/20240501/en/
```
The following two files will be created:  
90% encrypted data:
~/crypto_llm/data_management/tmp/output/encrypted_data/wikipedia_latin_poly_000000_None_True.jsonl  

10% unencrypted data:  
~/crypto_llm/data_management/tmp/output/encrypted_data/wikipedia_latin_poly_000000_None_True_no_encryption.jsonl  

## 4. Setup Training Environment
```sh
$ cd ~/crypto_llm/train/

$ conda deactivate

# Create a Python virtual environment.
$ conda create --name .venv_train python=3.9 -y

# Configure the virtual environment to automatically edit the `$LD_LIBRARY_PATH` when activated.
$ mkdir -p ~/miniconda3/envs/.venv_train/etc/conda/activate.d && \
    echo 'export ORIGINAL_LD_LIBRARY_PATH=$LD_LIBRARY_PATH' > ~/miniconda3/envs/.venv_train/etc/conda/activate.d/edit_environment_variable.sh && \
    echo 'export LD_LIBRARY_PATH="$HOME/miniconda3/envs/.venv_train/lib:$LD_LIBRARY_PATH"' >> ~/miniconda3/envs/.venv_train/etc/conda/activate.d/edit_environment_variable.sh && \
    chmod +x ~/miniconda3/envs/.venv_train/etc/conda/activate.d/edit_environment_variable.sh

# Configure the virtual environment to restore the `$LD_LIBRARY_PATH` when deactivated.
$ mkdir -p ~/miniconda3/envs/.venv_train/etc/conda/deactivate.d && \
    echo 'export LD_LIBRARY_PATH=$ORIGINAL_LD_LIBRARY_PATH' > ~/miniconda3/envs/.venv_train/etc/conda/deactivate.d/rollback_environment_variable.sh && \
    echo 'unset ORIGINAL_LD_LIBRARY_PATH' >> ~/miniconda3/envs/.venv_train/etc/conda/deactivate.d/rollback_environment_variable.sh && \
    chmod +x ~/miniconda3/envs/.venv_train/etc/conda/deactivate.d/rollback_environment_variable.sh

# Activate the created Python virtual environment.
$ conda activate .venv_train

# Install cuda-11.8.0.
(.venv_train) $ conda install nvidia/label/cuda-11.8.0::cuda-toolkit -y

# Install PyTorch with the specified version.
(.venv_train) $ conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Upgrade pip.
(.venv_train) $ pip install --U pip

# Install various packages from requirements.txt after installing the specified version of PyTorch.
(.venv_train) $ pip install -r requirements.txt

# Install DeepSpeed dependencies.
(.venv_train) $ pip install deepspeed-kernels

# Install DeepSpeed with the specified version, pre-building the "ops" extensions.
# https://www.deepspeed.ai/tutorials/advanced-install/#pre-install-deepspeed-ops
# Note: This may take some time.
(.venv_train) $ DS_BUILD_OPS=1 DS_BUILD_EVOFORMER_ATTN=0 DS_BUILD_SPARSE_ATTN=0 pip install deepspeed==0.12.4

# Verify that the DeepSpeed "ops" extensions are correctly installed.
(.venv_train) $ ds_report

# Clone the Megatron-DeepSpeed repository.
(.venv_train) $ git clone https://github.com/matsuolab/Megatron-DeepSpeed.git

# Checkout the specified tag to avoid errors on the main branch.
(.venv_train) $ cd ~/crypto_llm/train/Megatron-DeepSpeed/ && git fetch origin && git checkout refs/tags/ucllm_nedo_v20240415.1.0

# Install Megatron-DeepSpeed.
(.venv_train) $ cd ~/crypto_llm/train/Megatron-DeepSpeed/ && python setup.py install

# Clone the apex repository.
(.venv_train) $ git clone https://github.com/NVIDIA/apex.git

# Checkout the specified tag to avoid errors on the main branch.
(.venv_train) $ cd ~/crypto_llm/train/apex/ && git fetch origin && git checkout refs/tags/23.08

# Install apex.
# Note: This may take some time.
(.venv_train) $ cd ~/crypto_llm/train/apex/ && pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

(.venv_train) $ cd ~/crypto_llm/train/

# Reinstall ninja needed for Flash Attention 2 as a precaution.
(.venv_train) $ pip uninstall ninja -y && pip install ninja==1.11.1

# Install Flash Attention 2.
(.venv_train) $ pip install flash-attn==2.5.0 --no-build-isolation
```

## 5. Train the Tokenizer
```sh
(.venv_train) $ cd ~/crypto_llm/train/scripts/step1_train_tokenizer/

# Execute the training script.
(.venv_train) $ python ./train_sentencepiece_tokenizer.py \
    --input ../../../data_management/tmp/output/encrypted_data/wikipedia_latin_poly_000000_None_True.jsonl,../../../data_management/tmp/output/encrypted_data/wikipedia_latin_poly_000000_None_True_no_encryption.jsonl \
    --model_prefix ${YOUR_TOKENIZER_NAME} \
    --vocab_size 32000 \
    --input_sentence_size 3000000 \
    --shuffle_input_sentence True \
    --num_threads 16
```
## 6. Pre-train the Model
```sh
(.venv_train) $ cd ~/crypto_llm/train/scripts/step2_pretrain_model/

# Log in to W&B.
# https://wandb.ai/settings --> Danger Zone --> API keys --> Copy and paste the API key.
(.venv_train) $ wandb login

# Verify that you are logged into W&B.
(.venv_train) $ cat ~/.netrc

# Execute the pre-training script for encrypted data (the execution code may need to be modified according to your environment).
(.venv_train) $ bash ./gcp_honban_node-1_gpu-8/zero-0_dp-8_pp-1_tp-1_precision-fp32_flashattn2-on.sh \
    --input_tokenizer_file ../step1_train_tokenizer/${YOUR_TOKENIZER_NAME}.model \
    --output_model_dir ${OUTPUT_MODEL_DIR} \
    --save_interval 500 \
    --train_iters 9000 \
    --wandb_entity ${YOUR_WANDB_ACCOUNT} \
    --wandb_project encrypted_data_LLM \
    --data_base_dir ${OUTPUT_TOKENIZED_DATA_DIR} \
    --train_data_file ../../../data_management/tmp/output/encrypted_data/wikipedia_latin_poly_000000_None_True.jsonl \
    --data_name wikipedia_en_encryption

# Continue pre-training with unencrypted data.
(.venv_train) $ bash ./gcp_honban_node-1_gpu-8/zero-0_dp-8_pp-1_tp-1_precision-fp32_flashattn2-on.sh \
    --input_tokenizer_file ../step1_train_tokenizer/${YOUR_TOKENIZER_NAME}.model \
    --output_model_dir ${OUTPUT_MODEL_DIR} \
    --save_interval 500 \
    --train_iters 12000 \
    --wandb_entity ${YOUR_WANDB_ACCOUNT} \
    --wandb_project encrypted_data_LLM \
    --data_base_dir ${OUTPUT_TOKENIZED_DATA_DIR} \
    --train_data_file ../../../data_management/tmp/output/encrypted_data/wikipedia_latin_poly_000000_None_True_no_encryption.jsonl \
    --data_name wikipedia_en_no_encryption
```

## 7. Convert Model to Huggingface Format
```sh
(.venv_train) $ cd ~/crypto_llm/train/scripts/step3_upload_pretrained_model/
(.venv_train) $ bash ./convert_tokenizer_and_pretrained_model_to_huggingface_transformers.sh \
    --input_tokenizer_file ../step1_train_tokenizer/${YOUR_TOKENIZER_NAME}.model \
    --input_model_max_length 4096 \
    --input_model_dir ${MODEL_CHECKPOINT_DIR} \
    --output_tokenizer_and_model_dir ${OUTPUT_HF_MODEL_DIR}
```

## 8. Model Evaluation
The code for calculating perplexity was adapted from the Hugging Face documentation. For more details, you can visit the original Hugging Face page [here](https://huggingface.co/docs/transformers/perplexity).
```sh
(.venv_train) $ cd ~/crypto_llm/train/scripts/step6_evaluate_perplexity/
(.venv_train) $ python evaluate_perplexity.py ${OUTPUT_HF_MODEL_DIR}
```