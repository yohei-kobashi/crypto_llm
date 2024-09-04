#!/bin/bash

#PJM -L rscgrp=share-short
#PJM -L gpu=1
#PJM -L elapse=02:00:00
#PJM -g gb20
#PJM -j

###PJM -L node=1
set -e

echo "* Loading modules"
module load gcc/8.3.1
module load cuda/11.8
module load cudnn/8.8.0  
module list

export HOME=/work/gb20/b20048
export WORK_HOME=$HOME/crypto_llm/train

# NOTE if needed, conda installation
# conda env remove -n deep -y
#mkdir -p ~/miniconda3
#wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
#bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
#rm ~/miniconda3/miniconda.sh

echo "* Creating conda environment"
source $HOME/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda create -n deep python=3.9 -y
conda activate deep

echo "* Installing pytorch"
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
cd $WORK_HOME
pip install -r requirements.txt
pip install deepspeed-kernels

echo "* Installing deepspeed with pre-build operators"
DS_BUILD_OPS=1 DS_BUILD_EVOFORMER_ATTN=0 DS_BUILD_SPARSE_ATTN=0  pip install -v deepspeed==0.12.4

echo "* Confirming deepspeed installation"
ds_report

echo "* Installing NVIDIA apex"
cd $WORK_HOME
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

echo "* Installing Megatron-DeepSpeed"
cd $WORK_HOME
git clone https://github.com/microsoft/Megatron-DeepSpeed
cd Megatron-DeepSpeed
python setup.py install

cd $WORK_HOME
pip install --no-cache-dir flash-attn==2.5.0 --no-build-isolation

echo "* Confirming packages"
pip list