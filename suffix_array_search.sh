#!/bin/bash -l

#------ qsub option --------#
#PBS -q regular-c
#PBS -l select=1
#PBS -l walltime=48:00:00
#PBS -W group_list=gj26
#PBS -j oe

#------- Program execution -------#
module purge
module load cmake
module load gcc

cd crypto_llm
source env_crypto_llm/bin/activate
python script/preprocess/suffix_array_search.py build --corpus fineweb-edu/sample/100BT-10BT --spm_model tokenizers/plain.model --outdir fineweb-edu/index_100BT-10BT --shard_size 64000000 --workers 32
