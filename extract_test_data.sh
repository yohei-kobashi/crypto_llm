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
python script/preprocess/parquet_sampler.py --input-dir fineweb-edu/sample/100BT-10BT --output-dir fineweb-edu/sample/100BT-10BT-sampled2 --sampling-rate 1/9 --n_tasks 20
python script/preprocess/extract_name.py fineweb-edu/sample/100BT-10BT-sampled2 extract_names_from_fwe100b-10b_2 --n_tasks 20
python script/preprocess/filter_name.py fineweb-edu/sample/100BT-10BT-sampled2 extract_names_from_fwe100b-10b_2 --n_tasks 20
