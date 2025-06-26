#!/bin/bash -l

#------ qsub option --------#
#PBS -q small-c
#PBS -l select=1
#PBS -l walltime=24:00:00
#PBS -W group_list=go25
#PBS -j oe

#------- Program execution -------#
module purge
module load cmake
module load gcc

cd crypto_llm
source env_crypto_llm/bin/activate
python script/preprocess/extract_name.py extract_names_from_fwe100b-10b