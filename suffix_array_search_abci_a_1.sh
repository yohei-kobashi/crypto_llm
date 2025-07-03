#!/bin/bash -l
#PBS -q rt_HC
#PBS -l select=1
#PBS -l walltime=12:00:00
#PBS -P gcf51099

module load python/3.12

cd crypto_llm
source env_crypto_llm/bin/activate
# python script/preprocess/suffix_array_search.py build --parquet-dir fineweb-edu/sample/10BT-sample --out-dir fineweb-edu/index_10BT --workers 14
python script/preprocess/suffix_array_search.py query --index-dir index_10BT --input output/a_1 --window 35
