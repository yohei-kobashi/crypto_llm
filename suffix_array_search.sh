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
# python script/preprocess/suffix_array_search.py build --parquet-dir fineweb-edu/sample/10BT-sample --out-dir fineweb-edu/index_10BT --workers 14
python script/preprocess/suffix_array_search.py build --index-dir fineweb-edu/index_10BT --input output/e/stream_generation_output_20250702_033516.jsonl --window 35
python script/preprocess/suffix_array_search.py build --index-dir fineweb-edu/index_10BT --input output/e/stream_generation_output_20250702_033516.jsonl --window 30
python script/preprocess/suffix_array_search.py build --index-dir fineweb-edu/index_10BT --input output/e/stream_generation_output_20250702_033516.jsonl --window 25
python script/preprocess/suffix_array_search.py build --index-dir fineweb-edu/index_10BT --input output/e/stream_generation_output_20250702_033516.jsonl --window 20
python script/preprocess/suffix_array_search.py build --index-dir fineweb-edu/index_10BT --input output/e/stream_generation_output_20250702_033516.jsonl --window 15
python script/preprocess/suffix_array_search.py build --index-dir fineweb-edu/index_10BT --input output/e/stream_generation_output_20250702_033516.jsonl --window 10