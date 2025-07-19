#!/bin/bash -l
mkdir output/e
mkdir output/a_1
mkdir output/a_10
mkdir output/a_100
mkdir output/b
mkdir output/c
head -n 100000 output/stream_generation_output_20250702_033516.jsonl > output/e/100000.jsonl
head -n 100000 output/stream_generation_output_20250702_195255.jsonl > output/a_1/100000.jsonl
head -n 100000 output/stream_generation_output_20250702_195343.jsonl > output/a_10/100000.jsonl
head -n 100000 output/stream_generation_output_20250702_195501.jsonl > output/a_100/100000.jsonl
head -n 100000 output/stream_generation_output_20250702_195616.jsonl > output/b/100000.jsonl
head -n 100000 output/stream_generation_output_20250702_195659.jsonl > output/c/100000.jsonl
