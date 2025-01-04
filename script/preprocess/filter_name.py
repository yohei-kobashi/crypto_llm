import argparse
from name import FilterName
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.filters import SamplerFilter
from datatrove.pipeline.writers import JsonlWriter

parser = argparse.ArgumentParser("Filter an HF dataset and push the result to the hub")

parser.add_argument("input_dataset", type=str, help="HF dataset to filter")
parser.add_argument("job_name", type=str, help="job name", default="filter_name_temp")
parser.add_argument("--sample_ratio", type=float, help="sample ratio of the dataset", default=1.0)
parser.add_argument("--n_tasks", type=int, help="number of tasks", default=10)
parser.add_argument("--text_key", type=str, help="text column", default="text")
parser.add_argument("--seed", type=int, help="random seed", default=0)

ORG_NAME = "fumiyau"
LOCAL_PATH = "/home/uchiyama.fumiya/ucllm/cryptollm/working_dir/datatrove"
LOCAL_LOGS_PATH = "/home/uchiyama.fumiya/ucllm/cryptollm/logs/datatrove"

if __name__ == "__main__":
    args = parser.parse_args()
    JOB_NAME = args.job_name
    dist_executor = LocalPipelineExecutor(
        pipeline=[
            ParquetReader(args.input_dataset, glob_pattern="**/*.parquet", text_key=args.text_key),
            SamplerFilter(args.sample_ratio, seed=args.seed),
            FilterName(
                stat_path=f"{LOCAL_LOGS_PATH}/{JOB_NAME}_extract/stats.json",
                max_freq_to_drop=100,
                seed=args.seed,
                exclusion_writer=JsonlWriter(
                    f"{LOCAL_PATH}/{JOB_NAME}",
                    output_filename="data_dropped/${rank}.jsonl",
                    compression=None,
                ),
            ),
            JsonlWriter(
                f"{LOCAL_PATH}/{JOB_NAME}",
                output_filename="data_final/${rank}.jsonl",
                compression=None,
            ),
        ],
        tasks=args.n_tasks,
        skip_completed=False,
        logging_dir=f"{LOCAL_LOGS_PATH}/{JOB_NAME}_filter",
    )
    dist_executor.run()