import argparse
from name import ExtractName
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.filters import SamplerFilter

parser = argparse.ArgumentParser("Filter an HF dataset and push the result to the hub")

parser.add_argument("input_dataset", type=str, help="HF dataset to filter")
parser.add_argument("job_name", type=str, help="job name", default="filter_name_temp")
parser.add_argument("--sample_ratio", type=float, help="sample ratio of the dataset", default=1.0)
parser.add_argument("--n_tasks", type=int, help="number of tasks", default=10)
parser.add_argument("--text_key", type=str, help="text column", default="text")
parser.add_argument("--seed", type=int, help="random seed", default=0)

LOCAL_PATH = "working_dir/datatrove"
LOCAL_LOGS_PATH = "logs/datatrove"

if __name__ == "__main__":
    args = parser.parse_args()
    JOB_NAME = args.job_name
    dist_executor = LocalPipelineExecutor(
        pipeline=[
            ParquetReader(args.input_dataset, glob_pattern="**/*.parquet", text_key=args.text_key),
            SamplerFilter(args.sample_ratio ,seed=args.seed),
            ExtractName(f"{LOCAL_PATH}/{JOB_NAME}"),
        ],
        tasks=args.n_tasks,
        skip_completed=False,
        logging_dir=f"{LOCAL_LOGS_PATH}/{JOB_NAME}_extract",
    )
    dist_executor.run()