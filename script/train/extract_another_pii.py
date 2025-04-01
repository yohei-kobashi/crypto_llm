from name import FilterName

import argparse
from name import FilterName
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.filters import SamplerFilter
from datatrove.pipeline.writers import JsonlWriter

parser = argparse.ArgumentParser("Filter an HF dataset and push the result to the hub")

parser.add_argument("input_dataset", type=str, help="HF dataset to filter")
parser.add_argument("--job_name", type=str, help="job name", default="extract_names_from_fwe10b")
parser.add_argument("--n_tasks", type=int, help="number of tasks", default=10)
parser.add_argument("--text_key", type=str, help="text column", default="text")
parser.add_argument("--seed", type=int, help="random seed", default=1)
parser.add_argument("--drop_ratio", type=float, help="drop ratio", default=0.01)
parser.add_argument("--skip", type=int, help="skip the first n documents", default=0)
parser.add_argument("--limit", type=int, help="limit the number of documents", default=-1)
# seed is different from the original script to filter another PII

LOCAL_PATH = "working_dir/datatrove"
LOCAL_LOGS_PATH = "logs/datatrove"

if __name__ == "__main__":
    args = parser.parse_args()
    JOB_NAME = args.job_name
    dist_executor = LocalPipelineExecutor(
        pipeline=[
            JsonlReader(
                args.input_dataset,
                file_progress=True,
                doc_progress=True,
                glob_pattern="**/*.chunk.*.jsonl",
                skip=args.skip,
                limit=args.limit,
            ),
            FilterName(
                stat_path=f"{LOCAL_LOGS_PATH}/{JOB_NAME}_extract/stats.json",
                max_freq_to_drop=100,
                seed=args.seed,
                exclusion_writer=JsonlWriter(
                    f"{LOCAL_PATH}/{JOB_NAME}_b{str(args.skip)}_e{str(args.limit)}_seed{str(args.seed)}",
                    output_filename="data_dropped_${rank}.jsonl",
                    compression=None,
                ),
                drop_ratio=args.drop_ratio,
            ),
        ],
        tasks=args.n_tasks,
        skip_completed=False,
        logging_dir=f"{LOCAL_LOGS_PATH}/{JOB_NAME}_filter_b{str(args.skip)}_e{str(args.limit)}_seed{str(args.seed)}",
    )
    dist_executor.run()