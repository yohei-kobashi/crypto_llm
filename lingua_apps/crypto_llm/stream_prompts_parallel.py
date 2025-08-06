from omegaconf import OmegaConf
from dataclasses import dataclass, field
import time
import os
from pathlib import Path
import random
import json
from multiprocessing import Manager, Pool, cpu_count
from datasets import load_dataset

from lingua.args import dataclass_from_dict, dump_config
from apps.main.generate import build_tokenizer

@dataclass
class StreamGenerationArgs:
    ckpt: str = ""
    hf_source_path: str = "wikimedia/wikipedia"
    hf_source_name: str = "20231101.en"
    hf_source_split: str = "train"
    hf_source_streaming: bool = False
    max_prompt_sample_size: int = 100_000_000
    output_prompt_dir: str = "data/stream_generation_output"
    seed: int = 0
    num_workers: int = cpu_count()

def init_worker(consolidated_ckpt: str, seed: int):
    global tokenizer
    from omegaconf import OmegaConf
    from pathlib import Path
    from apps.main.generate import build_tokenizer

    ckpt_path = Path(consolidated_ckpt)
    config = OmegaConf.load(ckpt_path / "params.json")
    tokenizer = build_tokenizer(config.data.tokenizer.name, config.data.tokenizer.path)
    random.seed(seed)

def process_sample(sample):
    global tokenizer
    text = sample.get("text", "")
    tokens = tokenizer.encode(text, add_bos=False, add_eos=False)
    if len(tokens) <= 5:
        return None
    start = random.randint(0, len(tokens) - 5)
    return tokenizer.decode(tokens[start:start + 5])

def main():
    cli_cfg = OmegaConf.from_cli()
    args: StreamGenerationArgs = dataclass_from_dict(
        StreamGenerationArgs, cli_cfg, strict=False
    )
    print(args)

    os.makedirs(args.output_prompt_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(
        args.output_prompt_dir,
        f"stream_prompt_output_{ts}.jsonl"
    )

    dump_config(args, out_path[:-len(".jsonl")] + ".yaml", log_config=True)

    ds = load_dataset(
        args.hf_source_path,
        args.hf_source_name,
        split=args.hf_source_split,
        streaming=args.hf_source_streaming,
    )

    manager = Manager()
    counter = manager.Value('i', 0)
    counter_lock = manager.Lock()

    with open(out_path, "w") as fout, \
         Pool(
             processes=args.num_workers,
             initializer=init_worker,
             initargs=(args.ckpt, args.seed)
         ) as pool:
        
        while True:
            cnt = 0
            for decoded in pool.imap(process_sample, ds, chunksize=256):
                if decoded is None:
                    continue

                fout.write(json.dumps({"text": decoded}, ensure_ascii=False) + "\n")

                with counter_lock:
                    counter.value += 1
                    cnt = counter.value

                if cnt % 10000 == 0:
                    print(f"[{time.strftime('%H:%M:%S')}] Generated {cnt} samples")

            if cnt >= args.max_prompt_sample_size:
                break

    print(f"Finished: {counter.value} samples -> {out_path}")

if __name__ == "__main__":
    main()
