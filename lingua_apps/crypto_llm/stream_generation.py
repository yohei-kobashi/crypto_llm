from omegaconf import OmegaConf
from dataclasses import dataclass, field
import time
import os
import json
from typing import List, Optional

from lingua.args import dataclass_from_dict, dump_config
from apps.main.generate import (
    PackedCausalTransformerGeneratorArgs,
    PackedCausalTransformerGenerator,
    load_consolidated_model_and_tokenizer,
)

@dataclass
class StreamGenerationArgs:
    ckpt: str = ""  # Path to the consolidated model checkpoint
    gen_arg: PackedCausalTransformerGeneratorArgs = field(
        default_factory=PackedCausalTransformerGeneratorArgs
    )
    input_jsonl_path: Optional[str] = None
    sample_token_size: int = 1_000_000_000
    output_jsonl_dir: str = "data/stream_generation_lmoutput"
    batch_size: int = 64



def main():
    # Load CLI arguments (overrides) and combine with a YAML config
    cfg = OmegaConf.from_cli()
    stream_gen_cfg = dataclass_from_dict(
        StreamGenerationArgs, cfg, strict=False
    )
    gen_cfg = stream_gen_cfg.gen_arg
    print(cfg)

    model, tokenizer, _ = load_consolidated_model_and_tokenizer(cfg.ckpt)

    generator = PackedCausalTransformerGenerator(gen_cfg, model, tokenizer)

    # output into jsonl file
    if not os.path.exists(stream_gen_cfg.output_jsonl_dir):
        os.makedirs(stream_gen_cfg.output_jsonl_dir)

    time_str = time.strftime("%Y%m%d_%H%M%S")
    output_file_path = os.path.join(
        stream_gen_cfg.output_jsonl_dir,
        f"stream_generation_output_{time_str}.jsonl"
    )
    print(f"Output will be saved to {output_file_path}")

    dump_config(stream_gen_cfg, output_file_path[:-len(".jsonl")] + ".yaml", log_config=True)

    token_count = 0
    count = 0
    with open(output_file_path, "w") as output_file:
        with open(stream_gen_cfg.input_jsonl_path, "r") as input_file:
            prompt = []
            for line in input_file:
                sample = json.loads(line)
                text = sample.get("text", "")
                prompt.append(text)
                count += 1
                if count % stream_gen_cfg.batch_size == 0:
                    generation, _, _ = generator.generate(prompt)
                    for gen in generation:
                        output_file.write(json.dumps({"text": gen}) + "\n")
                        token_count += len(tokenizer.encode(gen, False, False))
                    prompt = []
                    print(f"Processed {count} samples, current token count: {token_count} of {stream_gen_cfg.sample_token_size}")
                if token_count >= stream_gen_cfg.sample_token_size:
                    break
    
    print(f"Finished processing {count} samples, total token count: {token_count}")


if __name__ == "__main__":
    main()