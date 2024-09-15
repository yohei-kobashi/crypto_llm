
import torch
import os
import sys
sys.path.append("/home/acf16449gb/crypto_llm/train/Megatron-DeepSpeed")
sys.path.append("/home/acf16449gb/crypto_llm/train/Megatron-DeepSpeed/tools/convert_checkpoint")
from deepspeed_checkpoint import DeepSpeedCheckpoint
from pretrain_gpt import model_provider
from deepspeed_to_megatron import _create_rank_checkpoint
import sentencepiece as spm

from megatron import get_args
from megatron.initialize import initialize_megatron
from megatron.training import get_model
from megatron.text_generation_utils import generate_samples_eval

def add_text_generate_args(parser):
    """Text generation arguments."""
    group = parser.add_argument_group(title="text generation")

    group.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature."
    )
    group.add_argument(
        "--greedy", action="store_true", default=False, help="Use greedy sampling."
    )
    group.add_argument("--top_p", type=float, default=0.0, help="Top p sampling.")
    group.add_argument("--top_k", type=int, default=0, help="Top k sampling.")
    group.add_argument(
        "--out-seq-length",
        type=int,
        default=1024,
        help="Size of the output generated text.",
    )
    group.add_argument(
        "--sample-input-file",
        type=str,
        default=None,
        help="Get input from file instead of interactive mode, "
        "each line is an input.",
    )
    group.add_argument(
        "--sample-output-file",
        type=str,
        default=None,
        help="Output file got from --sample-input-file",
    )
    group.add_argument(
        "--num-samples",
        type=int,
        default=0,
        help="Number of samples to generate unconditionally, "
        "defaults to 0 and interactive conditional sampling",
    )
    group.add_argument(
        "--genfile", type=str, help="Output file when generating unconditionally"
    )
    group.add_argument(
        "--recompute",
        action="store_true",
        help="During generation recompute all attention "
        "instead of using previously computed keys/values.",
    )
    group.add_argument(
        "--context-tokens", type=str, default="DeepSpeed is the greatest"
    )
    group.add_argument(
        "--tokenize-model",
        type=str,
        default=None,
        help="sentencepiece tokenizer file"
    )
    group.add_argument("--max-tokens", type=int, default=50)

    return parser

def override_args(args, override_args, skip_keys, skip_if_specified_keys):
    for k, v in vars(override_args).items():
        if k in skip_keys:
            continue
        if k in skip_if_specified_keys and getattr(args, k) is not None:
            continue
        setattr(args, k, v)

def main():
    # initialize megatron
    initialize_megatron(
        extra_args_provider=add_text_generate_args,
        args_defaults={
            "tokenizer_type": "SentencePieceTokenizer",
            "no_load_rng": True,
            "no_load_optim": True,
        },
    )
    args = get_args()

    # # GPUの設定
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model and load checkpoint.
    ds_checkpoint = DeepSpeedCheckpoint(args.load, tp_degree=1, pp_degree=1)

    # Merge the current args with the checkpoint args.
    cp_args = ds_checkpoint.get_args()
    skip_keys = ['world_size', 'rank', 'local_rank','device_count', 'micro_batch_size','global_batch_size', 'batch_size', 'tensorboard_dir', 'deepspeed', 'deepspeed_config',
                     'data_parallel_size', 'pipeline_model_parallel_size', 'tensor_model_parallel_size', 'moe_expert_parallel_size', 'moe_token_dropping', 'load', 'rampup_batch_size', 'iteration', 'inference', 'random_ltd', 'tokenizer_model']

    skip_if_specified = ['merge_file', 'vocab_file']

    override_args(args, cp_args, skip_keys, skip_if_specified)

    # tokenizerからvocab_sizeを取得してargsを更新
    # SentencePiece モデルファイルのパス
    model_file = args.tokenizer_model

    # SentencePiece トークナイザーの読み込み
    sp = spm.SentencePieceProcessor()
    sp.Load(model_file)

    # vocab_size の取得
    vocab_size = sp.GetPieceSize()

    print(f"Vocab size: {vocab_size}")
    args.vocab_size = vocab_size + 128

    input_state_dict = _create_rank_checkpoint(ds_checkpoint, 0, 0, True)

    model = get_model(model_provider)[0]

    model.load_state_dict(input_state_dict['model'], strict=True)

    # 推論
    args.micro_batch_size = 1
    print(generate_samples_eval(model, "My father is my sister's", 50, sp.eos_id()))
    

if __name__ == "__main__":
    main()
