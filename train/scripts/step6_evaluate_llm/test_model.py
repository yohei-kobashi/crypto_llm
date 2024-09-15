import torch
from megatron.model import GPTModel
from megatron.initialize import initialize_megatron
from megatron.training import get_model
from megatron.checkpointing import load_checkpoint
from megatron.text_generation_utils import generate_samples_input_from_file
from megatron import print_rank_0

from megatron.arguments import core_transformer_config_from_args
from megatron import get_args

def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    args = get_args()
    config = core_transformer_config_from_args(args)

    print_rank_0('building GPT model ...')
    model = GPTModel(config=config, num_tokentypes=0, parallel_output=False,
                     pre_process=pre_process, post_process=post_process,
                     return_moe_loss=False) # we need to set "return_moe_loss" for the inference_mode
    return model

def add_text_generate_args(parser):
    """Text generation arguments."""
    group = parser.add_argument_group(title='text generation')

    group.add_argument("--temperature", type=float, default=1.0,
                       help='Sampling temperature.')
    group.add_argument("--greedy", action='store_true', default=False,
                       help='Use greedy sampling.')
    group.add_argument("--top_p", type=float, default=0.0,
                       help='Top p sampling.')
    group.add_argument("--top_k", type=int, default=0,
                       help='Top k sampling.')
    group.add_argument("--out-seq-length", type=int, default=1024,
                       help='Size of the output generated text.')
    group.add_argument("--sample-input-file", type=str, default=None,
                       help='Get input from file instead of interactive mode, '
                       'each line is an input.')
    group.add_argument("--sample-output-file", type=str, default=None,
                       help='Output file got from --sample-input-file')
    group.add_argument("--num-samples", type=int, default=0,
                       help='Number of samples to generate unconditionally, '
                       'defaults to 0 and interactive conditional sampling')
    group.add_argument("--genfile", type=str,
                       help='Output file when generating unconditionally')
    group.add_argument("--recompute", action='store_true',
                       help='During generation recompute all attention '
                       'instead of using previously computed keys/values.')

    return parser

def main():
    """Main program."""
    initialize_megatron(extra_args_provider=add_text_generate_args,
                        args_defaults={'tokenizer_type': 'SentencePieceTokenizer',
                                       'no_load_rng': True,
                                       'no_load_optim': True})

    args = get_args()

    # # GPUの設定
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model and load checkpoint.
    model = get_model(model_provider)
    
    # # モデルを直接GPUに乗せる
    # model.to(device)

    if args.load is not None:
        _ = load_checkpoint(model, None, None, strict=False)

    assert len(model) == 1, "Above condition should have caught this"
    model = model[0]

    # 推論
    model.eval()

    args.micro_batch_size = 1
    generate_samples_input_from_file(model)
    

if __name__ == "__main__":

    main()
