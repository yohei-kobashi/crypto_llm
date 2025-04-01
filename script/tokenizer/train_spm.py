import argparse
import sentencepiece as spm

UNK_TOKEN = "<unk>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
PAD_TOKEN = "<pad>"
CLS_TOKEN = "<CLS>"
SEP_TOKEN = "<SEP>"
EOD_TOKEN = "<EOD>"
MASK_TOKEN = "<MASK>"
NEWLINE_TOKEN = "\n"

def main(args):
    user_defined_symbols=[
            BOS_TOKEN,
            EOS_TOKEN,
            PAD_TOKEN,
            CLS_TOKEN,
            SEP_TOKEN,
            EOD_TOKEN,
            MASK_TOKEN,
            NEWLINE_TOKEN,
        ],  # Note: `NEWLINE_TOKEN` is needed in `user_defined_symbols`.
    
    spm.SentencePieceTrainer.train(
        input=args.input, 
        model_prefix=args.model_prefix, 
        vocab_size=args.vocab_size, 
        model_type=args.model_type,
        byte_fallback=args.byte_fallback,
        split_digits=args.split_digits,
        allow_whitespace_only_pieces=args.allow_whitespace_only_pieces,
        remove_extra_whitespaces=args.remove_extra_whitespaces,
        input_sentence_size=args.input_sentence_size,
        shuffle_input_sentence=args.shuffle_input_sentence,
        train_extremely_large_corpus=args.train_extremely_large_corpus,
        user_defined_symbols=user_defined_symbols,
        )
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('--model_prefix', type=str, default='sp_model')
    parser.add_argument('--vocab_size', type=int, default=50000)
    parser.add_argument('--input_sentence_size', type=int, default=1000000)
    parser.add_argument('--shuffle_input_sentence', action='store_true')
    parser.add_argument('--train_extremely_large_corpus', action='store_true')
    parser.add_argument('--model_type', type=str, default='bpe')
    parser.add_argument('--byte_fallback', action='store_true')
    parser.add_argument('--split_digits', action='store_true')
    parser.add_argument('--allow_whitespace_only_pieces', action='store_true')
    parser.add_argument('--remove_extra_whitespaces', action='store_true')
    

    args = parser.parse_args()
    main(args)
# spm.SentencePieceTrainer.train(
#     input='test/botchan.txt', 
#     model_prefix='m', 
#     vocab_size=1000, 
#     user_defined_symbols=['foo', 'bar']
#     )
# sentencepiece_trainer.cc(73) LOG(INFO) Starts training with : 
# trainer_spec {
#   input: test/botchan.txt
#   .. snip
# unigram_model_trainer.cc(500) LOG(INFO) EM sub_iter=1 size=1188 obj=10.2839 num_tokens=32182 num_tokens/piece=27.0892
# unigram_model_trainer.cc(500) LOG(INFO) EM sub_iter=0 size=1100 obj=10.4269 num_tokens=33001 num_tokens/piece=30.0009
# unigram_model_trainer.cc(500) LOG(INFO) EM sub_iter=1 size=1100 obj=10.4069 num_tokens=33002 num_tokens/piece=30.0018
# trainer_interface.cc(595) LOG(INFO) Saving model: m.model
# trainer_interface.cc(619) LOG(INFO) Saving vocabs: m.vocab