# load jsonl files from given directories and evaluate tokenization compression
# tokenize each line and compare the number of tokens before and after tokenization

import argparse
import os
import random
import json
from tqdm import tqdm
import sentencepiece as spm

def main(args):

    assert len(args.input_dir) == len(args.probabilities)

    sp = spm.SentencePieceProcessor()
    sp.Load(args.model)

    for input_dir, prob in zip(args.input_dir, args.probabilities):
        for filename in os.listdir(input_dir):
            if random.random() < prob and filename.endswith('.jsonl') and (not filename.endswith('.val.jsonl')):
                with open(os.path.join(input_dir, filename)) as f2:
                    print(f'loading {os.path.join(input_dir, filename)}')
                    num_samples = 0
                    num_chars = 0
                    num_tokens = 0
                    count = 0
                    for line in tqdm(f2):
                        data = json.loads(line)
                        tokens = sp.encode(data['text'])
                        num_samples += 1
                        num_chars += len(data['text'])
                        num_tokens += len(tokens)
                        count += 1

                        if count % 10000 == 0:
                            break
                    print(f'num_samples: {num_samples}, num_chars: {num_chars}, avg_chars: {num_chars / num_samples}')
                    print(f'num_tokens: {num_tokens}, avg_tokens: {num_tokens / num_samples}')
                    print(f'compression rate: {num_tokens / num_chars}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, nargs='+')
    parser.add_argument('--probabilities', type=float, nargs='+')
    parser.add_argument('--model', type=str, default='sp_model.model')

    args = parser.parse_args()
    main(args)