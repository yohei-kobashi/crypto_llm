# load jsonl files from given directories and concatenate them into a single file
# output: raw_text.txt
#
# python make_raw_text.py \
# --input_dir data1/ data2/ data3/ \
# --probabilities 0.5 0.3 0.2 \
# --output working_dir/tokenizer/raw_text.txt

import argparse
import os
import random
import json
from tqdm import tqdm
def main(args):
    num_samples = 0
    num_chars = 0

    assert len(args.input_dir) == len(args.probabilities)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, 'w') as f:
        for input_dir, prob in zip(args.input_dir, args.probabilities):
            for filename in os.listdir(input_dir):
                if random.random() < prob and filename.endswith('.jsonl') and (not filename.endswith('.val.jsonl')):
                    with open(os.path.join(input_dir, filename)) as f2:
                        print(f'loading {os.path.join(input_dir, filename)}')
                        for line in tqdm(f2):
                            data = json.loads(line)
                            f.write(data['text'] + '\n')
                            num_samples += 1
                            num_chars += len(data['text'])

    print(f'num_samples: {num_samples}, num_chars: {num_chars}, avg_chars: {num_chars / num_samples}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, nargs='+')
    parser.add_argument('--probabilities', type=float, nargs='+')
    parser.add_argument('--output', type=str)

    args = parser.parse_args()
    main(args)
