import os
import sys
import glob
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import argparse

def process_file(file_name, split_rate=0.9, seed=None):
    original_seed = random.getstate()
    if seed is not None:
        random.seed(seed)
    encrypted_lines = []
    no_encrypted_lines = []
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            data = json.loads(line)
            if data['text']:
                if random.random() < split_rate:
                    encrypted_lines.append(json.dumps(data, ensure_ascii=False))
                else:
                    no_encrypted_lines.append(json.dumps(data, ensure_ascii=False))
    random.setstate(original_seed)

    return encrypted_lines, no_encrypted_lines

def main(args):
    # 引数として受け取りたい値一覧
    lang = args.lang
    split_rate = args.split_rate
    input_dir = args.input_dir
    output_dir = args.output_dir
    seed = args.seed
    
    os.makedirs(output_dir, exist_ok=True)

    if os.path.isdir(input_dir):
        wiki_files = glob.glob(os.path.join(input_dir, '*.jsonl'))
    else:
        wiki_files = [input_dir]
    
    lock = threading.Lock()  # ロックを作成

    with ThreadPoolExecutor() as executor:
        future_to_file = {executor.submit(process_file, file_name, split_rate, seed): file_name for file_name in wiki_files}
        
        with open(os.path.join(output_dir, 'wikipedia_{}_no_encryption_000000_{}.jsonl'.format(lang, seed)), 'w', encoding='utf-8') as out_file, \
                open(os.path.join(output_dir, 'wikipedia_{}_no_encryption_000000_{}_no_encryption.jsonl'.format(lang, seed)), 'w', encoding='utf-8') as out_file2:
            for future in as_completed(future_to_file):
                file_name = future_to_file[future]
                try:
                    encrypted_lines, no_encrypted_lines = future.result()
                    with lock:
                        for encrypted_line in encrypted_lines:
                            out_file.write(encrypted_line + '\n')
                        for line in no_encrypted_lines:
                            out_file2.write(line + '\n')
                except Exception as exc:
                    print(f'{file_name} generated an exception: {exc}')
    print('finished!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--lang', type=str, default='latin', help='Language of the input data')
    parser.add_argument('--split_rate', type=float, default=0.9, help='Rate of encryption')
    parser.add_argument('--input_dir', type=str, default='tmp/output/datasets/wikipedia/20240501/en/', help='Input directory')
    parser.add_argument('--output_dir', type=str, default='tmp/output/encrypted_data/', help='Output directory')
    parser.add_argument('--seed', type=int, default=None, help='Seed for random number generator')

    args = parser.parse_args()
    main(args)