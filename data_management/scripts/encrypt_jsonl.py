import os
import sys
import glob
import json
import random
from concurrent.futures import ThreadPoolExecutor
import threading
import argparse
from functools import partial

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

import encrypt

def process_line(line, encryptor):
    data = json.loads(line)
    data['text'] = encryptor.encrypt(data['text'])

    return data

def main(args):
    # 引数として受け取りたい値一覧
    encryption_type = args.encryption_type
    lang = args.lang
    input_file = args.input_file
    output_dir = args.output_dir
    key_length = args.key_length
    seed = args.seed
    reuse_key = args.reuse_key
    reuse_nonce = args.reuse_nonce
    
    os.makedirs(output_dir, exist_ok=True)

    # 暗号化して保存
    encryptor = encrypt.Encryptor(seed=seed)
    
    # 暗号化
    print('type:', encryption_type)

    if encryption_type == 'poly':
        print('key length:', key_length)
        encryptor.poly(key_length, reuse_key=reuse_key)
    else:
        encryptor.chacha20(reuse_key=reuse_key, reuse_nonce=reuse_nonce)
    sys.stdout.flush()

    with open(input_file, 'r', encoding='utf-8') as file, \
            open(os.path.join(output_dir, 'wikipedia_{}_{}_{:06d}_{}_{}.jsonl'.format(lang, encryption_type, key_length, seed, reuse_key)), 'w', encoding='utf-8') as out_file:
        with ThreadPoolExecutor() as executor:
            # 各行を並列に処理
            process_line_with_encryptor = partial(process_line, encryptor=encryptor)            
            for result in executor.map(process_line_with_encryptor, file):
                if result['text']:
                    out_file.write(json.dumps(result, ensure_ascii=False) + '\n')
                
    print('finished!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--lang', type=str, default='latin', help='Language of the input data')
    parser.add_argument('--encryption_type', type=str, default='poly', help='Type of encryption')
    parser.add_argument('--input_file', type=str, default='tmp/output/encrypted_data/wikipedia_latin_no_encryption_000000_1234_True.jsonl', help='Input directory')
    parser.add_argument('--output_dir', type=str, default='tmp/output/encrypted_data/', help='Output directory')
    parser.add_argument('--key_length', type=int, default=0, help='Length of the encryption key')
    parser.add_argument('--seed', type=int, default=None, help='Seed for random number generator')
    parser.add_argument('--reuse_key', type=bool, default=True, help='Reuse the same key')
    parser.add_argument('--reuse_nonce', type=bool, default=True, help='Reuse the same nonce')

    args = parser.parse_args()
    main(args)