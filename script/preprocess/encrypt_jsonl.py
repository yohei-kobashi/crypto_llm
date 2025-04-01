import os
import sys
import json
from concurrent.futures import ProcessPoolExecutor
import argparse
from functools import partial
import re

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

import encrypt_alpha as encrypt

def process_line(line, lang, encryption_type, key_length, seed, reuse_key, reuse_nonce):
    # Create an encryption instance within each worker
    encryptor = encrypt.Encryptor(seed=seed)
    if encryption_type == 'poly':
        encryptor.poly(key_length, lang=lang, reuse_key=reuse_key)
    else:
        encryptor.chacha20(reuse_key=reuse_key, reuse_nonce=reuse_nonce)
    
    data = json.loads(line)
    data['text'] = encryptor.encrypt(data['text'])
    return data

def main(args):
    encryption_type = args.encryption_type
    lang = args.lang
    input_file = args.input_file
    output_dir = args.output_dir
    key_length = args.key_length
    seed = args.seed
    reuse_key = args.reuse_key
    reuse_nonce = args.reuse_nonce
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing file: {input_file}")
    print('key length:', key_length)
    sys.stdout.flush()

    input_filename = input_file
    output_filename = os.path.join(
        output_dir,
        'encrypted_{}_{}_{}_{:06d}_{}_{}.jsonl'.format(
            re.sub(r".*/", "", input_file).replace(".jsonl", ""), lang, encryption_type, key_length, seed, reuse_key
        )
    )
    with open(input_filename, 'r', encoding='utf-8') as file, \
            open(output_filename, 'w', encoding='utf-8') as out_file:
        # Using ProcessPoolExecutor allows avoiding the impact of the GIL (Global Interpreter Lock)
        process_func = partial(
            process_line,
            lang=lang,
            encryption_type=encryption_type,
            key_length=key_length,
            seed=seed,
            reuse_key=reuse_key,
            reuse_nonce=reuse_nonce
        )
        with ProcessPoolExecutor() as executor:
            for result in executor.map(process_func, file, chunksize=100):
                if result.get('text'):
                    out_file.write(json.dumps(result, ensure_ascii=False) + '\n')
                            
    print('finished!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--lang', type=str, default='alpha', help='Language of the input data')
    parser.add_argument('--encryption_type', type=str, default='poly', help='Type of encryption')
    parser.add_argument('--input_file', type=str, default='test_input.jsonl', help='Input file prefix')
    parser.add_argument('--output_dir', type=str, default='./', help='Output directory')
    parser.add_argument('--key_length', type=int, default=0, help='Length of the encryption key')
    parser.add_argument('--seed', type=int, default=None, help='Seed for random number generator')
    parser.add_argument('--reuse_key', type=bool, default=True, help='Reuse the same key')
    parser.add_argument('--reuse_nonce', type=bool, default=True, help='Reuse the same nonce')

    args = parser.parse_args()
    main(args)