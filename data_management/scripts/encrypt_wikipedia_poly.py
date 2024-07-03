import os
import sys
import glob
import json
import encrypt
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def process_file(file_name, block_n, reset_key, encryptor, encryption_rate = 0.9):
    encrypted_lines = []
    no_encrypted_lines = []
    with open(file_name, 'r') as f:
        for line in f.readlines():
            data = json.loads(line)
            if data['text']:
                if random.random() < encryption_rate:
                    data['text'] = encryptor.encrypt(data['text'], block_n = block_n, reset_key = reset_key)
                    encrypted_lines.append(json.dumps(data, ensure_ascii=False))
                else:
                    no_encrypted_lines.append(json.dumps(data, ensure_ascii=False))

    return encrypted_lines, no_encrypted_lines

def main():
    lang = 'en'
    encryption_rate = 0.9
    wiki_dir = f'/storage9/encrypted_llm/datasets/wikipedia/20240520/{lang}/'
    output_dir = '/storage9/encrypted_llm/datasets/encrypted/'
    os.makedirs(output_dir, exist_ok=True)

    # 暗号化して保存
    encryptor = encrypt.Encryptor()
    
    # polyで3パターン
    for i,block_n,reset_key in [(10,0,False),(100,0,True)]:
        print('poly:', i, 'block_n:', block_n)
        sys.stdout.flush()
        encryptor.poly(i)
        if os.path.isdir(wiki_dir):
            wiki_files = glob.glob(os.path.join(wiki_dir, '*.jsonl'))
        else:
            wiki_files = [wiki_dir]
        
        lock = threading.Lock()  # ロックを作成

        with ThreadPoolExecutor() as executor:
            future_to_file = {executor.submit(process_file, file_name, block_n, reset_key, encryptor, encryption_rate): file_name for file_name in wiki_files}
            
            with open(os.path.join(output_dir, 'wikipedia_{}_poly_{:02d}s_reset_{}.jsonl'.format(lang, i, reset_key)), 'w') as out_file, \
                 open(os.path.join(output_dir, 'wikipedia_{}_poly_{:02d}s_reset_{}_no_encryption.jsonl'.format(lang, i, reset_key)), 'w') as out_file2:
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
    main()
