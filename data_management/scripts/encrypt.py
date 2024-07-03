import json
import glob
import os
import dask.bag as db
from dask import delayed, compute
import numpy as np
import random
import time
import unicodedata
from collections import Counter
from Crypto.Cipher import ChaCha20, AES
from Crypto.Util.Padding import pad
from base64 import b64encode

# 暗号化
class Encryptor():
    def __init__(self):
        self.lang = 'en'
        self.freq = Counter()
        self.symbols = []
        self.symbol2index = {}
        self.shift = 0
        self.key = None
        self.key_length = None
        self.nonce = None
        self.iv = None
    
    ###################### 共通 ######################
    # ランダムのバイト列を生成
    def get_random_bytes(self, length, seed=None):
        if seed:
            random.seed(seed)
        random_bytes = bytes(random.getrandbits(8) for _ in range(length))
        random.seed(None)
        
        return random_bytes
    
    # BMPの表示文字を取得
    def get_all_chars(self, seed=None):
        if self.lang == 'en':
            self.symbols = [chr(code_point) for code_point in range(0x024F+1) if unicodedata.category(chr(code_point)) not in {'Cc', 'Cf', 'Cs', 'Co', 'Cn'}]
        else:
            self.symbols = [chr(code_point) for code_point in range(0xFFFF+1) if unicodedata.category(chr(code_point)) not in {'Cc', 'Cf', 'Cs', 'Co', 'Cn'}]
        if seed:
            random.seed(seed)
        random.shuffle(self.symbols)
        random.seed(None)
        self.symbol2index = {symbol:i for i,symbol in enumerate(self.symbols)}
    
    # 置換先のシンボルを生成
    def generate_symbols(self, num_symbols=None, seed=None):
        if not len(self.symbols) or (num_symbols and len(self.symbols) < num_symbols):
            symbols = []
            # Unicode基本多言語面の範囲（制御文字や非表示文字を除外）
            
            symbol_set = set(self.symbols)
            for code_point in range(0x110000):
                char = chr(code_point)
                # Cc=制御文字, Cf=フォーマット文字, Cs=サロゲートペア, Co=プライベートユース文字, Cn=未割当文字の除外
                if char in symbol_set or unicodedata.category(char) in {'Cc', 'Cf', 'Cs', 'Co', 'Cn'}:
                    continue
                symbols.append(char)
            if seed:
                random.seed(seed)
            if num_symbols and len(symbols) > num_symbols - len(self.symbols):
                symbols = random.sample(symbols, num_symbols - len(self.symbols))
            else:
                random.shuffle(symbols)
            random.seed(None)
            self.symbols += symbols
        
        self.symbols = self.symbols[:num_symbols]
        
        # symbol2indexの辞書を作成
        self.symbol2index = {symbol:i for i,symbol in enumerate(self.symbols)}

    # 頻度分布を取得
    def aggregate_frequencies(self, data_dir, seed=None):
        if not data_dir or not os.path.exists(data_dir):
            return None
        if os.path.isdir(data_dir):
            files = glob.glob(os.path.join(data_dir, '*.jsonl'))
        else:
            files = [data_dir]
        
        # Daskでファイル読み込みと頻度計算を並列化
        @delayed
        def process_file(jsonl_file):
            text = ''
            with open(jsonl_file) as f:
                text += ''.join([char for line in f.readlines() for char in json.loads(line)['text']])
            return self.calculate_frequencies(text)

        # 各ファイルの処理を並列化
        tasks = [process_file(file) for file in files]
        results = compute(*tasks, scheduler='processes')
        
        # 結果を集計
        for counter in results:
            self.freq.update(counter)

        self.symbols = [char for char in self.freq.keys() if unicodedata.category(char) not in {'Cc', 'Cf', 'Cs', 'Co', 'Cn'} and ord(char) < 0xFFFF]
        if seed:
            random.seed(seed)
        random.shuffle(self.symbols)
        random.seed(None)
    
    def calculate_frequencies(self, text):
        num_workers = os.cpu_count()
        
        # Daskのbagにテキストを分割して読み込む
        bag = db.from_sequence(text, npartitions=num_workers)
        
        # 各チャンクでの文字数カウント
        counters = bag.map(lambda x: Counter(x)).compute()
        
        # カウンターの結果をマージ
        total_counter = Counter()
        for counter in counters:
            total_counter.update(counter)
        
        return total_counter

    # 転置式暗号
    def transpose(self, text, block_n, seed=None):
        if seed:
            random.seed(seed)
        transposed_text = ''
        for i in range(0, len(text), block_n):
            transposed_text += ''.join(random.sample(text[i:i+block_n], len(text[i:i+block_n])))
        random.seed(None)
        return transposed_text

    ###################### 単一換字式暗号 ######################
    def simple(self, lang='en', seed=None):
        self.lang = lang
        self.get_all_chars(seed=seed)
        if seed:
            random.seed(seed)
        self.shift = random.randint(1, len(self.symbols)-1)
        random.seed(None)
        self.encrypt = self.encrypt_simple

    def encrypt_simple(self, text, block_n=0, seed=None):
        if block_n:
            text = self.transpose(text, block_n, seed=seed)
        encrypted_chars = [self.symbols[(self.symbol2index[char] + self.shift) % len(self.symbols)] if char in self.symbol2index else char for char in text]
        encrypted_text = ''.join(encrypted_chars)
        return encrypted_text
        
    ###################### ホモフォニック置換暗号 ######################
    def homophonic(self, data_dir, seed=None):
        self.aggregate_frequencies(data_dir, seed=seed)
        self.create_homophonic_mapping(seed=seed)
        self.encrypt = self.encrypt_homophonic

    # マッピングを生成
    def create_homophonic_mapping(self, seed=None):
        self.generate_symbols(len(self.freq) * 3, seed=seed)
        remaining_symbols = self.symbols[:]
        if seed:
            random.seed(seed)
        random.shuffle(remaining_symbols)        
        random.seed(None)
        
        # 確率分布を計算
        total_frequencies = sum(self.freq.values())
        freqs = np.array(list(self.freq.values())) - total_frequencies / len(self.freq)
        freqs = np.maximum(freqs, 0)
        total_frequencies = np.sum(freqs)
        probs = freqs / total_frequencies

        # remaining_symbolsのsymbolを元のcharsに割り当て
        chars = list(self.freq.keys())
        # まず1つずつ割り当てる
        self.mapping = {char:[symbol] for symbol in remaining_symbols[:len(chars)]}
        remaining_symbols = remaining_symbols[len(chars):]
        
        # 残りは確率分布に
        if seed:
            np.random.seed(seed)
        choices = np.random.choice(len(chars), size=len(remaining_symbols), p=probs)
        np.random.seed(None)
        for choice in choices:
            char = chars[choice]
            self.mapping[char].append(remaining_symbols.pop())

    def encrypt_homophonic(self, text, block_n=0, seed=None):
        if block_n:
            text = self.transpose(text, block_n, seed=seed)
        encrypted_chars = [self.mapping[char] if char in self.mapping else char for char in text]
        encrypted_text = ''.join(encrypted_chars)
        return encrypted_text

    ###################### 多表式換字式暗号 ######################
    def poly(self, key_length=0, lang='en', seed=None):
        self.lang = lang
        self.get_all_chars(seed=seed)
        self.key_length = key_length
        if not key_length:
            key_length = 10000
        self.key = ''.join(random.choices(self.symbols, k=key_length))
        self.encrypt = self.encrypt_poly

    # 多表式換字式暗号による変換
    # 鍵を繰り返してテキストと同じ長さにする
    def extend_key(self, text, seed=None):
        if self.key_length:
            repeat_count = len(text) // self.key_length + 1
            extended_key = (self.key * repeat_count)[:len(text)]
        else:
            if len(self.key) < len(text):
                if seed:
                    random.seed(seed)
                self.key += ''.join(random.choices(self.symbols, k=len(text) - len(self.key)))
                random.seed(None)
                extended_key = self.key
            else:
                extended_key = self.key[:len(text)]
        
        return extended_key

    # 鍵に基づいて暗号化する
    def encrypt_poly(self, text, block_n=0, seed=None, reset_key=False):
        if block_n:
            text = self.transpose(text, block_n, seed=seed)
        if reset_key:
            self.get_all_chars(seed=seed)
        encrypted_text = ""
        extended_key = self.extend_key(text, seed=seed)
        
        for i in range(len(text)):
            char = text[i]
            key_char = extended_key[i]
            
            # インデックスをシフト値として使用
            shift = self.symbol2index[key_char]
            
            # charをmapping
            encrypted_text += self.symbols[(self.symbol2index[char] + shift) % len(self.symbols)] if char in self.symbol2index else char
            
        return encrypted_text

    ###################### ChaCha20（ストリーム暗号）######################
    # ストリーム暗号は文字単位で変換するためkeyとnonce固定なら単一換字式暗号と同等
    def chacha20(self, reuse_key=False, reuse_nonce=False, seed=None):
        self.key = None
        self.nonce = None
        if reuse_key:
            self.key = self.get_random_bytes(32, seed=seed)
        if reuse_nonce:
            self.nonce = self.get_random_bytes(8, seed=seed)
        self.encrypt = self.encrypt_chacha20
    
    def encrypt_chacha20(self, text, block_n=0, seed=None):
        if block_n:
            text = self.transpose(text, block_n, seed=seed)
        key = self.key if self.key else self.get_random_bytes(32, seed=seed)
        nonce = self.nonce if self.nonce else self.get_random_bytes(8, seed=seed)
        cipher = ChaCha20.new(key=key, nonce=nonce)
        encrypted_text = b64encode(cipher.encrypt(text.encode('utf-8'))).decode('utf-8')
        
        return encrypted_text

    ###################### AES（ブロック暗号）######################
    def aes(self, key_byte=32, reuse_key=False, reuse_iv=False, seed=None):
        # key_128 = os.urandom(16)  # 16 bytes key for AES-128
        # key_192 = os.urandom(24)  # 24 bytes key for AES-192
        # key_256 = os.urandom(32)  # 32 bytes key for AES-256
        # iv = os.urandom(16)  # 16 bytes IV for AES (CBC mode)
        self.key_byte = key_byte
        self.key = None
        self.iv = None
        if reuse_key:
            self.key = self.get_random_bytes(key_byte, seed=seed)
        if reuse_iv:
            self.iv = self.get_random_bytes(16, seed=seed)
        self.encrypt = self.encrypt_aes
    
    def encrypt_aes(self, text, block_n=0, seed=None):
        if block_n:
            text = self.transpose(text, block_n, seed=seed)
        key = self.key if self.key else self.get_random_bytes(self.key_byte, seed=seed)
        iv = self.iv if self.iv else self.get_random_bytes(16, seed=seed)
        
        # 暗号化
        cipher = AES.new(key, AES.MODE_CBC, iv)
        encrypted_text = b64encode(cipher.encrypt(pad(text.encode('utf-8'), AES.block_size))).decode('utf-8')
        
        return encrypted_text

def main():
    seed = 1234
    
    text = 'Hello!'
    # 単一換字式暗号
    encryptor = Encryptor()
    encryptor.simple(seed=seed)
    print('simple:', encryptor.encrypt(text, seed=seed))
    # ホモフォニック置換暗号
    # encryptor.homophonic('/groups/gcf51099/crypto_llm/data/datasets/wikipedia/20240301/ja/', seed=seed)
    # print('homophonic', encryptor.encrypt(text, seed))
    # 多表式換字式暗号
    encryptor.poly(0, seed=seed)
    print('poly:', encryptor.encrypt(text, seed=seed, reset_key=False))
    # ストリーム暗号
    encryptor.chacha20(reuse_key=True, reuse_nonce=True, seed=seed)
    print('chacha20', encryptor.encrypt(text, seed=seed))
    # ブロック暗号
    encryptor.aes(reuse_key=True, reuse_iv=True, seed=seed)
    print('aes', encryptor.encrypt(text, seed=seed))

if __name__ == '__main__':
    main()