import random
import unicodedata
from Crypto.Cipher import ChaCha20
from base64 import b64encode

# Encryption
class Encryptor():
    def __init__(self, seed=None):
        self.lang = 'latin'
        self.symbols = []
        self.symbol2index = {}
        self.shift = 0
        self.key = None
        self.key_length = None
        self.reuse_key = True
        self.nonce = None
        self.seed = seed
    
    ###################### Common ######################
    # Generate random bytes
    def get_random_bytes(self, length):
        original_seed = random.getstate()
        if self.seed is not None:
            random.seed(self.seed)
        random_bytes = bytes(random.getrandbits(8) for _ in range(length))
        random.setstate(original_seed)
        
        return random_bytes
    
    # Get display characters of BMP
    def get_all_chars(self):
        if self.lang == 'latin':
            self.symbols = [chr(code_point) for code_point in range(0x024F+1) if unicodedata.category(chr(code_point)) not in {'Cc', 'Cf', 'Cs', 'Co', 'Cn'}]
        else:
            self.symbols = [chr(code_point) for code_point in range(0xFFFF+1) if unicodedata.category(chr(code_point)) not in {'Cc', 'Cf', 'Cs', 'Co', 'Cn'}]

        original_seed = random.getstate()
        if self.seed is not None:
            random.seed(self.seed)
        random.shuffle(self.symbols)
        random.setstate(original_seed)
        self.symbol2index = {symbol:i for i,symbol in enumerate(self.symbols)}
    
    ###################### Polyalphabetic Cipher ######################
    def poly(self, key_length=0, lang='latin', reuse_key=True):
        self.lang = lang
        self.get_all_chars()
        self.key_length = key_length
        self.reuse_key = reuse_key

        if not key_length:
            # If key_length is 0, a key of length 10000 is temporarily generated.
            # The actual key length will be adjusted to match the text length in the extend_key method.
            key_length = 10000

        original_seed = random.getstate()
        if self.seed is not None:
            random.seed(self.seed)
        self.key = ''.join(random.choices(self.symbols, k=key_length))
        random.setstate(original_seed)
        self.encrypt = self.encrypt_poly

    # Convert using polyalphabetic cipher
    def extend_key(self, text):
        if self.key_length:
            # If self.key_length > 0, repeat the key to match the length of the text
            repeat_count = len(text) // self.key_length + 1
            extended_key = (self.key * repeat_count)[:len(text)]
        else:
            # If self.key_length == 0, extend the key to match the length of the text
            if len(self.key) < len(text):
                original_seed = random.getstate()
                if self.seed is not None:
                    random.seed(self.seed)
                self.key += ''.join(random.choices(self.symbols, k=len(text) - len(self.key)))
                random.setstate(original_seed)
                extended_key = self.key
            else:
                extended_key = self.key[:len(text)]
        
        return extended_key

    # Encrypt based on the key
    def encrypt_poly(self, text):
        if not self.reuse_key:
            self.get_all_chars()

        encrypted_text = ""
        extended_key = self.extend_key(text)
        
        for i in range(len(text)):
            char = text[i]
            key_char = extended_key[i]
            
            # Use index as shift value
            shift = self.symbol2index[key_char]
            
            # Map chars
            encrypted_text += self.symbols[(self.symbol2index[char] + shift) % len(self.symbols)] if char in self.symbol2index else char
            
        return encrypted_text

    ###################### ChaCha20 (Stream Cipher) ######################
    def chacha20(self, reuse_key=False, reuse_nonce=False):
        self.key = None
        self.nonce = None
        if reuse_key:
            self.key = self.get_random_bytes(32)
        if reuse_nonce:
            self.nonce = self.get_random_bytes(8)
        self.encrypt = self.encrypt_chacha20
    
    def encrypt_chacha20(self, text):
        key = self.key if self.key else self.get_random_bytes(32)
        nonce = self.nonce if self.nonce else self.get_random_bytes(8)
        
        cipher = ChaCha20.new(key=key, nonce=nonce)
        encrypted_text = b64encode(cipher.encrypt(text.encode('utf-8'))).decode('utf-8')
        
        return encrypted_text

def main():
    seed = 1234
    
    text = 'Hello!'

    encryptor = Encryptor(seed=seed)

    # Polyalphabetic cipher
    encryptor.poly(0, reuse_key=True)
    print('poly:', encryptor.encrypt(text))

    # Stream cipher
    encryptor.chacha20(reuse_key=True, reuse_nonce=True)
    print('chacha20', encryptor.encrypt(text))

if __name__ == '__main__':
    main()