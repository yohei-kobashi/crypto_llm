import timeit
import numpy as np
import random

import pyarrow.parquet as pq
from script.preprocess import encrypt_alpha
from script.preprocess.scrub_jsonl import scrubbing
from tqdm import tqdm


def encrypting_text(texts):
    encryptor = encrypt_alpha.Encryptor()
    encryptor.poly(100, lang="alpha", reuse_key=True)
    for text in texts:
        encryptor.encrypt(text)

def scrubbing_texts(texts):
    for text in tqdm(texts):
        scrubbing(text)

def main():
    dataset = pq.ParquetDataset("fineweb-edu/sample/10BT-sample/")
    texts = dataset.read(columns=["text"])
    texts = texts["text"].to_numpy()
    enc_times = []
    scrub_times = []
    for _ in range(10):
        samples = random.sample(list(texts), 1000)

        enc_timer = timeit.Timer(lambda: encrypting_text(samples))
        enc_times.append(enc_timer.timeit(number=1))
        
        scrub_timer = timeit.Timer(lambda: scrubbing_texts(samples))
        scrub_times.append(scrub_timer.timeit(number=1))
        
    print("encryption:", np.mean(enc_times), np.std(enc_times))
    print("scrubbing:", np.mean(scrub_times), np.std(scrub_times))

if __name__ == "__main__":
    main()