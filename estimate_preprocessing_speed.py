import timeit
import numpy as np
import random

import pyarrow.parquet as pq
from script.preprocess import encrypt_alpha
from script.preprocess.scrub_jsonl import scrubbing


def encrypting_text(texts):
    encryptor = encrypt_alpha.Encryptor()
    encryptor.poly(100, lang="alpha", reuse_key=True)
    for text in texts:
        encryptor.encrypt(text)

def scrubbing_texts(texts):
    for text in texts:
        scrubbing(text)

def main():
    dataset = pq.ParquetDataset("fineweb-edu/sample/10BT-sample/")
    texts = dataset.read(columns=["text"])
    texts = texts["text"].to_numpy()
    samples = random.sample(list(texts), 1000)

    enc_timer = timeit.Timer(lambda: encrypting_text(samples))
    times = enc_timer.repeat(repeat=10, number=1)
    print("encryption:", np.mean(times), np.std(times))
    
    scrub_timer = timeit.Timer(lambda: scrubbing_texts(samples))
    times = scrub_timer.repeat(repeat=10, number=1)
    print("scrubbing:", np.mean(times), np.std(times))

if __name__ == "__main__":
    main()