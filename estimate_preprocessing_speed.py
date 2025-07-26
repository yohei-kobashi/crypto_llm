import timeit
import numpy as np

import pyarrow.parquet as pq
from glob import glob
from script.preprocess import encrypt_alpha
from script.preprocess.scrub_jsonl import scrubbing


def encryption(texts):
    encryptor = encrypt_alpha.Encryptor()
    encryptor.poly(100, lang="alpha", reuse_key=True)
    for text in texts:
        encryptor.encrypt(text)

def scrubbing(texts):
    for text in texts:
        scrubbing(text)

def main():
    dataset = pq.ParquetDataset("fineweb-edu/sample/10BT-sample/")
    texts = dataset.read(columns=["text"])
    texts = texts["text"].to_numpy()

    times = timeit.repeat("encryption(texts)", globals=globals(), number=1, repeat=10)
    print("encryption:", np.mean(times), np.std(times))

    times = timeit.repeat("scrubbing(texts)", globals=globals(), number=1, repeat=10)
    print("scrubbing:", np.mean(times), np.std(times))

if __name__ == "__main__":
    main()