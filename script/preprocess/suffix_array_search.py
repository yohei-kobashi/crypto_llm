# suffix_array_pipeline (build/query + test-data + parquet splitter)
# -------------------------------------------------------------
# 2025-07-01: Added `gen_test` sub-command.
# 2025-07-01: Added `split` sub-command (fixed parts mode).
# 2025-07-02: Updated `query` to display total lines, hit count, and hit ratio.
# -------------------------------------------------------------
"""
Usage (main commands)
--------------------
# Build suffix-array index
python suffix_array_pipeline.py build \
       --parquet-dir fineweb-edu/sample-10BT \
       --out-dir ./index \
       --workers auto

# Query index
python suffix_array_pipeline.py query \
       --index-dir ./index \
       --input queries.txt \
       --workers auto

# Generate tiny HIT/MISS test set (40 lines)
python suffix_array_pipeline.py gen_test \
       --parquet-dir fineweb-edu/sample-10BT \
       --output queries.txt \
       --pairs 20

# Split each Parquet file into exactly 8 equal parts (by row count)
python suffix_array_pipeline.py split \
       --parquet-dir fineweb-edu/sample-10BT \
       --out-dir fineweb-edu/sample-10BT-split8 \
       --parts 8 \
       --workers auto
"""
from __future__ import annotations
import argparse, os, sys, pathlib, multiprocessing as mp, pickle, random, re
from functools import partial
from typing import List, Tuple, Dict
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

try:
    import pydivsufsort
except ImportError:
    raise SystemExit("Please `pip install pydivsufsort`.  (apt install libdivsufsort-dev)")

# Globals & helpers
WORD_RE = r"[\w'-]+"
W_RE = re.compile(WORD_RE)

def words(text: str) -> List[str]:
    return W_RE.findall(text.lower())

def get_n_workers(val: str | int | None) -> int:
    return min(72, mp.cpu_count()) if val in (None, "auto") else int(val)

# Parquet iterators

def _yield_text(parquet: str):
    pf = pq.ParquetFile(parquet)
    for batch in pf.iter_batches():
        for cell in batch.column("text"):
            yield cell.as_py()

def _yield_words(parquet: str):
    for txt in _yield_text(parquet):
        yield from words(txt)

# Build-phase workers

def scan_vocab_worker(pq_path: str) -> List[str]:
    seen: Dict[str, None] = {}
    for w in _yield_words(pq_path):
        seen[w] = None
    return list(seen)

VOCAB: Dict[str,int] | None = None

def enc_init(vocab_bytes: bytes):
    global VOCAB
    VOCAB = pickle.loads(vocab_bytes)

def encode_shard_worker(args: Tuple[str,str]) -> pathlib.Path:
    pq_path, out_dir = args
    assert VOCAB is not None
    ids = [VOCAB.get(w,0) for w in _yield_words(pq_path)]
    out = pathlib.Path(out_dir) / f"ids_{pathlib.Path(pq_path).stem}.npy"
    np.save(out, np.array(ids, dtype=np.int32))
    return out

# Build orchestrator

def build_index(pq_dir: str, out_dir: str, workers: int):
    os.makedirs(out_dir, exist_ok=True)
    shards = sorted(pathlib.Path(pq_dir).glob("*.parquet")) or sys.exit("No parquet shards found.")
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        ctx = mp.get_context("spawn")

    # Pass-1: vocab
    vocab: Dict[str,int] = {}
    next_id = 1
    with ctx.Pool(workers) as pool:
        for wordlist in tqdm(pool.imap_unordered(scan_vocab_worker, shards),
                              total=len(shards), desc="Scanning vocab"):
            for w in wordlist:
                if w not in vocab:
                    vocab[w] = next_id; next_id += 1
    # Sort for reproducibility
    vocab = {w:i+1 for i,w in enumerate(sorted(vocab.keys()))}
    np.save(os.path.join(out_dir,"vocab.npy"), np.array(list(vocab.keys()),dtype=object))

    # Pass-2: encode
    vb = pickle.dumps(vocab, pickle.HIGHEST_PROTOCOL)
    enc_args = [(str(p), out_dir) for p in shards]
    with ctx.Pool(workers, initializer=enc_init, initargs=(vb,)) as pool:
        id_files = list(tqdm(pool.imap_unordered(encode_shard_worker, enc_args),
                             total=len(enc_args), desc="Encoding shards"))

    # Concat ids
    total_len = sum(np.load(f, mmap_mode='r').shape[0] for f in id_files)
    all_ids_path = os.path.join(out_dir, 'all_ids.npy')
    all_ids = np.lib.format.open_memmap(all_ids_path, mode='w+', dtype=np.int32, shape=(total_len,))
    off = 0
    for f in tqdm(id_files, desc="Concatenating ids"):
        arr = np.load(f, mmap_mode='r')
        n = arr.shape[0]
        all_ids[off:off+n] = arr; off += n
    del all_ids

    # SA build
    print(f"Building SA for {total_len:,} ids…")
    sa = pydivsufsort.divsufsort(np.load(all_ids_path, mmap_mode='r'))
    np.save(os.path.join(out_dir,'suffix_array.npy'), sa)
    print("Index complete ✔︎")

# Query helpers

def binary_search(ids: np.ndarray, sa: np.ndarray, pattern: List[int]) -> bool:
    lo, hi = 0, sa.size
    m = len(pattern)
    pat = tuple(pattern)
    while lo < hi:
        mid = (lo + hi) // 2
        slice_cmp = tuple(ids[sa[mid]:sa[mid]+m])
        if slice_cmp < pat:
            lo = mid + 1
        else:
            hi = mid
    while lo < sa.size and tuple(ids[sa[lo]:sa[lo]+m]) == pat:
        return True
    return False


def query_index(idx_dir: str, in_txt: str, workers: int):
    vocab_arr = np.load(os.path.join(idx_dir,'vocab.npy'), allow_pickle=True)
    vocab = {w:i+1 for i,w in enumerate(vocab_arr)}
    ids = np.load(os.path.join(idx_dir,'all_ids.npy'), mmap_mode='r')
    sa = np.load(os.path.join(idx_dir,'suffix_array.npy'), mmap_mode='r')
    lines = [l.strip() for l in open(in_txt,encoding='utf-8') if l.strip()]
    win = 35
    results: List[Tuple[str,bool]] = []
    from multiprocessing.pool import ThreadPool
    with ThreadPool(workers) as pool:
        for ln, ok in tqdm(pool.imap_unordered(lambda line: (line, any(binary_search(ids, sa, seq) \
                                                                      for seq in (tuple(ids_seq := [vocab.get(w,0) for w in words(line)])[i:i+win] \
                                                                                   for i in range(len(ids_seq)-win+1)) if 0 not in seq)),
                                                      lines), total=len(lines), desc="Querying"):
            results.append((ln, ok))
            print(f"[{'HIT' if ok else 'MISS'}] {ln[:120]}{'…' if len(ln)>120 else ''}")
    total = len(results)
    hits = sum(1 for _,ok in results if ok)
    ratio = hits/total*100 if total else 0
    print(f"Total lines: {total}, Hits: {hits}, Hit ratio: {ratio:.2f}%")

# Test-data generator
def generate_test_queries(pq_dir: str, out_path: str, pairs: int, win: int=35):
    shards = list(pathlib.Path(pq_dir).glob("*.parquet")) or sys.exit("No parquet shards found.")
    random.seed(42)
    hits, misses = [], []
    for shard in tqdm(random.sample(shards, len(shards)), desc="Sampling shards"):
        for txt in _yield_text(str(shard)):
            w = words(txt)
            if len(w) < win: continue
            start = random.randint(0, len(w)-win)
            seg = w[start:start+win]
            hits.append(" ".join(seg))
            miss = seg.copy(); miss[win//2] = "xyzxyzxyzunique"; misses.append(" ".join(miss))
            if len(hits)>=pairs: break
        if len(hits)>=pairs: break
    with open(out_path,"w",encoding='utf-8') as f:
        for h,m in zip(hits, misses): f.write(h+"\n"); f.write(m+"\n")
    print(f"Wrote {len(hits)*2} lines to {out_path}")

# Parquet splitter
def split_parquet_file_parts(pq_path: str, out_dir: str, parts: int):
    tbl = pq.read_table(pq_path)
    rows = tbl.num_rows
    base = pathlib.Path(pq_path).stem
    size = rows // parts; rem = rows % parts
    for i in range(parts):
        start = i*size + min(i, rem)
        cnt = size + (1 if i < rem else 0)
        slice_tbl = tbl.slice(start, cnt)
        out = pathlib.Path(out_dir)/f"{base}_part{i}.parquet"
        pq.write_table(slice_tbl, out, compression='zstd')

def split_parquet_dir(pq_dir: str, out_dir: str, parts: int, workers: int):
    os.makedirs(out_dir, exist_ok=True)
    shards = sorted(pathlib.Path(pq_dir).glob("*.parquet")) or sys.exit("No parquet shards found.")
    try: ctx = mp.get_context("fork")
    except ValueError: ctx = mp.get_context("spawn")
    func = partial(split_parquet_file_parts, out_dir=out_dir, parts=parts)
    with ctx.Pool(workers) as pool:
        list(tqdm(pool.imap_unordered(func, shards), total=len(shards), desc="Splitting parquet"))
    print("Splitting complete ✔︎")

# CLI
def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build")
    p.add_argument("--parquet-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--workers", default="auto")
    q = sub.add_parser("query")
    q.add_argument("--index-dir", required=True)
    q.add_argument("--input", required=True)
    q.add_argument("--workers", default="auto")
    g = sub.add_parser("gen_test")
    g.add_argument("--parquet-dir", required=True)
    g.add_argument("--output", required=True)
    g.add_argument("--pairs", type=int, default=20)
    s = sub.add_parser("split")
    s.add_argument("--parquet-dir", required=True)
    s.add_argument("--out-dir", required=True)
    s.add_argument("--parts", type=int, required=True)
    s.add_argument("--workers", default="auto")
    args = parser.parse_args()
    if args.cmd == "build":
        w = get_n_workers(args.workers)
        build_index(args.parquet_dir, args.out_dir, w)
    elif args.cmd == "query":
        w = get_n_workers(args.workers)
        query_index(args.index_dir, args.input, w)
    elif args.cmd == "gen_test":
        generate_test_queries(args.parquet_dir, args.output, args.pairs)
    elif args.cmd == "split":
        w = get_n_workers(args.workers)
        split_parquet_dir(args.parquet_dir, args.out_dir, args.parts, w)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
