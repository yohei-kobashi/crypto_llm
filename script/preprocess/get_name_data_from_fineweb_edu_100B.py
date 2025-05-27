from datasets import load_dataset
import spacy
import json
import multiprocessing
import os

nlp = None

def init_worker():
    global nlp
    nlp = spacy.load("en_core_web_sm")

def process_row(row):
    global nlp
    doc = nlp(row["text"][0])
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return {k: v[0] for k, v in row.items()}
    return None

if __name__ == "__main__":
    dataset = load_dataset("yoheikobashi/fineweb-edu-100BT-samples-not-in-10BT", split="train")
    shuffled_dataset = dataset.shuffle(seed=42)

    max_processes = max(1, os.cpu_count() // 2)
    with multiprocessing.Pool(processes=max_processes, initializer=init_worker) as pool:
        results = pool.imap_unordered(process_row, shuffled_dataset, chunksize=10)

        output_path = "/home/yohei.kobashi/lingua/data/data_unlearned_PII_texts_md.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            count = 0
            for result in results:
                if result is not None:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    count += 1
                    if count >= 15000:
                        break
