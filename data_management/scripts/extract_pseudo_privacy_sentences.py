import sys
import json
import spacy
import csv
import re
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

nlp = spacy.load("en_core_web_md")

# jsonlファイル一覧
data_labels = [
    "wikipedia_latin_no_encryption_000000_1234_True_no_encryption",
]

def process_sentence(paragraph):
    pseudo_privacy_sentences = []
    freq_person = {}
    
    doc = nlp(paragraph)
    for sent in doc.sents:
        text = sent.text.strip()
        if re.search("\.$", text):
            persons = [ent.text for ent in sent.ents if ent.label_ == "PERSON"]
            if persons:
                pseudo_privacy_sentences.append(json.dumps({"text": text, "persons": persons}, ensure_ascii=False))
                for person in persons:
                    freq_person[person] = freq_person.get(person, 0) + 1

    return pseudo_privacy_sentences, freq_person

def process_file(data_label, batch_size=1000000):
    pseudo_privacy_sentences = []
    freq_person = {}

    with open(f"/groups/gcf51099/crypto_llm/data/encrypted/{data_label}.jsonl", "r") as infile:
        batch = []
        for row in infile:
            text = json.loads(row)["text"]
            batch.extend(text.split("\n"))
            
            # バッチがいっぱいになったら並列処理を行う
            if len(batch) >= batch_size:
                process_batch(batch, pseudo_privacy_sentences, freq_person)
                batch = []  # バッチをクリアする

        # 残りの段落を処理
        if batch:
            process_batch(batch, pseudo_privacy_sentences, freq_person)

    # 保存
    with open(f"/groups/gcf51099/crypto_llm/data/test/pseudo_privacy_{data_label}.jsonl", "w") as outfile:
        outfile.write("\n".join(pseudo_privacy_sentences))

    with open(f"/groups/gcf51099/crypto_llm/data/test/freq_pseudo_privacy_{data_label}.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter="\t")
        for k, v in freq_person.items():
            writer.writerow([k, v])

def process_batch(batch, pseudo_privacy_sentences, freq_person):
    with ProcessPoolExecutor(max_workers=50) as executor:
        futures = {executor.submit(process_sentence, paragraph): paragraph for paragraph in batch}
        
        for future in as_completed(futures):
            try:
                result_sentences, result_freq_person = future.result()
                pseudo_privacy_sentences.extend(result_sentences)
                for person, count in result_freq_person.items():
                    freq_person[person] = freq_person.get(person, 0) + count
            except Exception as e:
                print(f"Error processing paragraph: {e}")
    print("One batch has been processed.")
    sys.stdout.flush()

def main():
    for data_label in tqdm(data_labels):
        try:
            process_file(data_label)
            print(f"{data_label} processed successfully.")
        except Exception as e:
            print(f"Error processing {data_label}: {e}")

if __name__ == "__main__":
    main()
