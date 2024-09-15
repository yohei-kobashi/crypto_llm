from datasets import load_dataset
import spacy
import json
from tqdm import tqdm
import re

data_label = "oscar"

nlp = spacy.load("en_core_web_md")

# ignore_verifications=Trueをつけないとエラーとなる
oscar_dataset = load_dataset('oscar', 'unshuffled_deduplicated_en', 
                       split='train', 
                       ignore_verifications=True,
                       trust_remote_code=True,
                       streaming=True)

pseudo_privacy_sentences = []

for record in tqdm(oscar_dataset):
    for paragraph in record["text"].split("\n"):
        doc = nlp(paragraph)
        for sent in doc.sents:
            text = sent.text.strip()
            if re.search("\.$", text):
                persons = [ent.text for ent in sent.ents if ent.label_ == "PERSON"]
                if persons:
                    pseudo_privacy_sentences.append(json.dumps({"text": text, "persons": persons}, ensure_ascii=False))
                    if len(pseudo_privacy_sentences) == 1000:
                        break
        if len(pseudo_privacy_sentences) == 1000:
            break
    if len(pseudo_privacy_sentences) == 1000:
        break

# 保存
with open(f"/groups/gcf51099/crypto_llm/data/test/pseudo_privacy_{data_label}.jsonl", "w") as outfile:
    outfile.write("\n".join(pseudo_privacy_sentences))


