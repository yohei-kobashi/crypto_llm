import spacy
import json
import time
from tqdm import tqdm
import re
from datasets import load_dataset

max_count = 10000

def normalize_name(raw_name: str) -> str:
    # Convert to lowercase
    name = raw_name.lower()
    # Normalize extra whitespace
    name = re.sub(r'\s+', ' ', name.strip())
    return name

def canonicalize_name(raw_name: str) -> str:
    """
    Standardize so that the same person is recognized regardless of name order.
    Here, considering both "first last" and "last first" orders, return a unified key
    in sorted order.
    """
    name = normalize_name(raw_name)
    parts = name.split(' ')
    
    # If there's only one part, return it as is (though single-word names are unusual)
    # If there are two parts, normalize (sort) their order
    if len(parts) == 2:
        # Sort the name parts and join them with an underscore (fixed order)
        sorted_parts = sorted(parts)
        return "_".join(sorted_parts)
    else:
        return ""

nlp = spacy.load("en_core_web_sm")
data = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)

name_freq = {}

count = 0

start_time = time.time()
for doc in tqdm(data):
    name_set = set()
    result = nlp(doc["text"])
    for ent in result.ents:
        if ent.label_ == "PERSON":
            name_set.add(ent.text)

    canonical_name_set = set()
    for name in name_set:
        name_canonical = canonicalize_name(name)
        if name_canonical!="":
            canonical_name_set.add(name_canonical)

    for name_canonical in canonical_name_set:
        if name_canonical in name_freq:
            name_freq[name_canonical] += 1
        else:
            name_freq[name_canonical] = 1
    count += 1
    if count == max_count:
        end_time = time.time()
        break

print(len(name_freq))
print(end_time - start_time)
print((end_time - start_time) / max_count)



# save the name_freq dictionary to a file
with open("name_freq_key.json", "w") as f:
    name_freq_key = dict(sorted(name_freq.items(), key=lambda x: x[0]))
    json.dump(name_freq_key, f)

with open("name_freq_item.json", "w") as f:
    name_freq_value = dict(sorted(name_freq.items(), key=lambda x: x[1]))
    json.dump(name_freq_value, f)