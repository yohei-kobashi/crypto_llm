import json
import Levenshtein
import numpy as np
import csv
import time
from tqdm import tqdm

from model_and_tokenizer import model_and_tokenizer

def calculate_levenshtein_similarity(text1, text2):
    """Calculate Levenshtein similarity score"""
    return 1 - (Levenshtein.distance(text1, text2) / max(len(text1), len(text2)))

def main():
    start_time = time.time()

    eval_n = 1

    # models
    model_dirs = [
        "/groups/gcf51099/crypto_llm/models/hf/1.latin_wikipedia_no_encryption_000000_1234_True_step1500/",
        "/groups/gcf51099/crypto_llm/models/hf/1.latin_wikipedia_poly_000001_1234_True_step1500/",
        "/groups/gcf51099/crypto_llm/models/hf/1.latin_wikipedia_poly_000100_1234_True_step1500/",
        "/groups/gcf51099/crypto_llm/models/hf/1.latin_wikipedia_poly_010000_1234_True_step1500/",
        "/groups/gcf51099/crypto_llm/models/hf/1.latin_wikipedia_poly_000000_1234_True_step1500/",
    ]

    # Get pseudo privacy texts
    pseudo_privacy_texts = [json.loads(row)['text'] for row in open("/groups/gcf51099/crypto_llm/data/test/pseudo_privacy_oscar.jsonl")]

    # Analyze privacy texts:
    all_data = []
    for i,model_dir in enumerate(model_dirs):
        sims = []
        model = model_and_tokenizer(model_dir, device="cuda")
        for j,text in tqdm(enumerate(pseudo_privacy_texts)):
            token_ids = model.tokenize(text)
            token_length = len(token_ids[0])
            if token_length < 2:
                continue
            input_length = token_length // 2 if not token_length % 2 else token_length // 2 + 1
            for _ in range(eval_n):
                input_token_ids = token_ids[:, :input_length]
                generated_ids = model.generate_text(input_token_ids, max_length=token_length, decode_tokens=False)
                output_text = model.detokenize(token_ids[0][input_length:])
                generated_output_text = model.detokenize(generated_ids[input_length:])
                sim = calculate_levenshtein_similarity(output_text, generated_output_text)
                sims.append(sim)
                all_data.append((i,j,sim))
        
        print("model_dir", np.mean(sims))

    csv.writer(open("eval_privacy_data_oscar.tsv", "w"), delimiter='\t').writerows(all_data)

    print(time.time() - start_time)

if __name__ == "__main__":
    main()
