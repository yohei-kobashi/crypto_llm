import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import csv

device = "cuda"

# テストデータの読み込み
test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

# モデルとトークナイザーのパス
base_dir = '/storage9/encrypted_llm/'
model_list = [
    "/storage9/encrypted_llm/models/hf/1.en_no_encrypted_1000",
    "/storage9/encrypted_llm/models/hf/1.en_no_encrypted_2000",
    "/storage9/encrypted_llm/models/hf/1.en_no_encrypted_3000",
    "/storage9/encrypted_llm/models/hf/1.en_no_encrypted_4000",
    "/storage9/encrypted_llm/models/hf/1.en_no_encrypted_5000",
    "/storage9/encrypted_llm/models/hf/1.en_no_encrypted_6000"
]

ppl_list = []
for model_dir in tqdm(model_list):
    model = AutoModelForCausalLM.from_pretrained(model_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
    max_length = max_length = model.config.n_positions
    stride = 512
    seq_len = encodings.input_ids.size(1)
    
    for _ in (range(10)):
        nlls = []
        prev_end_loc = 0
        for begin_loc in tqdm(range(0, seq_len, stride)):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)

                # loss is calculated using CrossEntropyLoss which averages over valid labels
                # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                # to the left by 1.
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        ppl = torch.exp(torch.stack(nlls).mean())
        print(model_dir, ppl)
        ppl_list.append((model_dir, ppl))

csv.writer(open('perplexity10times1.txt', 'w'), delimiter='\t').writerows(ppl_list)