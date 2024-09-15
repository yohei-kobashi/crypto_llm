import torch
from transformers import AutoTokenizer, LlamaForCausalLM
import sys

def output_text(model_dir):
    device = "cpu"

    model = LlamaForCausalLM.from_pretrained(model_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    input_text = "My father is my sister's"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).to(device)
    output = model.generate(input_ids, attention_mask=attention_mask, max_length=50, num_return_sequences=1)

    generated_text = tokenizer.decode(output[0], skip_special_tokens=False)

    print(generated_text)

if __name__ == "__main__":
    model_dirs = [
        "/groups/gcf51099/crypto_llm/models/hf/1.latin_wikipedia_poly_000100_1234_True_step1500/",
        "/groups/gcf51099/crypto_llm/models/hf/1.latin_wikipedia_poly_010000_1234_True_step1500/",
        "/groups/gcf51099/crypto_llm/models/hf/1.latin_wikipedia_poly_000000_1234_True_step1500/",
    ]
    for model_dir in model_dirs:
        output_text(model_dir)