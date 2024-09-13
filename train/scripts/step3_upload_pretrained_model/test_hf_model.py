import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM
import sys

def output_text(model_dir, model_type):
    device = "cpu"

    if model_type == "llama":
        model = LlamaForCausalLM.from_pretrained(model_dir).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_dir).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)

    input_text = "My father is my sister's"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).to(device)
    output = model.generate(input_ids, attention_mask=attention_mask, max_length=50, num_return_sequences=1)

    generated_text = tokenizer.decode(output[0], skip_special_tokens=False)

    print(generated_text)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <model_dir>")
        sys.exit(1)
    
    model_dir = sys.argv[1]
    model_type = sys.argv[2] if len(sys.argv) > 2 else "llama"

    output_text(model_dir, model_type)