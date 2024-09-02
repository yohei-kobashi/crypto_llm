import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys

def output_text(model_dir):
    device = "cuda"

    model = AutoModelForCausalLM.from_pretrained(model_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    input_text = "I am a cat. Who"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output = model.generate(input_ids, max_length=50, num_return_sequences=1)

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    print(generated_text)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <model_dir>")
        sys.exit(1)
    
    model_dir = sys.argv[1]
    output_text(model_dir)