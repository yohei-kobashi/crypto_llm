from transformers import AutoTokenizer, LlamaForCausalLM

class model_and_tokenizer:
    def __init__(self, model_and_tokenizer_dir, device="cpu"):
        self.device = device
        self.model = LlamaForCausalLM.from_pretrained(model_and_tokenizer_dir).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_and_tokenizer_dir)

    def tokenize(self, input_text):
        return self.tokenizer.encode(input_text, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def generate_text(self, 
            input_ids, 
            max_length=50, 
            top_p=0.9, 
            top_k=50, 
            temperature=1.0, 
            num_return_sequences=1,
            do_sample=True,
            decode_tokens=True
        ):
        attention_mask = (input_ids != self.tokenizer.pad_token_id).to(self.device)

        # テキストの生成
        outputs = self.model.generate(
            input_ids,
            attention_mask=attention_mask, 
            max_length=max_length,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            do_sample=do_sample
        )
        output = outputs[0]

        if decode_tokens:
            output = self.tokenizer.decode(output, skip_special_tokens=True)
        
        return output