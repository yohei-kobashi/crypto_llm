import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
#import Levenshtein

# def calculate_levenshtein_similarity(text1, text2):
#     """Calculate Levenshtein similarity score"""
#     return 1 - (Levenshtein.distance(text1, text2) / max(len(text1), len(text2)))

def main():
    # コマンドライン引数を設定
    parser = argparse.ArgumentParser(description="Evaluate text reproduction using Levenshtein Distance")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory of the pre-trained model")
    # parser.add_argument("--jsonl_file", type=str, required=True, help="Path to the JSONL file containing text data")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--output_num", type=int, default=1, help="Number of output sequences to generate per input")
    args = parser.parse_args()

    # モデルとトークナイザーをロード
    device = "cuda"

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir).to(device)

    input_ids = tokenizer('I am a cat. Who', return_tensors="pt").input_ids.to(device)
    outputs = model.generate(input_ids)
    generated_texts = [tokenizer.decode(output, skip_special_tokens=True).strip() for output in outputs]
    print(generated_texts)

    # # JSONLファイルの読み込み
    # with open(args.jsonl_file, 'r') as f:
    #     lines = f.readlines()

    # # 各行のテキストに対して評価を行う
    # for line in lines:
    #     data = json.loads(line)
    #     input_text = data['text'].strip()
        
    #     # テキストを前半と後半に分割
    #     split_index = len(input_text) // 2
    #     input_text_front = input_text[:split_index]
    #     input_text_back = input_text[split_index:]
        
    #     # 前半部分をトークナイズ
    #     input_ids = tokenizer(input_text_front, return_tensors="pt").input_ids
        
    #     # テキストの生成
    #     outputs = model.generate(
    #         input_ids,
    #         max_length=len(input_ids[0]) + len(input_text_back.split()),
    #         top_p=args.top_p,
    #         top_k=args.top_k,
    #         temperature=args.temperature,
    #         num_return_sequences=args.output_num,
    #         do_sample=True
    #     )
        
    #     # 出力テキストのデコード
    #     generated_texts = [tokenizer.decode(output, skip_special_tokens=True)[len(input_text_front):].strip() for output in outputs]
        
    #     # Levenshtein Distanceで類似度を計算
    #     similarity_scores = [calculate_levenshtein_similarity(input_text_back, generated_text) for generated_text in generated_texts]
        
    #     # 出力が複数ある場合、平均スコアを計算
    #     average_similarity = sum(similarity_scores) / len(similarity_scores)
        
    #     print(f"Original Text Back: {input_text_back}")
    #     print(f"Generated Texts: {generated_texts}")
    #     print(f"Levenshtein Similarity Scores: {similarity_scores}")
    #     print(f"Average Similarity Score: {average_similarity:.4f}\n")

if __name__ == "__main__":
    main()
