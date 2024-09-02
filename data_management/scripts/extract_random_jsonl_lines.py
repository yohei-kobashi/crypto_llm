import json
import random
import argparse

def extract_random_lines(input_file, output_file, num_lines, seed=None):
    # seed値が指定されている場合、乱数生成器を初期化
    if seed is not None:
        random.seed(seed)

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # ランダムに指定された行数を選択
    selected_lines = random.sample(lines, num_lines)

    # 選択された行を新しいファイルに書き込む
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in selected_lines:
            f.write(line)

def main():
    parser = argparse.ArgumentParser(description="Extract random lines from a JSONL file.")
    parser.add_argument('--input_file', type=str, help="Path to the input JSONL file.")
    parser.add_argument('--output_file', type=str, help="Path to the output JSONL file.")
    parser.add_argument('--num_lines', type=int, help="Number of lines to extract.")
    parser.add_argument('--seed', type=int, default=None, help="Seed value for random number generator (optional).")

    args = parser.parse_args()

    extract_random_lines(args.input_file, args.output_file, args.num_lines, args.seed)

if __name__ == "__main__":
    main()
