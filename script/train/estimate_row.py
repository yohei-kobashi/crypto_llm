def get_line_number(file_path: str, position: int) -> int:
    line_number = 0
    with open(file_path, "r") as f:
        # ファイルの先頭から position まで読み込む
        while f.tell() < position:
            line = f.readline()
            if not line:  # EOFに到達した場合
                break
            line_number += 1
            if line_number % 10000 == 0:
                print(f"reading: {line_number}")

    print(f"Found line_number: {line_number}")
    return line_number

get_line_number("working_dir/datatrove/extract_names_from_fwe10b_n1/data_final_0.5_continual_chunked/data_final.chunk.00.jsonl", 7477931841)