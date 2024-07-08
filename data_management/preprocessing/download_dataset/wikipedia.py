import logging
import os
import bz2
import json
import shutil
import mwxml
import hashlib
import requests
import time
from tqdm import tqdm
logging.basicConfig(level=logging.INFO)

NUM_FILES = os.environ.get('NUM_FILES', 100)
RETRY_COUNT = 5  # リトライ回数
RETRY_DELAY = 10  # リトライ間の待機時間（秒）

def process_dump(page, path, file_index):
    with open(os.path.join(path, f"{file_index}.jsonl"), 'a', encoding='utf-8') as output_file:
        id = page.id
        title = page.title
        for revision in page:
            text = revision.text
            if text:
                break

        # 記事のタイトルとテキストをJSON形式に変換する
        article_json = json.dumps({
            'id': id,
            'title': title,
            'text': text,
        }, ensure_ascii=False)

        # JSONをファイルに書き込む
        output_file.write(article_json + '\n')

def download_with_resume(url, file_path, temp_file_path):
    headers = {}
    if os.path.exists(temp_file_path):
        resume_pos = os.path.getsize(temp_file_path)
        headers['Range'] = f"bytes={resume_pos}-"
    else:
        resume_pos = 0

    for attempt in range(RETRY_COUNT):
        try:
            with requests.get(url, stream=True, timeout=60, headers=headers) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0)) + resume_pos
                with open(temp_file_path, 'ab') as f:
                    for chunk in tqdm(r.iter_content(chunk_size=8192), total=total_size//8192, unit='KB', initial=resume_pos//8192):
                        if chunk:
                            f.write(chunk)
                            f.flush()
            os.rename(temp_file_path, file_path)
            logging.info(f"File saved successfully to {file_path}")
            return True
        except requests.exceptions.RequestException as e:
            logging.error(f"Error downloading the file: {e}")
            if attempt < RETRY_COUNT - 1:
                logging.info(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                logging.error("Max retries reached. Download failed.")
                return False


def download_dataset(date: str, output_base: str = "output", lang: str = "en") -> None:
    filename = f"{lang}wiki-{date}-pages-articles-multistream.xml.bz2"
    dump_path = os.path.join(output_base, f"tmp/wikipedia/{date}/{lang}")
    os.makedirs(dump_path, exist_ok=True)

    url = f"https://dumps.wikimedia.org/{lang}wiki/{date}/{filename}"
    file_path = os.path.join(dump_path, filename)
    temp_file_path = file_path + ".part"
    if not os.path.exists(file_path):
        logging.info(f"Downloading {url}")
        if not download_with_resume(url, file_path, temp_file_path):
            return  # ダウンロードに失敗した場合、関数を終了
    else:
        logging.info(f"File {file_path} already exists")
        logging.info(f"Skipping download\n")

    output_path = os.path.join(output_base, f"datasets/wikipedia/{date}/{lang}")
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    logging.info(f"Parse and process {file_path}")
    with bz2.open(file_path, 'rt', encoding='utf-8') as compressed_file:
        dump = mwxml.Dump.from_file(compressed_file)
        for index, page in enumerate(dump):
            if index > 0 and index % 1000 == 0:
                logging.info(f"Processed {index} articles")

            if page.namespace == 0 and page.redirect is None:
                file_index = int(hashlib.sha256(page.title.encode()).hexdigest(), 16) % NUM_FILES
                process_dump(page, output_path, file_index)