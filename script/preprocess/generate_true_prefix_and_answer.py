import json
import spacy
import argparse
import multiprocessing
import sentencepiece as spm

# Globals for spaCy and SentencePiece
nlp = None
sp_tokenizer = None

def init_spacy_and_tokenizer(sp_model_path: str):
    """
    Initialize spaCy English model and load SentencePiece tokenizer.
    """
    global nlp, sp_tokenizer
    nlp = spacy.load("en_core_web_md")
    sp_tokenizer = spm.SentencePieceProcessor()
    sp_tokenizer.load(sp_model_path)


def extract_sentences_with_person(record):
    """
    Extract sentences containing PERSON entities, splitting into S0, name, S1,
    and S1_keywords. Skip any sentence whose SentencePiece token length > 20.
    """
    text = record.get("text", "")
    results = []
    doc = nlp(text)

    for sent in doc.sents:
        # Skip sentences which do not include "PERSON"
        if not any(ent.label_ == "PERSON" for ent in sent.ents):
            continue
        
        sentence_text = sent.text.strip()
        # Compute token length using SentencePiece
        token_ids = sp_tokenizer.encode(sentence_text, out_type=int)
        if len(token_ids) > 20:
            # Skip sentences longer than 20 tokens
            continue

        # Only process sentences within token limit
        for ent in sent.ents:
            if ent.label_ == "PERSON":
                name = ent.text
                start = ent.start_char - sent.start_char
                end = ent.end_char - sent.start_char

                s0 = sentence_text[:start].strip()
                s1 = sentence_text[end:].strip()

                # Extract keywords from S1 (exclude punctuation and function words)
                s1_doc = nlp(s1)
                s1_keywords = [
                    token.text for token in s1_doc
                    if not token.is_punct and token.pos_ not in {"AUX", "PRON", "DET", "PART", "ADP", "CCONJ", "SCONJ", "INTJ"} and token_text.strip()
                ]

                # Compute token count for S0+name without re-tokenizing
                prefix_str = sentence_text[:end]
                prefix_token_count = None
                # Use decoded prefixes of token_ids to find token boundary
                for i in range(1, len(token_ids) + 1):
                    decoded = sp_tokenizer.decode(token_ids[:i])
                    if len(decoded) >= len(prefix_str):
                        prefix_token_count = i
                        break
                if prefix_token_count is None:
                    prefix_token_count = len(token_ids)

                results.append({
                    "S0": s0,
                    "name": name,
                    "S1": s1,
                    "S1_keywords": s1_keywords,
                    "total_token_count": len(token_ids),
                    "prefix_token_count": prefix_token_count
                })
                break  # Only first PERSON per sentence

    record["person_sentences"] = results
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract PERSON sentences (<=20 SentencePiece tokens) and keywords from JSONL"
    )
    parser.add_argument(
        "--input_path", type=str, required=True,
        help="Path to input JSONL file"
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
        help="Path to output JSONL file"
    )
    parser.add_argument(
        "--sp_model_path", type=str, required=True,
        help="Path to the SentencePiece model file (e.g., .model)"
    )
    args = parser.parse_args()

    # Use half of the available CPU cores
    num_workers = max(1, multiprocessing.cpu_count() // 2)

    # Load all records from JSONL
    with open(args.input_path, "r", encoding="utf-8") as infile:
        records = [json.loads(line) for line in infile]

    # Process records in parallel, initializing spaCy and SentencePiece in each worker
    with multiprocessing.Pool(
        processes=num_workers,
        initializer=init_spacy_and_tokenizer,
        initargs=(args.sp_model_path,)
    ) as pool:
        processed_records = pool.map(extract_sentences_with_person, records)

    # Write out processed records to JSONL
    with open(args.output_path, "w", encoding="utf-8") as outfile:
        for record in processed_records:
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
