# src/preprocess.py
import json
import os
from tqdm import tqdm

# Paths
RAW_PATH = "data/raw/triviaqa_raw.json"
OUT_PATH = "data/processed/triviaqa_clean.jsonl"

MAX_DOCS_PER_Q = 5

def load_raw():
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"Input file not found at: {RAW_PATH}")
    
    print(f"Loading raw data from {RAW_PATH}...")
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    # The TriviaQA JSON structure has a "Data" key holding the list
    return data["Data"]

def clean_text(text):
    if not text:
        return ""
    return " ".join(text.split())

def flatten_record(item):
    """
    Extracts relevant fields and formats documents.
    Returns None if the record is invalid.
    """
    # 1. Extract Question
    q = clean_text(item.get("Question", ""))

    # 2. Extract Answer
    answer_block = item.get("Answer") or {}
    ans = clean_text(answer_block.get("Value", ""))

    # 3. Extract Documents (SearchResults)
    raw_search_results = item.get("SearchResults") or []

    docs = []
    for sr in raw_search_results:
        title = clean_text(sr.get("Title", ""))
        desc = clean_text(sr.get("Description", ""))
        url = sr.get("Url", "")

        if not title and not desc:
            continue

        merged = f"{title}. {desc}"
        docs.append({
            "rank": sr.get("Rank", None),
            "title": title,
            "text": merged,
            "url": url
        })

    # 4. Limit to top 5 documents (Efficiency for RAG)
    docs = docs[:MAX_DOCS_PER_Q]
    qid = clean_text(item.get("QuestionId", ""))

    if not qid or not q or not ans or not docs:
        return None
    
    return {
        "id": qid,
        "question": q,
        "answer": ans,
        "docs": docs  # List of dicts
    }

def process_all():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    data = load_raw()
    total = len(data)
    saved_count = 0

    print(f"Found {total} records. Processing...")

    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for item in tqdm(data, desc="Cleaning Data"):
            flat = flatten_record(item)
            if flat:
                out.write(json.dumps(flat, ensure_ascii=False) + "\n")
                saved_count += 1

    print(f"\nProcessing Complete.")
    print(f"Total Raw Records: {total}")
    print(f"Saved Valid Records: {saved_count}")
    print(f"Skipped (Empty/Invalid): {total - saved_count}")
    print(f"Output File: {OUT_PATH}")


if __name__ == "__main__":
    process_all()
