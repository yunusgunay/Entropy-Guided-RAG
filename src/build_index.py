# src/build_index.py
import json
import pickle
from tqdm import tqdm
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

DATA_PATH = "data/processed/triviaqa_clean.jsonl"
INDEX_PATH = "data/processed/embeddings.faiss"
META_PATH = "data/processed/metadata.pkl"

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 64
USE_COSINE = True

def main():
    model = SentenceTransformer(MODEL_NAME)

    texts = []
    doc_id_strs = []
    question_ids = []
    questions = []
    answers = []
    ranks = []
    urls = []

    # Read question-level JSONL, flatten into passage-level lists
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading processed data"):
            item = json.loads(line)

            qid = item.get("id", "")
            q = item.get("question", "")
            a = item.get("answer", "")

            if not qid or not q or not a:
                continue

            for d in item.get("docs", []):
                text = d.get("text", "")
                if not text:
                    continue

                rank = d.get("rank", None)
                try:
                    rank_int = int(rank) if rank is not None else None
                except Exception:
                    rank_int = None
                
                # Stable doc id (needed for Layer-1 cache keys)
                faiss_id = len(texts)
                doc_id_str = f"{qid}_{rank_int}" if rank_int is not None else f"{qid}_{faiss_id}"

                texts.append(text)
                doc_id_strs.append(doc_id_str)
                question_ids.append(qid)
                questions.append(q)
                answers.append(a)
                ranks.append(rank_int)
                urls.append(d.get("url", ""))

    if len(texts) == 0:
        raise RuntimeError("No passages found. Check preprocess output format and DATA_PATH.")

    print(f"Loaded {len(texts):,} passages. Computing embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=USE_COSINE,
    ).astype("float32")

    dim = int(embeddings.shape[1])
    index = faiss.IndexFlatIP(dim) if USE_COSINE else faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)

    meta = {
        "texts": texts,
        "passages": texts, # alias for older scripts
        "question_ids": question_ids,
        "questions": questions,
        "answers": answers,
        "ranks": ranks,
        "urls": urls,
        "doc_ids_str": doc_id_strs, # Do NOT use these as L1 keys.
        "model_name": MODEL_NAME,
        "use_cosine": USE_COSINE,
        "faiss_dim": dim,
        "num_vectors": int(index.ntotal),
    }

    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)

    print(f"Saved FAISS index to: {INDEX_PATH}")
    print(f"Saved metadata to: {META_PATH}")
    print(f"Index dim={dim}, vectors={index.ntotal:,}")


if __name__ == "__main__":
    main()
