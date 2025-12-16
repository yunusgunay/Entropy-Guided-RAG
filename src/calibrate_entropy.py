# src/calibrate_entropy.py
# Calibrate entropy thresholds for retrieval confidence estimation
import json
import pickle
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

DATA_PATH = "data/processed/triviaqa_clean.jsonl" # Dataset (11620 samples)
INDEX_PATH = "data/processed/embeddings.faiss" # FAISS index of passage embeddings
META_PATH = "data/processed/metadata.pkl"

MODEL_NAME = "all-MiniLM-L6-v2" # Sentence-Transformer

K_MAX = 8
TEMPERATURE = 0.1 # softmax temperature
N_SAMPLES = None # None for full dataset

def softmax(x, tau):
    x = x / max(tau, 1e-8)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / (np.sum(ex) + 1e-12)

def entropy(scores, tau):
    p = softmax(scores, tau)
    p = np.clip(p, 1e-12, 1.0)
    H = float(-np.sum(p * np.log(p)))
    H_norm = H / np.log(len(p)) if len(p) > 1 else 0.0
    return H, H_norm

def contains_answer(texts, ans):
    a = ans.lower().strip()
    if not a:
        return False
    return any(a in t.lower() for t in texts)

def main():
    model = SentenceTransformer(MODEL_NAME)
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)

    doc_ids = meta["doc_ids"]
    texts = meta["texts"]
    use_cosine = meta.get("use_cosine", True)

    # (H_norm, best_k_needed) pairs
    pairs = []
    total = 0

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, desc="Calibrating")):
            if N_SAMPLES is not None and i >= N_SAMPLES:
                break
            item = json.loads(line)
            q = item["question"]
            ans = item["answer"]

            q_emb = model.encode([q], convert_to_numpy=True, normalize_embeddings=use_cosine).astype("float32")
            scores, idxs = index.search(q_emb, K_MAX)
            scores = scores[0]
            idxs = idxs[0]
            valid = idxs >= 0
            scores = scores[valid]
            idxs = idxs[valid]
            if len(idxs) == 0:
                continue

            H, Hn = entropy(scores, TEMPERATURE)

            # find smallest k that retrieves answer (proxy)
            top_texts = [texts[j] for j in idxs]
            best_k = None
            for k in range(1, len(top_texts) + 1):
                if contains_answer(top_texts[:k], ans):
                    best_k = k
                    break
            if best_k is None:
                # never found in top-K_MAX (treat as hard)
                best_k = len(top_texts)

            pairs.append((Hn, best_k))
            total += 1

    pairs = np.array(pairs, dtype=np.float32)
    Hn = pairs[:, 0]
    Kneed = pairs[:, 1]

    print(f"\nSamples used: {total}")
    print(f"H_norm: mean={Hn.mean():.3f}, std={Hn.std():.3f}")
    print(f"k_needed: mean={Kneed.mean():.2f}")

    # Simple thresholding by quantiles (3 buckets example)
    qs = [0.25, 0.50, 0.75]
    thr = np.quantile(Hn, qs).tolist()
    print("\nSuggested H_norm thresholds (quantiles):")
    for qv, tv in zip(qs, thr):
        print(f"  q={qv:.2f} -> H_norm={tv:.3f}")

    # Show mean k_needed per bucket
    buckets = [(-1, thr[0]), (thr[0], thr[1]), (thr[1], thr[2]), (thr[2], 1.1)]
    for bi, (lo, hi) in enumerate(buckets):
        mask = (Hn > lo) & (Hn <= hi)
        if mask.sum() == 0:
            continue
        print(f"Bucket {bi+1} ({lo:.3f},{hi:.3f}] count={mask.sum()}  mean k_needed={Kneed[mask].mean():.2f}")

if __name__ == "__main__":
    main()
