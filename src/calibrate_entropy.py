# src/calibrate_entropy.py
# Calibrate entropy thresholds for retrieval confidence estimation
"""
Output:
Samples used: 11260
H_norm: mean=0.907, std=0.084
k_needed: mean=4.54

Suggested H_norm thresholds (quantiles):
q=0.25 -> H_norm=0.880
q=0.50 -> H_norm=0.929
q=0.75 -> H_norm=0.963
Bucket 1 (-1.000,0.880] count=2815 mean k_needed=4.56
Bucket 2 (0.880,0.929] count=2815 mean k_needed=4.38
Bucket 3 (0.929,0.963] count=2815 mean k_needed=4.30
Bucket 4 (0.963,1.100] count=2815 mean k_needed=4.92
"""
import json
import pickle
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

DATA_PATH = "data/processed/triviaqa_clean.jsonl" # Dataset (11260 samples)
INDEX_PATH = "data/processed/embeddings.faiss" # FAISS index of passage embeddings
META_PATH = "data/processed/metadata.pkl"

MODEL_NAME = "all-MiniLM-L6-v2" # Sentence-Transformer

K_MAX = 8
TEMPERATURE = 0.1 # Softmax temperature
N_SAMPLES = None # None for full dataset

def softmax(x: np.ndarray, tau: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = x / max(float(tau), 1e-8)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / (np.sum(ex) + 1e-12)

def entropy_from_scores(scores: np.ndarray, tau: float) -> tuple[float, float]:
    # For cosine/IP, higher score = better, so use scores directly as logits
    p = softmax(scores, tau)
    p = np.clip(p, 1e-12, 1.0)
    H = float(-np.sum(p * np.log(p)))
    H_norm = float(H / np.log(len(p))) if len(p) > 1 else 0.0
    return H, H_norm

def contains_answer(texts: list[str], ans: str) -> bool:
    a = (ans or "").lower().strip()
    if not a:
        return False
    return any(a in (t or "").lower() for t in texts)

def main():
    model = SentenceTransformer(MODEL_NAME)
    index = faiss.read_index(INDEX_PATH)

    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)

    texts = meta.get("passages", meta.get("texts"))
    if texts is None:
        raise KeyError("metadata.pkl must contain 'texts' or 'passages'")
    
    use_cosine = bool(meta.get("use_cosine", True))

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
            if not q:
                continue

            q_emb = model.encode(
                [q],
                convert_to_numpy=True,
                normalize_embeddings=use_cosine
            ).astype("float32")

            scores, idxs = index.search(q_emb, K_MAX)
            scores = scores[0]
            idxs = idxs[0]

            valid = idxs >= 0
            scores = scores[valid]
            idxs = idxs[valid]
            if idxs.size == 0:
                continue

            H, Hn = entropy_from_scores(scores, TEMPERATURE)

            # Find smallest k that retrieves answer (proxy)
            top_texts = [texts[int(j)] for j in idxs]
            best_k = None
            for k in range(1, len(top_texts) + 1):
                if contains_answer(top_texts[:k], ans):
                    best_k = k
                    break
            if best_k is None:
                best_k = len(top_texts)

            pairs.append((Hn, float(best_k)))
            total += 1

    pairs = np.asarray(pairs, dtype=np.float32)
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
        print(f"q={qv:.2f} -> H_norm={tv:.3f}")

    # Show mean k_needed per bucket
    buckets = [(-1, thr[0]), (thr[0], thr[1]), (thr[1], thr[2]), (thr[2], 1.1)]
    for bi, (lo, hi) in enumerate(buckets):
        mask = (Hn > lo) & (Hn <= hi)
        if mask.sum() == 0:
            continue
        print(
            f"Bucket {bi+1} ({lo:.3f},{hi:.3f}] "
            f"count={int(mask.sum())} mean k_needed={Kneed[mask].mean():.2f}"
        )


if __name__ == "__main__":
    main()
