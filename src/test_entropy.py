# src/test_entropy.py
"""
Output:
--- Entropy OFF (fixed k=8) ---
Accuracy proxy: 7005/11260 = 62.21%
Avg retrieved tokens: 280.86
Avg embed ms: 9.945
Avg FAISS retr ms: 9.747

--- Entropy ON (adaptive k) ---
Accuracy proxy: 6612/11260 = 58.72%
Avg k: 6.00
Avg retrieved tokens: 210.78
Avg embed ms: 9.945
Avg FAISS retr ms: 9.747

--- Comparison ---
Accuracy delta (pp): -3.49
Token reduction% (Gain/Loss): +24.95%

--- Adaptive-k distribution ---
k= 4: count=  2816 hit-rate= 53.73%
k= 5: count=  2823 hit-rate= 60.33%
k= 7: count=  2808 hit-rate= 64.71%
k= 8: count=  2813 hit-rate= 56.13%
"""
import argparse
import json
import pickle
import time
import numpy as np
import faiss
from collections import defaultdict
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Paths
DATA_PATH = "data/processed/triviaqa_clean.jsonl"
INDEX_PATH = "data/processed/embeddings.faiss"
META_PATH = "data/processed/metadata.pkl"

MODEL_NAME = "all-MiniLM-L6-v2"
K_MAX = 8
TAU = 0.1
DEFAULT_THR = (0.880, 0.929, 0.963)
DEFAULT_KS = (4, 5, 7, 8)

def softmax(x: np.ndarray, tau: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    tau = max(float(tau), 1e-8)
    x = x / tau
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / (np.sum(ex) + 1e-12)

def compute_entropy_norm(scores: np.ndarray, tau: float = TAU, higher_is_better: bool = True) -> float:
    s = np.asarray(scores, dtype=np.float32)
    if s.size == 0:
        return 0.0
    logits = s if higher_is_better else (-s)
    p = softmax(logits, tau)
    p = np.clip(p, 1e-12, 1.0)
    H = float(-np.sum(p * np.log(p)))
    return float(H / np.log(len(p))) if len(p) > 1 else 0.0

def adaptive_k(H_norm: float, thr=DEFAULT_THR, ks=DEFAULT_KS) -> int:
    t1, t2, t3 = thr
    if H_norm <= t1: return int(ks[0])
    if H_norm <= t2: return int(ks[1])
    if H_norm <= t3: return int(ks[2])
    return int(ks[3])

def approx_tokens(text: str) -> int:
    return max(1, len((text or "").split()))

def contains_answer_in_docs(docs_text: list[str], ans: str) -> bool:
    a = (ans or "").lower().strip()
    if not a:
        return False
    return any(a in (t or "").lower() for t in docs_text)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit number of dataset questions (default: full).")
    args = ap.parse_args()

    thr = DEFAULT_THR
    ks = DEFAULT_KS
    kmax = K_MAX
    tau = TAU

    print("Loading model / index / metadata...")
    model = SentenceTransformer(MODEL_NAME)
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    passages = meta.get("passages", meta.get("texts"))
    if passages is None:
        raise KeyError("metadata.pkl must contain 'texts' or 'passages'")

    n = 0
    # OFF (fixed k=8)
    off_hits = 0
    off_tokens = 0
    off_embed_ms = 0.0
    off_retr_ms = 0.0
    # ON (adaptive k)
    on_hits = 0
    on_tokens = 0
    on_embed_ms = 0.0
    on_retr_ms = 0.0
    on_k_sum = 0
    H_list = []

    # Bucket stats for ON
    bucket_counts = defaultdict(int)
    bucket_hit_sum = defaultdict(int)

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            if args.limit is not None and n >= args.limit:
                break
            item = json.loads(line)
            q = item.get("question", "")
            ans = item.get("answer", "")
            if not q or not ans:
                continue

            # 1. Encode timing
            t0 = time.time()
            q_emb = model.encode([q], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
            embed_ms = (time.time() - t0) * 1000.0

            # 2. FAISS retrieval timing (single search used for both OFF/ON)
            t1 = time.time()
            scores, ids = index.search(q_emb, kmax)
            retr_ms = (time.time() - t1) * 1000.0

            scores = scores[0]
            ids = ids[0]
            valid = ids >= 0
            scores = scores[valid]
            ids = ids[valid]
            if ids.size == 0:
                continue

            # 3. Entropy + adaptive k
            Hn = compute_entropy_norm(scores, tau=tau, higher_is_better=True)
            k_adapt = adaptive_k(Hn, thr=thr, ks=ks)

            # Entropy OFF: fixed kmax
            ids_off = ids[:kmax].tolist()
            docs_off = [passages[int(i)] for i in ids_off]
            tok_off = sum(approx_tokens(t) for t in docs_off)
            hit_off = 1 if contains_answer_in_docs(docs_off, ans) else 0

            # Entropy ON: adaptive k
            ids_on = ids[:k_adapt].tolist()
            docs_on = [passages[int(i)] for i in ids_on]
            tok_on = sum(approx_tokens(t) for t in docs_on)
            hit_on = 1 if contains_answer_in_docs(docs_on, ans) else 0

            # Aggregates
            n += 1
            off_hits += hit_off
            off_tokens += tok_off
            off_embed_ms += embed_ms
            off_retr_ms += retr_ms
            on_hits += hit_on
            on_tokens += tok_on
            on_embed_ms += embed_ms
            on_retr_ms += retr_ms
            on_k_sum += k_adapt
            H_list.append(Hn)

            # Bucket logging
            bucket_key = k_adapt
            bucket_counts[bucket_key] += 1
            bucket_hit_sum[bucket_key] += hit_on

    if n == 0:
        raise RuntimeError("No valid samples processed. Check DATA_PATH format.")

    def pct(x): return 100.0 * x

    # Print results
    print("\n--- Entropy OFF (fixed k=8) ---")
    print(f"Accuracy proxy: {off_hits}/{n} = {pct(off_hits/n):.2f}%")
    print(f"Avg retrieved tokens: {off_tokens/n:.2f}")
    print(f"Avg embed ms: {off_embed_ms/n:.3f}")
    print(f"Avg FAISS retr ms: {off_retr_ms/n:.3f}")

    print("\n--- Entropy ON (adaptive k) ---")
    print(f"Accuracy proxy: {on_hits}/{n} = {pct(on_hits/n):.2f}%")
    print(f"Avg k: {on_k_sum/n:.2f}")
    print(f"Avg retrieved tokens: {on_tokens/n:.2f}")
    print(f"Avg embed ms: {on_embed_ms/n:.3f}")
    print(f"Avg FAISS retr ms: {on_retr_ms/n:.3f}")

    print("\n--- Comparison ---")
    print(f"Accuracy delta (pp): {pct(on_hits/n)-pct(off_hits/n):+.2f}")
    if off_tokens > 0:
        token_red_pct = pct((off_tokens - on_tokens) / off_tokens)
    else:
        token_red_pct = 0.0
    print(f"Token reduction% (Gain/Loss): {token_red_pct:+.2f}%")

    print("\n--- Adaptive-k distribution ---")
    for k in sorted(bucket_counts.keys()):
        c = bucket_counts[k]
        hr = bucket_hit_sum[k] / c if c else 0.0
        print(f"k={k:2d}: count={c:6d} hit-rate={pct(hr):6.2f}%")


if __name__ == "__main__":
    main()
