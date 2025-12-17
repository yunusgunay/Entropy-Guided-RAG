# src/query.py
"""
L1: Who was the man behind The Chipmunks?
Entropy H_norm = 0.986 → adaptive k=8
Entropy OFF: path=L1   | k=8 | retr=13.20 ms | TTFT gain=89.71% | acc=1 | L2 not checked
Entropy ON: path=L1   | k=8 | retr=9.11 ms | TTFT gain=89.71% | acc=1 | L2 not checked

L2: Which person created the Chipmunks?
Entropy H_norm = 0.973 → adaptive k=8
Entropy OFF: path=L2   | k=8 | retr=13.43 ms | TTFT gain=54.04% | acc=1 | L2 HIT (sim=0.969, skip_tokens=174)
Entropy ON: path=L2   | k=8 | retr=9.02 ms | TTFT gain=54.04% | acc=1 | L2 HIT (sim=0.969, skip_tokens=174)
"""
import os, json, time, pickle
import numpy as np
from tqdm import tqdm
import faiss
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import fuzz
from simulator import (compute_entropy_norm, adaptive_k, approx_tokens, KVCacheSim, Layer2KVAdapter)

# Configurations
INDEX_PATH = "data/processed/embeddings.faiss"
META_PATH = "data/processed/metadata.pkl"
DATA_PATH = "data/processed/triviaqa_clean.jsonl"
CACHE_FILE = "data/cache/cache_state.pkl"

BOOTSTRAP_N = 10
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K_MAX = 8
TAU = 0.1
FUZZY_THRESHOLD = 80
L2_SIM_TH = 0.83
L2_REUSE_RATIO = 0.6

def percent_gain(base, final):
    return 0.0 if base <= 1e-9 else 100.0 * (base - final) / base

def compute_accuracy(answer, docs):
    if not answer:
        return None
    a = answer.lower()
    for d in docs:
        if fuzz.partial_ratio(a, d.lower()) >= FUZZY_THRESHOLD:
            return 1
    return 0

# Bootstrap cache (L1+L2) for both situations: Entropy OFF (k=8) and Entropy ON (adaptive k)
def bootstrap_caches(model, index, passages, data):
    kv_off, kv_on = KVCacheSim(), KVCacheSim()
    l2_off = Layer2KVAdapter(model, sim_threshold=L2_SIM_TH, reuse_ratio=L2_REUSE_RATIO)
    l2_on = Layer2KVAdapter(model, sim_threshold=L2_SIM_TH, reuse_ratio=L2_REUSE_RATIO)

    for q, _ in tqdm(data, desc="Warming caches"):
        q_emb = model.encode([q], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        scores, ids = index.search(q_emb, TOP_K_MAX)
        scores = scores[0]
        ids = ids[0]

        # Entropy for IP/cosine: higher_is_better = True
        Hn = compute_entropy_norm(scores, tau=TAU, higher_is_better=True)
        k = adaptive_k(Hn)

        # Entropy OFF world always uses k=8
        doc_ids_off = ids[:TOP_K_MAX].tolist()
        docs_off = [passages[i] for i in doc_ids_off]
        sizes_off = [approx_tokens(d) for d in docs_off]
        kv_off.prefill_ttft_ms(doc_ids_off, sizes_off)
        kv_off.insert_path(doc_ids_off, sizes_off)
        l2_off.update_proto(docs_off)

        # Entropy ON world uses adaptive k
        doc_ids_on = ids[:k].tolist()
        docs_on = [passages[i] for i in doc_ids_on]
        sizes_on = [approx_tokens(d) for d in docs_on]
        kv_on.prefill_ttft_ms(doc_ids_on, sizes_on)
        kv_on.insert_path(doc_ids_on, sizes_on)
        l2_on.update_proto(docs_on)

    return kv_off, kv_on, l2_off, l2_on


# Load model, FAISS, data
print("Loading model and FAISS index...")
model = SentenceTransformer(MODEL_NAME)
index = faiss.read_index(INDEX_PATH)

with open(META_PATH, "rb") as f:
    meta = pickle.load(f)

passages = meta.get("passages", meta["texts"])

data = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= BOOTSTRAP_N:
            break
        item = json.loads(line)
        q = item["question"]
        a = item.get("answer", "")
        data.append((q, a))


# Load / Warmup
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "rb") as f:
        state = pickle.load(f)
    kv_off = state["kv_off"]; kv_on = state["kv_on"]
    l2_off = state["l2_off"]; l2_on = state["l2_on"]
    print(f"Loaded caches from {CACHE_FILE}")
else:
    kv_off, kv_on, l2_off, l2_on = bootstrap_caches(model, index, passages, data)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(
            dict(kv_off=kv_off, kv_on=kv_on, l2_off=l2_off, l2_on=l2_on),
            f
        )
    print(f"Bootstrapped caches and saved to {CACHE_FILE}")


# Dataset embeddings for nearest answer (accuracy proxy)
ds_qs, ds_as = zip(*data)
ds_emb = model.encode(list(ds_qs), convert_to_numpy=True, normalize_embeddings=True).astype("float32")

def nearest_dataset_answer(q, sim_th=0.8):
    q_emb = model.encode([q], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    sims = util.cos_sim(q_emb, ds_emb)[0].cpu().numpy()
    j = int(np.argmax(sims))
    return ds_as[j] if sims[j] >= sim_th else None


# Retrieval runner (one world)
def run_world(q, use_entropy, kv, l2):
    q_emb = model.encode([q], convert_to_numpy=True, normalize_embeddings=True).astype("float32")

    t0 = time.time()
    scores, ids = index.search(q_emb, TOP_K_MAX)
    retr_ms = (time.time() - t0) * 1000.0

    scores = scores[0]
    ids = ids[0]

    if use_entropy:
        Hn = compute_entropy_norm(scores, tau=TAU, higher_is_better=True)
        k = adaptive_k(Hn)
    else:
        Hn = np.nan
        k = TOP_K_MAX

    doc_ids = ids[:k].tolist()
    docs = [passages[i] for i in doc_ids]
    sizes = [approx_tokens(d) for d in docs]

    base, l1_ms, meta_l1 = kv.prefill_ttft_ms(doc_ids, sizes)
    final = l1_ms
    where = meta_l1["where"]

    l2_meta = {"reason": "not_checked", "sim": None, "skip_tokens": 0}
    if where == "NONE":
        ok, l2_meta = l2.try_reuse(docs, sizes)
        if ok:
            # L2 means: skip part of doc-prefill compute
            final = max(0.0, final - l2_meta["skip_tokens"] * kv.compute_ms)
            where = "L2"

    acc = compute_accuracy(nearest_dataset_answer(q), docs)
    return dict(
        Hn=Hn, k=k, retr_ms=retr_ms,
        base=base, final=final, where=where,
        acc=acc, l2_meta=l2_meta
    )


# Interactive Query Loop
print("System ready. Type 'exit' to quit.\n")
while True:
    q = input("Your question: ").strip()
    if not q or q.lower() in ("exit", "quit", ":q"):
        break

    # Show entropy for info
    q_emb = model.encode([q], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    tmp_s, _ = index.search(q_emb, TOP_K_MAX)
    Hn_info = compute_entropy_norm(tmp_s[0], tau=TAU, higher_is_better=True)
    k_info = adaptive_k(Hn_info)
    print(f"\nEntropy H_norm = {Hn_info:.3f} → adaptive k={k_info}")

    off = run_world(q, use_entropy=False, kv=kv_off, l2=l2_off)
    on = run_world(q, use_entropy=True,  kv=kv_on,  l2=l2_on)

    g_off = percent_gain(off["base"], off["final"])
    g_on = percent_gain(on["base"], on["final"])

    def fmt_acc(v): return "NA" if v is None else str(v)

    def fmt_l2(x):
        m = x["l2_meta"]
        if x["where"] == "L2":
            return f"L2 HIT (sim={m['sim']:.3f}, skip_tokens={m['skip_tokens']})"
        if x["where"] == "NONE":
            return f"L2 MISS (reason={m['reason']}, sim={m['sim']})"
        return "L2 not checked"

    print(f"Entropy OFF: path={off['where']:<4} | k=8 | retr={off['retr_ms']:.2f} ms | TTFT gain={g_off:.2f}% | acc={fmt_acc(off['acc'])} | {fmt_l2(off)}")
    print(f"Entropy ON: path={on['where']:<4} | k={on['k']} | retr={on['retr_ms']:.2f} ms | TTFT gain={g_on:.2f}% | acc={fmt_acc(on['acc'])} | {fmt_l2(on)}\n")

    # Online adaptation: update both caches with their own chosen k
    q_emb = model.encode([q], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q_emb, TOP_K_MAX)
    ids = ids[0].tolist()
    docs8 = [passages[i] for i in ids]
    sizes8 = [approx_tokens(d) for d in docs8]
    kv_off.insert_path(ids, sizes8)
    l2_off.update_proto(docs8)

    # Entropy ON world stores adaptive-k
    docsK = docs8[:k_info]
    sizesK = sizes8[:k_info]
    kv_on.insert_path(ids[:k_info], sizesK)
    l2_on.update_proto(docsK)
