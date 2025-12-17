# src/simulator.py
"""
RAGCache (L1) + Layer2 semantic reuse Simulator
===============================================
Core simulation architecture is:
1. Retrieval Entropy (How many docs do we actually need?)
2. L1 Cache (Have we seen this sequence of docs before?)
3. L2 Cache (Is this query semantically similar to a previous one?)
"""
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any

# --------------------
# 1. ENTROPY (Shannon)
# --------------------
def softmax(x: np.ndarray, tau: float) -> np.ndarray:
    # P(i) = exp(score_i / tau) / sum(exp(score_j / tau))
    x = np.asarray(x, dtype=np.float32)
    tau = max(float(tau), 1e-8)
    x = x / tau
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / (np.sum(ex) + 1e-12)

def compute_entropy_norm(scores: np.ndarray, tau: float = 0.1, higher_is_better: bool = True) -> float:
    # cosine/IP: higher is better -> logits = scores
    # L2 distance: lower is better -> logits = -scores
    s = np.asarray(scores, dtype=np.float32)
    if s.size == 0: return 0.0

    logits = s if higher_is_better else (-s)
    p = softmax(logits, tau)
    p = np.clip(p, 1e-12, 1.0)
    H = float(-np.sum(p * np.log(p)))

    # Normalize to [0,1] by dividing with log(k)
    H_norm = float(H / np.log(len(p))) if len(p) > 1 else 0.0
    return H_norm

def adaptive_k(H_norm: float,
               thr: Tuple[float, float, float] = (0.880, 0.929, 0.963),
               ks: Tuple[int, int, int, int] = (3, 5, 7, 8)) -> int:
    # Thresholds derived from "src/calibrate_entropy.py" quantiles
    t1, t2, t3 = thr
    if H_norm <= t1: return ks[0]
    if H_norm <= t2: return ks[1]
    if H_norm <= t3: return ks[2]
    return ks[3]

def approx_tokens(t: str) -> int:
    return max(1, len(t.split()))


# ----------------------------------------
# 2. L1 CACHE SIMULATOR (RAGCache Logic)
# ----------------------------------------
@dataclass
class TreeEntry:
    token_sum: int
    last_access_ts: float = field(default_factory=time.time)

class KVCacheSim:
    """
    Simulates RAGCache (Prefix Caching).
    Tracks exact sequences of document IDs.
    """
    def __init__(self, compute_ms_per_token: float = 0.05):
        self.tree: Dict[Tuple[int, ...], TreeEntry] = {}
        self.compute_ms = float(compute_ms_per_token)
        self.hits = 0
        self.misses = 0
    
    # Finds length of the longest cached prefix.
    def longest_prefix_hit(self, doc_ids: List[int]) -> int:
        for i in range(len(doc_ids), 0, -1):
            if tuple(doc_ids[:i]) in self.tree:
                return i
        return 0
    
    # Calculates the Time-To-First-Token (Latency) by (Tokens_To_Compute * ms_per_token)
    def prefill_ttft_ms(self, doc_ids: List[int], token_sizes: List[int], prompt_tokens: int = 32):
        total_tokens = int(prompt_tokens + sum(token_sizes))
        
        # 1. Baseline cost
        base_cost = total_tokens * self.compute_ms

        # 2. Check cache
        hit_len = self.longest_prefix_hit(doc_ids)

        # 3. Calculate final cost
        cached_tokens = int(sum(token_sizes[:hit_len]))
        remaining_tokens = total_tokens - cached_tokens
        final_cost = remaining_tokens * self.compute_ms

        if hit_len > 0:
            self.hits += 1
            where = "L1"
        else:
            self.misses += 1
            where = "NONE"
        
        meta = {"where": where, "cached_tokens": cached_tokens, "hit_len_docs": hit_len}
        return base_cost, final_cost, meta

    # Saves the current sequence into the Knowledge Tree.
    def insert_path(self, doc_ids: List[int], token_sizes: List[int]):
        running_key = []
        token_sum = 0
        for d, sz in zip(doc_ids, token_sizes):
            running_key.append(d)
            token_sum += sz
            key = tuple(running_key)
            if key not in self.tree:
                self.tree[key] = TreeEntry(token_sum=token_sum)
            self.tree[key].last_access_ts = time.time()

    def stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        return {
            "l1_hits": self.hits,
            "l1_misses": self.misses,
            "l1_hit_rate": (self.hits / total) if total else 0.0,
            "l1_num_cached_prefixes": len(self.tree),
        }


# ----------------------------------------
# 3. L2 CACHE SIMULATOR (Semantic Reuse)
# ----------------------------------------
class Layer2KVAdapter:
    """
    Simulates Semantic Reuse.
    If current retrieved-doc context is similar to a previous one, we assume we can
    reuse a fraction (reuse_ratio) of the document-prefill KV states.
    That means we skip reuse_ratio * sum(token_sizes) tokens worth of prefill compute.
    """
    def __init__(self, model, sim_threshold: float = 0.85, reuse_ratio: float = 0.6):
        self.model = model
        self.sim_th = float(sim_threshold)
        self.reuse_ratio = float(reuse_ratio)
        self.protos: List[np.ndarray] = [] # Normalized vectors
        self.accepts = 0
        self.rejects = 0
    
    def _encode_norm(self, docs_text: List[str]) -> np.ndarray:
        # Encode each document separately
        embs = self.model.encode(
            docs_text,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")
        v = embs.mean(axis=0)
        v = v / (np.linalg.norm(v) + 1e-12)
        return v
    
    def try_reuse(self, docs_text: List[str], token_sizes: List[int]) -> Tuple[bool, Dict[str, Any]]:
        if not self.protos:
            self.rejects += 1
            return False, {"reason": "no_history", "sim": None, "skip_tokens": 0}
        
        # Encode the current document set context and compare
        q = self._encode_norm(docs_text)
        best_sim = float(max(np.dot(q, p) for p in self.protos))
        
        # Decision
        if best_sim >= self.sim_th:
            self.accepts += 1
            doc_tokens = int(sum(token_sizes))
            skip_tokens = int(self.reuse_ratio * doc_tokens)
            return True, {"reason": "accepted", "sim": best_sim, "skip_tokens": skip_tokens}
        else:
            self.rejects += 1
            return False, {"reason": "low_sim", "sim": best_sim, "skip_tokens": 0}

    def update_proto(self, docs_text: List[str]):
        self.protos.append(self._encode_norm(docs_text))

    def stats(self) -> Dict[str, Any]:
        total = self.accepts + self.rejects
        return {
            "l2_accepts": self.accepts,
            "l2_rejects": self.rejects,
            "l2_accept_rate": (self.accepts / total) if total else 0.0,
            "l2_num_protos": len(self.protos),
            "reuse_ratio": self.reuse_ratio,
            "sim_threshold": self.sim_th,
        }
