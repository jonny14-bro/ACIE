# memory.py — ACIE Multi-Domain Memory System
#
# Upgrade over original:
#   • Multi-index FAISS: one index per domain (coding, debug, general, coding_python, …)
#   • Confidence-weighted save: only store high/medium quality exchanges
#   • Quality-ranked retrieval: returns results with a relevance score
#   • Session management unchanged (backward compatible)

import os
import json
import faiss
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer


# ── Quality thresholds ────────────────────────────────────────────────────────
QUALITY_SAVE_THRESHOLD = "medium"   # save "medium" and "high", skip "low"
_QUALITY_RANK = {"high": 2, "medium": 1, "low": 0}

# ── Default domains ───────────────────────────────────────────────────────────
DEFAULT_DOMAINS = ["coding", "debug", "general", "coding_python",
                   "coding_javascript", "coding_cpp"]


class MemorySystem:
    def __init__(self, dim: int = 384, domains: list[str] = None):
        self.dim = dim

        # ─── Paths ────────────────────────────────────────────────────────────
        self.base_dir    = "memory_store"
        self.session_dir = os.path.join(self.base_dir, "sessions")
        self.index_dir   = os.path.join(self.base_dir, "indexes")

        os.makedirs(self.session_dir, exist_ok=True)
        os.makedirs(self.index_dir,   exist_ok=True)

        # ─── Session ──────────────────────────────────────────────────────────
        self.session_id   = f"session_{int(datetime.now().timestamp())}"
        self.session_file = os.path.join(self.session_dir, f"{self.session_id}.json")
        self.history: list[dict] = []

        # ─── Embedder ────────────────────────────────────────────────────────
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # ─── Multi-domain indexes ─────────────────────────────────────────────
        self.domains = list(set((domains or []) + DEFAULT_DOMAINS))
        self.indexes: dict[str, faiss.Index]  = {}
        self.texts:   dict[str, list[dict]]   = {}   # {domain: [{text, quality, ts}]}

        for d in self.domains:
            idx, meta = self._load_domain(d)
            self.indexes[d] = idx
            self.texts[d]   = meta

        # Track which domains have been written to this session
        self._dirty: set[str] = set()

        # ─── Legacy global index (backward compat) ────────────────────────────
        self._legacy_index_file = os.path.join(self.base_dir, "global.faiss")
        self._legacy_meta_file  = os.path.join(self.base_dir, "global.meta.json")
        self.index, self.legacy_texts = self._load_legacy()

        print(f"[Memory] Session: {self.session_id}  |  "
              f"Domains: {len(self.domains)}  |  "
              f"Legacy entries: {len(self.legacy_texts)}")

    # ─────────────────────────────────────────────────────────────────────────
    # SESSION
    # ─────────────────────────────────────────────────────────────────────────

    def add_message(self, role: str, content: str):
        self.history.append({
            "role":    role,
            "content": content,
            "time":    str(datetime.now()),
        })

    def save_session(self):
        with open(self.session_file, "w") as f:
            json.dump(self.history, f, indent=2)

    def get_recent(self, k: int = 6) -> list[dict]:
        return self.history[-k:]

    # ─────────────────────────────────────────────────────────────────────────
    # DOMAIN INDEX  (multi-index memory)
    # ─────────────────────────────────────────────────────────────────────────

    def _domain_paths(self, domain: str):
        safe = domain.replace("/", "_")
        idx_path  = os.path.join(self.index_dir, f"{safe}.faiss")
        meta_path = os.path.join(self.index_dir, f"{safe}.meta.json")
        return idx_path, meta_path

    def _load_domain(self, domain: str):
        idx_path, meta_path = self._domain_paths(domain)
        if os.path.exists(idx_path):
            index = faiss.read_index(idx_path)
            meta  = json.load(open(meta_path))
        else:
            index = faiss.IndexFlatL2(self.dim)
            meta  = []
        return index, meta

    def _save_domain(self, domain: str):
        if domain not in self.indexes:
            return
        idx_path, meta_path = self._domain_paths(domain)
        faiss.write_index(self.indexes[domain], idx_path)
        with open(meta_path, "w") as f:
            json.dump(self.texts[domain], f)

    def _ensure_domain(self, domain: str):
        """Lazily create a new domain if it doesn't exist yet."""
        if domain not in self.indexes:
            self.indexes[domain] = faiss.IndexFlatL2(self.dim)
            self.texts[domain]   = []

    # ─────────────────────────────────────────────────────────────────────────
    # EMBEDDING HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def embed(self, text: str) -> np.ndarray:
        return self.embedder.encode(text).astype("float32")

    # ─────────────────────────────────────────────────────────────────────────
    # ADD / SEARCH  (domain-aware)
    # ─────────────────────────────────────────────────────────────────────────

    def add_to_domain(self, text: str, domain: str, quality: str = "medium"):
        """
        Add a text entry to a specific domain index.
        Skips low-quality entries to prevent memory pollution.
        """
        if _QUALITY_RANK.get(quality, 0) < _QUALITY_RANK[QUALITY_SAVE_THRESHOLD]:
            return   # skip low quality

        self._ensure_domain(domain)
        emb = self.embed(text)
        self.indexes[domain].add(np.array([emb]))
        self.texts[domain].append({
            "text":    text,
            "quality": quality,
            "ts":      str(datetime.now()),
        })
        self._dirty.add(domain)

    def search_domain(self, query: str, domain: str, k: int = 3) -> list[dict]:
        """
        Search a specific domain.
        Returns list of {text, quality, score} sorted by relevance.
        """
        self._ensure_domain(domain)
        entries = self.texts[domain]
        if not entries:
            return []

        q_emb = self.embed(query)
        k_actual = min(k, len(entries))
        D, I = self.indexes[domain].search(np.array([q_emb]), k_actual)

        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx < len(entries):
                entry = entries[idx].copy()
                entry["score"] = float(dist)   # lower = more similar
                results.append(entry)

        # Sort: quality bonus — high-quality answers surface first
        results.sort(key=lambda x: (x["score"] - _QUALITY_RANK.get(x["quality"], 0) * 0.3))
        return results

    def search_multi_domain(self, query: str, domains: list[str],
                            k_per_domain: int = 2) -> list[dict]:
        """Search across multiple domains, merge and re-rank."""
        all_results = []
        for d in domains:
            results = self.search_domain(query, d, k=k_per_domain)
            for r in results:
                r["domain"] = d
                all_results.append(r)
        all_results.sort(key=lambda x: x["score"])
        return all_results[:k_per_domain * len(domains)]

    def save_all_domains(self):
        for domain in self._dirty:
            self._save_domain(domain)
        self._dirty.clear()

    # ─────────────────────────────────────────────────────────────────────────
    # CONTEXT BUILDER  (domain-aware hybrid)
    # ─────────────────────────────────────────────────────────────────────────

    def build_context(self, user_input: str, domain: str = "general") -> str:
        """
        Build a rich context string for injection into the system prompt.
        Uses domain-specific memory + legacy global memory as fallback.
        """
        # Primary: domain-specific retrieval
        domain_results = self.search_domain(user_input, domain, k=3)

        # Secondary: always check general as a supplement
        general_results = []
        if domain != "general":
            general_results = self.search_domain(user_input, "general", k=2)

        # Legacy fallback
        legacy_results = self.search_global(user_input, k=2)

        recent = self.get_recent(3)
        context = ""

        # Domain memories
        relevant_texts = [r["text"] for r in domain_results]
        # Merge non-duplicates from general
        seen = set(relevant_texts)
        for r in general_results:
            if r["text"] not in seen:
                relevant_texts.append(r["text"])
                seen.add(r["text"])
        # Legacy fallback
        for t in legacy_results:
            if t not in seen:
                relevant_texts.append(t)
                seen.add(t)

        if relevant_texts:
            context += "\n[Relevant past interactions — use for accuracy]\n"
            for r in relevant_texts[:4]:
                context += f"• {r[:250]}\n"

        if recent:
            context += "\n[Current session]\n"
            for msg in recent:
                context += f"{msg['role']}: {msg['content']}\n"

        return context

    # ─────────────────────────────────────────────────────────────────────────
    # LEGACY GLOBAL MEMORY  (backward compatible)
    # ─────────────────────────────────────────────────────────────────────────

    def _load_legacy(self):
        if os.path.exists(self._legacy_index_file):
            index = faiss.read_index(self._legacy_index_file)
            texts = json.load(open(self._legacy_meta_file))
        else:
            index = faiss.IndexFlatL2(self.dim)
            texts = []
        return index, texts

    def save_global_memory(self):
        faiss.write_index(self.index, self._legacy_index_file)
        with open(self._legacy_meta_file, "w") as f:
            json.dump(self.legacy_texts, f)

    def add_global_memory(self, text: str):
        emb = self.embed(text)
        self.index.add(np.array([emb]))
        self.legacy_texts.append(text)

    def search_global(self, query: str, k: int = 2) -> list[str]:
        if not self.legacy_texts:
            return []
        q_emb = self.embed(query)
        k_actual = min(k, len(self.legacy_texts))
        D, I = self.index.search(np.array([q_emb]), k_actual)
        return [self.legacy_texts[i] for i in I[0] if i < len(self.legacy_texts)]
