"""Hybrid retrieval: BM25 + multilingual dense embeddings, fused with Reciprocal Rank
Fusion, optionally re-ranked by a cross-encoder. The cross-encoder score is also used by
the agent router to decide when to abstain."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .corpus import Chunk

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
STOPWORDS = set("""a an the of for to in on at by with and or is are was were be been what which who whom how
when where why do does did should can could would will shall may might i my me we our you your it its this that
these those from as per much many about into than then there their them they he she his her not no yes if please
tell give use used using get""".split())


def tokenize(text: str) -> list[str]:
    toks = _TOKEN.findall((text or "").lower())
    return [t for t in toks if t not in STOPWORDS]


@dataclass
class Hit:
    chunk: Chunk
    score: float          # score of the final stage
    rerank_score: float | None = None

    def to_dict(self) -> dict:
        return {"chunk_id": self.chunk.chunk_id, "doc_id": self.chunk.doc_id, "title": self.chunk.title,
                "section": self.chunk.section, "score": round(float(self.score), 4),
                "rerank_score": None if self.rerank_score is None else round(float(self.rerank_score), 4),
                "source_url": self.chunk.source_url, "text": self.chunk.text}


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [tokenize(f"{c.title} {c.section} {c.text}") for c in chunks]
        self.avgdl = sum(map(len, self.docs)) / max(1, len(self.docs))
        df: dict[str, int] = {}
        for d in self.docs:
            for t in set(d):
                df[t] = df.get(t, 0) + 1
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = []
        for d in self.docs:
            tf: dict[str, int] = {}
            for t in d:
                tf[t] = tf.get(t, 0) + 1
            self.tf.append(tf)

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        q = tokenize(query)
        scores = []
        for i, tf in enumerate(self.tf):
            dl = len(self.docs[i])
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            scores.append((i, s))
        scores.sort(key=lambda x: -x[1])
        return scores[:k]


class DenseIndex:
    """multilingual-e5 embeddings (needs 'query: ' / 'passage: ' prefixes)."""

    def __init__(self, chunks: list[Chunk], model_name: str, device: str | None = None):
        from sentence_transformers import SentenceTransformer
        import numpy as np
        self.np = np
        self.chunks = chunks
        self.model = SentenceTransformer(model_name, device=device)
        passages = [f"passage: {c.title}. {c.section}. {c.text}" for c in chunks]
        self.emb = self.model.encode(passages, batch_size=32, normalize_embeddings=True, show_progress_bar=False)

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        q = self.model.encode([f"query: {query}"], normalize_embeddings=True, show_progress_bar=False)[0]
        sims = self.emb @ q
        idx = self.np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in idx]


class Reranker:
    def __init__(self, model_name: str, device: str | None = None):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name, device=device, max_length=512)

    def score(self, query: str, chunks: list[Chunk]) -> list[float]:
        if not chunks:
            return []
        pairs = [(query, f"{c.title}. {c.section}. {c.text}") for c in chunks]
        return [float(s) for s in self.model.predict(pairs, show_progress_bar=False)]


class HybridRetriever:
    MODES = ("bm25", "dense", "hybrid", "hybrid_rerank")

    def __init__(self, chunks: list[Chunk], embedder: str | None = None, reranker: str | None = None,
                 device: str | None = None, candidate_k: int = 10, rrf_k: int = 60):
        self.chunks = chunks
        self.bm25 = BM25Index(chunks)
        self.dense = DenseIndex(chunks, embedder, device) if embedder else None
        self.reranker = Reranker(reranker, device) if reranker else None
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k

    def available_modes(self) -> list[str]:
        modes = ["bm25"]
        if self.dense:
            modes += ["dense", "hybrid"]
            if self.reranker:
                modes.append("hybrid_rerank")
        return modes

    def _rrf(self, ranked_lists: list[list[tuple[int, float]]]) -> list[tuple[int, float]]:
        agg: dict[int, float] = {}
        for lst in ranked_lists:
            for rank, (i, _) in enumerate(lst):
                agg[i] = agg.get(i, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        return sorted(agg.items(), key=lambda x: -x[1])

    def search(self, query: str, top_k: int = 3, mode: str = "hybrid_rerank",
               dense_query: str | None = None) -> list[Hit]:
        """`dense_query` lets the dense retriever see a different (e.g. original-language) query."""
        if mode not in self.available_modes():
            mode = self.available_modes()[-1]
        k = self.candidate_k
        if mode == "bm25":
            ranked = self.bm25.search(query, k)
        elif mode == "dense":
            ranked = self.dense.search(dense_query or query, k)
        else:
            ranked = self._rrf([self.bm25.search(query, k), self.dense.search(dense_query or query, k)])[:k]
        hits = [Hit(self.chunks[i], s) for i, s in ranked]
        if mode == "hybrid_rerank" and hits:
            scores = self.reranker.score(query, [h.chunk for h in hits])
            for h, s in zip(hits, scores):
                h.rerank_score = s
                h.score = s
            hits.sort(key=lambda h: -h.score)
        return hits[:top_k]

    def relevance(self, query: str, hits: list[Hit]) -> float | None:
        """Best cross-encoder score among hits (computed if the mode had no reranker)."""
        if not hits or not self.reranker:
            return None
        if hits[0].rerank_score is not None:
            return max(h.rerank_score for h in hits)
        return max(self.reranker.score(query, [h.chunk for h in hits]))
