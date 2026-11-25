"""Hybrid retrieval: BM25 + dense (bge-small in LanceDB), fused with RRF, then a cross-encoder rerank."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

from . import config
from .ingest import Chunk, load_corpus

Mode = Literal["bm25", "dense", "hybrid", "rerank"]
MODES: tuple[Mode, ...] = ("bm25", "dense", "hybrid", "rerank")


@dataclass
class Hit:
    chunk: Chunk
    score: float
    ranks: dict[str, int] = field(default_factory=dict)  # 1-based rank in each stage

    def to_dict(self) -> dict:
        c = self.chunk
        return {
            "id": c.id,
            "pep": c.pep,
            "title": c.title,
            "status": c.status,
            "section": c.section,
            "url": c.url,
            "text": c.text,
            "score": round(float(self.score), 4),
            "ranks": self.ranks,
        }


def rrf(rankings: list[list[int]], k: int = config.RRF_K) -> list[tuple[int, float]]:
    """Reciprocal rank fusion over lists of row ids. Returns (row, score) best first."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, row in enumerate(ranking):
            scores[row] = scores.get(row, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


class Reranker(Protocol):
    def predict(self, pairs: list[tuple[str, str]], batch_size: int = ...) -> list[float]: ...


class Retriever:
    def __init__(self, index_dir: Path = config.INDEX_DIR, load_reranker: bool = True):
        import bm25s
        import lancedb
        from sentence_transformers import CrossEncoder, SentenceTransformer

        self.chunks: list[Chunk] = load_corpus(index_dir / "chunks.jsonl")
        self.bm25 = bm25s.BM25.load(str(index_dir / "bm25"))
        self.table = lancedb.connect(str(index_dir / "lancedb")).open_table("chunks")
        self.embed = SentenceTransformer(config.EMBED_MODEL)
        self.reranker: Reranker | None = CrossEncoder(config.RERANK_MODEL) if load_reranker else None
        self.timings: dict[str, float] = {}

    # --- single stages -------------------------------------------------
    def bm25_rows(self, query: str, n: int) -> list[int]:
        import bm25s

        tokens = bm25s.tokenize([query], stopwords="en", show_progress=False)
        docs, scores = self.bm25.retrieve(tokens, k=n, show_progress=False)
        return [int(d) for d, s in zip(docs[0], scores[0], strict=False) if s > 0]

    def dense_rows(self, query: str, n: int) -> list[int]:
        vec = self.embed.encode(config.EMBED_QUERY_PREFIX + query, normalize_embeddings=True)
        res = self.table.search(vec.tolist()).metric("cosine").limit(n).select(["row"]).to_list()
        return [int(r["row"]) for r in res]

    # --- full pipeline ---------------------------------------------------
    def search(self, query: str, mode: Mode = "rerank", k: int = config.TOP_K, candidates: int = config.CANDIDATES) -> list[Hit]:
        t = {}
        ranks: dict[int, dict[str, int]] = {}

        def note(stage: str, rows: list[int]):
            for i, r in enumerate(rows):
                ranks.setdefault(r, {})[stage] = i + 1

        s = time.perf_counter()
        bm = self.bm25_rows(query, candidates) if mode != "dense" else []
        t["bm25_ms"] = (time.perf_counter() - s) * 1000
        note("bm25", bm)
        s = time.perf_counter()
        dn = self.dense_rows(query, candidates) if mode != "bm25" else []
        t["dense_ms"] = (time.perf_counter() - s) * 1000
        note("dense", dn)

        if mode == "bm25":
            fused = [(r, 1.0 / (i + 1)) for i, r in enumerate(bm)]
        elif mode == "dense":
            fused = [(r, 1.0 / (i + 1)) for i, r in enumerate(dn)]
        else:
            fused = rrf([bm, dn])
            note("fused", [r for r, _ in fused])

        if mode == "rerank" and self.reranker is not None and fused:
            pool = [r for r, _ in fused[: candidates]]
            s = time.perf_counter()
            scores = self.reranker.predict([(query, self.chunks[r].index_text) for r in pool], batch_size=32)
            t["rerank_ms"] = (time.perf_counter() - s) * 1000
            fused = sorted(zip(pool, (float(x) for x in scores), strict=False), key=lambda kv: -kv[1])
            note("rerank", [r for r, _ in fused])

        self.timings = {key: round(v, 1) for key, v in t.items()}
        return [Hit(self.chunks[r], score, ranks.get(r, {})) for r, score in fused[:k]]
