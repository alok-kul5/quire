from __future__ import annotations

from pathlib import Path

import pytest

from quire.ingest import parse_pep
from quire.retrieve import Hit

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def chunks():
    return parse_pep(FIXTURES / "pep-9999.rst")


class FakeRetriever:
    """Keyword-overlap retriever over the fixture chunks; same interface as quire.retrieve.Retriever."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.timings = {}

    def search(self, query, mode="rerank", k=6, candidates=40):
        q = set(query.lower().split())
        scored = sorted(self.chunks, key=lambda c: -len(q & set(c.index_text.lower().split())))
        hits = []
        for i, c in enumerate(scored[:k]):
            overlap = len(q & set(c.index_text.lower().split()))
            ranks = {"bm25": i + 1, "dense": i + 1, "fused": i + 1}
            if mode == "rerank":
                ranks["rerank"] = i + 1
            hits.append(Hit(c, float(overlap) if overlap else -10.0, ranks))
        self.timings = {"bm25_ms": 1.0, "dense_ms": 2.0, "rerank_ms": 3.0}
        return hits


@pytest.fixture
def retriever(chunks):
    return FakeRetriever(chunks)
