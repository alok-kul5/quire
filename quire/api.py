"""FastAPI service: /api/ask, /api/search, /api/health, /api/evals, plus the web UI.

  uvicorn quire.api:app --port 8000
"""

from __future__ import annotations

import json
import time
from functools import lru_cache
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__, config
from .answer import answer

WEB = config.ROOT / "web"
RESULTS = config.ROOT / "evals" / "results"


class AskIn(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    mode: Literal["bm25", "dense", "hybrid", "rerank"] = "rerank"
    k: int = Field(default=config.TOP_K, ge=1, le=12)


class SearchIn(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    mode: Literal["bm25", "dense", "hybrid", "rerank"] = "rerank"
    k: int = Field(default=10, ge=1, le=30)


def create_app(retriever=None, llm=None) -> FastAPI:
    app = FastAPI(title="Quire", version=__version__, description="Cited answers from the Python PEPs")

    @lru_cache(maxsize=1)
    def get_retriever():
        if retriever is not None:
            return retriever
        from .retrieve import Retriever

        return Retriever()

    @lru_cache(maxsize=1)
    def get_llm():
        if llm is not None:
            return llm
        from .llm import OllamaLLM

        return OllamaLLM()

    @app.get("/api/health")
    def health():
        r = get_retriever()
        m = get_llm()
        ok = m.available() if hasattr(m, "available") else True
        return {
            "status": "ok",
            "chunks": len(r.chunks),
            "peps": len({c.pep for c in r.chunks}),
            "model": m.name,
            "llm_available": ok,
            "embed_model": config.EMBED_MODEL,
            "rerank_model": config.RERANK_MODEL,
        }

    @app.post("/api/search")
    def search(body: SearchIn):
        r = get_retriever()
        hits = r.search(body.query, mode=body.mode, k=body.k)
        return {"query": body.query, "mode": body.mode, "hits": [h.to_dict() for h in hits], "timings": r.timings}

    @app.post("/api/ask")
    def ask(body: AskIn):
        r = get_retriever()
        s = time.perf_counter()
        hits = r.search(body.question, mode=body.mode, k=body.k)
        retrieval_ms = round((time.perf_counter() - s) * 1000, 1)
        try:
            a = answer(body.question, hits, get_llm())
        except Exception as e:  # LLM down or timed out
            raise HTTPException(503, f"LLM unavailable: {e}") from e
        out = a.to_dict()
        out["mode"] = body.mode
        out["timings"] = {**r.timings, "retrieval_ms": retrieval_ms, **a.timings}
        return out

    @app.get("/api/evals")
    def evals():
        out = {}
        for name in ("retrieval", "generation"):
            p = RESULTS / f"{name}.json"
            if p.exists():
                data = json.loads(p.read_text())
                out[name] = data.get("modes") or data.get("summary")
        return out

    if WEB.exists():
        app.mount("/static", StaticFiles(directory=WEB), name="static")

        @app.get("/", include_in_schema=False)
        def index():
            return FileResponse(WEB / "index.html")

    return app


app = create_app()
