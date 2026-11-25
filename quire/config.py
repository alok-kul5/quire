"""Paths, model names and tunables. Override any of them with QUIRE_* env vars."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.environ.get(f"QUIRE_{name}", default)


RAW_DIR = Path(_env("RAW_DIR", str(ROOT / "data" / "raw" / "peps" / "peps")))
INDEX_DIR = Path(_env("INDEX_DIR", str(ROOT / "data" / "index")))

EMBED_MODEL = _env("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RERANK_MODEL = _env("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

OLLAMA_URL = _env("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = _env("LLM_MODEL", "qwen2.5:7b")
JUDGE_MODEL = _env("JUDGE_MODEL", "gemma3:4b")  # a different model family than the answerer

CHUNK_WORDS = int(_env("CHUNK_WORDS", "220"))
CHUNK_OVERLAP = int(_env("CHUNK_OVERLAP", "40"))

CANDIDATES = int(_env("CANDIDATES", "40"))  # per retriever, before fusion
RRF_K = 60
TOP_K = int(_env("TOP_K", "6"))  # chunks handed to the LLM
