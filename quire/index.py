"""Build the two indexes: a BM25 index (bm25s) and a dense LanceDB table.

Run:  python -m quire.index
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from . import config
from .ingest import build_corpus


def embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBED_MODEL)


def tokenize(texts: list[str]):
    import bm25s

    return bm25s.tokenize(texts, stopwords="en", show_progress=False)


def build(raw_dir: Path = config.RAW_DIR, index_dir: Path = config.INDEX_DIR, batch: int = 64) -> dict:
    import bm25s
    import lancedb

    t0 = time.time()
    index_dir.mkdir(parents=True, exist_ok=True)
    chunks = build_corpus(raw_dir, index_dir / "chunks.jsonl")
    texts = [c.index_text for c in chunks]
    print(f"parsed {len(chunks)} chunks from {len({c.pep for c in chunks})} PEPs in {time.time() - t0:.1f}s")

    t1 = time.time()
    bm = bm25s.BM25()
    bm.index(tokenize(texts), show_progress=False)
    bm.save(str(index_dir / "bm25"))
    print(f"bm25 index in {time.time() - t1:.1f}s")

    t2 = time.time()
    model = embedder()
    vecs = model.encode(texts, batch_size=batch, normalize_embeddings=True, show_progress_bar=True)
    rows = [{"id": c.id, "row": i, "vector": v.tolist()} for i, (c, v) in enumerate(zip(chunks, vecs))]
    db = lancedb.connect(str(index_dir / "lancedb"))
    db.create_table("chunks", data=rows, mode="overwrite")
    print(f"embedded {len(rows)} chunks in {time.time() - t2:.1f}s")

    stats = {
        "chunks": len(chunks),
        "peps": len({c.pep for c in chunks}),
        "embed_model": config.EMBED_MODEL,
        "dim": int(vecs.shape[1]),
        "build_seconds": round(time.time() - t0, 1),
    }
    (index_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=config.RAW_DIR)
    ap.add_argument("--out", type=Path, default=config.INDEX_DIR)
    args = ap.parse_args()
    print(build(args.raw, args.out))
