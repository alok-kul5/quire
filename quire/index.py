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


def build(raw_dir: Path = config.RAW_DIR, index_dir: Path = config.INDEX_DIR, batch: int = 8) -> dict:
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
    model.max_seq_length = 384
    db = lancedb.connect(str(index_dir / "lancedb"))
    table = None
    step = 512  # embed and write in slices to keep peak RAM low
    for start in range(0, len(texts), step):
        for attempt in range(30):  # survive transient memory pressure on a shared machine
            try:
                vecs = model.encode(texts[start : start + step], batch_size=batch, normalize_embeddings=True)
                break
            except RuntimeError as e:
                if "memory" not in str(e) or attempt == 29:
                    raise
                print(f"  low memory at {start}, retrying in 20s", flush=True)
                time.sleep(20)
        rows = [{"id": chunks[start + j].id, "row": start + j, "vector": v.tolist()} for j, v in enumerate(vecs)]
        if table is None:
            table = db.create_table("chunks", data=rows, mode="overwrite")
        else:
            table.add(rows)
        print(f"  embedded {min(start + step, len(texts))}/{len(texts)}", flush=True)
    dim = len(rows[0]["vector"])
    print(f"embedded {len(texts)} chunks in {time.time() - t2:.1f}s")

    stats = {
        "chunks": len(chunks),
        "peps": len({c.pep for c in chunks}),
        "embed_model": config.EMBED_MODEL,
        "dim": dim,
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
