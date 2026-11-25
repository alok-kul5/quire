"""Pure metric functions: retrieval quality, citation checks and heuristic faithfulness."""

from __future__ import annotations

import json
import re

STOP = set(
    "a an the of to in on for and or but is are was were be been being it its this that these those with as by at from "
    "which who what when where why how can could should would will may might must do does did not no yes than then so "
    "such into about over under also only just more most other some any each all both either neither there their they "
    "them he she we you i his her our your if else while because via per pep peps python".split()
)
WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\sA-Za-z0-9]{1,3}")


def first_relevant_rank(retrieved_peps: list[int], gold: set[int]) -> int | None:
    for i, p in enumerate(retrieved_peps, 1):
        if p in gold:
            return i
    return None


def recall_at_k(retrieved_peps: list[int], gold: set[int], k: int) -> float:
    r = first_relevant_rank(retrieved_peps[:k], gold)
    return 1.0 if r is not None else 0.0


def reciprocal_rank(retrieved_peps: list[int], gold: set[int], k: int = 10) -> float:
    r = first_relevant_rank(retrieved_peps[:k], gold)
    return 1.0 / r if r else 0.0


def content_tokens(text: str) -> set[str]:
    return {t.lower() for t in WORD.findall(text) if t.lower() not in STOP and (len(t) > 2 or not t.isalpha())}


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9`\"'(])", text.strip())
    return [p for p in parts if len(p.split()) >= 3]


SENT_CITE = re.compile(r"\[(\d+)\]")


def heuristic_faithfulness(answer: str, passages: list[str], threshold: float = 0.5) -> dict:
    """Per sentence: is it cited, and do most of its content words appear in the cited passages?

    supported = cited AND overlap(sentence tokens, cited passage tokens) >= threshold.
    A cheap, deterministic proxy that catches invented specifics and missing citations.
    """
    sents = split_sentences(answer) or ([answer] if answer.strip() else [])
    rows = []
    for s in sents:
        cites = [int(n) for n in SENT_CITE.findall(s) if 1 <= int(n) <= len(passages)]
        toks = content_tokens(SENT_CITE.sub("", s))
        ctx = set().union(*(content_tokens(passages[c - 1]) for c in cites)) if cites else set()
        overlap = len(toks & ctx) / len(toks) if toks else 1.0
        rows.append({"sentence": s, "cited": bool(cites), "overlap": round(overlap, 3), "supported": bool(cites) and overlap >= threshold})
    n = len(rows) or 1
    return {
        "sentences": len(rows),
        "cited_rate": sum(r["cited"] for r in rows) / n,
        "supported_rate": sum(r["supported"] for r in rows) / n,
        "mean_overlap": sum(r["overlap"] for r in rows) / n,
        "rows": rows,
    }


def keyword_hit(answer: str, keywords: list[str]) -> bool:
    a = answer.lower()
    return any(k.lower() in a for k in keywords) if keywords else False


def parse_judge(raw: str) -> dict:
    """Extract {"faithful": bool, "score": 1-5} from a judge reply; tolerant of extra prose."""
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            score = int(obj.get("score", 0))
            faithful = obj.get("faithful")
            if isinstance(faithful, str):
                faithful = faithful.strip().lower() in {"yes", "true"}
            if faithful is None:
                faithful = score >= 4
            return {"score": max(0, min(5, score)), "faithful": bool(faithful), "unsupported": obj.get("unsupported", [])}
        except (ValueError, TypeError):
            pass
    m = re.search(r"score\D{0,5}([1-5])", raw, flags=re.I)
    score = int(m.group(1)) if m else 0
    return {"score": score, "faithful": score >= 4, "unsupported": []}
