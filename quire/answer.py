"""Grounded answering: numbered context in, inline [n] citations out, refusal when unsupported."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from .llm import LLM
from .retrieve import Hit

REFUSAL_TOKEN = "INSUFFICIENT_CONTEXT"
REFUSAL_TEXT = "I can't answer that from the PEPs I retrieved. Try rephrasing, or name the PEP you mean."

# Cross-encoder logit below which the best passage is treated as off-topic and the
# LLM is never called. Calibrated on the unanswerable split of the eval set.
MIN_RERANK_SCORE = -4.0

SYSTEM = f"""You answer questions about Python Enhancement Proposals (PEPs) using ONLY the numbered passages provided.

Rules:
- Every sentence that states a fact must end with the citation of the passage(s) that support it, like [1] or [2][4].
- Use only facts found in the passages. Do not use outside knowledge, even if you know the answer.
- If the passages do not contain the answer, reply with exactly: {REFUSAL_TOKEN}
- Be concise: 2 to 5 sentences. Mention PEP numbers when helpful. No preamble."""

CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


@dataclass
class Answer:
    question: str
    text: str
    refused: bool
    citations: list[int]  # 1-based passage numbers actually cited
    hits: list[Hit]
    model: str
    timings: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.text,
            "refused": self.refused,
            "citations": self.citations,
            "sources": [h.to_dict() | {"n": i + 1, "cited": (i + 1) in self.citations} for i, h in enumerate(self.hits)],
            "model": self.model,
            "timings": self.timings,
        }


def format_context(hits: list[Hit]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        c = h.chunk
        blocks.append(f"[{i}] PEP {c.pep} - {c.title} > {c.section}\n{c.text}")
    return "\n\n".join(blocks)


def build_prompt(question: str, hits: list[Hit]) -> str:
    return f"Passages:\n\n{format_context(hits)}\n\nQuestion: {question}\nAnswer:"


def parse_citations(text: str, n_passages: int) -> tuple[str, list[int]]:
    """Return text with out-of-range citations removed, and the sorted valid citation numbers."""
    used: set[int] = set()

    def keep(m: re.Match) -> str:
        nums = [int(x) for x in m.group(1).split(",")]
        ok = [x for x in nums if 1 <= x <= n_passages]
        used.update(ok)
        return "".join(f"[{x}]" for x in ok)

    cleaned = CITE.sub(keep, text)
    return re.sub(r"\s+([.,;])", r"\1", cleaned).strip(), sorted(used)


def answer(question: str, hits: list[Hit], llm: LLM, guard: bool = True) -> Answer:
    if not hits or (guard and "rerank" in hits[0].ranks and hits[0].score < MIN_RERANK_SCORE):
        return Answer(question, REFUSAL_TEXT, True, [], hits, llm.name, {"llm_ms": 0.0})
    s = time.perf_counter()
    raw = llm.chat(SYSTEM, build_prompt(question, hits))
    ms = round((time.perf_counter() - s) * 1000, 1)
    if REFUSAL_TOKEN in raw or not raw.strip():
        return Answer(question, REFUSAL_TEXT, True, [], hits, llm.name, {"llm_ms": ms})
    text, cites = parse_citations(raw, len(hits))
    return Answer(question, text, False, cites, hits, llm.name, {"llm_ms": ms})
