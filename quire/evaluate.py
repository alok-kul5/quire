"""Eval harness.

  python -m quire.evaluate retrieval      # recall@k / MRR for bm25, dense, hybrid, rerank
  python -m quire.evaluate generation     # answers + faithfulness (heuristic and LLM judge) + refusals
  python -m quire.evaluate report         # write evals/results/REPORT.md from the saved JSON
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from . import config
from .answer import answer, format_context
from .metrics import heuristic_faithfulness, keyword_hit, parse_judge, recall_at_k, reciprocal_rank
from .retrieve import MODES, Retriever

EVAL_DIR = config.ROOT / "evals"
RESULTS = EVAL_DIR / "results"
KS = (1, 3, 5, 10)

JUDGE_SYSTEM = """You are a strict fact-checker. You get numbered source passages and an answer that cites them like [n].
Decide whether EVERY factual claim in the answer is directly supported by the passages it cites.
Reply with JSON only: {"score": <1-5>, "faithful": <true|false>, "unsupported": [<short claims that are not supported>]}
5 = every claim supported; 4 = supported with trivial wording drift; 3 = one claim unsupported; 1-2 = mostly unsupported or invented."""


def load_questions(path: Path = EVAL_DIR / "questions.jsonl") -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_retrieval(retriever: Retriever, questions: list[dict]) -> dict:
    qs = [q for q in questions if q["answerable"]]
    out: dict = {"n_questions": len(qs), "modes": {}, "per_question": {}}
    for mode in MODES:
        rows, lat = [], []
        for q in qs:
            s = time.perf_counter()
            hits = retriever.search(q["question"], mode=mode, k=10)
            lat.append((time.perf_counter() - s) * 1000)
            peps = [h.chunk.pep for h in hits]
            gold = set(q["gold_peps"])
            row = {f"recall@{k}": recall_at_k(peps, gold, k) for k in KS}
            row["mrr@10"] = reciprocal_rank(peps, gold, 10)
            rows.append(row)
            out["per_question"].setdefault(q["id"], {})[mode] = {"top_peps": peps, **row}
        agg = {m: round(statistics.mean(r[m] for r in rows), 3) for m in rows[0]}
        agg["p50_ms"] = round(statistics.median(lat), 1)
        agg["p95_ms"] = round(sorted(lat)[int(0.95 * (len(lat) - 1))], 1)
        out["modes"][mode] = agg
        print(mode, agg)
    return out


def run_generation(retriever: Retriever, questions: list[dict], llm, judge, mode: str = "rerank", limit: int | None = None) -> dict:
    qs = questions[:limit] if limit else questions
    per = []
    for i, q in enumerate(qs, 1):
        hits = retriever.search(q["question"], mode=mode)
        a = answer(q["question"], hits, llm)
        passages = [h.chunk.text for h in hits]
        row = {
            "id": q["id"],
            "question": q["question"],
            "answerable": q["answerable"],
            "answer": a.text,
            "refused": a.refused,
            "citations": a.citations,
            "cited_peps": sorted({hits[c - 1].chunk.pep for c in a.citations}),
            "top_score": round(hits[0].score, 3) if hits else None,
            "llm_ms": a.timings.get("llm_ms"),
        }
        if q["answerable"] and not a.refused:
            gold = set(q["gold_peps"])
            row["keyword_hit"] = keyword_hit(a.text, q["keywords"])
            row["citation_precision"] = (sum(hits[c - 1].chunk.pep in gold for c in a.citations) / len(a.citations)) if a.citations else 0.0
            h = heuristic_faithfulness(a.text, passages)
            row["heuristic"] = {k: v for k, v in h.items() if k != "rows"}
            if judge is not None:
                raw = judge.chat(JUDGE_SYSTEM, f"Passages:\n\n{format_context(hits)}\n\nAnswer:\n{a.text}\n\nJSON:", max_tokens=200)
                row["judge"] = parse_judge(raw)
        per.append(row)
        print(f"[{i}/{len(qs)}] {q['id']} refused={a.refused} {row.get('judge', {}).get('score', '-')} | {a.text[:90]!r}")

    ans = [r for r in per if r["answerable"]]
    unans = [r for r in per if not r["answerable"]]
    answered = [r for r in ans if not r["refused"]]

    def mean(xs):
        xs = list(xs)
        return round(statistics.mean(xs), 3) if xs else None

    summary = {
        "mode": mode,
        "model": llm.name,
        "judge": getattr(judge, "name", None),
        "n_answerable": len(ans),
        "n_unanswerable": len(unans),
        "answer_rate": mean(not r["refused"] for r in ans),
        "keyword_accuracy": mean(r["keyword_hit"] for r in answered),
        "citation_rate": mean(bool(r["citations"]) for r in answered),
        "citation_precision": mean(r["citation_precision"] for r in answered),
        "heuristic_supported": mean(r["heuristic"]["supported_rate"] for r in answered),
        "judge_faithful": mean(r["judge"]["faithful"] for r in answered if "judge" in r),
        "judge_score": mean(r["judge"]["score"] for r in answered if "judge" in r),
        "refusal_rate_unanswerable": mean(r["refused"] for r in unans),
        "false_refusal_rate": mean(r["refused"] for r in ans),
        "llm_p50_ms": round(statistics.median([r["llm_ms"] for r in per if r["llm_ms"]] or [0]), 1),
    }
    print(json.dumps(summary, indent=2))
    return {"summary": summary, "per_question": per}


def fmt(x, pct=True):
    if x is None:
        return "-"
    return f"{x * 100:.1f}%" if pct else f"{x}"


def write_report() -> Path:
    lines = ["# Quire eval report", ""]
    rp = RESULTS / "retrieval.json"
    if rp.exists():
        r = json.loads(rp.read_text())
        lines += [
            f"## Retrieval ablation ({r['n_questions']} hand-checked questions, PEP-level relevance)",
            "",
            "| Retriever | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 | p50 latency |",
            "|---|---|---|---|---|---|---|",
        ]
        names = {"bm25": "BM25", "dense": "Dense (bge-small)", "hybrid": "Hybrid (RRF)", "rerank": "Hybrid + cross-encoder rerank"}
        for m, a in r["modes"].items():
            lines.append(
                f"| {names[m]} | {fmt(a['recall@1'])} | {fmt(a['recall@3'])} | {fmt(a['recall@5'])} | {fmt(a['recall@10'])} | {a['mrr@10']:.3f} | {a['p50_ms']:.0f} ms |"
            )
        lines.append("")
    gp = RESULTS / "generation.json"
    if gp.exists():
        s = json.loads(gp.read_text())["summary"]
        lines += [
            f"## Generation ({s['model']}, judged by {s['judge']})",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Answer rate (answerable) | {fmt(s['answer_rate'])} |",
            f"| Keyword accuracy | {fmt(s['keyword_accuracy'])} |",
            f"| Answers with citations | {fmt(s['citation_rate'])} |",
            f"| Citation precision (cited chunk from a gold PEP) | {fmt(s['citation_precision'])} |",
            f"| Heuristic faithfulness (sentences cited and grounded) | {fmt(s['heuristic_supported'])} |",
            f"| LLM-judge faithful | {fmt(s['judge_faithful'])} |",
            f"| LLM-judge mean score (1-5) | {fmt(s['judge_score'], pct=False)} |",
            f"| Refusal rate on unanswerable | {fmt(s['refusal_rate_unanswerable'])} |",
            f"| False refusal rate | {fmt(s['false_refusal_rate'])} |",
            f"| LLM p50 latency | {s['llm_p50_ms'] / 1000:.1f} s |",
            "",
        ]
    out = RESULTS / "REPORT.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["retrieval", "generation", "report"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-judge", action="store_true")
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.what == "report":
        write_report()
        return
    questions = load_questions()
    retriever = Retriever()
    if args.what == "retrieval":
        res = run_retrieval(retriever, questions)
        (RESULTS / "retrieval.json").write_text(json.dumps(res, indent=2))
    else:
        from .llm import OllamaLLM

        judge = None if args.no_judge else OllamaLLM(config.JUDGE_MODEL)
        res = run_generation(retriever, questions, OllamaLLM(), judge, limit=args.limit)
        (RESULTS / "generation.json").write_text(json.dumps(res, indent=2))
    write_report()


if __name__ == "__main__":
    main()
