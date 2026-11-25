from fastapi.testclient import TestClient

from quire.api import create_app
from quire.evaluate import run_generation
from quire.llm import FakeLLM


def client(retriever, llm=None):
    return TestClient(create_app(retriever=retriever, llm=llm or FakeLLM()))


def test_health(retriever):
    r = client(retriever).get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["peps"] == 1 and body["model"] == "fake"


def test_ask_returns_answer_citations_and_sources(retriever):
    r = client(retriever).post("/api/ask", json={"question": "why is frobnicate a builtin", "mode": "rerank"})
    assert r.status_code == 200
    d = r.json()
    assert d["refused"] is False
    assert d["citations"] == [1]
    assert d["sources"][0]["url"].startswith("https://peps.python.org/pep-9999/")
    assert {"retrieval_ms", "llm_ms"} <= set(d["timings"])


def test_ask_validates_input(retriever):
    c = client(retriever)
    assert c.post("/api/ask", json={"question": "hi"}).status_code == 422
    assert c.post("/api/ask", json={"question": "valid question", "mode": "magic"}).status_code == 422


def test_search(retriever):
    r = client(retriever).post("/api/search", json={"query": "rejected keyword", "k": 2})
    assert r.status_code == 200 and len(r.json()["hits"]) == 2


def test_llm_failure_is_503(retriever):
    class Boom(FakeLLM):
        def chat(self, *a, **k):
            raise RuntimeError("connection refused")

    r = client(retriever, Boom()).post("/api/ask", json={"question": "why is frobnicate a builtin"})
    assert r.status_code == 503


def test_index_page_served(retriever):
    r = client(retriever).get("/")
    assert r.status_code == 200 and "Quire" in r.text


def test_generation_eval_plumbing(retriever):
    qs = [
        {"id": "a", "question": "why is frobnicate a builtin", "gold_peps": [9999], "keywords": ["frob"], "answerable": True},
        {"id": "u", "question": "zzz qqq", "gold_peps": [], "keywords": [], "answerable": False},
    ]
    judge = FakeLLM('{"score": 5, "faithful": true}')
    judge.name = "fake-judge"
    res = run_generation(retriever, qs, FakeLLM(), judge)
    s = res["summary"]
    assert s["answer_rate"] == 1.0
    assert s["citation_precision"] == 1.0
    assert s["judge_faithful"] == 1.0
    assert s["refusal_rate_unanswerable"] == 1.0
