from quire.answer import REFUSAL_TEXT, answer, build_prompt, parse_citations
from quire.llm import FakeLLM
from quire.retrieve import rrf


def test_rrf_rewards_agreement():
    fused = rrf([[1, 2, 3], [3, 1, 4]])
    order = [r for r, _ in fused]
    assert order[0] == 1  # ranked 1st and 2nd
    assert set(order) == {1, 2, 3, 4}
    assert fused[0][1] > fused[-1][1]


def test_parse_citations_drops_out_of_range():
    text, cites = parse_citations("It is new [1]. Also old [7]. Both [2, 3].", 3)
    assert cites == [1, 2, 3]
    assert "[7]" not in text
    assert "[2][3]" in text


def test_prompt_numbers_passages(retriever):
    hits = retriever.search("frobnicate builtin")
    p = build_prompt("What is frobnicate?", hits)
    assert "[1] PEP 9999" in p and p.rstrip().endswith("Answer:")


def test_answer_cites_with_fake_llm(retriever):
    hits = retriever.search("why a builtin frobnicate")
    a = answer("why a builtin frobnicate", hits, FakeLLM())
    assert not a.refused
    assert a.citations == [1]
    d = a.to_dict()
    assert d["sources"][0]["cited"] is True and d["sources"][0]["n"] == 1


def test_llm_refusal_token_becomes_refusal(retriever):
    hits = retriever.search("frobnicate")
    a = answer("unrelated", hits, FakeLLM("INSUFFICIENT_CONTEXT"))
    assert a.refused and a.text == REFUSAL_TEXT and a.citations == []


def test_low_rerank_score_refuses_without_calling_llm(retriever):
    hits = retriever.search("zzz qqq")  # no overlap -> score -10
    llm = FakeLLM()
    a = answer("zzz qqq", hits, llm)
    assert a.refused and llm.calls == []


def test_no_hits_refuses():
    assert answer("x", [], FakeLLM()).refused


def test_reasoning_is_stripped():
    from quire.llm import strip_reasoning

    assert strip_reasoning("Okay, let me think.\n</think>\n\nIt is := [1].") == "It is := [1]."
    assert strip_reasoning("<think>hmm</think>Answer [1].") == "Answer [1]."
    assert strip_reasoning("Plain [1].") == "Plain [1]."
