from quire.metrics import heuristic_faithfulness, keyword_hit, parse_judge, recall_at_k, reciprocal_rank


def test_recall_and_mrr():
    assert recall_at_k([5, 8, 572], {572}, 3) == 1.0
    assert recall_at_k([5, 8, 572], {572}, 2) == 0.0
    assert reciprocal_rank([5, 572, 572], {572}) == 0.5
    assert reciprocal_rank([1, 2], {9}) == 0.0


def test_heuristic_flags_uncited_and_invented():
    passages = ["The walrus operator := assigns a value inside an expression."]
    good = heuristic_faithfulness("The walrus operator := assigns a value inside an expression [1].", passages)
    assert good["supported_rate"] == 1.0
    uncited = heuristic_faithfulness("The walrus operator assigns values inside expressions.", passages)
    assert uncited["cited_rate"] == 0.0 and uncited["supported_rate"] == 0.0
    invented = heuristic_faithfulness("Guido vetoed lambdas during the Amsterdam sprint in 1994 [1].", passages)
    assert invented["supported_rate"] == 0.0


def test_keyword_hit():
    assert keyword_hit("Use 4 spaces", ["4"])
    assert not keyword_hit("tabs", ["4"])
    assert not keyword_hit("anything", [])


def test_parse_judge_variants():
    assert parse_judge('{"score": 5, "faithful": true, "unsupported": []}')["faithful"] is True
    assert parse_judge('Sure! {"score": 2, "faithful": "no"}')["faithful"] is False
    assert parse_judge("Score: 4 overall")["score"] == 4
    assert parse_judge("garbage")["score"] == 0
