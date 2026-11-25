from quire.ingest import clean_inline, pack, parse_header, slugify, split_sections


def test_header_parsing_handles_continuation_lines():
    meta, start = parse_header(["PEP: 1", "Author: A,", "        B", "Status: Final", "", "body"])
    assert meta["pep"] == "1"
    assert meta["author"] == "A, B"
    assert meta["status"] == "Final"
    assert start == 5


def test_chunks_carry_metadata_and_section_urls(chunks):
    assert chunks, "fixture should produce chunks"
    c = chunks[0]
    assert c.pep == 9999
    assert c.title == "A Test Proposal for frobnicate"
    assert c.status == "Final"
    assert c.section == "Abstract"
    assert c.url == "https://peps.python.org/pep-9999/#abstract"
    assert c.index_text.startswith("PEP 9999 - A Test Proposal for frobnicate > Abstract")


def test_copyright_is_skipped_and_subsections_kept(chunks):
    sections = {c.section for c in chunks}
    assert "Copyright" not in sections
    assert {"Abstract", "Rationale", "Rejected Ideas"} <= sections
    assert all("public domain" not in c.text for c in chunks)


def test_inline_markup_is_cleaned(chunks):
    abstract = chunks[0].text
    assert ":func:" not in abstract and ":pep:" not in abstract
    assert "PEP 572" in abstract
    assert "frob()" in abstract and "``" not in abstract
    assert "https://example.org" not in abstract


def test_clean_inline_roles():
    assert clean_inline(":pep:`Assignment <572>`") == "PEP 572"
    assert clean_inline(":rfc:`2119`") == "RFC 2119"
    assert clean_inline(":class:`~typing.Protocol`") == "typing.Protocol"


def test_pack_respects_window_and_overlap():
    paras = [" ".join(f"w{i}" for i in range(n0, n0 + 50)) for n0 in range(0, 500, 50)]
    out = list(pack(paras, max_words=120, overlap=20))
    assert all(len(c.split()) <= 120 for c in out)
    # consecutive chunks share the overlap words
    assert out[0].split()[-20:] == out[1].split()[:20]
    # every word appears somewhere
    assert {w for c in out for w in c.split()} == {f"w{i}" for i in range(500)}


def test_pack_splits_one_giant_paragraph():
    out = list(pack([" ".join(str(i) for i in range(1000))], max_words=200, overlap=0))
    assert len(out) == 5 and all(len(c.split()) == 200 for c in out)


def test_split_sections_and_slug():
    secs = split_sections(["intro", "", "Title Here", "==========", "text", "", "Sub", "---", "more"])
    assert [t for t, _ in secs] == ["Preamble", "Title Here", "Sub"]
    assert slugify("Why not `foo`?") == "why-not-foo"
