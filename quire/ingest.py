"""Parse PEP reStructuredText into section-aware chunks with metadata.

Chunking strategy:
  1. Read the RFC-822 style header (PEP number, title, status, type, created...).
  2. Split the body on reST section underlines so a chunk never spans two sections.
  3. Clean inline markup (roles, literals, hyperlinks) so BM25 sees plain words.
  4. Pack paragraphs into ~CHUNK_WORDS windows with a word overlap, never cutting
     a paragraph unless it alone is longer than the window.
  5. Prepend a short context header ("PEP 572 - Assignment Expressions > Rationale")
     to the text that gets embedded and indexed, which helps both BM25 and dense
     retrieval find chunks whose body never names the PEP.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

from . import config

UNDERLINE = re.compile(r"^([=\-~^\"'`#*+])\1{2,}\s*$")
HEADER_FIELD = re.compile(r"^([A-Za-z][\w-]*):\s*(.*)$")
SKIP_SECTIONS = {"copyright", "references", "footnotes", "acknowledgements", "acknowledgments"}


@dataclass
class Chunk:
    id: str
    pep: int
    title: str
    status: str
    type: str
    created: str
    section: str
    url: str
    text: str  # cleaned body text shown to users and the LLM
    index_text: str  # context header + text, used for BM25 and embeddings

    def to_dict(self) -> dict:
        return asdict(self)


def slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug


def clean_inline(text: str) -> str:
    text = re.sub(r":pep:`(?:[^`<]*<)?(\d+)(?:#[^`>]*)?>?`", r"PEP \1", text)
    text = re.sub(r":rfc:`(\d+)`", r"RFC \1", text)
    text = re.sub(r":[a-z:]+:`~?([^`<]+?)(?:\s*<[^`]*>)?`", r"\1", text)
    text = re.sub(r"`([^`<]+?)\s*<[^`]+>`__?", r"\1", text)
    text = re.sub(r"``([^`]+)``", r"\1", text)
    text = re.sub(r"`([^`]+)`_*", r"\1", text)
    text = re.sub(r"\[#?[\w-]*\]_", "", text)
    return text


def parse_header(lines: list[str]) -> tuple[dict, int]:
    meta: dict[str, str] = {}
    key = None
    i = 0
    for i, line in enumerate(lines):
        if not line.strip():
            break
        m = HEADER_FIELD.match(line)
        if m and not line.startswith((" ", "\t")):
            key = m.group(1).lower()
            meta[key] = m.group(2).strip()
        elif key:
            meta[key] += " " + line.strip()
    return meta, i + 1


def split_sections(lines: list[str]) -> list[tuple[str, list[str]]]:
    sections: list[tuple[str, list[str]]] = [("Preamble", [])]
    i = 0
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        # overlined titles: ====\nTitle\n====
        if UNDERLINE.match(line) and i + 2 < len(lines) and UNDERLINE.match(lines[i + 2]) and lines[i + 1].strip():
            sections.append((lines[i + 1].strip(), []))
            i += 3
            continue
        if line.strip() and not line.startswith(" ") and UNDERLINE.match(nxt) and len(nxt.strip()) >= len(line.strip()) - 1:
            sections.append((line.strip(), []))
            i += 2
            continue
        sections[-1][1].append(line)
        i += 1
    return [(t, body) for t, body in sections if any(b.strip() for b in body)]


def paragraphs(body: list[str]) -> list[str]:
    out, cur = [], []
    for line in body:
        s = line.rstrip()
        if s.lstrip().startswith((".. contents", ".. _", ".. toctree", ".. raw", ".. image", ".. figure")):
            continue
        if re.match(r"^\s*\.\. (code-block|code|sourcecode)::", s):
            s = ""
        if not s.strip():
            if cur:
                out.append(" ".join(cur))
                cur = []
            continue
        cur.append(s.strip())
    if cur:
        out.append(" ".join(cur))
    return [clean_inline(p).strip() for p in out if p.strip() and p.strip() != "::"]


def pack(paras: list[str], max_words: int, overlap: int) -> Iterator[str]:
    window: list[str] = []
    fresh = 0  # words added since the last emitted chunk
    for para in paras:
        words = para.split()
        if fresh and len(window) + len(words) > max_words:
            yield " ".join(window)
            window, fresh = window[-overlap:] if overlap else [], 0
        window.extend(words)
        fresh += len(words)
        while len(window) > max_words:  # a single paragraph longer than the window
            yield " ".join(window[:max_words])
            window = window[max_words - overlap :]
            fresh = len(window) - overlap
    if fresh > 0:
        yield " ".join(window)


def parse_pep(path: Path, max_words: int = config.CHUNK_WORDS, overlap: int = config.CHUNK_OVERLAP) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    lines = raw.splitlines()
    meta, start = parse_header(lines)
    if "pep" not in meta or not meta["pep"].strip().isdigit():
        return []
    pep = int(meta["pep"])
    title = clean_inline(meta.get("title", f"PEP {pep}"))
    chunks: list[Chunk] = []
    seen_slugs: dict[str, int] = {}
    for section, body in split_sections(lines[start:]):
        if section.lower() in SKIP_SECTIONS:
            continue
        slug = slugify(section)
        seen_slugs[slug] = seen_slugs.get(slug, 0) + 1
        anchor = slug if seen_slugs[slug] == 1 else f"{slug}-{seen_slugs[slug] - 1}"
        url = f"https://peps.python.org/pep-{pep:04d}/" + ("" if section == "Preamble" else f"#{anchor}")
        texts = [t for t in pack(paragraphs(body), max_words, overlap) if len(t.split()) >= 4]
        for n, text in enumerate(texts):
            header = f"PEP {pep} - {title} > {section}"
            chunks.append(
                Chunk(
                    id=f"pep{pep:04d}-{anchor or 'preamble'}-{n}",
                    pep=pep,
                    title=title,
                    status=meta.get("status", ""),
                    type=meta.get("type", ""),
                    created=meta.get("created", ""),
                    section=section,
                    url=url,
                    text=text,
                    index_text=f"{header}\n{text}",
                )
            )
    return chunks


def iter_pep_files(raw_dir: Path) -> Iterable[Path]:
    yield from sorted(p for p in raw_dir.glob("pep-*.rst"))


def build_corpus(raw_dir: Path = config.RAW_DIR, out: Path | None = None) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in iter_pep_files(raw_dir):
        chunks.extend(parse_pep(path))
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    return chunks


def load_corpus(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]
