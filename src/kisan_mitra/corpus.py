"""Loads the curated advisory corpus (markdown + YAML front matter) and chunks it."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    meta: dict = field(default_factory=dict)

    @property
    def source_url(self) -> str:
        return self.meta.get("source_url", "")

    @property
    def source_org(self) -> str:
        return self.meta.get("source_org", "")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    source_url: str

    def as_context(self) -> str:
        return f"[{self.title} - {self.section}] {self.text}"


_FM = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.S)


def load_corpus(corpus_dir: str | Path) -> list[Document]:
    docs = []
    for p in sorted(Path(corpus_dir).glob("*.md")):
        raw = p.read_text(encoding="utf-8")
        m = _FM.match(raw)
        meta = yaml.safe_load(m.group(1)) if m else {}
        body = raw[m.end():] if m else raw
        doc_id = meta.get("id", p.stem)
        docs.append(Document(doc_id=doc_id, title=meta.get("title", p.stem), text=body.strip(), meta=meta))
    if not docs:
        raise FileNotFoundError(f"No .md documents found in {corpus_dir}")
    return docs


def _sections(body: str) -> list[tuple[str, str]]:
    """Split markdown body on '## ' headings. Text before the first '##' is 'Overview'."""
    out, cur_head, cur = [], "Overview", []
    for line in body.splitlines():
        if line.startswith("# ") and not line.startswith("## "):
            continue  # document H1 duplicates the title
        if line.startswith("## "):
            if "\n".join(cur).strip():
                out.append((cur_head, "\n".join(cur).strip()))
            cur_head, cur = line[3:].strip(), []
        else:
            cur.append(line)
    if "\n".join(cur).strip():
        out.append((cur_head, "\n".join(cur).strip()))
    return out


def chunk_documents(docs: list[Document], chunk_words: int = 160, overlap: int = 30) -> list[Chunk]:
    chunks: list[Chunk] = []
    for d in docs:
        for s_idx, (head, text) in enumerate(_sections(d.text)):
            text = re.sub(r"\s*Source: https?://\S+", "", text)  # URLs live in citations, not in the evidence
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n+", " ", text).strip()
            words = text.split()
            if len(words) <= chunk_words:
                windows = [words]
            else:
                step = max(1, chunk_words - overlap)
                windows = [words[i:i + chunk_words] for i in range(0, len(words), step)]
                if len(windows) > 1 and len(windows[-1]) < overlap + 10:
                    windows[-2] = words[(len(windows) - 2) * step:]
                    windows.pop()
            for w_idx, w in enumerate(windows):
                chunks.append(Chunk(
                    chunk_id=f"{d.doc_id}#{s_idx}.{w_idx}", doc_id=d.doc_id, title=d.title,
                    section=head, text=" ".join(w), source_url=d.source_url,
                ))
    return chunks
