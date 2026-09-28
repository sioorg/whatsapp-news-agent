#!/usr/bin/env python
"""Add your own documents to the shared local knowledge base.

    PYTHONPATH=. .venv/bin/python scripts/ingest_docs.py notes.md report.pdf

Supports .txt, .md and .pdf. Each file is split into ~1500-character chunks
on paragraph boundaries and indexed the same way an auto-cached search
result is (see app/rag.py) — rag_search finds it right alongside real news.

Run this against whichever database CHECKPOINT_DB's sibling RAG_DB_PATH
points at for the environment you mean to update — locally that's your
.env, in production it's the path already set in docker-compose.yml, so run
it there (or point RAG_DB_PATH at the same mounted file) rather than on a
throwaway local copy.
"""

import sys
from pathlib import Path

from app import rag

CHUNK_SIZE = 1500


def _read(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        return "\n\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)

    return path.read_text(encoding="utf-8", errors="ignore")


def _chunks(text: str, size: int = CHUNK_SIZE) -> list[str]:
    """Split on paragraph boundaries, packing consecutive paragraphs up to
    ~size characters — mirrors the spirit of connectors/common.split_message,
    but for indexing rather than WhatsApp's outbound length cap."""

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return [text.strip()] if text.strip() else []

    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if current and len(current) + len(paragraph) + 2 > size:
            chunks.append(current)
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}" if current else paragraph
    if current:
        chunks.append(current)

    return chunks


def main(paths: list[str]) -> None:
    if not paths:
        print(__doc__)
        raise SystemExit(1)

    total = 0
    for arg in paths:
        path = Path(arg)
        if not path.exists():
            print(f"skip: {path} not found")
            continue
        if path.suffix.lower() not in {".txt", ".md", ".pdf"}:
            print(f"skip: {path} — supported types are .txt, .md, .pdf")
            continue

        chunks = _chunks(_read(path))
        for i, chunk in enumerate(chunks, start=1):
            rag.add_document(
                chunk, source=str(path), title=f"{path.name} (part {i}/{len(chunks)})"
            )
        print(f"{path}: indexed {len(chunks)} chunk(s)")
        total += len(chunks)

    print(f"done: {total} chunk(s) indexed")


if __name__ == "__main__":
    main(sys.argv[1:])
