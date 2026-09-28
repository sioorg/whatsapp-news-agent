"""Local knowledge store, shared across every user's conversation.

This is deliberately separate from the per-thread conversation memory in
app.agent (keyed by WhatsApp sender). This store has one namespace that
everyone's searches feed into and everyone's questions can draw from. Two
things write to it: every real news_search/web_search result (see
app.tools.cache_search_results), and whatever gets ingested by hand with
scripts/ingest_docs.py.

Backed by LangGraph's SqliteStore — the same on-disk pattern already used
for CHECKPOINT_DB, so this needs no separate database service — indexed
with fastembed (ONNX, CPU-only, no API key, no per-call cost). See
app.agent's routing logic for how a question is decided to need this,
a live web search, or both.
"""

import sqlite3
import uuid
from functools import lru_cache
from pathlib import Path

from fastembed import TextEmbedding
from langgraph.store.base import SearchItem
from langgraph.store.sqlite import SqliteStore

from app.config import settings

# Everything lives in one namespace: there's no per-user separation here,
# by design — a search result one person triggers is fair game for anyone
# else's question too.
NAMESPACE = ("knowledge",)


@lru_cache(maxsize=1)
def _embedder() -> TextEmbedding:
    kwargs = {"model_name": settings.rag_embedding_model}
    if settings.rag_model_cache_dir:
        kwargs["cache_dir"] = settings.rag_model_cache_dir
    return TextEmbedding(**kwargs)


def _embed(texts: list[str]) -> list[list[float]]:
    return [vector.tolist() for vector in _embedder().embed(texts)]


@lru_cache(maxsize=1)
def _store() -> SqliteStore:
    path = settings.rag_db_path or ":memory:"
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    # check_same_thread=False: BackgroundTasks runs the agent on worker
    # threads, same reason app.agent's checkpointer connection needs it.
    # isolation_level=None (autocommit): SqliteStore issues its own explicit
    # BEGIN/COMMIT; sqlite3's default implicit transaction wrapping
    # conflicts with that ("cannot start a transaction within a
    # transaction") the moment put() is called.
    connection = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
    store = SqliteStore(
        connection,
        index={
            "dims": settings.rag_embedding_dims,
            "embed": _embed,
            "fields": ["text"],
        },
    )
    store.setup()
    return store


def add_document(text: str, *, source: str, **metadata: str) -> None:
    """Index one chunk of text. Silently skips blanks.

    The same call indexes a manually-ingested document and an auto-cached
    search result — only ``source`` and the extra metadata differ.
    """

    text = text.strip()
    if not text:
        return

    _store().put(NAMESPACE, str(uuid.uuid4()), {"text": text, "source": source, **metadata})


def cache_search_results(results: list[dict]) -> None:
    """Persist Tavily results so a repeat or follow-up question can be
    answered from local knowledge instead of paying for another search."""

    for item in results:
        text = f"{item.get('title', '')}\n{item.get('content', '')}"
        add_document(
            text,
            source=item.get("url", ""),
            title=item.get("title", ""),
            published=item.get("published_date", ""),
        )


def search(query: str, k: int | None = None) -> list[SearchItem]:
    """Return the best matches for query, best first.

    Empty if the store has nothing yet — SqliteStore just returns [] rather
    than raising, so a cold store behaves like "no results found", not an
    error.
    """

    return _store().search(NAMESPACE, query=query, limit=k or settings.rag_top_k)
