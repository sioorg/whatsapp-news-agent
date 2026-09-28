"""Local knowledge store: put/search against a real SqliteStore, but with a
fake, deterministic embedding function standing in for fastembed's real
model — that keeps this in the fast/free mocked layer, with no network call
and no ~130MB download. See tests/test_agent.py for the router tests that
decide when rag_search gets offered at all.
"""

import zlib

import pytest

import app.rag as rag_module
from app.config import settings

DIMS = 8


def _fake_embed(texts: list[str]) -> list[list[float]]:
    """A crude but deterministic bag-of-words embedding: good enough to
    separate obviously-unrelated texts by a stable hash, bad at anything
    subtler — which is fine, this only needs to exercise put/search, not
    prove real semantic quality (that's fastembed's job, not app.rag's)."""

    vectors = []
    for text in texts:
        vector = [0.0] * DIMS
        for word in text.lower().split():
            vector[zlib.crc32(word.encode()) % DIMS] += 1.0
        norm = sum(v * v for v in vector) ** 0.5 or 1.0
        vectors.append([v / norm for v in vector])
    return vectors


@pytest.fixture(autouse=True)
def _fresh_store(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "rag_db_path", str(tmp_path / "rag.sqlite"))
    monkeypatch.setattr(settings, "rag_embedding_dims", DIMS)
    monkeypatch.setattr(rag_module, "_embed", _fake_embed)
    rag_module._store.cache_clear()
    yield
    rag_module._store.cache_clear()


def test_search_on_an_empty_store_returns_nothing():
    assert rag_module.search("anything") == []


def test_a_document_is_findable_by_its_own_content():
    rag_module.add_document("The Eiffel Tower is in Paris.", source="manual")

    hits = rag_module.search("Eiffel Tower")

    assert len(hits) == 1
    assert "Eiffel Tower" in hits[0].value["text"]
    assert hits[0].value["source"] == "manual"


def test_search_ranks_the_more_relevant_document_first():
    rag_module.add_document("SpaceX launched a rocket from Texas.", source="a")
    rag_module.add_document("The Eiffel Tower is a famous landmark in Paris.", source="b")

    hits = rag_module.search("Where is the Eiffel Tower?")

    assert hits[0].value["source"] == "b"


def test_blank_text_is_not_indexed():
    rag_module.add_document("   ", source="manual")

    assert rag_module.search("anything") == []


def test_cache_search_results_indexes_title_content_and_metadata():
    rag_module.cache_search_results(
        [
            {
                "title": "AI regulation update",
                "content": "The EU passed a new AI law.",
                "url": "https://example.com/a",
                "published_date": "2026-09-01",
            }
        ]
    )

    hits = rag_module.search("AI regulation")

    assert len(hits) == 1
    assert hits[0].value["source"] == "https://example.com/a"
    assert hits[0].value["title"] == "AI regulation update"
    assert hits[0].value["published"] == "2026-09-01"


def test_cache_search_results_handles_missing_optional_fields():
    """Tavily's own contract already defaults these (see app/tools.py's
    _format), but cache_search_results reads the same dicts independently
    and must not KeyError just because published_date/url are absent."""

    rag_module.cache_search_results([{"title": "Sparse result"}])

    hits = rag_module.search("Sparse result")

    assert len(hits) == 1
    assert hits[0].value["source"] == ""
    assert hits[0].value["published"] == ""


def test_store_persists_across_reconnects():
    """A restart must not lose what's already been indexed — same guarantee
    the conversation checkpointer gives (see tests/test_agent.py)."""

    rag_module.add_document("Persistent knowledge.", source="manual")
    rag_module._store.cache_clear()  # simulate a fresh process reopening the file

    hits = rag_module.search("Persistent knowledge")

    assert len(hits) == 1


def test_search_respects_the_configured_top_k(monkeypatch):
    monkeypatch.setattr(settings, "rag_top_k", 2)
    for i in range(5):
        rag_module.add_document(f"Document number {i} about news.", source=str(i))

    assert len(rag_module.search("news")) == 2
