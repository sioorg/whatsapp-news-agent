"""Checkpointer selection.

The agent's reasoning needs a live LLM, so it is not exercised here. What is
worth pinning down is which conversation store gets chosen, because getting
that wrong silently loses every user's history on restart.

These tests monkeypatch the settings object rather than reloading app.config.
Reloading would rebind app.config.settings to a new instance while app.main
kept the old one, so later tests would mutate an object nothing reads.
"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from app.agent import _build_checkpointer
from app.config import settings


def test_uses_in_memory_store_when_unset(monkeypatch):
    monkeypatch.setattr(settings, "checkpoint_db", "")

    assert isinstance(_build_checkpointer(), InMemorySaver)


def test_uses_sqlite_when_a_path_is_given(monkeypatch, tmp_path):
    db = tmp_path / "checkpoints.sqlite"
    monkeypatch.setattr(settings, "checkpoint_db", str(db))

    checkpointer = _build_checkpointer()

    assert isinstance(checkpointer, SqliteSaver)
    assert db.exists()


def test_creates_missing_parent_directories(monkeypatch, tmp_path):
    db = tmp_path / "nested" / "dir" / "checkpoints.sqlite"
    monkeypatch.setattr(settings, "checkpoint_db", str(db))

    _build_checkpointer()

    assert db.exists()


def test_schema_is_reusable_across_checkpointers(monkeypatch, tmp_path):
    """A second checkpointer over the same file works on the existing schema.

    This is what lets history outlive container replacement.
    """

    db = tmp_path / "checkpoints.sqlite"
    monkeypatch.setattr(settings, "checkpoint_db", str(db))

    _build_checkpointer()
    _build_checkpointer()

    import sqlite3

    tables = {
        row[0]
        for row in sqlite3.connect(db).execute(
            "select name from sqlite_master where type='table'"
        )
    }

    assert {"checkpoints", "writes"} <= tables
