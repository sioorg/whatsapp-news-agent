"""Checkpointer selection.

The agent's reasoning needs a live LLM, so it is not exercised here. What is
worth pinning down is which conversation store gets chosen, because getting
that wrong silently loses every user's history on restart.

These tests monkeypatch the settings object rather than reloading app.config.
Reloading would rebind app.config.settings to a new instance while app.main
kept the old one, so later tests would mutate an object nothing reads.
"""

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from tavily.errors import InvalidAPIKeyError

import app.agent as agent_module
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


class _StubChatModel:
    """A fake bound-tools chat model that always calls news_search once.

    Simulates exactly the shape a real provider produces: first turn picks
    the tool, second turn (once the tool result — success or error — is in
    state) writes the final reply. bind_tools() returns self so agent.py's
    ``build_llm().bind_tools(TOOLS)`` chain works unchanged.
    """

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        # A real dangling-tool_call bug means this second call never even
        # gets a well-formed message list to invoke with; asserting on the
        # shape here is what would have caught it.
        has_tool_message = any(isinstance(m, ToolMessage) for m in messages)

        if not has_tool_message:
            return AIMessage(
                content="",
                tool_calls=[{"name": "news_search", "args": {"query": "AI"}, "id": "call_1"}],
            )
        return AIMessage(content="Here's what I found (or didn't).")


def test_a_failed_tool_call_does_not_corrupt_the_thread(monkeypatch):
    """A Tavily failure must end the turn cleanly, not crash with a dangling
    tool_call left in the checkpoint — see [[whatsapp-news-agent-deployment]]
    for what happens to the user's thread if this regresses: every later
    message on it is rejected forever, with no way for them to recover.
    """

    monkeypatch.setattr(settings, "checkpoint_db", "")
    monkeypatch.setattr(agent_module, "build_llm", lambda: _StubChatModel())
    monkeypatch.setattr(
        "app.tools._client",
        lambda: type("FakeClient", (), {"search": lambda self, **_: (_ for _ in ()).throw(
            InvalidAPIKeyError("bad key")
        )})(),
    )
    agent_module.build_graph.cache_clear()

    try:
        reply = agent_module.answer("latest AI news", thread_id="corruption-test")

        assert reply  # the turn finished; nothing propagated out of invoke()

        state = agent_module.build_graph().get_state(
            {"configurable": {"thread_id": "corruption-test"}}
        )
        messages = state.values["messages"]

        # Every AIMessage with tool_calls must be immediately followed by a
        # matching ToolMessage — that's the invariant a real LLM API enforces,
        # and the one a swallowed crash used to violate.
        for i, message in enumerate(messages):
            if isinstance(message, AIMessage) and message.tool_calls:
                assert i + 1 < len(messages)
                assert isinstance(messages[i + 1], ToolMessage)

        # The thread must still work for a normal follow-up.
        follow_up = agent_module.answer("anything else?", thread_id="corruption-test")
        assert follow_up
    finally:
        agent_module.build_graph.cache_clear()
