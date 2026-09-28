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


class _RoutingStubLLM:
    """Reports a fixed route classification and records which tools
    call_model bound it with, so the router's tool-gating can be asserted
    on directly rather than inferred from which tool the model happened to
    call."""

    def __init__(self, route_choice="web"):
        self.route_choice = route_choice
        self.bound_tool_names: list[str] | None = None

    def with_structured_output(self, schema):
        choice = self.route_choice

        class _Router:
            def invoke(self, messages):
                return schema(choice=choice)

        return _Router()

    def bind_tools(self, tools):
        self.bound_tool_names = [t.name for t in tools]

        class _Bound:
            def invoke(self, messages):
                return AIMessage(content="stub reply")

        return _Bound()


def _route_and_get_bound_tools(monkeypatch, route_choice):
    stub = _RoutingStubLLM(route_choice)
    monkeypatch.setattr(settings, "checkpoint_db", "")
    monkeypatch.setattr(agent_module, "build_llm", lambda: stub)
    agent_module.build_graph.cache_clear()

    try:
        agent_module.answer("does this need fresh info?", thread_id=f"route-{route_choice}")
        return stub.bound_tool_names
    finally:
        agent_module.build_graph.cache_clear()


def test_rag_route_only_offers_rag_search(monkeypatch):
    assert _route_and_get_bound_tools(monkeypatch, "rag") == ["rag_search"]


def test_web_route_offers_only_the_web_tools(monkeypatch):
    assert _route_and_get_bound_tools(monkeypatch, "web") == ["news_search", "web_search"]


def test_both_route_offers_every_tool(monkeypatch):
    assert _route_and_get_bound_tools(monkeypatch, "both") == [
        "rag_search",
        "news_search",
        "web_search",
    ]


def test_a_broken_router_fails_open_to_every_tool(monkeypatch):
    """The classification call itself can fail (rate limit, bad output,
    whatever) — that must cost an unnecessary tool offer, never a blocked
    reply. See route_query's docstring in app/agent.py."""

    class _BrokenRouter:
        def with_structured_output(self, schema):
            raise RuntimeError("router is down")

        def bind_tools(self, tools):
            self.bound_tool_names = [t.name for t in tools]

            class _Bound:
                def invoke(self, messages):
                    return AIMessage(content="stub reply")

            return _Bound()

    stub = _BrokenRouter()
    monkeypatch.setattr(settings, "checkpoint_db", "")
    monkeypatch.setattr(agent_module, "build_llm", lambda: stub)
    agent_module.build_graph.cache_clear()

    try:
        reply = agent_module.answer("anything", thread_id="broken-router")
        assert reply
        assert stub.bound_tool_names == ["rag_search", "news_search", "web_search"]
    finally:
        agent_module.build_graph.cache_clear()
