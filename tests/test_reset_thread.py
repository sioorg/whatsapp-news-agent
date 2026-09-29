"""scripts/reset_thread.py: clearing one stuck conversation must never touch
another thread's history, on either checkpointer backend."""

import sys

from langchain_core.messages import AIMessage, HumanMessage

import app.agent as agent_module
from app.config import settings

sys.path.insert(0, "scripts")
import reset_thread  # noqa: E402


class _StubLLM:
    """No real LLM call in this file — it's only exercising checkpointer
    behavior, not routing or answering."""

    def with_structured_output(self, schema):
        class _Router:
            def invoke(self, messages):
                return schema(choice="rag")

        return _Router()

    def bind_tools(self, tools):
        class _Bound:
            def invoke(self, messages):
                return AIMessage(content="stub reply")

        return _Bound()


def _run_two_threads(monkeypatch, thread_a: str, thread_b: str):
    monkeypatch.setattr(agent_module, "build_llm", lambda: _StubLLM())
    agent_module.build_graph.cache_clear()
    graph = agent_module.build_graph()
    for thread_id in (thread_a, thread_b):
        graph.invoke(
            {"messages": [HumanMessage(content="hi")]},
            config={"configurable": {"thread_id": thread_id}},
        )
    return graph


def test_refuses_in_memory_rather_than_silently_doing_nothing(monkeypatch):
    """A separate process can never reach a running server's in-memory
    state — a fresh MemorySaver here would be disconnected from whatever
    the live server actually holds. Must say so, not print "cleared" while
    touching nothing anyone reads."""

    monkeypatch.setattr(settings, "checkpoint_db", "")

    try:
        reset_thread.main("anything")
        raised = False
    except SystemExit:
        raised = True

    assert raised


def test_clears_only_the_named_thread_sqlite(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "checkpoint_db", str(tmp_path / "checkpoints.sqlite"))
    graph = _run_two_threads(monkeypatch, "stuck", "fine")

    reset_thread.main("stuck")

    assert graph.get_state({"configurable": {"thread_id": "stuck"}}).values == {}
    assert graph.get_state({"configurable": {"thread_id": "fine"}}).values != {}
    agent_module.build_graph.cache_clear()


def test_the_cleared_thread_works_again_afterwards(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "checkpoint_db", str(tmp_path / "checkpoints.sqlite"))
    graph = _run_two_threads(monkeypatch, "stuck", "fine")

    reset_thread.main("stuck")

    result = graph.invoke(
        {"messages": [HumanMessage(content="hi again")]},
        config={"configurable": {"thread_id": "stuck"}},
    )
    assert result["messages"]
    agent_module.build_graph.cache_clear()
