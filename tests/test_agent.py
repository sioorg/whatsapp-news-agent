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
from app import metrics
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


def test_rag_route_offers_rag_search_and_weather(monkeypatch):
    """get_weather/generate_image ride along on every route — see
    ROUTE_TOOLS's comment in app/agent.py for why they aren't gated like
    the search tools are."""

    assert _route_and_get_bound_tools(monkeypatch, "rag") == [
        "rag_search",
        "get_weather",
        "generate_image",
    ]


def test_web_route_offers_the_web_tools_and_weather(monkeypatch):
    assert _route_and_get_bound_tools(monkeypatch, "web") == [
        "news_search",
        "web_search",
        "get_weather",
        "generate_image",
    ]


def test_both_route_offers_every_tool(monkeypatch):
    assert _route_and_get_bound_tools(monkeypatch, "both") == [
        "rag_search",
        "news_search",
        "web_search",
        "get_weather",
        "generate_image",
    ]


def test_router_decision_is_recorded_in_metrics(monkeypatch):
    before = metrics.ROUTER_DECISIONS.labels(route="rag")._value.get()

    _route_and_get_bound_tools(monkeypatch, "rag")

    assert metrics.ROUTER_DECISIONS.labels(route="rag")._value.get() == before + 1


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
        assert stub.bound_tool_names == [
            "rag_search",
            "news_search",
            "web_search",
            "get_weather",
            "generate_image",
        ]
    finally:
        agent_module.build_graph.cache_clear()


class _StatelessStubLLM:
    """Records the system prompt and full message history call_model built,
    and returns a fixed reply — via plain .invoke(), matching call_model's
    real code exactly (it never calls .stream() itself).

    LangGraph's stream_mode="messages" only splits a reply into multiple
    token-level chunks when the underlying provider truly streams — a stub's
    .invoke() always comes back as exactly one chunk (verified empirically:
    a real Groq call under the same stream_mode does arrive in several
    pieces, a stub's doesn't). Good enough here to test stream_reply's
    filtering and wiring; real incremental streaming is what the manual
    verification against Groq during development actually proved, not
    something these stub-based tests can demonstrate on their own.
    """

    def __init__(self, reply_text="stub reply"):
        self.reply_text = reply_text
        self.seen_system_prompt = None
        self.seen_message_count = None

    def with_structured_output(self, schema):
        class _Router:
            def invoke(self, messages):
                return schema(choice="rag")

        return _Router()

    def bind_tools(self, tools):
        outer = self

        class _Bound:
            def invoke(self, messages):
                outer.seen_system_prompt = messages[0].content
                outer.seen_message_count = len(messages) - 1  # exclude the system message
                return AIMessage(content=outer.reply_text)

        return _Bound()


def test_stream_reply_forwards_the_full_message_history(monkeypatch):
    stub = _StatelessStubLLM()
    monkeypatch.setattr(agent_module, "build_llm", lambda: stub)
    agent_module.build_stateless_graph.cache_clear()

    from langchain_core.messages import HumanMessage

    history = [
        HumanMessage(content="hello"),
        AIMessage(content="hi there"),
        HumanMessage(content="tell me more"),
    ]
    chunks = list(agent_module.stream_reply(history))

    assert "".join(chunks) == "stub reply"
    assert stub.seen_message_count == len(history)
    agent_module.build_stateless_graph.cache_clear()


def test_stream_reply_uses_the_web_prompt_not_whatsapps(monkeypatch):
    stub = _StatelessStubLLM()
    monkeypatch.setattr(agent_module, "build_llm", lambda: stub)
    agent_module.build_stateless_graph.cache_clear()

    from langchain_core.messages import HumanMessage

    list(agent_module.stream_reply([HumanMessage(content="hi")]))

    assert "standard Markdown" in stub.seen_system_prompt
    assert "WhatsApp" not in stub.seen_system_prompt
    agent_module.build_stateless_graph.cache_clear()


def test_stream_reply_never_touches_the_checkpoint_db(monkeypatch, tmp_path):
    """The stateless graph must never accumulate rows in CHECKPOINT_DB —
    each web request would otherwise be one more dead thread_id forever,
    since (unlike WhatsApp's ~5 phone numbers) there's no natural cap on
    how many web requests come in."""

    import sqlite3

    from langchain_core.messages import HumanMessage

    db_path = tmp_path / "checkpoints.sqlite"
    monkeypatch.setattr(settings, "checkpoint_db", str(db_path))
    stub = _StatelessStubLLM()
    monkeypatch.setattr(agent_module, "build_llm", lambda: stub)
    agent_module.build_stateless_graph.cache_clear()

    # Force the sqlite file and its schema to actually exist first.
    agent_module._build_checkpointer()
    before = sqlite3.connect(db_path).execute("select count(*) from checkpoints").fetchone()[0]

    for _ in range(3):
        list(agent_module.stream_reply([HumanMessage(content="hi")]))

    after = sqlite3.connect(db_path).execute("select count(*) from checkpoints").fetchone()[0]
    assert before == after == 0
    agent_module.build_stateless_graph.cache_clear()


class _ImageStubLLM:
    """Calls generate_image on the very first invoke, then always replies
    with plain text — for testing answer()/stream_reply()'s image_out
    extraction, including answer()'s stale-history guard across separate
    turns on the same thread."""

    def __init__(self):
        self.calls = 0

    def with_structured_output(self, schema):
        class _Router:
            def invoke(self, messages):
                return schema(choice="both")

        return _Router()

    def bind_tools(self, tools):
        outer = self

        class _Bound:
            def invoke(self, messages):
                outer.calls += 1
                if outer.calls == 1:
                    return AIMessage(
                        content="",
                        tool_calls=[
                            {"name": "generate_image", "args": {"prompt": "a cat"}, "id": "call_1"}
                        ],
                    )
                return AIMessage(content="Here you go.")

        return _Bound()


def test_answer_extracts_a_generated_image(monkeypatch):
    """End-to-end through the real graph (ToolNode included, not stubbed)
    — checks both that the image_out param surfaces what generate_image
    produced, and that a later turn on the same (checkpointed) thread that
    never calls the tool again does NOT resurface it. That second check is
    the whole reason answer() scopes its scan to messages after the latest
    HumanMessage rather than the full, ever-growing checkpointed history —
    see the comment there."""

    monkeypatch.setattr(settings, "checkpoint_db", "")
    monkeypatch.setattr("app.image_gen.generate", lambda prompt: (b"fake-bytes", "image/jpeg"))
    stub = _ImageStubLLM()
    monkeypatch.setattr(agent_module, "build_llm", lambda: stub)
    agent_module.build_graph.cache_clear()

    try:
        image_out: dict = {}
        reply = agent_module.answer("draw me a cat", thread_id="image-test", image_out=image_out)
        assert reply
        assert image_out.get("image") == (b"fake-bytes", "image/jpeg")

        image_out2: dict = {}
        agent_module.answer("anything else?", thread_id="image-test", image_out=image_out2)
        assert image_out2.get("image") is None
    finally:
        agent_module.build_graph.cache_clear()


def test_stream_reply_extracts_a_generated_image(monkeypatch):
    """Same as test_answer_extracts_a_generated_image but for the
    streaming path — deliberately calls the real stream_reply()/real
    LangGraph .stream(), not a mock, and fully drains the generator the
    way a real caller must. This is the test that was missing when
    image_out (then a contextvars side channel) shipped: every existing
    test at the time mocked app.main.stream_reply directly, so nothing
    ever exercised the real interaction between LangGraph's streaming API
    and the extraction logic — which is exactly where the bug turned out
    to be (confirmed in production 2026-09-30: images generated correctly
    but never reached the user over the streaming web path)."""

    monkeypatch.setattr("app.image_gen.generate", lambda prompt: (b"fake-bytes", "image/jpeg"))
    stub = _ImageStubLLM()
    monkeypatch.setattr(agent_module, "build_llm", lambda: stub)
    agent_module.build_stateless_graph.cache_clear()

    from langchain_core.messages import HumanMessage

    try:
        image_out: dict = {}
        chunks = list(
            agent_module.stream_reply([HumanMessage(content="draw me a cat")], image_out=image_out)
        )
        assert chunks
        assert image_out.get("image") == (b"fake-bytes", "image/jpeg")
    finally:
        agent_module.build_stateless_graph.cache_clear()


# --- Image understanding (the reverse direction: images sent TO the bot) ---


def test_human_message_is_plain_text_without_an_image():
    msg = agent_module._human_message("hello", None)

    assert msg.content == "hello"


def test_human_message_builds_multimodal_content_with_an_image():
    msg = agent_module._human_message("what is this", (b"fake-bytes", "image/png"))

    blocks = msg.content_blocks
    assert blocks[0]["type"] == "text"
    assert blocks[0]["text"] == "what is this"
    assert blocks[1]["type"] == "image"
    assert blocks[1]["mime_type"] == "image/png"
    import base64

    assert blocks[1]["base64"] == base64.b64encode(b"fake-bytes").decode()


def test_human_message_defaults_the_prompt_for_a_captionless_image():
    msg = agent_module._human_message("", (b"fake-bytes", "image/png"))

    assert msg.content_blocks[0]["text"] == agent_module.DEFAULT_IMAGE_PROMPT


def test_answer_passes_the_image_to_the_llm(monkeypatch):
    """End-to-end through the real graph: a stub LLM records the actual
    messages it was invoked with, confirming the image genuinely reaches
    the model as multimodal content — not just that _human_message builds
    the right shape in isolation."""

    monkeypatch.setattr(settings, "checkpoint_db", "")

    class _RecordingStub:
        def __init__(self):
            self.seen_messages = None

        def with_structured_output(self, schema):
            class _Router:
                def invoke(self, messages):
                    return schema(choice="both")

            return _Router()

        def bind_tools(self, tools):
            outer = self

            class _Bound:
                def invoke(self, messages):
                    outer.seen_messages = messages
                    return AIMessage(content="it's a cat")

            return _Bound()

    stub = _RecordingStub()
    monkeypatch.setattr(agent_module, "build_llm", lambda: stub)
    agent_module.build_graph.cache_clear()

    try:
        reply = agent_module.answer(
            "what is this",
            thread_id="vision-test",
            image_in=(b"fake-bytes", "image/jpeg"),
        )
        assert reply == "it's a cat"

        human = stub.seen_messages[-1]
        blocks = human.content_blocks
        assert blocks[0]["text"] == "what is this"
        assert blocks[1]["type"] == "image"
        assert blocks[1]["mime_type"] == "image/jpeg"
    finally:
        agent_module.build_graph.cache_clear()
