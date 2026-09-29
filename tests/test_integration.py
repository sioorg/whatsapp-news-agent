"""Layer 3: tests that call real APIs.

Excluded from CI by the `integration` marker — they cost money and depend on
live services. Run nightly via agent-eval.yml, or locally with:

    pytest -m integration -v

What these assert is deliberately loose. The reply text is non-deterministic,
so pinning exact strings would fail constantly. Instead they check the
properties that actually matter: a tool was called, a real URL came back,
and the answer fits in one WhatsApp message.
"""

import os
import re

import pytest

pytestmark = pytest.mark.integration

_URL = re.compile(r"https?://\S+")


@pytest.fixture(autouse=True)
def require_keys():
    missing = [k for k in ("GROQ_API_KEY", "TAVILY_API_KEY") if not os.getenv(k)]
    if missing:
        pytest.skip(f"missing credentials: {', '.join(missing)}")


def test_tavily_returns_recent_articles():
    """The tool contract: results carry titles and URLs in the expected shape."""

    from app.tools import news_search

    output = news_search.invoke("artificial intelligence")

    assert "Title:" in output
    assert _URL.search(output), "no source URL in tool output"


def test_get_weather_returns_a_real_reading():
    """Open-Meteo needs no credentials — this only depends on GROQ_API_KEY
    via the shared require_keys fixture for consistency with the rest of
    this file, not because weather.py itself needs it."""

    from app.weather import get_report

    report = get_report("Bengaluru")

    assert "Bengaluru" in report
    assert "°C" in report


def test_agent_calls_get_weather_for_a_weather_question():
    """A weather question must trigger get_weather, on every route — see
    ROUTE_TOOLS in app/agent.py for why it's never gated like the search
    tools are."""

    from langchain_core.messages import HumanMessage, ToolMessage

    from app.agent import build_graph

    result = build_graph().invoke(
        {"messages": [HumanMessage(content="What's the weather like in Bengaluru right now?")]},
        config={"configurable": {"thread_id": "ci-weather"}, "recursion_limit": 12},
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert any(m.name == "get_weather" for m in tool_messages), (
        "agent answered a weather question without calling get_weather"
    )


def test_agent_searches_rather_than_answering_from_memory():
    """A news question must trigger a tool call, not a recalled answer."""

    from langchain_core.messages import HumanMessage, ToolMessage

    from app.agent import build_graph

    result = build_graph().invoke(
        {"messages": [HumanMessage(content="what is the latest AI news today?")]},
        config={"configurable": {"thread_id": "ci-integration"}, "recursion_limit": 12},
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]

    assert tool_messages, "agent answered without calling a search tool"
    assert any(m.name in {"news_search", "web_search"} for m in tool_messages)


def test_reply_fits_a_single_whatsapp_message():
    """Guards against a prompt edit that makes replies balloon."""

    from app.agent import answer
    from app.connectors.meta_whatsapp import MAX_BODY_CHARS

    reply = answer("latest news on semiconductors", thread_id="ci-length")

    assert reply
    assert len(reply) <= MAX_BODY_CHARS, (
        f"reply was {len(reply)} chars, over the {MAX_BODY_CHARS} limit, "
        "so it would arrive split across messages"
    )


def test_reply_cites_sources():
    """The system prompt requires source URLs; this catches silent drift."""

    from app.agent import answer

    reply = answer("latest news on electric vehicles", thread_id="ci-sources")

    assert _URL.search(reply), "reply contained no source URL"


def test_conversation_memory_works_across_turns():
    from app.agent import answer

    answer(
        "My name is CI Bot and I only follow robotics news. Just acknowledge.",
        thread_id="ci-memory",
    )
    reply = answer("What is my name? Do not search.", thread_id="ci-memory")

    assert "ci bot" in reply.lower()


def test_a_repeated_question_is_answered_from_the_local_cache():
    """Proves the router + rag_search loop end to end, with a real
    embedding model and a real router classification — not just the stubs
    in tests/test_agent.py and tests/test_rag.py.

    The first ask hits the web (and auto-caches the results, see
    app.tools). A second, fresh thread asking the same thing has no
    conversation memory of the first — so any use of rag_search on it can
    only be the shared local cache the first call just populated.
    """

    from langchain_core.messages import HumanMessage, ToolMessage

    from app.agent import build_graph

    query = "What is the latest news about the James Webb Space Telescope?"
    config = {"recursion_limit": 12}

    first = build_graph().invoke(
        {"messages": [HumanMessage(content=query)]},
        config={**config, "configurable": {"thread_id": "ci-rag-first"}},
    )
    first_tools = {m.name for m in first["messages"] if isinstance(m, ToolMessage)}
    assert first_tools & {"news_search", "web_search"}, "first ask should hit the web"

    second = build_graph().invoke(
        {"messages": [HumanMessage(content=query)]},
        config={**config, "configurable": {"thread_id": "ci-rag-second"}},
    )
    second_tools = {m.name for m in second["messages"] if isinstance(m, ToolMessage)}
    assert "rag_search" in second_tools, (
        "a repeated question should hit the local cache, not just the web again"
    )
