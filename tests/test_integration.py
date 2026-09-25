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
