"""Search tools exposed to the LangGraph agent: local knowledge first, then
Tavily for the web. See app.agent for how a question is routed to one, the
other, or both, and app.rag for the local store itself."""

import logging
from functools import lru_cache

from langchain_core.tools import tool
from tavily import TavilyClient

from app import rag
from app.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _client() -> TavilyClient:
    return TavilyClient(api_key=settings.tavily_api_key())


def _format(results: list[dict]) -> str:
    if not results:
        return "No results found."

    blocks = []
    for item in results:
        published = item.get("published_date", "")
        header = f"Title: {item.get('title', 'Untitled')}"
        if published:
            header += f"\nPublished: {published}"

        blocks.append(
            f"{header}\n"
            f"URL: {item.get('url', '')}\n"
            f"Content: {item.get('content', '')[:600]}"
        )

    return "\n\n---\n\n".join(blocks)


def _cache_quietly(results: list[dict]) -> None:
    """Feed a successful search into the local store. Caching is a bonus,
    never a requirement — a hiccup here must not turn a good search result
    into a failed tool call."""

    try:
        rag.cache_search_results(results)
    except Exception:
        logger.exception("failed to cache search results locally")


def _format_rag(hits: list) -> str:
    if not hits:
        return "No results found in local knowledge base."

    blocks = []
    for hit in hits:
        value = hit.value
        header = f"Title: {value.get('title') or 'Untitled'}"
        published = value.get("published", "")
        if published:
            header += f"\nPublished: {published}"

        blocks.append(
            f"{header}\n"
            f"URL: {value.get('source', '')}\n"
            f"Content: {value.get('text', '')[:600]}"
        )

    return "\n\n---\n\n".join(blocks)


@tool
def rag_search(query: str) -> str:
    """Search previously fetched news and any manually ingested documents.

    Instant and free, unlike news_search/web_search — try this first for
    anything that might already be covered: a follow-up question, something
    likely searched before, or general background. Falls back to an empty
    result (not an error) if the local store has nothing relevant yet.
    """

    return _format_rag(rag.search(query))


@tool
def news_search(query: str) -> str:
    """Search recent news articles for a topic.

    Use this for anything about current events, latest happenings, or
    "what's new" style questions. Returns headlines with dates and source URLs.
    """

    response = _client().search(
        query=query,
        topic="news",
        days=settings.tavily_search_days,
        max_results=settings.tavily_max_results,
        include_answer=False,
    )

    results = response.get("results", [])
    _cache_quietly(results)
    return _format(results)


@tool
def web_search(query: str) -> str:
    """Search the general web for background or reference information.

    Use this when the question is not time-sensitive, or when news results
    lacked the detail needed to answer a follow-up.
    """

    response = _client().search(
        query=query,
        topic="general",
        max_results=settings.tavily_max_results,
        include_answer=False,
    )

    results = response.get("results", [])
    _cache_quietly(results)
    return _format(results)


TOOLS = [rag_search, news_search, web_search]
