"""Tavily-backed search tools exposed to the LangGraph agent."""

from functools import lru_cache

from langchain_core.tools import tool
from tavily import TavilyClient

from app.config import settings


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

    return _format(response.get("results", []))


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

    return _format(response.get("results", []))


TOOLS = [news_search, web_search]
