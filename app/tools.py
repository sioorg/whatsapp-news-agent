"""Tools exposed to the LangGraph agent: local knowledge, Tavily for the
web, live weather, and image generation. See app.agent for how a question
is routed to rag, web, or both (get_weather/generate_image are offered on
every route — see there for why), app.rag for the local store, and
app.weather/app.image_gen for those backends."""

import logging
from functools import lru_cache

from langchain_core.tools import tool
from tavily import TavilyClient

from app import image_gen, metrics, rag, weather
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

    metrics.TOOL_CALLS.labels(tool="rag_search").inc()
    hits = rag.search(query)
    metrics.RAG_LOOKUPS.labels(result="hit" if hits else "miss").inc()
    return _format_rag(hits)


@tool
def news_search(query: str) -> str:
    """Search recent news articles for a topic.

    Use this for anything about current events, latest happenings, or
    "what's new" style questions. Returns headlines with dates and source URLs.
    """

    metrics.TOOL_CALLS.labels(tool="news_search").inc()

    with metrics.track_api_call("tavily"):
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

    metrics.TOOL_CALLS.labels(tool="web_search").inc()

    with metrics.track_api_call("tavily"):
        response = _client().search(
            query=query,
            topic="general",
            max_results=settings.tavily_max_results,
            include_answer=False,
        )

    results = response.get("results", [])
    _cache_quietly(results)
    return _format(results)


@tool
def get_weather(location: str, unit: str = "celsius", days: int = 1) -> str:
    """Get the current weather and forecast for a place.

    Prefer a place's current official name over a well-known alias, to avoid
    an ambiguous match — e.g. "Bengaluru" not "Bangalore", "Mumbai" not
    "Bombay", "Kolkata" not "Calcutta", "Chennai" not "Madras". Add the
    country if the name alone could mean more than one place internationally
    (e.g. "Springfield, Illinois" or "Cambridge, UK").

    unit: "celsius" (default) or "fahrenheit" — use fahrenheit only if the
    user asks for it explicitly, or the conversation makes clear they think
    in it (e.g. they mentioned a US location or gave a temperature in °F).

    days: how many days of forecast to include, 1 (default, today only) up
    to 7. Use more than 1 only when the question is about future days —
    "this weekend", "next week", "the next few days" — not for "right now"
    or "today" questions.
    """

    metrics.TOOL_CALLS.labels(tool="get_weather").inc()
    return weather.get_report(location, unit=unit, days=days)


@tool(response_format="content_and_artifact")
def generate_image(prompt: str) -> tuple[str, tuple[bytes, str]]:
    """Generate an image from a text description and send it to the user.

    Use this whenever the user explicitly asks for an image, picture,
    photo, drawing, or illustration of something — not for describing an
    existing image (this tool only creates new ones from text).

    prompt: a clear, self-contained description of what to draw — expand a
    short/vague user request into something concrete rather than passing
    their exact words verbatim (e.g. "a golden retriever puppy playing in
    autumn leaves, photorealistic" rather than just "a dog").
    """

    metrics.TOOL_CALLS.labels(tool="generate_image").inc()
    image_bytes, mime_type = image_gen.generate(prompt)
    # The image itself never goes to the LLM as text (that would mean
    # base64-encoding it into the conversation, wasting a huge number of
    # tokens for something the model can't even see) — it rides along as
    # this ToolMessage's `artifact`, which app.agent.answer()/stream_reply()
    # pull out directly to actually send, bypassing the model entirely.
    return f"Generated an image for: {prompt}", (image_bytes, mime_type)


TOOLS = [rag_search, news_search, web_search, get_weather, generate_image]
