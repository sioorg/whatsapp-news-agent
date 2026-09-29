"""app.tools' own formatting and the caching guard.

Tavily itself is never called here — that's test_integration.py's job for
the real thing, and test_agent.py's job for exercising a failing tool
through the full graph. This file only covers the plain functions in
between: turning results into WhatsApp-ready text, and making sure a
caching hiccup can never surface as a failed search.
"""

import logging

import app.tools as tools_module


def test_format_reports_no_results_found_on_an_empty_list():
    assert tools_module._format([]) == "No results found."


def test_format_includes_title_url_published_and_content():
    result = tools_module._format(
        [
            {
                "title": "Headline",
                "url": "https://example.com/a",
                "content": "body text",
                "published_date": "2026-01-01",
            }
        ]
    )

    assert "Headline" in result
    assert "https://example.com/a" in result
    assert "2026-01-01" in result
    assert "body text" in result


def test_format_truncates_content_to_600_chars():
    result = tools_module._format([{"title": "t", "url": "u", "content": "x" * 1000}])

    assert "x" * 600 in result
    assert "x" * 601 not in result


def test_format_rag_reports_no_results_found_on_an_empty_list():
    assert tools_module._format_rag([]) == "No results found in local knowledge base."


class _FakeHit:
    def __init__(self, value):
        self.value = value


def test_format_rag_includes_title_source_and_content():
    result = tools_module._format_rag(
        [_FakeHit({"title": "Cached", "source": "https://example.com/b", "text": "cached body"})]
    )

    assert "Cached" in result
    assert "https://example.com/b" in result
    assert "cached body" in result


def test_format_rag_falls_back_to_untitled():
    result = tools_module._format_rag([_FakeHit({"source": "u", "text": "t"})])

    assert "Untitled" in result


def test_cache_quietly_swallows_a_failure(monkeypatch, caplog):
    def _boom(results):
        raise RuntimeError("embedding backend down")

    monkeypatch.setattr(tools_module.rag, "cache_search_results", _boom)

    with caplog.at_level(logging.ERROR):
        tools_module._cache_quietly([{"title": "x"}])  # must not raise

    assert "failed to cache" in caplog.text


def test_news_search_caches_its_results(monkeypatch):
    cached = []
    monkeypatch.setattr(tools_module.rag, "cache_search_results", cached.append)
    monkeypatch.setattr(
        tools_module,
        "_client",
        lambda: type(
            "FakeClient",
            (),
            {"search": lambda self, **_: {"results": [{"title": "t", "url": "u"}]}},
        )(),
    )

    tools_module.news_search.invoke({"query": "AI"})

    assert cached == [[{"title": "t", "url": "u"}]]


def test_get_weather_delegates_to_the_weather_module(monkeypatch):
    monkeypatch.setattr(tools_module.weather, "get_report", lambda location: f"report for {location}")

    result = tools_module.get_weather.invoke({"location": "Bengaluru"})

    assert result == "report for Bengaluru"


def test_get_weather_is_never_cached_into_rag(monkeypatch):
    """Unlike news_search/web_search, weather must never be fed into the
    shared local store — a stale cached reading served up later as
    "current" would be actively wrong, not just outdated context (see
    app/weather.py's module docstring)."""

    cached = []
    monkeypatch.setattr(tools_module.rag, "cache_search_results", cached.append)
    monkeypatch.setattr(tools_module.weather, "get_report", lambda location: "sunny")

    tools_module.get_weather.invoke({"location": "Bengaluru"})

    assert cached == []
