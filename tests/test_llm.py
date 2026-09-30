"""app.llm's provider selection — just confirms build_llm wires each
provider's client with the right constructor arguments. Never calls a
real API (that's test_integration.py's job)."""

import pytest

from app.config import settings
from app.llm import build_llm


def test_groq_client_retries_more_than_langchain_groqs_own_default(monkeypatch):
    """langchain_groq defaults max_retries to 2, which wasn't enough to
    clear a real rate-limit hit (needed ~48s; see app/llm.py's comment) —
    pins this higher on purpose. Would regress silently if left
    untested, since the default only bites under real load, never in the
    mocked test suite."""

    monkeypatch.setattr(settings, "llm_provider", "groq")

    llm = build_llm()

    assert llm.max_retries > 2


def test_unknown_provider_raises(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "not-a-real-provider")

    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        build_llm()
