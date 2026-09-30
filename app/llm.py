"""LLM factory. Provider is chosen with the LLM_PROVIDER env var."""

from langchain_core.language_models import BaseChatModel

from app.config import settings


def build_llm() -> BaseChatModel:
    """Return a chat model for the configured provider."""

    provider = settings.llm_provider

    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key(),
            temperature=0.2,
            # The underlying groq SDK already retries 429s automatically,
            # honoring the exact wait time Groq's error response specifies
            # (see groq._base_client._calculate_retry_timeout) — no need to
            # hand-roll that. Its default of 2 wasn't enough for a real
            # rate-limit hit (needed ~48s to clear; confirmed via a live
            # 429 on 2026-09-30, ITPM limit on qwen/qwen3.8-27b's free
            # tier), so raised here rather than left at langchain_groq's
            # own default.
            max_retries=5,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        # No temperature here: Claude 5 models reject it as deprecated.
        return ChatAnthropic(
            model=settings.anthropic_model,
            api_key=settings.anthropic_api_key(),
            max_tokens=2048,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER: {provider!r}. Expected 'groq' or 'anthropic'."
    )
