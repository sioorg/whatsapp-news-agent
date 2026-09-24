"""Environment-backed settings for the WhatsApp news agent."""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

# Real environment variables win over .env, so deployments and one-off shell
# overrides don't get silently clobbered by a local file.
load_dotenv(override=False)


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            "Copy .env.example to .env and fill it in."
        )
    return value


@dataclass
class Settings:
    """All runtime configuration, resolved once at import time."""

    llm_provider: str = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "groq").lower()
    )
    groq_model: str = field(
        default_factory=lambda: os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    )
    anthropic_model: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")
    )

    # Path to the SQLite conversation store. Empty = in-process memory only.
    checkpoint_db: str = field(
        default_factory=lambda: os.getenv("CHECKPOINT_DB", "")
    )

    tavily_max_results: int = field(
        default_factory=lambda: int(os.getenv("TAVILY_MAX_RESULTS", "5"))
    )
    tavily_search_days: int = field(
        default_factory=lambda: int(os.getenv("TAVILY_SEARCH_DAYS", "7"))
    )

    # Twilio sandbox default number. Override once you move to a paid number.
    twilio_whatsapp_from: str = field(
        default_factory=lambda: os.getenv("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")
    )
    validate_twilio_signature: bool = field(
        default_factory=lambda: os.getenv("VALIDATE_TWILIO_SIGNATURE", "false").lower()
        == "true"
    )
    public_base_url: str = field(
        default_factory=lambda: os.getenv("PUBLIC_BASE_URL", "")
    )

    # --- Meta WhatsApp Cloud API ---
    meta_graph_version: str = field(
        default_factory=lambda: os.getenv("META_GRAPH_VERSION", "v21.0")
    )
    validate_meta_signature: bool = field(
        default_factory=lambda: os.getenv("VALIDATE_META_SIGNATURE", "false").lower()
        == "true"
    )

    def tavily_api_key(self) -> str:
        return _require("TAVILY_API_KEY")

    def groq_api_key(self) -> str:
        return _require("GROQ_API_KEY")

    def anthropic_api_key(self) -> str:
        return _require("ANTHROPIC_API_KEY")

    def twilio_account_sid(self) -> str:
        return _require("TWILIO_ACCOUNT_SID")

    def twilio_auth_token(self) -> str:
        return _require("TWILIO_AUTH_TOKEN")

    def meta_phone_number_id(self) -> str:
        return _require("META_PHONE_NUMBER_ID")

    def meta_access_token(self) -> str:
        return _require("META_ACCESS_TOKEN")

    def meta_verify_token(self) -> str:
        return _require("META_VERIFY_TOKEN")

    def meta_app_secret(self) -> str:
        return _require("META_APP_SECRET")


settings = Settings()
