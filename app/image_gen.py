"""Image generation, provider-pluggable like app.llm's LLM_PROVIDER. Used
by the generate_image tool in app/tools.py.

Groq (this project's default LLM provider) has no image-generation API, so
this is a genuinely new dependency either way. Defaults to Pollinations —
free, keyless, no signup — the same "no cost, no account" bar Open-Meteo
was picked for in app/weather.py, so the feature works with zero setup.
OpenAI's gpt-image-1 is implemented too, for when quality matters more
than cost, but IMAGE_PROVIDER defaults to "pollinations" — switching to
"openai" is an explicit opt-in (and needs OPENAI_API_KEY set), never
automatic, since every call then costs real money.
"""

import base64
import urllib.parse
from functools import lru_cache

import requests

from app import metrics
from app.config import settings

POLLINATIONS_URL = "https://image.pollinations.ai/prompt/{prompt}"
TIMEOUT_SECONDS = 60

# Without an account/token, Pollinations' free tier stamps a small logo on
# the image (see its APIDOCS.md) — a fair tradeoff for zero-setup/zero-cost,
# not a bug. Not attempting to suppress it (that needs a Pollinations
# account this project doesn't have).
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024


def _pollinations(prompt: str) -> bytes:
    url = POLLINATIONS_URL.format(prompt=urllib.parse.quote(prompt))
    with metrics.track_api_call("pollinations"):
        response = requests.get(
            url,
            params={"width": DEFAULT_WIDTH, "height": DEFAULT_HEIGHT},
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.content  # JPEG


@lru_cache(maxsize=1)
def _openai_client():
    from openai import OpenAI

    return OpenAI(api_key=settings.openai_api_key())


def _openai(prompt: str) -> bytes:
    with metrics.track_api_call("openai_images"):
        response = _openai_client().images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size=f"{DEFAULT_WIDTH}x{DEFAULT_HEIGHT}",
        )
        return base64.b64decode(response.data[0].b64_json)  # PNG


def generate(prompt: str) -> tuple[bytes, str]:
    """Generate an image from a text prompt. Returns (image_bytes,
    mime_type) — the two providers return different formats (Pollinations:
    JPEG, OpenAI: PNG), and callers (WhatsApp media upload, the web
    connector's inline markdown) need the real one, not a guess, or Meta's
    API rejects a mislabeled upload. Provider chosen by IMAGE_PROVIDER —
    see this module's docstring."""

    provider = settings.image_provider

    if provider == "pollinations":
        return _pollinations(prompt), "image/jpeg"
    if provider == "openai":
        return _openai(prompt), "image/png"

    raise ValueError(
        f"Unknown IMAGE_PROVIDER: {provider!r}. Expected 'pollinations' or 'openai'."
    )
