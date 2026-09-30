"""app.image_gen: provider dispatch against mocked HTTP/SDK calls, never a
real API call (that's tests/test_integration.py's job)."""

import base64

import pytest

import app.image_gen as image_gen
from app.config import settings


class _FakeResponse:
    def __init__(self, content: bytes, status: int = 200):
        self.content = content
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


def test_pollinations_returns_jpeg_bytes(monkeypatch):
    monkeypatch.setattr(settings, "image_provider", "pollinations")
    monkeypatch.setattr(
        image_gen.requests, "get", lambda url, params=None, timeout=None: _FakeResponse(b"fake-jpeg-bytes")
    )

    image_bytes, mime_type = image_gen.generate("a cat")

    assert image_bytes == b"fake-jpeg-bytes"
    assert mime_type == "image/jpeg"


def test_pollinations_url_encodes_the_prompt(monkeypatch):
    monkeypatch.setattr(settings, "image_provider", "pollinations")
    seen = {}

    def fake_get(url, params=None, timeout=None):
        seen["url"] = url
        return _FakeResponse(b"x")

    monkeypatch.setattr(image_gen.requests, "get", fake_get)

    image_gen.generate("a cat & a dog")

    assert "image.pollinations.ai/prompt/" in seen["url"]
    assert " " not in seen["url"]


def test_openai_returns_png_bytes(monkeypatch):
    monkeypatch.setattr(settings, "image_provider", "openai")

    raw_png = b"fake-png-bytes"
    encoded = base64.b64encode(raw_png).decode()

    class _FakeImages:
        def generate(self, **kwargs):
            class _Data:
                b64_json = encoded

            class _Response:
                data = [_Data()]

            return _Response()

    class _FakeClient:
        images = _FakeImages()

    monkeypatch.setattr(image_gen, "_openai_client", lambda: _FakeClient())

    image_bytes, mime_type = image_gen.generate("a cat")

    assert image_bytes == raw_png
    assert mime_type == "image/png"


def test_unknown_provider_raises(monkeypatch):
    monkeypatch.setattr(settings, "image_provider", "not-a-real-provider")

    with pytest.raises(ValueError, match="Unknown IMAGE_PROVIDER"):
        image_gen.generate("a cat")
