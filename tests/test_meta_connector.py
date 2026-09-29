"""Meta Cloud API payload parsing and signature verification."""

import hashlib
import hmac
import json

from app.connectors import meta_whatsapp as meta

from .conftest import APP_SECRET


def test_parses_a_text_message(meta_payload):
    messages = meta.parse_inbound(meta_payload)

    assert len(messages) == 1
    assert messages[0].sender == "919902245562"
    assert messages[0].body == "latest AI news"
    assert messages[0].profile_name == "Sio"


def test_ignores_delivery_receipts():
    """Status callbacks arrive on the same endpoint and carry no messages."""

    payload = {
        "entry": [
            {"changes": [{"value": {"statuses": [{"id": "wamid.X", "status": "delivered"}]}}]}
        ]
    }

    assert meta.parse_inbound(payload) == []


def test_ignores_non_text_messages():
    payload = {
        "entry": [
            {"changes": [{"value": {"messages": [{"from": "91", "type": "image"}]}}]}
        ]
    }

    assert meta.parse_inbound(payload) == []


def test_ignores_empty_body():
    payload = {
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "messages": [
                                {"from": "91", "type": "text", "text": {"body": "   "}}
                            ]
                        }
                    }
                ]
            }
        ]
    }

    assert meta.parse_inbound(payload) == []


def test_parses_several_messages_in_one_payload():
    payload = {
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "contacts": [{"profile": {"name": "A"}, "wa_id": "111"}],
                            "messages": [
                                {"from": "111", "type": "text", "text": {"body": "one"}},
                                {"from": "222", "type": "text", "text": {"body": "two"}},
                            ],
                        }
                    }
                ]
            }
        ]
    }

    messages = meta.parse_inbound(payload)

    assert [m.body for m in messages] == ["one", "two"]
    # Only the first sender has a contact entry; the second falls back to "".
    assert [m.profile_name for m in messages] == ["A", ""]


def test_empty_payload_is_harmless():
    assert meta.parse_inbound({}) == []


def test_parses_a_voice_message():
    """Body stays empty and audio_media_id is set instead — see
    meta_whatsapp.parse_inbound's docstring for why transcription doesn't
    happen here."""

    payload = {
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "contacts": [{"profile": {"name": "Sio"}, "wa_id": "919902245562"}],
                            "messages": [
                                {
                                    "from": "919902245562",
                                    "type": "audio",
                                    "audio": {"id": "MEDIA123", "mime_type": "audio/ogg; codecs=opus"},
                                }
                            ],
                        }
                    }
                ]
            }
        ]
    }

    messages = meta.parse_inbound(payload)

    assert len(messages) == 1
    assert messages[0].sender == "919902245562"
    assert messages[0].body == ""
    assert messages[0].audio_media_id == "MEDIA123"
    assert messages[0].reply_as_voice is True
    assert messages[0].profile_name == "Sio"


def test_ignores_an_audio_message_with_no_media_id():
    payload = {
        "entry": [{"changes": [{"value": {"messages": [{"from": "91", "type": "audio", "audio": {}}]}}]}]
    }

    assert meta.parse_inbound(payload) == []


def test_download_media_fetches_the_url_then_the_bytes(monkeypatch):
    """Two requests, both bearer-authenticated: resolve the media id to a
    CDN URL, then download from it — verified against the real API before
    writing this test."""

    calls = []

    class _Response:
        def __init__(self, json_body=None, content=b""):
            self._json = json_body
            self.content = content

        def json(self):
            return self._json

        def raise_for_status(self):
            pass

    def fake_get(url, headers=None, timeout=None):
        calls.append((url, headers))
        if "MEDIA123" in url:
            return _Response(json_body={"url": "https://cdn.example.com/x"})
        return _Response(content=b"audio-bytes")

    monkeypatch.setattr(meta.requests, "get", fake_get)

    result = meta.download_media("MEDIA123")

    assert result == b"audio-bytes"
    assert len(calls) == 2
    assert "MEDIA123" in calls[0][0]
    assert calls[1][0] == "https://cdn.example.com/x"
    # Same bearer token on both requests.
    assert calls[0][1]["Authorization"] == calls[1][1]["Authorization"]


def test_upload_media_returns_the_media_id(monkeypatch):
    def fake_post(url, headers=None, files=None, data=None, timeout=None):
        assert data == {"messaging_product": "whatsapp"}
        assert "file" in files

        class _Response:
            status_code = 200

            def json(self):
                return {"id": "NEWMEDIA456"}

        return _Response()

    monkeypatch.setattr(meta.requests, "post", fake_post)

    assert meta.upload_media(b"fake-ogg-bytes") == "NEWMEDIA456"


def test_upload_media_raises_on_failure(monkeypatch):
    def fake_post(url, headers=None, files=None, data=None, timeout=None):
        class _Response:
            status_code = 400
            text = "bad request"

        return _Response()

    monkeypatch.setattr(meta.requests, "post", fake_post)

    try:
        meta.upload_media(b"x")
        assert False, "expected a RuntimeError"
    except RuntimeError as e:
        assert "400" in str(e)


def test_send_voice_message_uploads_then_sends(monkeypatch):
    monkeypatch.setattr(meta, "upload_media", lambda audio_bytes: "UPLOADED_ID")

    sent = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        sent.update(json)

        class _Response:
            status_code = 200

            def json(self):
                return {"messages": [{"id": "wamid.X"}]}

        return _Response()

    monkeypatch.setattr(meta.requests, "post", fake_post)

    meta.send_voice_message("919902245562", b"fake-audio")

    assert sent["type"] == "audio"
    assert sent["audio"] == {"id": "UPLOADED_ID"}
    assert sent["to"] == "919902245562"


def test_send_voice_message_raises_on_send_failure(monkeypatch):
    monkeypatch.setattr(meta, "upload_media", lambda audio_bytes: "UPLOADED_ID")

    def fake_post(url, headers=None, json=None, timeout=None):
        class _Response:
            status_code = 500
            text = "server error"

        return _Response()

    monkeypatch.setattr(meta.requests, "post", fake_post)

    try:
        meta.send_voice_message("919902245562", b"fake-audio")
        assert False, "expected a RuntimeError"
    except RuntimeError as e:
        assert "500" in str(e)


def _sign(body: bytes) -> str:
    return "sha256=" + hmac.new(APP_SECRET.encode(), body, hashlib.sha256).hexdigest()


def test_accepts_a_valid_signature(meta_payload):
    raw = json.dumps(meta_payload).encode()

    assert meta.is_valid_signature(_sign(raw), raw) is True


def test_rejects_a_tampered_body(meta_payload):
    raw = json.dumps(meta_payload).encode()

    assert meta.is_valid_signature(_sign(raw), raw + b"x") is False


def test_rejects_a_malformed_header(meta_payload):
    raw = json.dumps(meta_payload).encode()

    assert meta.is_valid_signature("garbage", raw) is False
    assert meta.is_valid_signature("", raw) is False


def test_rejects_a_signature_from_a_different_secret(meta_payload):
    raw = json.dumps(meta_payload).encode()
    wrong = "sha256=" + hmac.new(b"not-the-secret", raw, hashlib.sha256).hexdigest()

    assert meta.is_valid_signature(wrong, raw) is False
