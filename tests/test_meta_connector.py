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
