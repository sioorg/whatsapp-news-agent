"""HTTP surface: health, both webhooks, and the local chat endpoint."""

import hashlib
import hmac
import json
from unittest.mock import patch

import pytest

from .conftest import APP_SECRET, VERIFY_TOKEN


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# --- Meta verification handshake -----------------------------------------


def test_verification_echoes_the_challenge(client):
    response = client.get(
        "/webhook/meta",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": VERIFY_TOKEN,
            "hub.challenge": "1158201444",
        },
    )

    assert response.status_code == 200
    assert response.text == "1158201444"


def test_verification_rejects_a_wrong_token(client):
    response = client.get(
        "/webhook/meta",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "wrong",
            "hub.challenge": "x",
        },
    )

    assert response.status_code == 403


def test_verification_rejects_a_wrong_mode(client):
    response = client.get(
        "/webhook/meta",
        params={
            "hub.mode": "unsubscribe",
            "hub.verify_token": VERIFY_TOKEN,
            "hub.challenge": "x",
        },
    )

    assert response.status_code == 403


# --- Meta inbound --------------------------------------------------------


def test_inbound_message_triggers_a_reply(client, meta_payload):
    sent = []

    with patch("app.connectors.meta_whatsapp.send_message", lambda to, body: sent.append((to, body))):
        with patch("app.main.answer", return_value="stub reply"):
            response = client.post("/webhook/meta", json=meta_payload)

    assert response.status_code == 200
    assert sent == [("919902245562", "stub reply")]


def test_thread_id_is_the_sender(client, meta_payload):
    """Memory is keyed on the sender, so each user gets their own history."""

    with patch("app.connectors.meta_whatsapp.send_message"):
        with patch("app.main.answer", return_value="stub") as answer:
            client.post("/webhook/meta", json=meta_payload)

    assert answer.call_args.kwargs["thread_id"] == "919902245562"


def test_status_callback_sends_nothing(client):
    payload = {"entry": [{"changes": [{"value": {"statuses": [{"status": "read"}]}}]}]}
    sent = []

    with patch("app.connectors.meta_whatsapp.send_message", lambda to, body: sent.append(to)):
        response = client.post("/webhook/meta", json=payload)

    assert response.status_code == 200
    assert sent == []


def test_malformed_json_is_rejected(client):
    response = client.post(
        "/webhook/meta",
        content=b"not json",
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 400


def test_an_agent_failure_still_sends_something(client, meta_payload):
    """A crash in the agent must not leave the user with silence."""

    sent = []

    with patch("app.connectors.meta_whatsapp.send_message", lambda to, body: sent.append(body)):
        with patch("app.main.answer", side_effect=RuntimeError("boom")):
            response = client.post("/webhook/meta", json=meta_payload)

    assert response.status_code == 200
    assert len(sent) == 1
    assert "went wrong" in sent[0].lower()


def test_a_send_failure_does_not_crash_the_worker(client, meta_payload):
    with patch("app.connectors.meta_whatsapp.send_message", side_effect=RuntimeError("meta down")):
        with patch("app.main.answer", return_value="stub"):
            response = client.post("/webhook/meta", json=meta_payload)

    assert response.status_code == 200


# --- Meta signature verification ------------------------------------------


@pytest.fixture
def signature_required():
    from app.config import settings

    settings.validate_meta_signature = True
    yield
    settings.validate_meta_signature = False


def test_signed_request_is_accepted(client, meta_payload, signature_required):
    raw = json.dumps(meta_payload).encode()
    signature = "sha256=" + hmac.new(APP_SECRET.encode(), raw, hashlib.sha256).hexdigest()
    sent = []

    with patch("app.connectors.meta_whatsapp.send_message", lambda to, body: sent.append(to)):
        with patch("app.main.answer", return_value="stub"):
            response = client.post(
                "/webhook/meta",
                content=raw,
                headers={"content-type": "application/json", "X-Hub-Signature-256": signature},
            )

    assert response.status_code == 200
    assert sent == ["919902245562"]


def test_unsigned_request_is_rejected(client, meta_payload, signature_required):
    response = client.post("/webhook/meta", json=meta_payload)

    assert response.status_code == 403


def test_wrongly_signed_request_is_rejected(client, meta_payload, signature_required):
    response = client.post(
        "/webhook/meta",
        content=json.dumps(meta_payload).encode(),
        headers={"content-type": "application/json", "X-Hub-Signature-256": "sha256=deadbeef"},
    )

    assert response.status_code == 403


# --- Twilio ---------------------------------------------------------------


def test_twilio_inbound_returns_twiml(client):
    with patch("app.connectors.twilio_whatsapp.send_message"):
        with patch("app.main.answer", return_value="stub"):
            response = client.post(
                "/webhook/whatsapp",
                data={"From": "whatsapp:+351912345678", "Body": "hi"},
            )

    assert response.status_code == 200
    # text/xml, not application/xml: Twilio rejects the latter with 12300.
    assert response.headers["content-type"].startswith("text/xml")


def test_twilio_status_callback_is_ignored(client):
    response = client.post("/webhook/whatsapp", data={"MessageStatus": "delivered"})

    assert response.status_code == 204


# --- Local test endpoint --------------------------------------------------


def test_chat_endpoint(client):
    with patch("app.main.answer", return_value="stub reply"):
        response = client.post("/chat", json={"message": "hello"})

    assert response.status_code == 200
    assert response.json() == {"reply": "stub reply"}
