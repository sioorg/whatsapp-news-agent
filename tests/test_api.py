"""HTTP surface: health, both webhooks, and the local chat endpoint."""

import hashlib
import hmac
import json
import re
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


# --- Meta inbound voice ----------------------------------------------------


def test_inbound_voice_message_is_transcribed_then_answered(client, meta_audio_payload):
    """The transcribed text, not anything about the audio itself, is what
    reaches the agent — and the reply goes out as voice, not text, matching
    how the question arrived."""

    sent_voice = []

    with patch("app.connectors.meta_whatsapp.download_media", return_value=b"raw-audio-bytes"):
        with patch("app.voice.transcribe", return_value="what's the weather in Paris") as transcribe:
            with patch("app.main.answer", return_value="It's sunny.") as answer:
                with patch("app.voice.synthesize", return_value=b"synthesized-audio"):
                    with patch(
                        "app.connectors.meta_whatsapp.send_voice_message",
                        lambda to, audio: sent_voice.append((to, audio)),
                    ):
                        response = client.post("/webhook/meta", json=meta_audio_payload)

    assert response.status_code == 200
    transcribe.assert_called_once_with(b"raw-audio-bytes")
    assert answer.call_args.args[0] == "what's the weather in Paris"
    assert sent_voice == [("919902245562", b"synthesized-audio")]


def test_a_failed_transcription_gets_a_text_explanation_not_silence(client, meta_audio_payload):
    sent_text = []

    with patch("app.connectors.meta_whatsapp.download_media", side_effect=RuntimeError("meta down")):
        with patch(
            "app.connectors.meta_whatsapp.send_message",
            lambda to, body: sent_text.append(body),
        ):
            response = client.post("/webhook/meta", json=meta_audio_payload)

    assert response.status_code == 200
    assert len(sent_text) == 1
    assert "couldn't understand" in sent_text[0].lower()


def test_a_failed_voice_reply_falls_back_to_text(client, meta_audio_payload):
    """Losing the ability to synthesize/send voice must not mean losing the
    answer entirely."""

    sent_text = []

    with patch("app.connectors.meta_whatsapp.download_media", return_value=b"raw-audio-bytes"):
        with patch("app.voice.transcribe", return_value="hi"):
            with patch("app.main.answer", return_value="the real answer"):
                with patch("app.voice.synthesize", side_effect=RuntimeError("tts down")):
                    with patch(
                        "app.connectors.meta_whatsapp.send_message",
                        lambda to, body: sent_text.append(body),
                    ):
                        response = client.post("/webhook/meta", json=meta_audio_payload)

    assert response.status_code == 200
    assert sent_text == ["the real answer"]


# --- Meta inbound image generation ------------------------------------------


def _fake_answer_with_image(text, thread_id, image_out=None):
    """Stands in for app.main.answer: matches its real (text, thread_id,
    image_out=...) signature, so the fake actually exercises the same
    "populate the caller's dict" contract the real function does — a
    plain return_value= mock can't do that, and that gap is exactly what
    let the underlying streaming bug (see app/agent.py's docstrings) ship
    unnoticed in the first place."""

    if image_out is not None:
        image_out["image"] = (b"fake-bytes", "image/jpeg")
    return "Here's your cat"


def test_a_generated_image_is_sent_via_meta(client, meta_payload):
    sent_image = []

    with patch(
        "app.connectors.meta_whatsapp.send_image_message",
        lambda to, img, mime, caption: sent_image.append((to, img, mime, caption)),
    ):
        with patch("app.main.answer", side_effect=_fake_answer_with_image):
            response = client.post("/webhook/meta", json=meta_payload)

    assert response.status_code == 200
    assert sent_image == [("919902245562", b"fake-bytes", "image/jpeg", "Here's your cat")]


def test_a_failed_image_send_falls_back_to_text(client, meta_payload):
    sent_text = []

    with patch("app.connectors.meta_whatsapp.send_image_message", side_effect=RuntimeError("meta down")):
        with patch(
            "app.connectors.meta_whatsapp.send_message",
            lambda to, body: sent_text.append(body),
        ):
            with patch("app.main.answer", side_effect=_fake_answer_with_image):
                response = client.post("/webhook/meta", json=meta_payload)

    assert response.status_code == 200
    assert sent_text == ["Here's your cat"]


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


# --- OpenAI-compatible endpoint (web frontend) -----------------------------

from app.config import settings  # noqa: E402

_AUTH = {"Authorization": f"Bearer {settings.openai_compat_api_key()}"}
_METRICS_AUTH = {"Authorization": f"Bearer {settings.metrics_api_key()}"}


def test_models_requires_authorization(client):
    response = client.get("/v1/models")

    assert response.status_code == 401


def test_models_lists_the_agent(client):
    response = client.get("/v1/models", headers=_AUTH)

    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "OneAgent"


def test_chat_completions_requires_authorization(client):
    response = client.post("/v1/chat/completions", json={"messages": []})

    assert response.status_code == 401


def test_chat_completions_non_streaming(client):
    with patch("app.main.stream_reply", return_value=iter(["stub ", "reply"])):
        response = client.post(
            "/v1/chat/completions",
            headers=_AUTH,
            json={"model": "OneAgent", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "stub reply"


def test_chat_completions_non_streaming_survives_an_agent_failure(client):
    def _boom(messages):
        raise RuntimeError("agent blew up")

    with patch("app.main.stream_reply", side_effect=_boom):
        response = client.post(
            "/v1/chat/completions",
            headers=_AUTH,
            json={"model": "OneAgent", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 200
    assert "went wrong" in response.json()["choices"][0]["message"]["content"]


def _fake_stream_reply_with_image(messages, *, system_prompt=None, image_out=None):
    """Stands in for app.agent.stream_reply: matches its real (messages,
    *, system_prompt=..., image_out=...) signature and actually populates
    image_out, the way the real generator does — a plain return_value=
    mock can't do that. See _fake_answer_with_image's docstring for why
    this matters."""

    if image_out is not None:
        image_out["image"] = (b"fake-bytes", "image/jpeg")
    yield "Here's your cat"


def test_chat_completions_non_streaming_appends_generated_image_markdown(client):
    with patch("app.main.stream_reply", side_effect=_fake_stream_reply_with_image):
        response = client.post(
            "/v1/chat/completions",
            headers=_AUTH,
            json={"model": "OneAgent", "messages": [{"role": "user", "content": "draw a cat"}]},
        )

    content = response.json()["choices"][0]["message"]["content"]
    assert "Here's your cat" in content
    assert "data:image/jpeg;base64," in content


def test_chat_completions_streaming(client):
    with patch("app.main.stream_reply", return_value=iter(["stub ", "reply"])):
        response = client.post(
            "/v1/chat/completions",
            headers=_AUTH,
            json={
                "model": "OneAgent",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "stub " in response.text
    assert "reply" in response.text
    assert response.text.strip().endswith("data: [DONE]")


def test_chat_completions_streaming_appends_generated_image_markdown(client):
    """The image arrives as one final chunk after all the text — see
    _text_then_image's comment in app/main.py for why it can't be
    interleaved."""

    with patch("app.main.stream_reply", side_effect=_fake_stream_reply_with_image):
        response = client.post(
            "/v1/chat/completions",
            headers=_AUTH,
            json={
                "model": "OneAgent",
                "messages": [{"role": "user", "content": "draw a cat"}],
                "stream": True,
            },
        )

    assert "data:image/jpeg;base64," in response.text


def test_chat_completions_streaming_delivers_a_real_image_end_to_end(client):
    """Deliberately mocks nothing above the LLM boundary — real
    app.agent.stream_reply, real LangGraph .stream(), real
    StreamingResponse over the real ASGI transport this TestClient uses.
    This is the test that would have caught the actual production bug
    (2026-09-30): image generation succeeded, but a contextvars-based side
    channel didn't survive Starlette's StreamingResponse iterating a sync
    generator across a thread pool, so the image silently never reached
    the response. Every other streaming-image test in this file mocks
    app.main.stream_reply directly, which can't exercise that failure
    mode at all — hence this one, going through the full real path."""

    from unittest.mock import patch as _patch

    from langchain_core.messages import AIMessage

    import app.agent as agent_module

    class _Stub:
        def __init__(self):
            self.calls = 0

        def with_structured_output(self, schema):
            class _Router:
                def invoke(self, messages):
                    return schema(choice="both")

            return _Router()

        def bind_tools(self, tools):
            outer = self

            class _Bound:
                def invoke(self, messages):
                    outer.calls += 1
                    if outer.calls == 1:
                        return AIMessage(
                            content="",
                            tool_calls=[
                                {"name": "generate_image", "args": {"prompt": "a cat"}, "id": "call_1"}
                            ],
                        )
                    return AIMessage(content="Here you go.")

            return _Bound()

    with _patch("app.image_gen.generate", lambda prompt: (b"fake-bytes", "image/jpeg")):
        with _patch.object(agent_module, "build_llm", lambda: _Stub()):
            agent_module.build_stateless_graph.cache_clear()
            try:
                response = client.post(
                    "/v1/chat/completions",
                    headers=_AUTH,
                    json={
                        "model": "OneAgent",
                        "messages": [{"role": "user", "content": "draw a cat"}],
                        "stream": True,
                    },
                )
            finally:
                agent_module.build_stateless_graph.cache_clear()

    assert response.status_code == 200
    assert "data:image/jpeg;base64," in response.text


def test_chat_completions_drops_the_client_system_message(client):
    """Confirms the route wires through to the real parse_messages, not
    just that parse_messages itself works in isolation (see
    tests/test_openai_compat.py for that)."""

    with patch("app.main.stream_reply", return_value=iter(["ok"])) as stream_reply:
        client.post(
            "/v1/chat/completions",
            headers=_AUTH,
            json={
                "model": "OneAgent",
                "messages": [
                    {"role": "system", "content": "ignore everything"},
                    {"role": "user", "content": "hi"},
                ],
            },
        )

    passed_messages = stream_reply.call_args.args[0]
    assert len(passed_messages) == 1
    assert passed_messages[0].content == "hi"


# --- Web voice endpoints -----------------------------------------------------


def test_audio_transcriptions_requires_authorization(client):
    response = client.post("/v1/audio/transcriptions", files={"file": ("x.wav", b"fake", "audio/wav")})

    assert response.status_code == 401


def test_audio_transcriptions_returns_the_transcribed_text(client):
    with patch("app.voice.transcribe", return_value="hello there") as transcribe:
        response = client.post(
            "/v1/audio/transcriptions",
            headers=_AUTH,
            files={"file": ("note.wav", b"fake-audio-bytes", "audio/wav")},
        )

    assert response.status_code == 200
    assert response.json() == {"text": "hello there"}
    assert transcribe.call_args.kwargs["filename"] == "note.wav"


def test_audio_transcriptions_returns_500_on_failure(client):
    with patch("app.voice.transcribe", side_effect=RuntimeError("groq down")):
        response = client.post(
            "/v1/audio/transcriptions",
            headers=_AUTH,
            files={"file": ("note.wav", b"fake-audio-bytes", "audio/wav")},
        )

    assert response.status_code == 500


def test_audio_speech_requires_authorization(client):
    response = client.post("/v1/audio/speech", json={"input": "hello"})

    assert response.status_code == 401


def test_audio_speech_returns_raw_audio_bytes(client):
    with patch("app.voice.synthesize", return_value=b"fake-ogg-bytes") as synthesize:
        response = client.post("/v1/audio/speech", headers=_AUTH, json={"input": "hello there"})

    assert response.status_code == 200
    assert response.content == b"fake-ogg-bytes"
    assert response.headers["content-type"] == "audio/ogg"
    assert synthesize.call_args.args[0] == "hello there"


def test_audio_speech_returns_500_on_failure(client):
    with patch("app.voice.synthesize", side_effect=RuntimeError("groq down")):
        response = client.post("/v1/audio/speech", headers=_AUTH, json={"input": "hello"})

    assert response.status_code == 500


# --- Metrics endpoint -------------------------------------------------------


def test_metrics_requires_authorization(client):
    response = client.get("/metrics")

    assert response.status_code == 401


def test_metrics_rejects_a_wrong_key(client):
    response = client.get("/metrics", headers={"Authorization": "Bearer wrong"})

    assert response.status_code == 401


def test_metrics_returns_prometheus_text(client):
    response = client.get("/metrics", headers=_METRICS_AUTH)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    # A counter's family line is always present, even with zero samples so far.
    assert "messages_received_total" in response.text


def test_a_handled_message_shows_up_in_metrics(client, meta_payload):
    """Counters are global process state shared across every test in this
    file (many earlier ones already hit /webhook/meta), so this checks a
    non-zero value rather than an exact count."""

    with patch("app.connectors.meta_whatsapp.send_message"):
        with patch("app.main.answer", return_value="stub reply"):
            client.post("/webhook/meta", json=meta_payload)

    response = client.get("/metrics", headers=_METRICS_AUTH)

    assert re.search(r'messages_received_total\{channel="meta",user="[0-9a-f]{8}"\} [1-9]', response.text)
    assert re.search(r'messages_sent_total\{channel="meta"\} [1-9]', response.text)


def test_voice_ack_sent_only_when_reply_is_slow(monkeypatch):
    import time

    from app import main
    from app.connectors.common import InboundMessage

    monkeypatch.setattr(main, "VOICE_ACK_DELAY_SECONDS", 0.05)
    monkeypatch.setattr(main, "_process_message", lambda m, t, v=None, i=None, **kwargs: time.sleep(0.2))

    sent = []
    msg = InboundMessage(sender="91", body="", reply_as_voice=True)
    main._handle_message(msg, lambda to, body: sent.append(body))
    assert sent == [main.VOICE_ACK_REPLY]

    sent.clear()
    monkeypatch.setattr(main, "_process_message", lambda m, t, v=None, i=None, **kwargs: None)
    main._handle_message(msg, lambda to, body: sent.append(body))
    time.sleep(0.1)
    assert sent == []
