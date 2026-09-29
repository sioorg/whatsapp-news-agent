"""FastAPI service exposing the WhatsApp webhooks and the web connector.

Each connector gets its own path, so there is no global provider switch to
get wrong:

  /webhook/meta      Meta WhatsApp Cloud API   (free-form replies, free tier)
  /webhook/whatsapp  Twilio                    (needs an upgraded account)
  /v1/*              OpenAI-compatible         (web frontend, e.g. Open WebUI)
"""

import json
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Callable

from fastapi import BackgroundTasks, FastAPI, File, Request, Response, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel

from app import rag, voice
from app.agent import answer, stream_reply
from app.config import settings
from app.connectors import meta_whatsapp as meta
from app.connectors import openai_compat
from app.connectors import twilio_whatsapp as twilio
from app.connectors.common import InboundMessage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("whatsapp-news-agent")


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the local embedding model at boot, not on a user's first message.

    Uvicorn (and so the deploy's health check) won't start accepting
    connections until this returns, which is the point: a cold ~130MB
    HuggingFace download/load happens once here rather than adding several
    seconds to whichever message happens to trigger the first search after
    a deploy. Never fails startup — a warm-up error (e.g. no network for a
    first-ever download) just means rag_search retries lazily on first use.
    """

    try:
        rag._embedder()
    except Exception:
        logger.exception("failed to warm up the local embedding model at startup")

    yield


app = FastAPI(title="WhatsApp News Agent", lifespan=_lifespan)

FAILURE_REPLY = "Sorry, something went wrong fetching that news. Try again in a moment."
VOICE_FAILURE_REPLY = "Sorry, I couldn't understand that voice message. Try again, or type instead."


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "llm_provider": settings.llm_provider}


def _handle_message(
    message: InboundMessage,
    send_text: Callable[[str, str], None],
    send_voice: Callable[[str, bytes], None] | None = None,
) -> None:
    """Run the agent and send the reply. Executed off the webhook request.

    A voice note arrives here with an empty ``body`` and ``audio_media_id``
    set (see meta_whatsapp.parse_inbound for why the download/transcription
    happens here, in the background task, rather than synchronously while
    parsing the webhook). If transcription fails, there's no text to answer
    with, so this replies with a plain explanation and stops rather than
    calling the agent on empty input.

    ``send_voice`` is only ever passed for Meta (Twilio has no voice-send
    support). If the original message was voice, replying with voice is
    attempted first; any failure there — synthesis or the send itself —
    falls back to a text reply rather than leaving the user with nothing.
    """

    if message.audio_media_id and not message.body:
        try:
            audio_bytes = meta.download_media(message.audio_media_id)
            message.body = voice.transcribe(audio_bytes)
        except Exception:
            logger.exception("failed to transcribe voice note from %s", message.sender)
            try:
                send_text(message.sender, VOICE_FAILURE_REPLY)
            except Exception:
                logger.exception("failed to send transcription-failure reply to %s", message.sender)
            return

    logger.info("handling message from %s: %s", message.sender, message.body[:80])

    try:
        reply = answer(message.body, thread_id=message.sender)
    except Exception:
        logger.exception("agent failed for %s", message.sender)
        reply = FAILURE_REPLY

    if message.reply_as_voice and send_voice is not None:
        try:
            send_voice(message.sender, voice.synthesize(reply))
            return
        except Exception:
            logger.exception(
                "failed to synthesize/send a voice reply to %s, falling back to text",
                message.sender,
            )

    try:
        send_text(message.sender, reply)
    except Exception:
        logger.exception("failed to send reply to %s", message.sender)


# --------------------------------------------------------------------------
# Meta WhatsApp Cloud API
# --------------------------------------------------------------------------


@app.get("/webhook/meta")
def meta_verify(request: Request) -> Response:
    """Answer Meta's subscription handshake.

    Meta GETs this once when you save the callback URL, and expects the
    hub.challenge value echoed back as plain text.
    """

    params = request.query_params

    if params.get("hub.mode") == "subscribe" and params.get(
        "hub.verify_token"
    ) == settings.meta_verify_token():
        logger.info("meta webhook verified")
        return PlainTextResponse(params.get("hub.challenge", ""))

    logger.warning("meta webhook verification failed")
    return Response(status_code=403)


@app.post("/webhook/meta")
async def meta_webhook(request: Request, background: BackgroundTasks) -> Response:
    """Receive inbound messages from the Cloud API.

    Meta retries any webhook it does not get a prompt 200 from, so the reply
    is produced in the background and delivered through the Graph API.
    """

    raw = await request.body()

    if settings.validate_meta_signature:
        signature = request.headers.get("X-Hub-Signature-256", "")

        if not meta.is_valid_signature(signature, raw):
            logger.warning("rejected request with invalid Meta signature")
            return Response(status_code=403)

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("meta webhook received non-JSON body")
        return Response(status_code=400)

    messages = meta.parse_inbound(payload)

    if not messages:
        # Delivery/read receipts and non-text messages arrive here too.
        return Response(status_code=200)

    for message in messages:
        background.add_task(_handle_message, message, meta.send_message, meta.send_voice_message)

    return Response(status_code=200)


# --------------------------------------------------------------------------
# Twilio
# --------------------------------------------------------------------------


@app.post("/webhook/whatsapp")
async def twilio_webhook(request: Request, background: BackgroundTasks) -> Response:
    """Receive an inbound WhatsApp message from Twilio.

    Twilio gives the webhook ~15s before it times out, and a news lookup plus
    LLM call can exceed that. So we acknowledge immediately and send the reply
    asynchronously through the REST API.
    """

    form = dict(await request.form())

    if settings.validate_twilio_signature:
        url = settings.public_base_url.rstrip("/") + "/webhook/whatsapp"
        signature = request.headers.get("X-Twilio-Signature", "")

        if not twilio.is_valid_signature(signature, url, form):
            logger.warning("rejected request with invalid Twilio signature")
            return Response(status_code=403)

    message = twilio.parse_inbound(form)

    if message is None:
        logger.info("ignoring non-text webhook event")
        return Response(status_code=204)

    background.add_task(_handle_message, message, twilio.send_message)

    # Empty TwiML: acknowledge without sending an inline reply.
    # text/xml, not application/xml — Twilio rejects the latter with 12300.
    return Response(
        content='<?xml version="1.0" encoding="UTF-8"?><Response></Response>',
        media_type="text/xml",
    )


# --------------------------------------------------------------------------
# OpenAI-compatible (web frontend, e.g. Open WebUI)
# --------------------------------------------------------------------------


def _unauthorized() -> JSONResponse:
    return JSONResponse(status_code=401, content=openai_compat.unauthorized_response())


@app.get("/v1/models")
def list_models(request: Request):
    if not openai_compat.is_authorized(request.headers.get("authorization")):
        return _unauthorized()
    return openai_compat.models_payload()


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible chat endpoint for the web frontend.

    Meant to be reachable only from inside the Docker network (see the
    README) — unlike the WhatsApp/Twilio webhooks above, which have no
    choice but to be public, this one never needs its own public hostname.
    Stateless per call: see app.agent's module docstring for why this uses
    stream_reply/build_stateless_graph rather than answer()/build_graph().
    """

    if not openai_compat.is_authorized(request.headers.get("authorization")):
        return _unauthorized()

    body = await request.json()
    messages = openai_compat.parse_messages(body)

    if body.get("stream"):
        return StreamingResponse(
            openai_compat.stream_sse(stream_reply(messages)),
            media_type="text/event-stream",
        )

    try:
        content = "".join(stream_reply(messages))
    except Exception:
        logger.exception("agent failed for a web chat request")
        content = FAILURE_REPLY

    return openai_compat.completion_payload(content)


@app.post("/v1/audio/transcriptions")
async def audio_transcriptions(request: Request, file: UploadFile = File(...)):
    """OpenAI-compatible speech-to-text, for Open WebUI's voice input when
    it's configured to use a backend engine rather than its own built-in
    browser Web API option (see the README)."""

    if not openai_compat.is_authorized(request.headers.get("authorization")):
        return _unauthorized()

    audio_bytes = await file.read()

    try:
        text = voice.transcribe(audio_bytes, filename=file.filename or "audio.webm")
    except Exception:
        logger.exception("transcription failed for a web request")
        return JSONResponse(
            status_code=500, content={"error": {"message": "transcription failed"}}
        )

    return {"text": text}


@app.post("/v1/audio/speech")
async def audio_speech(request: Request):
    """OpenAI-compatible text-to-speech, for Open WebUI's voice output when
    it's configured to use a backend engine rather than its own built-in
    browser Web API option (see the README). Returns raw Ogg/Opus bytes,
    the same format sent to WhatsApp."""

    if not openai_compat.is_authorized(request.headers.get("authorization")):
        return _unauthorized()

    body = await request.json()
    text = body.get("input", "")

    try:
        audio_bytes = voice.synthesize(text)
    except Exception:
        logger.exception("speech synthesis failed for a web request")
        return JSONResponse(
            status_code=500, content={"error": {"message": "speech synthesis failed"}}
        )

    return Response(content=audio_bytes, media_type="audio/ogg")


# --------------------------------------------------------------------------
# Local testing
# --------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "local-test"


@app.post("/chat")
def chat(payload: ChatRequest) -> dict:
    """Test the agent without WhatsApp in the loop."""

    return {"reply": answer(payload.message, thread_id=payload.thread_id)}
