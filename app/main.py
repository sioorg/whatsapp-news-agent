"""FastAPI service exposing the WhatsApp webhooks and the web connector.

Each connector gets its own path, so there is no global provider switch to
get wrong:

  /webhook/meta      Meta WhatsApp Cloud API   (free-form replies, free tier)
  /webhook/whatsapp  Twilio                    (needs an upgraded account)
  /v1/*              OpenAI-compatible         (web frontend, e.g. Open WebUI)
"""

import base64
import json
import logging
import threading
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncIterator, Callable, Iterator

from fastapi import BackgroundTasks, FastAPI, File, Request, Response, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from app import metrics, rag, voice
from app.agent import answer, pop_pending_image, stream_reply
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
VOICE_ACK_REPLY = "🎙️ Got your voice note, preparing a voice reply…"
# Voice replies take longer (download, transcribe, agent, synthesize, upload),
# and the Cloud API can't show "recording audio…", so if one is still in
# flight after this long, say so explicitly.
VOICE_ACK_DELAY_SECONDS = 5
VOICE_FAILURE_REPLY = "Sorry, I couldn't understand that voice message. Try again, or type instead."


def _image_markdown(image: tuple[bytes, str]) -> str:
    """Inline-image markdown for the web connector — Open WebUI (and most
    Markdown-rendering OpenAI-compatible clients) render a base64 data URI
    directly, no file hosting needed."""

    image_bytes, mime_type = image
    encoded = base64.b64encode(image_bytes).decode()
    return f"\n\n![generated image](data:{mime_type};base64,{encoded})"


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "llm_provider": settings.llm_provider}


def _metrics_authorized(authorization_header: str | None) -> bool:
    """Same pattern as openai_compat.is_authorized, its own separate secret.

    A dedicated key rather than reusing OPENAI_COMPAT_API_KEY: rotating one
    shouldn't force rotating Prometheus's scrape config too, and vice
    versa. Fails closed like that one does — see its docstring."""

    if not authorization_header or not authorization_header.startswith("Bearer "):
        return False
    token = authorization_header.removeprefix("Bearer ")
    try:
        expected = settings.metrics_api_key()
    except RuntimeError:
        return False
    return token == expected


@app.get("/metrics")
def metrics_endpoint(request: Request) -> Response:
    """Business metrics for Prometheus — see app/metrics.py for what's
    tracked and why. Gated the same way /v1/* is: this route shares the
    public Cloudflare hostname with everything else, no path isolation, and
    the numbers here (usage volume, error rates) aren't meant to be public."""

    if not _metrics_authorized(request.headers.get("authorization")):
        return Response(status_code=401)
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# WhatsApp clears a typing indicator after ~25s, so refresh a bit sooner.
TYPING_REFRESH_SECONDS = 20


@contextmanager
def _typing_keepalive(
    message: InboundMessage, send_typing: Callable[[str], None] | None
) -> Iterator[None]:
    """Show "typing…" to the sender until the block exits.

    Best effort: a failed indicator is logged and never blocks the reply.
    The indicator also disappears on its own once the reply is sent.
    """

    if send_typing is None or not message.message_id:
        yield
        return

    done = threading.Event()

    def _run() -> None:
        while not done.is_set():
            try:
                send_typing(message.message_id)
            except Exception:
                logger.warning("typing indicator failed for %s", message.sender, exc_info=True)
            done.wait(TYPING_REFRESH_SECONDS)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    try:
        yield
    finally:
        done.set()


def _handle_message(
    message: InboundMessage,
    send_text: Callable[[str, str], None],
    send_voice: Callable[[str, bytes], None] | None = None,
    send_typing: Callable[[str], None] | None = None,
    send_image: Callable[[str, bytes, str, str], None] | None = None,
    channel: str = "unknown",
) -> None:
    """Run the agent and send the reply, with a typing indicator meanwhile.

    Voice notes additionally get a one-off acknowledgement text if the reply
    hasn't been sent within VOICE_ACK_DELAY_SECONDS.
    """

    ack = None
    if message.reply_as_voice:
        ack = threading.Timer(
            VOICE_ACK_DELAY_SECONDS, _send_voice_ack, args=(message, send_text)
        )
        ack.daemon = True
        ack.start()

    try:
        with _typing_keepalive(message, send_typing):
            _process_message(message, send_text, send_voice, send_image, channel=channel)
    finally:
        if ack is not None:
            ack.cancel()


def _send_voice_ack(message: InboundMessage, send_text: Callable[[str, str], None]) -> None:
    try:
        send_text(message.sender, VOICE_ACK_REPLY)
    except Exception:
        logger.warning("failed to send voice acknowledgement to %s", message.sender, exc_info=True)


def _process_message(
    message: InboundMessage,
    send_text: Callable[[str, str], None],
    send_voice: Callable[[str, bytes], None] | None = None,
    send_image: Callable[[str, bytes, str, str], None] | None = None,
    channel: str = "unknown",
) -> None:
    """Run the agent and send the reply. Executed off the webhook request.

    A voice note arrives here with an empty ``body`` and ``audio_media_id``
    set (see meta_whatsapp.parse_inbound for why the download/transcription
    happens here, in the background task, rather than synchronously while
    parsing the webhook). If transcription fails, there's no text to answer
    with, so this replies with a plain explanation and stops rather than
    calling the agent on empty input.

    ``send_voice``/``send_image`` are only ever passed for Meta (Twilio has
    neither media-send capability in this project). If app.tools.generate_image
    ran this turn, sending the image (with the reply as its caption) takes
    priority over a voice reply — reading "generated an image" aloud would be
    pointless when the actual picture is what was asked for. Failing either
    falls back to a plain text reply rather than leaving the user with nothing.

    ``channel`` only feeds Prometheus labels (app/metrics.py) — it changes
    no behavior here.
    """

    metrics.MESSAGES_RECEIVED.labels(
        channel=channel, user=metrics.hash_user(message.sender)
    ).inc()

    if message.audio_media_id and not message.body:
        metrics.VOICE_MESSAGES.labels(direction="in", channel=channel).inc()
        try:
            with metrics.track_api_call("meta_download"):
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
        with metrics.TURN_DURATION.labels(channel=channel).time():
            reply = answer(message.body, thread_id=message.sender)
    except Exception:
        logger.exception("agent failed for %s", message.sender)
        metrics.AGENT_ERRORS.labels(channel=channel).inc()
        reply = FAILURE_REPLY

    image = pop_pending_image()
    if image is not None and send_image is not None:
        image_bytes, mime_type = image
        try:
            # Meta caps an image caption at 1024 chars — shorter than a
            # text body's own cap, so truncate defensively rather than
            # risk the whole send being rejected over a long reply.
            with metrics.track_api_call(f"{channel}_send"):
                send_image(message.sender, image_bytes, mime_type, reply[:1024])
            metrics.MESSAGES_SENT.labels(channel=channel).inc()
            return
        except Exception:
            logger.exception(
                "failed to send a generated image to %s, falling back to text",
                message.sender,
            )

    if message.reply_as_voice and send_voice is not None:
        try:
            with metrics.track_api_call(f"{channel}_send"):
                send_voice(message.sender, voice.synthesize(reply))
            metrics.MESSAGES_SENT.labels(channel=channel).inc()
            metrics.VOICE_MESSAGES.labels(direction="out", channel=channel).inc()
            return
        except Exception:
            logger.exception(
                "failed to synthesize/send a voice reply to %s, falling back to text",
                message.sender,
            )

    try:
        with metrics.track_api_call(f"{channel}_send"):
            send_text(message.sender, reply)
        metrics.MESSAGES_SENT.labels(channel=channel).inc()
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
        background.add_task(
            _handle_message,
            message,
            meta.send_message,
            meta.send_voice_message,
            meta.send_typing_indicator,
            meta.send_image_message,
            channel="meta",
        )

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

    background.add_task(_handle_message, message, twilio.send_message, channel="twilio")

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
    metrics.MESSAGES_RECEIVED.labels(channel="web", user="web").inc()

    if body.get("stream"):
        # Counts a stream that was started, not necessarily one a client
        # read to completion — the alternative (only counting on full
        # delivery) would need the same generator-consumption tracking
        # timed_generator already does for TURN_DURATION, for one counter.
        metrics.MESSAGES_SENT.labels(channel="web").inc()

        def _text_then_image() -> Iterator[str]:
            # pop_pending_image() only has something once the underlying
            # generator is fully drained (generate_image's artifact is
            # captured mid-stream by app.agent.stream_reply, but there's no
            # way to know "no more text coming" until exhaustion) — so the
            # image, if any, always arrives as one final chunk after all
            # the text, not interleaved with it.
            yield from metrics.timed_generator("web", stream_reply(messages))
            image = pop_pending_image()
            if image is not None:
                yield _image_markdown(image)

        return StreamingResponse(
            openai_compat.stream_sse(_text_then_image()),
            media_type="text/event-stream",
        )

    try:
        with metrics.TURN_DURATION.labels(channel="web").time():
            content = "".join(stream_reply(messages))
    except Exception:
        logger.exception("agent failed for a web chat request")
        metrics.AGENT_ERRORS.labels(channel="web").inc()
        content = FAILURE_REPLY
    else:
        metrics.MESSAGES_SENT.labels(channel="web").inc()
        image = pop_pending_image()
        if image is not None:
            content += _image_markdown(image)

    return openai_compat.completion_payload(content)


@app.post("/v1/audio/transcriptions")
async def audio_transcriptions(request: Request, file: UploadFile = File(...)):
    """OpenAI-compatible speech-to-text, for Open WebUI's voice input when
    it's configured to use a backend engine rather than its own built-in
    browser Web API option (see the README)."""

    if not openai_compat.is_authorized(request.headers.get("authorization")):
        return _unauthorized()

    audio_bytes = await file.read()
    metrics.VOICE_MESSAGES.labels(direction="in", channel="web").inc()

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

    metrics.VOICE_MESSAGES.labels(direction="out", channel="web").inc()
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
