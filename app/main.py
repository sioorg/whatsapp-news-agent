"""FastAPI service exposing the WhatsApp webhooks.

Two providers are wired up, each on its own path. The route decides which
connector sends the reply, so there is no global provider switch to get wrong:

  /webhook/meta      Meta WhatsApp Cloud API   (free-form replies, free tier)
  /webhook/whatsapp  Twilio                    (needs an upgraded account)
"""

import json
import logging
from typing import Callable

from fastapi import BackgroundTasks, FastAPI, Request, Response
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from app.agent import answer
from app.config import settings
from app.connectors import meta_whatsapp as meta
from app.connectors import twilio_whatsapp as twilio
from app.connectors.common import InboundMessage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("whatsapp-news-agent")

app = FastAPI(title="WhatsApp News Agent")

FAILURE_REPLY = "Sorry, something went wrong fetching that news. Try again in a moment."


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "llm_provider": settings.llm_provider}


def _handle_message(message: InboundMessage, send: Callable[[str, str], None]) -> None:
    """Run the agent and send the reply. Executed off the webhook request."""

    logger.info("handling message from %s: %s", message.sender, message.body[:80])

    try:
        reply = answer(message.body, thread_id=message.sender)
    except Exception:
        logger.exception("agent failed for %s", message.sender)
        reply = FAILURE_REPLY

    try:
        send(message.sender, reply)
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
        background.add_task(_handle_message, message, meta.send_message)

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
# Local testing
# --------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "local-test"


@app.post("/chat")
def chat(payload: ChatRequest) -> dict:
    """Test the agent without WhatsApp in the loop."""

    return {"reply": answer(payload.message, thread_id=payload.thread_id)}
