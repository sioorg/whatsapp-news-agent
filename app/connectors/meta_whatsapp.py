"""Meta WhatsApp Cloud API connector.

Unlike Twilio's trial tier, Meta allows free-form replies inside the 24-hour
customer service window with no template gate, so the agent can answer with
whatever text it produces.
"""

import hashlib
import hmac
import logging

import requests

from app.config import settings
from app.connectors.common import InboundMessage, split_message

logger = logging.getLogger(__name__)

# Meta caps a text body at 4096 characters. Leave headroom.
MAX_BODY_CHARS = 3800

TIMEOUT_SECONDS = 20


def _endpoint() -> str:
    return (
        f"https://graph.facebook.com/{settings.meta_graph_version}"
        f"/{settings.meta_phone_number_id()}/messages"
    )


def _media_info_endpoint(media_id: str) -> str:
    return f"https://graph.facebook.com/{settings.meta_graph_version}/{media_id}"


def _media_upload_endpoint() -> str:
    return (
        f"https://graph.facebook.com/{settings.meta_graph_version}"
        f"/{settings.meta_phone_number_id()}/media"
    )


def parse_inbound(payload: dict) -> list[InboundMessage]:
    """Extract text and voice messages from a Cloud API webhook payload.

    One payload can carry several messages, and most payloads carry none —
    delivery receipts and read receipts arrive on the same endpoint as
    ``statuses`` rather than ``messages``.

    A voice note's ``body`` is left empty here, with only its
    ``audio_media_id`` set — downloading and transcribing it is a real
    network round trip (Meta, then Groq), and this function runs
    synchronously inside the webhook request, before Meta's retry timeout,
    not in the background task everything else (the LLM call, Tavily, TTS)
    already runs in. main.py's _handle_message does that download/
    transcribe step itself, in the background, before calling the agent.
    """

    messages: list[InboundMessage] = []

    for entry in payload.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {})

            # Map wa_id -> profile name so we can label the sender.
            names = {
                contact.get("wa_id"): contact.get("profile", {}).get("name", "")
                for contact in value.get("contacts", [])
            }

            for message in value.get("messages", []):
                msg_type = message.get("type")
                sender = message.get("from", "")

                if not sender:
                    continue

                if msg_type == "text":
                    body = message.get("text", {}).get("body", "").strip()
                    if not body:
                        continue

                    messages.append(
                        InboundMessage(sender=sender, body=body, profile_name=names.get(sender, ""))
                    )

                elif msg_type == "audio":
                    media_id = message.get("audio", {}).get("id")
                    if not media_id:
                        continue

                    messages.append(
                        InboundMessage(
                            sender=sender,
                            body="",
                            audio_media_id=media_id,
                            reply_as_voice=True,
                            profile_name=names.get(sender, ""),
                        )
                    )

                else:
                    logger.info("ignoring %s message", msg_type)

    return messages


def download_media(media_id: str) -> bytes:
    """Fetch a media attachment's bytes. Two requests, both bearer-token
    authenticated: the first resolves the media id to a temporary CDN URL,
    the second downloads from it. Verified directly against a real
    uploaded file, byte-for-byte, before relying on this shape."""

    headers = {"Authorization": f"Bearer {settings.meta_access_token()}"}

    info = requests.get(_media_info_endpoint(media_id), headers=headers, timeout=TIMEOUT_SECONDS)
    info.raise_for_status()

    media = requests.get(info.json()["url"], headers=headers, timeout=TIMEOUT_SECONDS)
    media.raise_for_status()
    return media.content


def upload_media(audio_bytes: bytes) -> str:
    """Upload audio and return its media id, ready to reference in an
    outbound message."""

    headers = {"Authorization": f"Bearer {settings.meta_access_token()}"}
    files = {"file": ("voice.ogg", audio_bytes, "audio/ogg; codecs=opus")}
    data = {"messaging_product": "whatsapp"}

    response = requests.post(
        _media_upload_endpoint(), headers=headers, files=files, data=data, timeout=TIMEOUT_SECONDS
    )

    if response.status_code >= 400:
        raise RuntimeError(f"Meta media upload failed ({response.status_code}): {response.text[:500]}")

    return response.json()["id"]


def send_message(to: str, body: str) -> None:
    """Send a WhatsApp reply, chunked if it exceeds the length cap."""

    headers = {
        "Authorization": f"Bearer {settings.meta_access_token()}",
        "Content-Type": "application/json",
    }

    for chunk in split_message(body, MAX_BODY_CHARS):
        response = requests.post(
            _endpoint(),
            headers=headers,
            json={
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": to,
                "type": "text",
                "text": {"preview_url": False, "body": chunk},
            },
            timeout=TIMEOUT_SECONDS,
        )

        if response.status_code >= 400:
            # Meta returns a JSON error body worth surfacing verbatim.
            raise RuntimeError(
                f"Meta send failed ({response.status_code}): {response.text[:500]}"
            )

        message_id = (response.json().get("messages") or [{}])[0].get("id", "?")
        logger.info("sent message %s to %s", message_id, to)


def send_voice_message(to: str, audio_bytes: bytes) -> None:
    """Send a voice note: upload the audio, then send a message referencing
    it. ``audio_bytes`` must already be Ogg/Opus (see app.voice.synthesize)
    — that's the one format WhatsApp renders as a real, playable voice-note
    bubble rather than a generic file attachment, verified by sending a
    real message end to end and checking how it rendered.

    Unlike send_message, there's no chunking: a voice reply is one clip,
    however long the underlying text was. app.agent's system prompts don't
    currently shorten replies with a voice reply in mind.
    """

    media_id = upload_media(audio_bytes)

    headers = {
        "Authorization": f"Bearer {settings.meta_access_token()}",
        "Content-Type": "application/json",
    }
    response = requests.post(
        _endpoint(),
        headers=headers,
        json={
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "audio",
            "audio": {"id": media_id},
        },
        timeout=TIMEOUT_SECONDS,
    )

    if response.status_code >= 400:
        raise RuntimeError(f"Meta voice send failed ({response.status_code}): {response.text[:500]}")

    message_id = (response.json().get("messages") or [{}])[0].get("id", "?")
    logger.info("sent voice message %s to %s", message_id, to)


def is_valid_signature(signature: str, raw_body: bytes) -> bool:
    """Verify the X-Hub-Signature-256 header against the raw request body."""

    if not signature.startswith("sha256="):
        return False

    expected = hmac.new(
        settings.meta_app_secret().encode(),
        raw_body,
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(expected, signature.removeprefix("sha256="))
