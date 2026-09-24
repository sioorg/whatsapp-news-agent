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


def parse_inbound(payload: dict) -> list[InboundMessage]:
    """Extract text messages from a Cloud API webhook payload.

    One payload can carry several messages, and most payloads carry none —
    delivery receipts and read receipts arrive on the same endpoint as
    ``statuses`` rather than ``messages``.
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
                if message.get("type") != "text":
                    logger.info("ignoring %s message", message.get("type"))
                    continue

                sender = message.get("from", "")
                body = message.get("text", {}).get("body", "").strip()

                if not sender or not body:
                    continue

                messages.append(
                    InboundMessage(
                        sender=sender,
                        body=body,
                        profile_name=names.get(sender, ""),
                    )
                )

    return messages


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
