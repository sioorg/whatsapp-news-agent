"""Twilio WhatsApp connector: outbound sending and inbound request parsing.

Note: Twilio *trial* accounts cannot deliver custom replies — free-form REST
sends return `21654 ContentSid Required`, the Content API for creating
templates is blocked, and inline TwiML is unsupported. This connector needs an
upgraded account. See app/connectors/meta_whatsapp.py for the free alternative.
"""

import logging
from functools import lru_cache

from twilio.request_validator import RequestValidator
from twilio.rest import Client

from app.config import settings
from app.connectors.common import InboundMessage, split_message

logger = logging.getLogger(__name__)

# WhatsApp hard-caps a message body at 1600 characters over Twilio.
MAX_BODY_CHARS = 1500


def parse_inbound(form: dict) -> InboundMessage | None:
    """Build a message from Twilio's form POST. None for non-text events."""

    sender = (form.get("From") or "").strip()
    body = (form.get("Body") or "").strip()

    if not sender or not body:
        # Status callbacks, media-only messages, and delivery receipts all
        # land on the same webhook. Nothing to answer.
        return None

    return InboundMessage(
        sender=sender,
        body=body,
        profile_name=(form.get("ProfileName") or "").strip(),
    )


@lru_cache(maxsize=1)
def _client() -> Client:
    return Client(settings.twilio_account_sid(), settings.twilio_auth_token())


def send_message(to: str, body: str) -> None:
    """Send a WhatsApp reply, chunked if it exceeds the length cap."""

    for chunk in split_message(body, MAX_BODY_CHARS):
        message = _client().messages.create(
            from_=settings.twilio_whatsapp_from,
            to=to,
            body=chunk,
        )
        logger.info("sent message %s to %s", message.sid, to)


def is_valid_signature(signature: str, url: str, form: dict) -> bool:
    """Verify the X-Twilio-Signature header against the request."""

    validator = RequestValidator(settings.twilio_auth_token())
    return validator.validate(url, form, signature or "")
