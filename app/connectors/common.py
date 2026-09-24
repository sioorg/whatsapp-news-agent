"""Pieces shared by every WhatsApp connector."""

from dataclasses import dataclass


@dataclass
class InboundMessage:
    """A normalised inbound WhatsApp message, provider-agnostic.

    ``sender`` is whatever address the provider expects back as a recipient:
    ``whatsapp:+3519…`` for Twilio, a bare ``3519…`` wa_id for Meta. It also
    keys conversation memory, so it must be stable per user.
    """

    sender: str
    body: str
    profile_name: str = ""


def split_message(text: str, limit: int) -> list[str]:
    """Split a reply into provider-sized chunks, preferring line boundaries."""

    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    remaining = text

    while len(remaining) > limit:
        window = remaining[:limit]
        split_at = window.rfind("\n")

        if split_at < limit // 2:
            split_at = window.rfind(" ")
        if split_at < limit // 2:
            split_at = limit

        chunks.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()

    if remaining:
        chunks.append(remaining)

    return chunks
