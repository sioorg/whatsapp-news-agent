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
    # True when this message arrived as a voice note — signals that the
    # reply should be synthesized back to voice too, matching how the
    # person asked (see main.py's _handle_message).
    reply_as_voice: bool = False
    # Set instead of ``body`` for an inbound voice note (Meta only, for
    # now): downloading and transcribing it is a real network round trip,
    # too slow to do inside parse_inbound, which runs synchronously in the
    # webhook request before Meta's retry timeout. main.py's
    # _handle_message does that step itself, in the background task
    # everything else already runs in, filling ``body`` in before calling
    # the agent.
    audio_media_id: str | None = None
    # Set instead of/alongside ``body`` for an inbound image (Meta only,
    # for now): ``body`` holds the caption if one was sent (may be empty),
    # downloading the image bytes is a real network round trip, same
    # reasoning and same place (main.py's _handle_message) as
    # audio_media_id above.
    image_media_id: str | None = None
    # From the webhook payload's own mime_type field — never assumed, since
    # the model needs the real value to correctly interpret base64 image
    # data (see app.agent's _human_message).
    image_mime_type: str = ""
    # Provider's id for this inbound message (Meta's ``wamid…``). Needed to
    # show a typing indicator, which is attached to the message being
    # answered.
    message_id: str | None = None


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
