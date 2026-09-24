"""Message chunking, shared by both connectors."""

import pytest

from app.connectors.common import InboundMessage, split_message


def test_short_text_is_not_split():
    assert split_message("hello", 1500) == ["hello"]


def test_text_at_exactly_the_limit_is_not_split():
    text = "x" * 1500
    assert split_message(text, 1500) == [text]


@pytest.mark.parametrize("limit", [1500, 3800])
def test_every_chunk_respects_the_limit(limit):
    text = "word " * 4000

    chunks = split_message(text, limit)

    assert len(chunks) > 1
    assert all(len(chunk) <= limit for chunk in chunks)


def test_split_prefers_line_boundaries():
    # Three lines, each comfortably under the limit but over it combined.
    text = "\n".join(["a" * 40] * 3)

    chunks = split_message(text, 50)

    # No chunk should end mid-line.
    assert all(chunk == "a" * 40 for chunk in chunks)


def test_no_content_is_lost_when_splitting():
    text = "word " * 1000

    rejoined = "".join(split_message(text, 200)).replace(" ", "")

    assert rejoined == text.replace(" ", "")


def test_unbroken_text_still_splits():
    """A single token longer than the limit must not loop forever."""

    chunks = split_message("x" * 5000, 1000)

    assert len(chunks) == 5
    assert all(len(chunk) <= 1000 for chunk in chunks)


def test_inbound_message_defaults_profile_name():
    message = InboundMessage(sender="919902245562", body="hi")

    assert message.profile_name == ""
