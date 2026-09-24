"""Twilio form parsing."""

from app.connectors import twilio_whatsapp as twilio


def test_parses_a_text_message():
    message = twilio.parse_inbound(
        {"From": "whatsapp:+351912345678", "Body": "hi", "ProfileName": "Sio"}
    )

    assert message is not None
    assert message.sender == "whatsapp:+351912345678"
    assert message.body == "hi"
    assert message.profile_name == "Sio"


def test_ignores_status_callbacks():
    assert twilio.parse_inbound({"MessageStatus": "delivered"}) is None


def test_ignores_media_only_messages():
    assert (
        twilio.parse_inbound({"From": "whatsapp:+351912345678", "NumMedia": "1"}) is None
    )


def test_ignores_whitespace_only_body():
    assert twilio.parse_inbound({"From": "whatsapp:+1", "Body": "   "}) is None


def test_profile_name_is_optional():
    message = twilio.parse_inbound({"From": "whatsapp:+1", "Body": "hi"})

    assert message is not None
    assert message.profile_name == ""
