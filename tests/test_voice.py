"""app.voice: the Groq STT/TTS wrapper logic against a fake client (never
the real API — that's tests/test_integration.py's job), plus the ffmpeg
conversion step against a real, tiny synthetic WAV file (ffmpeg is a local,
deterministic, offline operation, unlike a network call, so there's no
reason to mock it — skipped gracefully if ffmpeg isn't installed, so a
future environment change can't break the fast/mocked layer)."""

import shutil
import struct
import wave
from io import BytesIO

import pytest

import app.voice as voice


class _FakeTranscriptions:
    def __init__(self, text=" transcribed text "):
        self.text = text
        self.last_call = None

    def create(self, **kwargs):
        self.last_call = kwargs
        return self.text


class _FakeSpeech:
    def __init__(self, audio_bytes=b"fake-wav-bytes"):
        self.audio_bytes = audio_bytes
        self.last_call = None

    def create(self, **kwargs):
        self.last_call = kwargs

        class _Response:
            def read(_self):
                return self.audio_bytes

        return _Response()


class _FakeGroqClient:
    def __init__(self, transcriptions=None, speech=None):
        class _Audio:
            pass

        self.audio = _Audio()
        self.audio.transcriptions = transcriptions or _FakeTranscriptions()
        self.audio.speech = speech or _FakeSpeech()


def test_transcribe_strips_and_passes_through(monkeypatch):
    fake = _FakeGroqClient(transcriptions=_FakeTranscriptions(" hello world "))
    monkeypatch.setattr(voice, "_client", lambda: fake)

    result = voice.transcribe(b"audio-bytes", filename="note.ogg")

    assert result == "hello world"
    assert fake.audio.transcriptions.last_call["model"] == voice.STT_MODEL
    assert fake.audio.transcriptions.last_call["file"] == ("note.ogg", b"audio-bytes")


def test_synthesize_wav_uses_the_tts_model_and_requested_voice(monkeypatch):
    fake = _FakeGroqClient()
    monkeypatch.setattr(voice, "_client", lambda: fake)

    voice._synthesize_wav("hello", "daniel")

    call = fake.audio.speech.last_call
    assert call["model"] == voice.TTS_MODEL
    assert call["voice"] == "daniel"
    # Confirmed empirically that this model only accepts "wav" server-side,
    # despite the SDK's type hint listing more formats.
    assert call["response_format"] == "wav"


def test_synthesize_falls_back_to_the_default_voice_for_an_unknown_one(monkeypatch):
    fake = _FakeGroqClient()
    monkeypatch.setattr(voice, "_client", lambda: fake)
    monkeypatch.setattr(voice, "_wav_to_ogg_opus", lambda wav_bytes: b"converted")

    voice.synthesize("hello", voice="not-a-real-voice")

    assert fake.audio.speech.last_call["voice"] == voice.DEFAULT_TTS_VOICE


def test_synthesize_accepts_every_documented_voice(monkeypatch):
    fake = _FakeGroqClient()
    monkeypatch.setattr(voice, "_client", lambda: fake)
    monkeypatch.setattr(voice, "_wav_to_ogg_opus", lambda wav_bytes: b"converted")

    for v in voice.TTS_VOICES:
        voice.synthesize("hello", voice=v)
        assert fake.audio.speech.last_call["voice"] == v


def _tiny_wav() -> bytes:
    """A real, minimal, valid WAV file — silence, 0.1s, 8kHz mono — built
    with the stdlib so this test needs no network and no Groq call."""

    buf = BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(struct.pack("<800h", *([0] * 800)))
    return buf.getvalue()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_wav_to_ogg_opus_produces_real_opus_audio():
    ogg_bytes = voice._wav_to_ogg_opus(_tiny_wav())

    assert ogg_bytes[:4] == b"OggS"  # the Ogg container's magic bytes


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_wav_to_ogg_opus_raises_on_garbage_input():
    with pytest.raises(Exception):
        voice._wav_to_ogg_opus(b"not a real wav file")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_synthesize_end_to_end_with_real_ffmpeg_but_a_fake_groq_call(monkeypatch):
    """Only the Groq call is faked — the actual conversion step this
    feature depends on runs for real."""

    fake = _FakeGroqClient(speech=_FakeSpeech(audio_bytes=_tiny_wav()))
    monkeypatch.setattr(voice, "_client", lambda: fake)

    result = voice.synthesize("hello")

    assert result[:4] == b"OggS"
