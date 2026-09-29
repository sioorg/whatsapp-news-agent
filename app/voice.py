"""Speech-to-text and text-to-speech via Groq — shared by every connector
that needs voice: WhatsApp's inbound/outbound audio
(app.connectors.meta_whatsapp) and the web backend's /v1/audio/* endpoints
(app.connectors.openai_compat).

Two separate Groq models, not one: Whisper for transcription, Orpheus for
synthesis. Both model names were verified directly against the live API
before being pinned here, not taken from docs/search results, which were
stale — Groq had just decommissioned `playai-tts` in favor of
`canopylabs/orpheus-v1-english`, a different model with a different (much
narrower) set of supported voices and output formats.
"""

import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

from groq import Groq

from app.config import settings

STT_MODEL = "whisper-large-v3"
TTS_MODEL = "canopylabs/orpheus-v1-english"

# The only voices canopylabs/orpheus-v1-english actually accepts — found by
# deliberately sending an invalid one and reading the resulting error
# message, since neither Groq's docs nor the SDK's type hints list them.
TTS_VOICES = ("autumn", "diana", "hannah", "austin", "daniel", "troy")
DEFAULT_TTS_VOICE = "autumn"

FFMPEG_TIMEOUT_SECONDS = 30


@lru_cache(maxsize=1)
def _client() -> Groq:
    return Groq(api_key=settings.groq_api_key())


def transcribe(audio_bytes: bytes, filename: str = "audio.ogg") -> str:
    """Transcribe speech to text.

    ``filename`` only needs a plausible audio extension — Groq uses it to
    guess the container format, it isn't validated against the actual
    sender in any way.
    """

    result = _client().audio.transcriptions.create(
        model=STT_MODEL,
        file=(filename, audio_bytes),
        response_format="text",
    )
    return str(result).strip()


def _synthesize_wav(text: str, voice: str) -> bytes:
    response = _client().audio.speech.create(
        input=text,
        model=TTS_MODEL,
        voice=voice,
        # Despite the SDK's type hint listing flac/mp3/mulaw/ogg/wav as
        # valid for *some* Groq TTS model, canopylabs/orpheus-v1-english
        # itself only actually accepts "wav" — confirmed by trying "ogg"
        # for real and reading the resulting 400 error.
        response_format="wav",
    )
    return response.read()


def _wav_to_ogg_opus(wav_bytes: bytes) -> bytes:
    """Convert to the one outbound audio format WhatsApp's Cloud API
    renders as a native, playable voice-note bubble.

    Verified directly, not assumed: WhatsApp's Cloud API doesn't accept
    audio/wav for outbound messages at all, and of the formats it does
    accept (audio/ogg;codecs=opus, audio/mpeg, audio/amr, audio/mp4,
    audio/aac), only ogg+opus renders as a real voice note — anything else
    shows up as a generic file attachment. Confirmed by sending a real
    message end to end and checking it played correctly.

    16kHz mono is standard for speech-quality Opus and keeps the file
    small; ffmpeg is a real subprocess dependency (see the Dockerfile),
    not a Python package — there's no pure-Python Opus encoder worth
    trusting over it.
    """

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "in.wav"
        ogg_path = Path(tmp) / "out.ogg"
        wav_path.write_bytes(wav_bytes)

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", str(wav_path),
                "-c:a", "libopus",
                "-ar", "16000",
                "-ac", "1",
                "-b:a", "32k",
                str(ogg_path),
            ],
            check=True,
            capture_output=True,
            timeout=FFMPEG_TIMEOUT_SECONDS,
        )
        return ogg_path.read_bytes()


def synthesize(text: str, voice: str = DEFAULT_TTS_VOICE) -> bytes:
    """Synthesize speech, returned as Ogg/Opus bytes ready to send to
    WhatsApp or serve from /v1/audio/speech."""

    if voice not in TTS_VOICES:
        voice = DEFAULT_TTS_VOICE

    wav_bytes = _synthesize_wav(text, voice)
    return _wav_to_ogg_opus(wav_bytes)
