# syntax=docker/dockerfile:1

FROM python:3.12-slim AS base

# Fail fast and keep logs unbuffered so docker logs shows output immediately.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ffmpeg: converts Groq's TTS output (WAV) to Ogg/Opus, the one outbound
# audio format WhatsApp's Cloud API renders as a real, playable voice-note
# bubble rather than a generic file attachment (verified directly — see
# app/voice.py). A system package, not a Python one: no pure-Python Opus
# encoder is worth trusting over it. Installed before the pip layer since
# it never changes with app code, keeping that layer's cache valid longer.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Dependencies next so code edits don't invalidate the install layer.
COPY requirements.lock.txt ./
RUN pip install --no-cache-dir -r requirements.lock.txt

COPY app ./app
# Maintenance CLIs (scripts/ingest_docs.py, scripts/reset_thread.py), meant
# to be run with `docker exec` against the live container.
COPY scripts ./scripts

# Writable home for the conversation store; owned by the unprivileged user.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /data \
    && chown -R appuser:appuser /data /app
USER appuser

EXPOSE 8000

# Bind 0.0.0.0: the container's loopback is not reachable from the host.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
