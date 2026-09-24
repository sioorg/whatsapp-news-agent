# syntax=docker/dockerfile:1

FROM python:3.12-slim AS base

# Fail fast and keep logs unbuffered so docker logs shows output immediately.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first so code edits don't invalidate the install layer.
COPY requirements.lock.txt ./
RUN pip install --no-cache-dir -r requirements.lock.txt

COPY app ./app

# Writable home for the conversation store; owned by the unprivileged user.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /data \
    && chown -R appuser:appuser /data /app
USER appuser

EXPOSE 8000

# Bind 0.0.0.0: the container's loopback is not reachable from the host.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
