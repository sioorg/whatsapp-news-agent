"""Prometheus metrics for what this agent actually does — message volume,
router/tool usage, turn latency, third-party API outcomes, voice usage, and
RAG cache effectiveness. The monitoring stack's cadvisor/node-exporter
already cover CPU/memory/restarts; these are the business-level signals on
top of that, scraped from GET /metrics (see app/main.py).

Every helper here is pure observability: track_api_call() re-raises
whatever the wrapped call raises, unchanged, so a metrics bug can never
turn a working request into a failed one, and a real failure can never be
hidden by a metrics-recording problem.
"""

import hashlib
import time
from contextlib import contextmanager
from typing import Iterator

from prometheus_client import Counter, Histogram

MESSAGES_RECEIVED = Counter(
    "messages_received_total",
    "Inbound messages received, by channel and sender.",
    ["channel", "user"],
)
MESSAGES_SENT = Counter(
    "messages_sent_total",
    "Replies successfully sent, by channel.",
    ["channel"],
)

ROUTER_DECISIONS = Counter(
    "router_decisions_total",
    "Which knowledge route (rag/web/both) the router picked for a turn.",
    ["route"],
)
TOOL_CALLS = Counter(
    "tool_calls_total",
    "Which tool the agent actually invoked.",
    ["tool"],
)

TURN_DURATION = Histogram(
    "turn_duration_seconds",
    "End-to-end time to produce a reply for one user turn.",
    ["channel"],
)
AGENT_ERRORS = Counter(
    "agent_errors_total",
    "Turns where the agent raised and a fallback reply was sent instead.",
    ["channel"],
)

API_CALLS = Counter(
    "api_calls_total",
    "Outbound calls to third-party APIs, by outcome.",
    ["api", "status"],
)
API_CALL_DURATION = Histogram(
    "api_call_duration_seconds",
    "Latency of outbound calls to third-party APIs.",
    ["api"],
)

VOICE_MESSAGES = Counter(
    "voice_messages_total",
    "Voice notes handled, by direction (in/out) and channel.",
    ["direction", "channel"],
)

IMAGES_UNDERSTOOD = Counter(
    "images_understood_total",
    "Images the user sent for the model to see/describe, by channel — "
    "the reverse direction of generate_image (which is a tool_calls_total entry).",
    ["channel"],
)

RAG_LOOKUPS = Counter(
    "rag_lookups_total",
    "Local knowledge-store lookups, by whether they found anything.",
    ["result"],
)


def hash_user(identifier: str) -> str:
    """Short, non-reversible label for a sender. A raw phone number/thread
    id must never become a metric label — Prometheus/Grafana are queried
    well beyond this app's own access control (see the /metrics auth note
    in app/main.py), and metric labels are far stickier than a log line."""

    return hashlib.sha256(identifier.encode()).hexdigest()[:8]


@contextmanager
def track_api_call(api: str) -> Iterator[None]:
    """Time a third-party call and record success/error. Re-raises
    unchanged — see this module's docstring."""

    start = time.monotonic()
    status = "success"
    try:
        yield
    except Exception:
        status = "error"
        raise
    finally:
        API_CALL_DURATION.labels(api=api).observe(time.monotonic() - start)
        API_CALLS.labels(api=api, status=status).inc()


def timed_generator(channel: str, items: Iterator[str]) -> Iterator[str]:
    """Wrap a text-chunk generator, recording TURN_DURATION from the first
    item pulled to exhaustion. For the streaming web path, where the actual
    end-to-end time can't be measured with a plain before/after timer since
    the caller (StreamingResponse) consumes the generator itself, on its
    own schedule."""

    start = time.monotonic()
    try:
        yield from items
    finally:
        TURN_DURATION.labels(channel=channel).observe(time.monotonic() - start)
