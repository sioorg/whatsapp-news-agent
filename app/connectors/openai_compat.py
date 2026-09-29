"""OpenAI-compatible connector: makes the agent look like an OpenAI chat
model, so any client that speaks that API can use it with almost no
integration work. This project points Open WebUI (already self-hosted on
the same box) at it as a custom model.

Unlike the WhatsApp/Twilio connectors, there's no inbound webhook here — a
client calls this directly and gets a response (or a stream of one) back on
the same request. The protocol is also stateless server-side: the caller
resends the full conversation every time, which is why app.agent uses
build_stateless_graph/stream_reply for this path rather than the
checkpointed one — see that module's docstring for the reasoning.
"""

import json
import logging
import time
import uuid
from typing import Iterator

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

from app.config import settings

logger = logging.getLogger(__name__)

MODEL_ID = "news-agent"


def is_authorized(authorization_header: str | None) -> bool:
    """Check a Bearer token against the configured key.

    This endpoint is meant to be reachable only from inside the Docker
    network (Open WebUI's container, not the public internet — see the
    README), so this is defense in depth rather than the primary access
    control. It also happens to be what any OpenAI-style client expects to
    send anyway.
    """

    if not authorization_header or not authorization_header.startswith("Bearer "):
        return False
    token = authorization_header.removeprefix("Bearer ")
    return token == settings.openai_compat_api_key()


def models_payload() -> dict:
    """The GET /v1/models response body Open WebUI needs before it will
    even let you select this as a model."""

    return {
        "object": "list",
        "data": [{"id": MODEL_ID, "object": "model", "created": 0, "owned_by": "sioorg"}],
    }


def _content_text(content) -> str:
    """OpenAI messages usually carry plain string content, but some
    clients send a list of content-part dicts instead. Normalize to a
    plain string either way — same pattern app.agent.answer() already
    uses for Anthropic's own content-block replies."""

    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content or ""


def parse_messages(body: dict) -> list[AnyMessage]:
    """Convert an OpenAI chat-completions request body into LangChain
    messages.

    Any system message the client sends is dropped: the agent always uses
    its own WEB_SYSTEM_PROMPT (app/agent.py), since its tool-calling and
    routing behavior depends on a prompt this project controls, not
    whatever a client happens to send. Anything besides user/assistant
    (e.g. a "tool" role) is skipped too — Open WebUI's own resent history
    never includes those.
    """

    messages: list[AnyMessage] = []
    for m in body.get("messages", []):
        role = m.get("role")
        text = _content_text(m.get("content"))
        if role == "user":
            messages.append(HumanMessage(content=text))
        elif role == "assistant":
            messages.append(AIMessage(content=text))
    return messages


def _chunk(request_id: str, created: int, delta: dict, finish_reason: str | None) -> str:
    payload = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": MODEL_ID,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(payload)}\n\n"


def stream_sse(text_chunks: Iterator[str]) -> Iterator[str]:
    """Wrap a stream of plain text (from app.agent.stream_reply) as
    OpenAI-style SSE chat-completion-chunk events.

    A failure partway through can't turn into a different HTTP status —
    the response has already started — so it's caught here and folded into
    one last visible chunk instead of just cutting the stream off with no
    explanation.
    """

    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    yield _chunk(request_id, created, {"role": "assistant"}, None)
    try:
        for text in text_chunks:
            yield _chunk(request_id, created, {"content": text}, None)
    except Exception:
        logger.exception("agent failed mid-stream")
        yield _chunk(
            request_id,
            created,
            {"content": "\n\nSorry, something went wrong answering that."},
            None,
        )
    yield _chunk(request_id, created, {}, "stop")
    yield "data: [DONE]\n\n"


def completion_payload(content: str) -> dict:
    """The non-streaming POST /v1/chat/completions response body."""

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def unauthorized_response() -> dict:
    return {"error": {"message": "Invalid API key", "type": "invalid_request_error"}}
