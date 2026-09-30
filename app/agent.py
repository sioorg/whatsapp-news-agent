"""LangGraph news agent with per-conversation memory and a knowledge router.

Every turn first passes through route_query, which classifies the question
as answerable from the local store (app.rag), needing a live web search, or
possibly both — and constrains which tools call_model is allowed to reach
for that turn accordingly. See app/rag.py for what the local store is and
how it's filled.

Two compiled graphs share the same nodes (_build_graph_builder): build_graph
is checkpointed per-thread (WhatsApp, Twilio, /chat — server-side memory
keyed on sender), build_stateless_graph has no checkpointer at all (the
OpenAI-compatible web connector, app/connectors/openai_compat.py — that
protocol has the client resend full history every call, so server-side
memory would be redundant, and worse: each such call needs its own throwaway
thread_id, which would accumulate one dead row per web message forever in
CHECKPOINT_DB's file if it went through the checkpointed graph instead).
"""

import contextvars
from datetime import date
from functools import lru_cache
from typing import Annotated, Iterator, Literal

import sqlite3
from pathlib import Path

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from app import metrics
from app.config import settings
from app.llm import build_llm
from app.tools import (
    TOOLS,
    generate_image,
    get_weather,
    news_search,
    rag_search,
    web_search,
)

# Set by answer()/stream_reply() when app.tools.generate_image ran this
# turn, read (and cleared) by pop_pending_image() right after — see there
# for why this side channel exists instead of putting bytes in the reply
# text. contextvars, not a plain module dict: each request/background task
# gets its own isolated value, so concurrent turns from different users can
# never see each other's pending image. Value is (image_bytes, mime_type)
# — see app.image_gen.generate for why the mime type has to travel with
# the bytes rather than being assumed.
_pending_image: contextvars.ContextVar[tuple[bytes, str] | None] = contextvars.ContextVar(
    "_pending_image", default=None
)


def pop_pending_image() -> tuple[bytes, str] | None:
    """The (image_bytes, mime_type) generate_image produced during the
    most recent answer()/stream_reply() call in this context (None if it
    didn't run, or wasn't called this turn), and clears it. Call once,
    immediately after — see app/main.py's callers."""

    image = _pending_image.get()
    _pending_image.set(None)
    return image


# WhatsApp/Twilio: plaintext, WhatsApp's own *bold*/_italic_ convention (not
# standard Markdown — a single asterisk is italic in most Markdown flavors),
# and a hard length cap matched to one WhatsApp message.
WHATSAPP_SYSTEM_PROMPT = """You are a news assistant that replies over WhatsApp.

Rules:
- If rag_search is available, try it first — it's instant and free. Only \
reach for news_search/web_search if rag_search doesn't have enough, or isn't \
offered to you this turn.
- For anything about current events or "latest" news, call the news_search tool. \
Never answer from memory about recent events.
- For weather questions, call the get_weather tool. Never guess at current \
conditions from memory — weather changes hour to hour.
- If asked to draw, make, or generate an image or picture, call the \
generate_image tool. Never claim to have made an image without calling it.
- Keep replies under 1200 characters. WhatsApp is a chat, not a report.
- Lead with a one-line summary, then up to 5 bullets. Each bullet: headline, \
one sentence of context, then the source URL on the same line.
- Use plain text. WhatsApp supports *bold* and _italic_ only, no markdown headings \
or link syntax.
- If the search returns nothing relevant, say so plainly instead of speculating.
"""

# The web connector (Open WebUI, or any OpenAI-compatible client): a real
# Markdown renderer and no message-length cap, so neither WhatsApp
# constraint applies.
WEB_SYSTEM_PROMPT = """You are a news assistant.

Rules:
- If rag_search is available, try it first — it's instant and free. Only \
reach for news_search/web_search if rag_search doesn't have enough, or isn't \
offered to you this turn.
- For anything about current events or "latest" news, call the news_search tool. \
Never answer from memory about recent events.
- For weather questions, call the get_weather tool. Never guess at current \
conditions from memory — weather changes hour to hour.
- If asked to draw, make, or generate an image or picture, call the \
generate_image tool. Never claim to have made an image without calling it.
- Use standard Markdown: **bold**, _italic_, headings, and bullet lists as \
appropriate. Cite sources as Markdown links, e.g. [source name](https://...).
- If the search returns nothing relevant, say so plainly instead of speculating.
"""

ROUTER_PROMPT = """Classify what kind of information the user's most recent \
message needs, to decide which search tools to offer the assistant this turn.

- "rag": likely answerable from earlier searches or documents already on \
file — a follow-up question, something you'd expect to have been looked up \
before, or general background that doesn't change day to day.
- "web": clearly needs fresh, current information — mentions "latest", \
"today", "breaking", or anything about an ongoing event.
- "both": unsure, or the question plausibly needs both cached context and a \
fresh check.

Classify only the single most recent user message below."""


class Route(BaseModel):
    """Which knowledge sources a turn should draw on."""

    choice: Literal["rag", "web", "both"] = Field(
        description=(
            "'rag' for something likely already covered by earlier searches "
            "or documents; 'web' for something that clearly needs fresh, "
            "current information; 'both' if unsure or it may need both."
        )
    )


# Route -> the tool subset call_model is allowed to use that turn. ToolNode
# itself always gets the full TOOLS list (below) so it can execute whatever
# the model actually called — this only constrains what the model is
# offered, keeping "web" turns from skipping the free local check, and "rag"
# turns from spending a Tavily call on something already on file.
#
# get_weather/generate_image ride along on every route, unlike the search
# tools: neither is about rag-vs-fresh-knowledge at all — an image request
# (or a weather question) can attach to any kind of turn, and gating either
# behind the router would risk a genuine failure mode — a request the
# router misclassifies as "rag" would otherwise have no way to reach it
# (see app/weather.py for get_weather's own version of this reasoning).
ROUTE_TOOLS: dict[str, list] = {
    "rag": [rag_search, get_weather, generate_image],
    "web": [news_search, web_search, get_weather, generate_image],
    "both": TOOLS,
}


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    route: str
    # Which channel's prompt call_model should use this turn. Set once, in
    # the initial state passed to invoke()/stream() — never mutated by a
    # node — so a single conversation always stays on one channel's voice.
    system_prompt: str


def _build_checkpointer():
    """Conversation store: SQLite when CHECKPOINT_DB is set, else in-process.

    Deployments should set CHECKPOINT_DB to a path on a mounted volume so
    history survives container replacement. Left unset, the agent keeps
    history in memory, which is fine for tests and local runs.
    """

    if not settings.checkpoint_db:
        return MemorySaver()

    path = Path(settings.checkpoint_db)
    path.parent.mkdir(parents=True, exist_ok=True)

    # check_same_thread=False: BackgroundTasks runs the agent on worker
    # threads, so the connection is used from a different thread than it
    # was created on.
    connection = sqlite3.connect(path, check_same_thread=False)

    saver = SqliteSaver(connection)
    saver.setup()
    return saver


def _build_graph_builder() -> StateGraph:
    """Construct the shared node/edge wiring. Called once per compiled
    variant (build_graph, build_stateless_graph) — each gets its own
    build_llm() client, a harmless minor duplication, so that either
    variant can be built and cached independently of the other."""

    llm = build_llm()

    def route_query(state: State) -> dict:
        """Classify the latest message so call_model knows which tools to
        offer this turn. Runs once per invoke() — i.e. once per user
        message, before any tool round-trips — and that classification
        holds for the whole turn even across multiple tool calls.

        Fails open to "both" on any router hiccup: a bad classification
        just costs an unnecessary tool offer, never a blocked reply.
        """

        last_human = next(
            (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            None,
        )
        text = last_human.content if last_human else ""

        try:
            with metrics.track_api_call("llm_router"):
                route = llm.with_structured_output(Route).invoke(
                    [SystemMessage(content=ROUTER_PROMPT), HumanMessage(content=text)]
                )
            choice = route.choice
        except Exception:
            choice = "both"

        metrics.ROUTER_DECISIONS.labels(route=choice).inc()
        return {"route": choice}

    def call_model(state: State) -> dict:
        today_line = f"Today's date is {date.today().isoformat()}.\n\n"
        prompt_body = state.get("system_prompt") or WHATSAPP_SYSTEM_PROMPT
        system = SystemMessage(content=today_line + prompt_body)
        tools_for_turn = ROUTE_TOOLS.get(state.get("route", "web"), TOOLS)
        with metrics.track_api_call("llm_answer"):
            response = llm.bind_tools(tools_for_turn).invoke([system, *state["messages"]])
        return {"messages": [response]}

    builder = StateGraph(State)
    builder.add_node("route", route_query)
    builder.add_node("agent", call_model)
    # ToolNode gets every tool regardless of route — it only executes
    # whatever the model actually called, never chooses on its own.
    #
    # handle_tool_errors=True: ToolNode's default only catches LangChain's own
    # ToolInvocationError (bad args), not a real Tavily failure (rate limit,
    # bad key, timeout). Left uncaught, that exception propagates out of
    # invoke() with a dangling, unanswered tool_call already checkpointed —
    # every later message on this thread then resends that malformed history
    # to the LLM and gets rejected, permanently, with no way for the user to
    # recover. Catching it here turns a search failure into a normal
    # ToolMessage instead, so the turn always finishes and the next message
    # starts from a valid state.
    builder.add_node("tools", ToolNode(TOOLS, handle_tool_errors=True))

    builder.add_edge(START, "route")
    builder.add_edge("route", "agent")
    # tools_condition routes to "tools" on a tool call, otherwise to END.
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")

    return builder


@lru_cache(maxsize=1)
def build_graph():
    """Compile the agent graph once and reuse it across requests.

    Checkpointed per-thread — this is the WhatsApp/Twilio/`/chat` path,
    where thread_id is a stable identity (a phone number, or a caller-chosen
    string) and server-side memory across separate requests is the point.
    """

    return _build_graph_builder().compile(checkpointer=_build_checkpointer())


@lru_cache(maxsize=1)
def build_stateless_graph():
    """Same graph, no checkpointer — nothing is ever written to disk here.

    For the OpenAI-compatible web connector: that protocol has the caller
    resend the full conversation on every request, so there's no stable
    per-conversation id to key server-side memory on, and no need for one —
    the input already carries everything. Compiling without a checkpointer
    means each call is correctly one-shot, with no dead thread_id rows
    accumulating in CHECKPOINT_DB.
    """

    return _build_graph_builder().compile()


def answer(text: str, thread_id: str) -> str:
    """Run one user turn through the graph and return the reply text.

    ``thread_id`` scopes conversation memory. Pass the sender's WhatsApp
    number so each user gets their own history.
    """

    _pending_image.set(None)

    result = build_graph().invoke(
        {"messages": [HumanMessage(content=text)]},
        config={
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 12,
        },
    )

    messages = result["messages"]

    # build_graph is checkpointed, so `messages` is the WHOLE conversation
    # history, not just this turn — scanning it naively for a generate_image
    # ToolMessage would resurface an OLD image on a later turn that never
    # asked for one. Scoping to messages after the latest HumanMessage
    # (always this turn's, since it was just appended) avoids that.
    last_human_idx = max(
        (i for i, m in enumerate(messages) if isinstance(m, HumanMessage)),
        default=-1,
    )
    for m in reversed(messages[last_human_idx + 1 :]):
        if isinstance(m, ToolMessage) and m.name == "generate_image" and getattr(m, "artifact", None):
            _pending_image.set(m.artifact)
            break

    reply = messages[-1].content

    if isinstance(reply, list):
        # Anthropic returns content blocks; join the text parts.
        reply = "".join(
            block.get("text", "") for block in reply if isinstance(block, dict)
        )

    return reply.strip() or "I couldn't put together an answer for that one."


def stream_reply(
    messages: list[AnyMessage], *, system_prompt: str = WEB_SYSTEM_PROMPT
) -> Iterator[str]:
    """Stream the final answer's text as it's generated, for a full,
    caller-supplied conversation. Used by the OpenAI-compatible web
    connector — see this module's docstring for why it has no thread_id and
    uses build_stateless_graph rather than build_graph.

    Only forwards content from the "agent" node: route_query's own
    classification call also emits stream chunks (verified empirically —
    it shows up tagged with that node name, carrying a synthetic tool call
    for the Route schema, never real answer text), and call_model's
    tool-picking round produces tool_calls with empty content, not text.
    Filtering on "node is agent AND has content" is what's left after
    excluding both.

    Also watches every chunk (not just "agent" ones) for a generate_image
    ToolMessage, stashing its artifact for pop_pending_image() — no stale-
    history risk here the way answer() has to guard against, since
    build_stateless_graph has no checkpointer: every call starts fresh from
    exactly the caller-supplied ``messages``, nothing accumulated.
    """

    _pending_image.set(None)

    for chunk, metadata in build_stateless_graph().stream(
        {"messages": messages, "system_prompt": system_prompt},
        config={"recursion_limit": 12},
        stream_mode="messages",
    ):
        if (
            isinstance(chunk, ToolMessage)
            and chunk.name == "generate_image"
            and getattr(chunk, "artifact", None)
        ):
            _pending_image.set(chunk.artifact)

        if metadata.get("langgraph_node") == "agent" and chunk.content:
            content = chunk.content
            if isinstance(content, list):
                # Anthropic returns content blocks; join the text parts.
                content = "".join(
                    block.get("text", "") for block in content if isinstance(block, dict)
                )
            if content:
                yield content
