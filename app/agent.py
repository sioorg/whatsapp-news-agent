"""LangGraph news agent with per-conversation memory and a knowledge router.

Every turn first passes through route_query, which classifies the question
as answerable from the local store (app.rag), needing a live web search, or
possibly both — and constrains which tools call_model is allowed to reach
for that turn accordingly. See app/rag.py for what the local store is and
how it's filled.
"""

from datetime import date
from functools import lru_cache
from typing import Annotated, Literal

import sqlite3
from pathlib import Path

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from app.config import settings
from app.llm import build_llm
from app.tools import TOOLS, news_search, rag_search, web_search

SYSTEM_PROMPT = """You are a news assistant that replies over WhatsApp.

Today's date is {today}.

Rules:
- If rag_search is available, try it first — it's instant and free. Only \
reach for news_search/web_search if rag_search doesn't have enough, or isn't \
offered to you this turn.
- For anything about current events or "latest" news, call the news_search tool. \
Never answer from memory about recent events.
- Keep replies under 1200 characters. WhatsApp is a chat, not a report.
- Lead with a one-line summary, then up to 5 bullets. Each bullet: headline, \
one sentence of context, then the source URL on the same line.
- Use plain text. WhatsApp supports *bold* and _italic_ only, no markdown headings \
or link syntax.
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
ROUTE_TOOLS: dict[str, list] = {
    "rag": [rag_search],
    "web": [news_search, web_search],
    "both": TOOLS,
}


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    route: str


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


@lru_cache(maxsize=1)
def build_graph():
    """Compile the agent graph once and reuse it across requests."""

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
            route = llm.with_structured_output(Route).invoke(
                [SystemMessage(content=ROUTER_PROMPT), HumanMessage(content=text)]
            )
            choice = route.choice
        except Exception:
            choice = "both"

        return {"route": choice}

    def call_model(state: State) -> dict:
        system = SystemMessage(content=SYSTEM_PROMPT.format(today=date.today().isoformat()))
        tools_for_turn = ROUTE_TOOLS.get(state.get("route", "web"), TOOLS)
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

    return builder.compile(checkpointer=_build_checkpointer())


def answer(text: str, thread_id: str) -> str:
    """Run one user turn through the graph and return the reply text.

    ``thread_id`` scopes conversation memory. Pass the sender's WhatsApp
    number so each user gets their own history.
    """

    result = build_graph().invoke(
        {"messages": [HumanMessage(content=text)]},
        config={
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 12,
        },
    )

    reply = result["messages"][-1].content

    if isinstance(reply, list):
        # Anthropic returns content blocks; join the text parts.
        reply = "".join(
            block.get("text", "") for block in reply if isinstance(block, dict)
        )

    return reply.strip() or "I couldn't put together an answer for that one."
