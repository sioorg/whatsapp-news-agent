"""LangGraph news agent with per-conversation memory."""

from datetime import date
from functools import lru_cache
from typing import Annotated

import sqlite3
from pathlib import Path

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from app.config import settings
from app.llm import build_llm
from app.tools import TOOLS

SYSTEM_PROMPT = """You are a news assistant that replies over WhatsApp.

Today's date is {today}.

Rules:
- For anything about current events or "latest" news, call the news_search tool. \
Never answer from memory about recent events.
- Keep replies under 1200 characters. WhatsApp is a chat, not a report.
- Lead with a one-line summary, then up to 5 bullets. Each bullet: headline, \
one sentence of context, then the source URL on the same line.
- Use plain text. WhatsApp supports *bold* and _italic_ only, no markdown headings \
or link syntax.
- If the search returns nothing relevant, say so plainly instead of speculating.
"""


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


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

    llm_with_tools = build_llm().bind_tools(TOOLS)

    def call_model(state: State) -> dict:
        system = SystemMessage(content=SYSTEM_PROMPT.format(today=date.today().isoformat()))
        response = llm_with_tools.invoke([system, *state["messages"]])
        return {"messages": [response]}

    builder = StateGraph(State)
    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(TOOLS))

    builder.add_edge(START, "agent")
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
