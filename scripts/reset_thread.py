#!/usr/bin/env python
"""Clear one WhatsApp number's stuck conversation history.

    PYTHONPATH=. .venv/bin/python scripts/reset_thread.py 919902245562

In Docker, run it inside the running container so it uses the same
CHECKPOINT_DB path and dependencies the live app does. PYTHONPATH=/app is
required here — `docker exec`'s working directory is /app (the image's
WORKDIR), but Python only puts a script's own directory on sys.path, not
the CWD, so `from app...` fails to find the app package without it:

    docker exec -e PYTHONPATH=/app whatsapp-news-agent \
        python scripts/reset_thread.py 919902245562

This deletes ONLY that thread's checkpoint rows. Everyone else's history,
and this thread's own next message onward, are untouched — it's the same
effect as that person never having messaged the bot before.

Only works with a SQLite-backed CHECKPOINT_DB, which production uses. With
CHECKPOINT_DB unset (in-memory), the running server's state lives only in
its own process — a separate script process can't reach it, and this says
so rather than silently doing nothing.

Needed only for a thread that got stuck before app/agent.py's
handle_tool_errors=True fix: a dangling, unanswered tool_call left in the
checkpoint gets resent to the LLM and rejected on every message from that
number, forever, since there's no way for the person on WhatsApp to reset
it themselves. That specific cause can't recur going forward, but this
stays useful as a general "unstick one conversation" tool for whatever
future cause turns up.
"""

import sys

from app.agent import _build_checkpointer
from app.config import settings


def main(thread_id: str) -> None:
    if not settings.checkpoint_db:
        # In-memory storage lives only inside the running server's own
        # process. This script is always a separate process, so it would
        # build a brand-new, disconnected MemorySaver and "succeed" at
        # clearing state nothing else ever reads — silently doing nothing
        # useful. There's no fix for that short of restarting the server
        # itself, which is the only thing that actually reaches that memory.
        print("CHECKPOINT_DB is unset (in-memory storage) — this script")
        print("can't reach a running server's in-memory state from outside")
        print("it. Restart the server instead; that clears every thread's")
        print("history, not just this one.")
        raise SystemExit(1)

    checkpointer = _build_checkpointer()
    checkpointer.delete_thread(thread_id)
    print(f"cleared checkpoint history for thread {thread_id!r}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(1)
    main(sys.argv[1])
