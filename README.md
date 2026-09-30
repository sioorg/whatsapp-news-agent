# WhatsApp News Agent

A LangGraph agent that answers news questions over WhatsApp. Inbound messages
arrive via a webhook; a router first decides whether the question can be
answered from a local knowledge base, needs a live Tavily search, or both;
the reply is delivered back to the sender.

```
WhatsApp ──▶ Meta Cloud API ──▶ POST /webhook/meta ──▶ LangGraph agent
                                      │                     │
                               (200 immediately)         route (rag/web/both)
                                                             │
                                                    rag_search (local, free)
                                                    news_search / web_search
                                                         (Tavily — also
                                                          caches into rag)
                                                             │
WhatsApp ◀── Graph API send ◀──── background task ◀──────────┘
```

Three connectors are supported, each on its own route, so there is no global
switch to misconfigure:

| Route | Connector | Status |
| --- | --- | --- |
| `/webhook/meta` | Meta WhatsApp Cloud API | **Default.** Free tier, free-form replies |
| `/webhook/whatsapp` | Twilio | Requires an **upgraded** account — see below |
| `/v1/chat/completions` | OpenAI-compatible (web frontend) | For Open WebUI or any OpenAI-style client — see below |

## Layout

| File | Purpose |
| --- | --- |
| [app/config.py](app/config.py) | Env-backed settings, resolved once at import |
| [app/llm.py](app/llm.py) | Chat model factory — Groq or Anthropic |
| [app/rag.py](app/rag.py) | Local knowledge store — LangGraph `SqliteStore` + local embeddings |
| [app/weather.py](app/weather.py) | Live weather via Open-Meteo — free, keyless |
| [app/voice.py](app/voice.py) | Speech-to-text/text-to-speech via Groq, plus the ffmpeg conversion voice replies need |
| [app/image_gen.py](app/image_gen.py) | Image generation — Pollinations (free, default) or OpenAI `gpt-image-1` (paid, opt-in) |
| [app/tools.py](app/tools.py) | `rag_search` (local), Tavily's `news_search`/`web_search`, `get_weather`, `generate_image` |
| [app/agent.py](app/agent.py) | LangGraph graph — router + agent/tools loop + per-sender memory |
| [app/connectors/common.py](app/connectors/common.py) | `InboundMessage`, chunking — shared by both WhatsApp connectors |
| [app/connectors/meta_whatsapp.py](app/connectors/meta_whatsapp.py) | Cloud API send/receive, media upload/download, HMAC signature |
| [app/connectors/twilio_whatsapp.py](app/connectors/twilio_whatsapp.py) | Twilio send, form parsing, signature |
| [app/connectors/openai_compat.py](app/connectors/openai_compat.py) | OpenAI-style message conversion, `/v1/models`, SSE streaming |
| [app/main.py](app/main.py) | FastAPI webhooks, `/v1/*`, `/chat` test endpoint, RAG model warm-up |
| [app/metrics.py](app/metrics.py) | Prometheus business metrics — `GET /metrics`, see the Metrics section below |
| [scripts/ingest_docs.py](scripts/ingest_docs.py) | CLI to add your own `.txt`/`.md`/`.pdf` files to the knowledge base |
| [scripts/reset_thread.py](scripts/reset_thread.py) | CLI to clear one WhatsApp number's stuck conversation history |

## Code flow

### The graph

```
            ┌─────────┐
            │  START  │
            └────┬────┘
                 │
                 ▼
          ┌─────────────┐
          │    route    │   route_query()  — agent.py:121
          │             │   classifies rag / web / both — agent.py:138
          └──────┬──────┘
                 │  sets state["route"]; decided once per user
                 │  message, unchanged across this turn's tool calls
                 ▼
          ┌─────────────┐
          │    agent    │   call_model()  — agent.py:147
          │             │   invokes the LLM at agent.py:150, but only
          │             │   offers it ROUTE_TOOLS[route] — see agent.py:79
          └──┬───────┬──┘
             │       │
 tool_calls  │       │  no tool_calls
   present   │       │
             ▼       ▼
       ┌─────────┐  ┌─────┐
       │  tools  │  │ END │
       └────┬────┘  └─────┘
            │        ToolNode(TOOLS) — always holds all 4 tools regardless
            │        of route (agent.py:168): it only executes whatever the
            │        model actually called, never chooses on its own
            │        rag_search / news_search / web_search / get_weather
            └────────────┐
                         │  results appended as a ToolMessage
                         ▼
                    back to agent
```

Wiring lives in [`build_graph()`](app/agent.py#L116). The branch after `agent`
is `tools_condition`, a LangGraph prebuilt: it inspects the last message and
routes to `tools` if it carries tool calls, otherwise to `END`. `route` itself
is unconditional — every turn passes through it exactly once, before `agent`
ever runs.

### One turn, step by step

`call_model` is never called by this codebase — it is *registered* as a node and
LangGraph invokes it. Same pattern as a FastAPI route handler. Compiling the
graph runs nothing; `graph.invoke()` is what starts the engine.

```
answer(text, thread_id)                                   agent.py:179
  └─ graph.invoke({"messages": [HumanMessage]})           agent.py:186
       │
       │  LangGraph engine takes over
       │
       ├─ visit 0 ─▶ route_query(state)        state = [Human]
       │               └─ llm.with_structured_output(Route).invoke(...)
       │                    ◀── Route(choice="web")
       │               state["route"] = "web"
       │
       ├─ visit 1 ─▶ call_model(state)         state = [Human]
       │               └─ llm.bind_tools([news_search, web_search]).invoke(...)
       │                    ◀── AIMessage(tool_calls=['news_search'])
       │
       ├─ tools_condition ──▶ tools
       │               └─ news_search("...") ─▶ Tavily API, then caches
       │                    the results into app/rag.py for next time
       │                    ◀── ToolMessage (~4k chars of articles)
       │
       ├─ visit 2 ─▶ call_model(state)         state = [Human, AI, Tool]
       │               └─ llm.bind_tools([news_search, web_search]).invoke(...)
       │                    ◀── AIMessage(content="Here's the latest…")
       │
       └─ no tool_calls ─▶ END, invoke() returns
```

Three LLM calls for a single question that needs a fresh search: one to
classify the route, one to choose the search, one to write the answer. A
fourth happens if the model searches twice before answering. A question the
router sends down `"rag"` only ever offers `rag_search`, so it can finish in
two LLM calls if the local store already has an answer.

Key points that are easy to miss:

- **Two different LLM calls happen every turn, not one.**
  [agent.py:138](app/agent.py#L138) classifies the question before anything
  else runs; [agent.py:150](app/agent.py#L150) is the one that actually
  answers or picks a tool, and — like before — runs more than once per turn
  because LangGraph re-enters `call_model` after each tool call.
- **The route is decided once per turn, not once per tool round-trip.**
  `route_query` only sits between `START` and `agent`
  ([agent.py:170](app/agent.py#L170)); once a route is chosen, every
  re-entry into `call_model` within that same turn reuses it, even across
  several tool calls in a row.
- **A broken router fails open, never closed.** If the classification call
  itself errors, `route_query` falls back to `"both"`
  ([agent.py:142](app/agent.py#L142)) — offering every tool — rather than
  ever silently restricting what the model can reach.
- **Tavily output goes to the LLM, never to the user.** The `ToolMessage` is
  merged into state by the `add_messages` reducer, so visit 2 sees the raw
  articles as context and summarizes them. The user only ever receives
  LLM-written text.
- **`state` is supplied by LangGraph**, not built by you. Visit 1 receives one
  message, visit 2 receives three.
- **`recursion_limit=12`** ([agent.py:190](app/agent.py#L190)) caps the
  agent↔tools loop so a confused model cannot search forever.

### Tracing it yourself

```bash
PYTHONPATH=. .venv/bin/python -c "
import logging; logging.disable(logging.INFO)
from app.agent import build_graph
from langchain_core.messages import HumanMessage
r = build_graph().invoke(
    {'messages': [HumanMessage(content='latest news on SpaceX')]},
    config={'configurable': {'thread_id': 'trace'}})
for m in r['messages']:
    m.pretty_print()
"
```

## Local knowledge base (router + RAG)

A shared store — not per-sender, unlike the conversation checkpointer —
that everyone's searches feed into and everyone's questions can draw from.
Backed by LangGraph's own `SqliteStore` (see [app/rag.py](app/rag.py)),
which already depends on `sqlite-vec` via `langgraph-checkpoint-sqlite`, so
this needs no separate database service. Embeddings run locally via
[fastembed](https://github.com/qdrant/fastembed) (ONNX, CPU-only) — no API
key, no per-call cost, and far lighter than a torch-based alternative.

**What fills it:**

- Every successful `news_search`/`web_search` result, automatically. A
  caching failure is logged and swallowed — it can never turn a good search
  result into a failed reply (see `_cache_quietly` in
  [app/tools.py](app/tools.py)).
- Whatever you add by hand:

  ```bash
  PYTHONPATH=. .venv/bin/python scripts/ingest_docs.py notes.md report.pdf
  ```

  Point `RAG_DB_PATH` at the same file the running app uses (in Docker,
  that's the path already set in `docker-compose.yml`) — otherwise you're
  populating a database nothing reads from.

**How the router decides what to offer:** every turn, before `call_model`
runs, a small LLM call classifies the latest message as `rag` (likely
already covered — a follow-up, or something you'd expect to have searched
before), `web` (clearly needs fresh information — "latest", "today",
"breaking"), or `both` (unsure). That classification constrains which tools
`call_model` can reach for the rest of the turn — see `ROUTE_TOOLS` in
[app/agent.py](app/agent.py). `ToolNode` itself always holds every tool
regardless, so this only ever narrows what the model is *offered*, never
what it's capable of running if it already called something.

**Config** (see `.env.example`):

| Variable | Default | Notes |
| --- | --- | --- |
| `RAG_DB_PATH` | *(empty)* | SQLite file path. Empty = in-memory, lost on restart |
| `RAG_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Any fastembed-supported model |
| `RAG_EMBEDDING_DIMS` | `384` | Must match the model above if you change it |
| `RAG_MODEL_CACHE_DIR` | *(empty)* | Where the ~130MB model is cached after its first download |
| `RAG_TOP_K` | `4` | Results returned per `rag_search` call |

**Startup warm-up:** the embedding model loads once at boot
([app/main.py](app/main.py)'s `_lifespan`), not on a user's first message —
uvicorn (and so the deploy's health check) won't accept connections until
that finishes. Without `RAG_MODEL_CACHE_DIR` pointed at a mounted volume,
every container restart redownloads the model and pays that cost again.

**Testing:** the default `pytest` layer never touches the real model —
[tests/test_rag.py](tests/test_rag.py) swaps in a small deterministic fake
embedding function, and the router tests in
[tests/test_agent.py](tests/test_agent.py) stub the classification call
entirely. One `integration`-marked test in
[tests/test_integration.py](tests/test_integration.py) proves the real
loop end to end: a first question hits the web and caches its results, a
second, unrelated thread asking the same thing gets it from `rag_search`.

## Weather

[app/weather.py](app/weather.py) calls [Open-Meteo](https://open-meteo.com) —
free, no API key, no signup, no rate limit at this scale — for the
`get_weather` tool. Two calls per question: geocode the place name to
coordinates, then fetch current conditions plus a forecast. The model picks
`unit` (`celsius`, the default, or `fahrenheit`) and `days` (1–7, default 1)
from context — e.g. "in Fahrenheit" or "this weekend" — no fixed
configuration for either.

**Offered on every route** (`rag`, `web`, and `both` — see `ROUTE_TOOLS` in
[app/agent.py](app/agent.py)), unlike the search tools, which the router
does gate. There's no Tavily quota to protect here, and gating it would risk
a real failure: a weather question the router misclassifies as `rag` would
otherwise have no way to get a real answer. It's also never cached into the
local knowledge base (see [Local knowledge base](#local-knowledge-base-router--rag)
above) — unlike a news article, a stale cached reading served up later as
"current" would be actively wrong, not just outdated.

**A real limitation worth knowing, found during development, not fully
solved:** Open-Meteo's geocoding only matches the literal name given, not
aliases. Searching "Bangalore" returns only a tiny, unrelated town in
Pakistan — the actual 8.5-million-person city is indexed solely as
"Bengaluru". When several places *do* share a name (e.g. "Springfield"
matches five US cities), `geocode()` picks the most populous one, which
handles that case — but a well-known alias resolving to the *wrong* place
entirely isn't something population-ranking can catch. Mitigated, not
solved: the tool's docstring asks the calling model to prefer current
official names (leaning on its own general knowledge of common aliases —
"Bengaluru" not "Bangalore", "Mumbai" not "Bombay"), and every report names
the resolved place, region, and country up front, so a wrong match is at
least visible rather than silently trusted.

## Image generation

[app/image_gen.py](app/image_gen.py) backs the `generate_image` tool —
"draw/make/generate me an image of X". Provider-pluggable via
`IMAGE_PROVIDER`, same pattern as `LLM_PROVIDER`:

| | Provider | Cost | Notes |
| --- | --- | --- | --- |
| Default | `pollinations` | Free, keyless | [Pollinations.ai](https://pollinations.ai) — same "no signup, no cost" bar Open-Meteo was picked for. Free-tier images carry a small logo (no account configured). Returns JPEG. |
| Opt-in | `openai` | Paid, per image | `gpt-image-1`, needs `OPENAI_API_KEY`. Better quality/prompt-following; real money per call, so never the default. Returns PNG. |

The two providers return different image formats, which matters: `generate()`
returns `(image_bytes, mime_type)`, not just bytes, so every caller (Meta's
media upload, the web connector's inline markdown) labels the file correctly
regardless of provider — Meta rejects an upload whose declared type doesn't
match the actual bytes.

**Offered on every route**, like `get_weather` and for the same reason: an
image request isn't a rag-vs-fresh-knowledge question, and gating it behind
the router risks the same failure mode (see [Weather](#weather) above).

**How the image actually gets delivered — no reliance on the model
echoing anything.** The image never goes to the LLM as text (that would
mean base64-encoding it into the conversation for a model that can't even
see it, burning a huge number of tokens for nothing). Instead, the tool is
defined with `response_format="content_and_artifact"`: its text return value
goes to the LLM as normal, but the raw bytes ride along on the resulting
`ToolMessage.artifact`, invisible to the model. `app.agent.answer()` and
`stream_reply()` pull it out of the graph's own message history directly
(scoped to the current turn only — `answer()`'s checkpointed history is
otherwise ever-growing, and a naive scan would resurface a stale image from
an earlier turn) and write it to an explicit `image_out: dict` parameter the
caller passes in, which `app/main.py` reads right after. **Not** a
`contextvars` side channel — that was tried first and reverted after a real
production failure (2026-09-30): Starlette's `StreamingResponse` iterates a
sync generator chunk-by-chunk across a thread pool, and a
`contextvars.ContextVar.set()` made while producing one chunk doesn't
reliably survive to a later chunk's resumption, since each resumption can
run in a freshly copied context. A plain dict has no such problem — mutating
and reading it doesn't depend on which thread does it. On WhatsApp this
becomes a real image message (reply text as its caption, capped at Meta's
1024-char limit); on the web connector it's appended as inline Markdown
(`![...](data:image/jpeg;base64,...)`), which Open WebUI and most
Markdown-rendering clients render directly — for a streaming response, as
one final chunk after all the text, since there's no way to know "no more
text is coming" before the underlying generator is fully drained.

**Twilio gets no image support** — same reasoning as voice: Twilio has no
media-send path wired up in this project (trial-tier limitations), so a
Twilio user who asks for an image just gets the model's text description
of what it made, not the picture itself.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env    # then fill in the keys
```

### Keys you need

- `TAVILY_API_KEY` — https://app.tavily.com
- `GROQ_API_KEY` — https://console.groq.com (or `ANTHROPIC_API_KEY`)
- `TWILIO_ACCOUNT_SID` / `TWILIO_AUTH_TOKEN` — Twilio console

Real environment variables take precedence over `.env`, so you can override any
setting per-run: `LLM_PROVIDER=anthropic .venv/bin/uvicorn app.main:app`.

## Run it

```bash
.venv/bin/uvicorn app.main:app --reload --port 8000
```

Test the agent without WhatsApp in the loop:

```bash
curl -s localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"message": "latest news on AI regulation in the EU"}' | jq -r .reply
```

## Connect WhatsApp (Meta Cloud API)

### 1. Create the Meta app

1. Go to [developers.facebook.com/apps](https://developers.facebook.com/apps)
   and create an app of type **Business**.
2. Add the **WhatsApp** product. Meta issues a free test phone number and a
   temporary access token automatically.
3. On **WhatsApp → API Setup**, copy into `.env`:
   - **Phone number ID** → `META_PHONE_NUMBER_ID`
   - **Temporary access token** → `META_ACCESS_TOKEN` (expires in 24h; swap
     for a permanent System User token once it works)
4. On **Settings → Basic**, copy **App Secret** → `META_APP_SECRET`.
5. Invent any string and set it as `META_VERIFY_TOKEN` — Meta only checks that
   the value you configure matches the value this app returns.
6. Still on **API Setup**, add your own number under **To**, and confirm the
   code WhatsApp sends you. Test numbers can only message verified recipients.

### 2. Run and expose the server

```bash
.venv/bin/uvicorn app.main:app --port 8000 --reload
```

In another terminal — `cloudflared` needs no account, unlike ngrok which now
requires signup and an authtoken:

```bash
brew install cloudflared
cloudflared tunnel --url http://localhost:8000
```

### 3. Point Meta at the webhook

On **WhatsApp → Configuration → Webhook**, click **Edit**:

- **Callback URL**: `https://<random>.trycloudflare.com/webhook/meta`
- **Verify token**: the `META_VERIFY_TOKEN` value from `.env`

Click **Verify and save**. Meta GETs the URL immediately and expects
`hub.challenge` echoed back — the log line `meta webhook verified` confirms it
worked.

Then **Manage** the webhook fields and subscribe to **messages**. Without that
subscription Meta verifies the URL but never posts anything to it.

### 4. Message it

WhatsApp a question to the test number shown on the API Setup page. The log
should show `handling message from …` then `sent message wamid.… to …`.

### Notes

- The tunnel URL changes every `cloudflared` restart, so step 3 has to be
  repeated each time. `HTTP 502` through the tunnel means the tunnel is fine
  and uvicorn is down; a timeout means the tunnel itself died.
- Replies are free-form and only allowed inside the 24-hour window opened by
  the user's last message. Outside it Meta requires an approved template.
- Set `VALIDATE_META_SIGNATURE=true` once you have a stable URL. It verifies
  `X-Hub-Signature-256` against `META_APP_SECRET`.

### Troubleshooting Meta

#### Nothing reaches the webhook, but every console setting looks right

The console's **Verify and save** configures the *app's* callback URL. It does
not always create the second, separate link: the **WhatsApp Business Account →
app subscription**. When that link is missing, Meta routes events to its own
first-party app and your URL is never called. Nothing in the UI shows this.

Check which apps your WABA actually notifies:

```bash
.venv/bin/python -c "
import json, requests
from app.config import settings
waba = '<YOUR_WABA_ID>'   # WhatsApp -> API Setup, under the phone number ID
r = requests.get(f'https://graph.facebook.com/{settings.meta_graph_version}/{waba}/subscribed_apps',
                 params={'access_token': settings.meta_access_token()}, timeout=20)
print(json.dumps(r.json(), indent=2))
"
```

If your own app ID is absent — typically only
`WA DevX Webhook Events 1P App` is listed — subscribe it:

```bash
.venv/bin/python -c "
import requests
from app.config import settings
waba = '<YOUR_WABA_ID>'
r = requests.post(f'https://graph.facebook.com/{settings.meta_graph_version}/{waba}/subscribed_apps',
                  params={'access_token': settings.meta_access_token()}, timeout=20)
print(r.status_code, r.text)
"
```

#### The bot stops replying after a day

The token on the API Setup page is **temporary and expires in 24 hours**.
Check any token's expiry and scopes:

```bash
.venv/bin/python -c "
import requests, json
from app.config import settings
t = settings.meta_access_token()
print(json.dumps(requests.get('https://graph.facebook.com/debug_token',
      params={'input_token': t, 'access_token': t}).json(), indent=2))
"
```

For a non-expiring token, create a **System User** in
[Business Settings](https://business.facebook.com/settings) → **Users → System
Users**, assign it the WhatsApp app with `whatsapp_business_messaging` and
`whatsapp_business_management`, and generate a token with **no expiry**.

#### Other quick checks

| Symptom | Cause |
| --- | --- |
| `500` on the verify handshake | Server started before `.env` had the `META_*` values. `--reload` watches `.py` only — restart it. |
| `HTTP 502` through the tunnel | Tunnel fine, uvicorn down |
| Timeout through the tunnel | Tunnel died. Restart it, then update the callback URL in Meta. |
| Replies send but never arrive | Recipient not on the test number's verified list |

## Connect WhatsApp (Twilio) — needs an upgraded account

A Twilio **trial** account cannot deliver custom replies. All three paths are
closed, verified against a live trial account:

| Path | Result |
| --- | --- |
| REST send with `Body` | `21654 ContentSid Required`, even inside the 24h window |
| REST send with a template | Content API returns `20003 not available on a Trial account` |
| Inline TwiML reply | `12300` on every inbound, no message created — trial docs state *"Direct TwiML XML is not supported during response"* |

Once the account is upgraded:

- Register an approved WhatsApp sender and set `TWILIO_WHATSAPP_FROM` to it.
- Point the sender's inbound webhook at `/webhook/whatsapp`.
- Set `VALIDATE_TWILIO_SIGNATURE=true` and `PUBLIC_BASE_URL` to your https
  origin, exactly as Twilio calls it.

## Connect a web frontend (OpenAI-compatible)

`app/connectors/openai_compat.py` makes the agent look like an OpenAI chat
model — `GET /v1/models` and `POST /v1/chat/completions`, with real token
streaming — so any client that speaks that API can use it. This project
points [Open WebUI](https://github.com/open-webui/open-webui) at it, already
self-hosted on the same box, rather than building a bespoke frontend.

**Architecture difference from WhatsApp, worth knowing:** the OpenAI chat
API is stateless server-side — the client resends the full conversation on
every request, rather than the server remembering it by a stable id.
Because of that, this connector uses `stream_reply`/`build_stateless_graph`
(see [app/agent.py](app/agent.py)) instead of `answer`/`build_graph`: no
checkpointer, no thread_id, nothing written to `CHECKPOINT_DB`. A web
conversation lives in Open WebUI's own history, entirely separate from any
WhatsApp thread. The one thing that *is* still shared across every channel
is the local knowledge base — a search triggered from WhatsApp benefits a
later web question and vice versa, since that store was never per-thread to
begin with. The web connector also uses its own system prompt
(`WEB_SYSTEM_PROMPT`) — real Markdown, no WhatsApp-style `*bold*`/length cap
— any system message the client itself sends is dropped, since the agent's
tool-calling and routing behavior depends on a prompt this project controls.

**Security model — read this before exposing it.** These routes live on the
*same* public hostname as `/webhook/*`, `/chat`, and `/health` — there is no
path-level restriction at the Cloudflare tunnel, so `/v1/chat/completions`
is reachable from the internet exactly like everything else in this app.
`OPENAI_COMPAT_API_KEY` (checked by `is_authorized` in the connector) is the
*only* thing gating it. Generate a long random value and treat it as a real
secret, not a formality:

```bash
openssl rand -hex 32
```

**Setup, once the box has this deployed:**

1. Add the generated key to `.env` on the box as `OPENAI_COMPAT_API_KEY`.
   This is a `.env`-only change, not a code change, so pushing to `main`
   does **not** pick it up — recreate the container directly instead:

   ```bash
   cd /home/mysio/my-apps/whatsapp-news-agent
   docker compose up -d --force-recreate
   ```

   (`docker restart` is not enough — it reuses the environment the
   container was originally created with, it doesn't re-read `.env`.)

2. Work out how Open WebUI can reach the agent, which depends on how its
   container is networked:

   ```bash
   docker inspect open-webui --format '{{.HostConfig.NetworkMode}}'
   ```

   - **`host`** — it shares the box's own network stack directly. Since
     this compose file already publishes the agent to `127.0.0.1:8000` on
     the host (for local `curl` debugging), it's already reachable with no
     further setup: use `http://localhost:8000/v1` as the base URL below.
     This is the tighter setup — that port is loopback-only, never exposed
     to the LAN or internet.
   - **`bridge`/`default`** — join it to the same Docker network as the
     agent, so it can reach it by container name:

     ```bash
     docker network connect edge open-webui
     ```

     then use `http://whatsapp-news-agent:8000/v1` as the base URL.
   - **`container:<name>`** — it shares another container's network
     namespace; reachability depends on what that container can already
     reach, figure out from there.

3. In Open WebUI, go to **Admin Settings → Connections**, add an OpenAI API
   connection:
   - **Base URL:** whichever of the two above applies, from step 2
   - **API key:** the same value as `OPENAI_COMPAT_API_KEY`
4. "OneAgent" should now appear as a selectable model. Pick it and send a
   message — replies stream in token by token, with real Markdown and
   source links.
5. **Restrict who can use it.** Open WebUI's own login is the main gate for
   *people*, separate from the API key above (which only gates the
   machine-to-machine connection). Set `ENABLE_SIGNUP=false` on Open WebUI's
   own container/compose file (managed separately from this repo) so only
   accounts you create can sign in — otherwise anyone who finds the public
   Open WebUI URL can register themselves an account and start spending
   your Groq/Tavily quota, which has no natural cap the way WhatsApp's
   5-verified-recipient test-number limit does.

## Voice

[app/voice.py](app/voice.py) adds speech-to-text and text-to-speech, shared
by WhatsApp (a full round trip: send a voice note, get one back) and the web
connector (backend `/v1/audio/*` endpoints, for when Open WebUI's built-in
browser voice isn't what you want — see below).

**Models — pinned to what's actually live, not what docs/search describe.**
Groq's TTS lineup changed recently: `playai-tts` is decommissioned, and the
current model is `canopylabs/orpheus-v1-english`, which has a much narrower
voice/format list than the old one. Both this and `whisper-large-v3` need to
be individually enabled at **console.groq.com/settings/project/limits** (the
same project-level allowlist as the chat model — see "Model availability"
below) — the TTS model separately needs its terms accepted at
**console.groq.com/playground?model=canopylabs%2Forpheus-v1-english**. Until
both are done, transcription/synthesis calls fail with `model_permission_
blocked_project` or `model_terms_required`.

| | Model | Notes |
| --- | --- | --- |
| STT | `whisper-large-v3` | Free tier: 2,000 requests/day, ~8h audio/day |
| TTS | `canopylabs/orpheus-v1-english` | Paid, per character (~$50/1M chars) — one of: `autumn`, `diana`, `hannah`, `austin`, `daniel`, `troy` |

**A real format conversion is unavoidable for WhatsApp, not optional.**
Verified directly, not assumed:
- Groq's TTS model only actually outputs `wav`, despite the SDK's type hint
  listing more formats.
- WhatsApp's Cloud API doesn't accept `audio/wav` for outbound messages at
  all, and of what it does accept, only `audio/ogg;codecs=opus` renders as a
  real, playable voice-note bubble — everything else shows as a generic file
  attachment.

So every synthesized reply goes through **ffmpeg** (`wav` → 16kHz mono
Opus/Ogg, a real system dependency — see the Dockerfile, not a Python
package: no pure-Python Opus encoder is worth trusting over it). Confirmed
end to end by sending a real message to a real number and checking it
played back correctly, not just that the conversion produced a
plausible-looking file.

**WhatsApp inbound is handled in two stages, not one.** A voice note's
download and transcription is real network work (Meta, then Groq) that's
too slow to do while `meta_whatsapp.parse_inbound` is still parsing the
webhook — that function runs synchronously, before Meta's retry timeout.
So `parse_inbound` only records the voice note's media id
(`InboundMessage.audio_media_id`); `main.py`'s `_handle_message` downloads
and transcribes it in the background task everything else (the LLM call,
Tavily, TTS) already runs in. If transcription fails, the person gets a
plain text explanation rather than the agent running on empty input.

**A failed voice reply falls back to text, never silence.** If synthesis or
the send itself fails, `_handle_message` logs it and sends the same reply as
plain text instead of leaving the question unanswered.

**The web side has two independent voice paths, and you likely only need
one.** Open WebUI ships its own browser-based voice (Settings → Audio →
Web API) — free, zero backend work, using whatever your browser/OS
provides. `/v1/audio/transcriptions` and `/v1/audio/speech` exist for
configuring Open WebUI to use *this* backend instead, so voice quality is
identical to the WhatsApp side (the same Groq models) rather than varying
by browser. Same auth as `/v1/chat/completions` — see that section above
for why these routes are public-hostname-reachable and a real
`OPENAI_COMPAT_API_KEY` matters.

**Testing:** the default `pytest` layer mocks every Groq call —
[tests/test_voice.py](tests/test_voice.py) fakes the client entirely for
the STT/TTS wrapper logic, but runs **real ffmpeg** for the conversion step
(a local, deterministic, offline operation, unlike a network call — skipped
gracefully if ffmpeg isn't installed). One `integration`-marked test
synthesizes real speech then transcribes that same audio back, checking
recognizable words survived the round trip — no OS-specific tooling or
committed audio fixture needed for that.

## Metrics

`GET /metrics` ([app/metrics.py](app/metrics.py)) exposes Prometheus-format
business metrics: message/conversation volume (by channel and a hashed,
non-reversible sender id — never a raw phone number), router/tool usage,
per-turn latency, third-party API call outcomes (LLM, Tavily, Meta,
Open-Meteo, Groq STT/TTS), voice usage, and RAG cache hit rate.

**Same access-control model as `/v1/*`, not a formality.** This route lives
on the same public Cloudflare hostname as everything else — `METRICS_API_KEY`
(a separate secret from `OPENAI_COMPAT_API_KEY`, so rotating one never
affects the other) is the real gate, checked as a Bearer token, failing
closed if unset.

Meant to be scraped by the `sioorg/monitoring` stack: the container needs to
join that stack's `monitoring` Docker network (already wired in
`docker-compose.yml`), and Prometheus needs the same `METRICS_API_KEY` value
in its own scrape config — see that repo's README.

## Behaviour notes

- **Why the reply is async.** Twilio times the webhook out at ~15s, and a
  Tavily search plus LLM call regularly exceeds that. The webhook returns empty
  TwiML immediately and the answer is delivered through the REST API from a
  background task.
- **Memory** is keyed on the sender's WhatsApp number, so follow-ups like
  "tell me more about the second one" work. Uses SQLite when `CHECKPOINT_DB`
  is set (production does) so it survives container replacement, in-process
  `MemorySaver` otherwise — see `_build_checkpointer` in
  [app/agent.py](app/agent.py).
- **The local knowledge base is shared, not per-sender** — see
  [Local knowledge base (router + RAG)](#local-knowledge-base-router--rag)
  above. One person's search can answer someone else's later question.
- **Long replies** are split on line boundaries into 1500-char chunks, under
  WhatsApp's 1600-char cap.
- **Non-text, non-audio events** (delivery receipts, images, documents,
  etc.) return 200/204 and are ignored — see [Voice](#voice) above for the
  one exception: Meta audio messages are transcribed, not ignored. Twilio's
  connector still ignores everything but text; voice was only built for
  Meta.

## Model availability

`GROQ_MODEL` defaults to `qwen/qwen3.8-27b`. Groq enables different models per
project — if you get a 403 `model_permission_blocked_project`, check which
models your project allows. The same restriction applies to the STT/TTS
models [Voice](#voice) uses; see that section for both of them.

```bash
.venv/bin/python -c "
import os; from dotenv import load_dotenv; load_dotenv()
from groq import Groq
print([m.id for m in Groq(api_key=os.environ['GROQ_API_KEY']).models.list().data])
"
```
