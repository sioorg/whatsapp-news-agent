# WhatsApp News Agent

A LangGraph agent that answers news questions over WhatsApp. Inbound messages
arrive via a webhook, the agent searches Tavily for recent articles, and the
reply is delivered back to the sender.

```
WhatsApp ──▶ Meta Cloud API ──▶ POST /webhook/meta ──▶ LangGraph agent
                                      │                     │
                               (200 immediately)      news_search / web_search
                                                         (Tavily)
                                                             │
WhatsApp ◀── Graph API send ◀──── background task ◀──────────┘
```

Two providers are supported, each on its own route, so there is no global
switch to misconfigure:

| Route | Provider | Status |
| --- | --- | --- |
| `/webhook/meta` | Meta WhatsApp Cloud API | **Default.** Free tier, free-form replies |
| `/webhook/whatsapp` | Twilio | Requires an **upgraded** account — see below |

## Layout

| File | Purpose |
| --- | --- |
| [app/config.py](app/config.py) | Env-backed settings, resolved once at import |
| [app/llm.py](app/llm.py) | Chat model factory — Groq or Anthropic |
| [app/tools.py](app/tools.py) | Tavily `news_search` and `web_search` tools |
| [app/agent.py](app/agent.py) | LangGraph graph + per-sender conversation memory |
| [app/connectors/common.py](app/connectors/common.py) | `InboundMessage`, chunking — shared by both connectors |
| [app/connectors/meta_whatsapp.py](app/connectors/meta_whatsapp.py) | Cloud API send, payload parsing, HMAC signature |
| [app/connectors/twilio_whatsapp.py](app/connectors/twilio_whatsapp.py) | Twilio send, form parsing, signature |
| [app/main.py](app/main.py) | FastAPI webhooks and `/chat` test endpoint |

## Code flow

### The graph

```
            ┌─────────┐
            │  START  │
            └────┬────┘
                 │
                 ▼
          ┌─────────────┐
          │    agent    │   call_model()  — agent.py:43
          │             │   invokes the LLM at agent.py:45
          └──┬───────┬──┘
             │       │
 tool_calls  │       │  no tool_calls
   present   │       │
             ▼       ▼
       ┌─────────┐  ┌─────┐
       │  tools  │  │ END │
       └────┬────┘  └─────┘
            │        ToolNode(TOOLS) runs
            │        news_search / web_search
            └────────────┐
                         │  results appended as a ToolMessage
                         ▼
                    back to agent
```

Wiring lives in [`build_graph()`](app/agent.py#L38). The branch is
`tools_condition`, a LangGraph prebuilt: it inspects the last message and routes
to `tools` if it carries tool calls, otherwise to `END`.

### One turn, step by step

`call_model` is never called by this codebase — it is *registered* as a node and
LangGraph invokes it. Same pattern as a FastAPI route handler. Compiling the
graph runs nothing; `graph.invoke()` is what starts the engine.

```
answer(text, thread_id)                                   agent.py:63
  └─ graph.invoke({"messages": [HumanMessage]})           agent.py:70
       │
       │  LangGraph engine takes over
       │
       ├─ visit 1 ─▶ call_model(state)         state = [Human]
       │               └─ llm_with_tools.invoke([system, *messages])
       │                    ◀── AIMessage(tool_calls=['news_search'])
       │
       ├─ route ──▶ tools
       │               └─ news_search("...") ─▶ Tavily API
       │                    ◀── ToolMessage (~4k chars of articles)
       │
       ├─ visit 2 ─▶ call_model(state)         state = [Human, AI, Tool]
       │               └─ llm_with_tools.invoke([system, *messages])
       │                    ◀── AIMessage(content="Here's the latest…")
       │
       └─ no tool_calls ─▶ END, invoke() returns
```

Two LLM calls for a single question: one to choose the search, one to write the
answer from the results. A third happens when the model searches twice before
answering.

Key points that are easy to miss:

- **[agent.py:45](app/agent.py#L45) is the only place the LLM is invoked.** It
  runs more than once per turn because LangGraph re-enters the same function
  after each tool call.
- **Tavily output goes to the LLM, never to the user.** The `ToolMessage` is
  merged into state by the `add_messages` reducer, so visit 2 sees the raw
  articles as context and summarizes them. The user only ever receives
  LLM-written text.
- **`state` is supplied by LangGraph**, not built by you. Visit 1 receives one
  message, visit 2 receives three.
- **`recursion_limit=12`** ([agent.py:74](app/agent.py#L74)) caps the
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

## Behaviour notes

- **Why the reply is async.** Twilio times the webhook out at ~15s, and a
  Tavily search plus LLM call regularly exceeds that. The webhook returns empty
  TwiML immediately and the answer is delivered through the REST API from a
  background task.
- **Memory** is keyed on the sender's WhatsApp number, so follow-ups like
  "tell me more about the second one" work. It is in-process
  (`MemorySaver`) and is lost on restart — swap in a persistent checkpointer
  if you need durability across deploys.
- **Long replies** are split on line boundaries into 1500-char chunks, under
  WhatsApp's 1600-char cap.
- **Non-text events** (delivery receipts, media-only messages) return 204 and
  are ignored.

## Model availability

`GROQ_MODEL` defaults to `qwen/qwen3.8-27b`. Groq enables different models per
project — if you get a 403 `model_permission_blocked_project`, check which
models your project allows:

```bash
.venv/bin/python -c "
import os; from dotenv import load_dotenv; load_dotenv()
from groq import Groq
print([m.id for m in Groq(api_key=os.environ['GROQ_API_KEY']).models.list().data])
"
```
