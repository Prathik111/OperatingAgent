# API reference — the Stage C HTTP surface (`packages/api`)

The native agent's core is headless: the CLI, the evaluation runner and this HTTP server are
all *peer clients* of the same `AgentService`. This package puts that service behind REST +
Server-Sent Events so a browser, a script or another service can drive a run, answer its
permission prompts and read its receipts.

The whole surface is built on a **framework-free ASGI toolkit** (`packages/api/src/api/asgi.py`)
— just enough routing, JSON and SSE, on the standard library. There is no Starlette/FastAPI
dependency; the app is a plain ASGI callable, which is what lets the entire HTTP contract be
tested offline by driving that callable directly with a fake `receive`/`send` pair. The
handlers live in `packages/api/src/api/server.py`.

## Getting an app

```python
from api.server import create_app, serve

app = create_app(service)          # service: an agent_native AgentService
```

`create_app(service)` returns an ASGI application (`async def app(scope, receive, send)`).
Run it under any ASGI server. For convenience, `serve(service, host="127.0.0.1", port=8080)`
runs it with uvicorn — imported lazily, so it is needed only when you actually serve over a
socket (the optional `serve` extra: `uv sync --package api --extra serve`).

## Endpoints

Base paths, exactly as registered in `create_app`:

| Method | Path | Purpose |
|---|---|---|
| GET | `/healthz` | Liveness. Returns `{"status": "ok"}`. |
| GET | `/` , `/ui` | The single-page console (HTML) — a thin client over the routes below. |
| POST | `/sessions` | Create a session. **201.** |
| GET | `/sessions` | List sessions, each with its latest run. |
| GET | `/sessions/{id}` | One session, with its full run history. |
| DELETE | `/sessions/{id}` | Delete a session and everything under it. |
| POST | `/sessions/{id}/messages` | Send a message; **stream the run's events as SSE.** |
| POST | `/sessions/{id}/resume` | Carry an interrupted run to completion; returns its receipt (not streamed). |
| POST | `/sessions/{id}/fork` | Branch the conversation into a new session. **201.** |
| GET | `/sessions/{id}/events` | Replay stored events after a cursor (JSON), or live-stream them (SSE). |
| GET | `/runs/{run_id}` | One run's receipt. |
| GET | `/permissions` | The permission prompts currently waiting on a human. |
| POST | `/permissions/{call_id}` | Answer a waiting prompt. |
| GET | `/memories` | Recall notes (`?q=`) or list recent ones. |
| POST | `/memories` | Add a note. |

### Sessions

`POST /sessions` — body fields, all optional: `agent` (default `"build"`), `title`,
`working_directory` (default `"."`). Returns the created session.

`GET /sessions` — query: `working_directory` (filter), `limit` (0 = all). Each entry carries a
`latest_run` receipt (or `null`).

`GET /sessions/{id}` — the session plus a `runs` array of receipts. 404 if unknown.

`DELETE /sessions/{id}` — removes the session's messages, events, runs, grants and notes in one
shot. 404 if it did not exist.

`POST /sessions/{id}/fork` — body: `title` (optional). Copies the message history verbatim (the
system prompt included) into a new session with its own event stream numbered from one; the
response adds `forked_from`. **201.**

### Running a turn (SSE)

`POST /sessions/{id}/messages` — body: `message` (or `text`), plus any run limits (see
[Run limits](#run-limits)). The run starts as a background task while the request **streams the
events it produces** as SSE. The stream subscribes from the current tip, so only *this* run's
events flow; events from a helper (sub-agent) run — whose `run_id` carries a `/` — pass through
but do **not** end the stream. The stream ends on the top-level `RUN_FINISHED`.

Because every event is stored and numbered, a client that drops can resume without loss:
reconnect to `GET /sessions/{id}/events?from=<last id>&stream=1`. The SSE `id:` line is the
event's sequence number — exactly what an `EventSource` sends back as `Last-Event-ID` — so a
reconnect is a replay from that id.

`GET /sessions/{id}/events` — query: `from=N` (the last sequence the client already has;
default 0). Without `stream`, returns the stored tail after `N` as JSON. With `stream=1`,
catches up from `N` and then stays open as SSE.

`POST /sessions/{id}/resume` — body: run limits (optional). Carries the session's interrupted
run to completion and returns its receipt (the `GET /runs/{run_id}` shape plus `final_text`).
This does **not** stream: if the run had already finished, resuming rebuilds the receipt with no
model call and there would be nothing to stream. A client that wants the live events watches
`GET /sessions/{id}/events?stream=1` while this runs.

**SSE frame.** Each event is one frame:

```
id: <sequence>
event: <event type>
data: <event as JSON>

```

Response headers: `content-type: text/event-stream`, `cache-control: no-cache`,
`connection: keep-alive`, `x-accel-buffering: no` (so a buffering proxy cannot defeat streaming).

### Runs

`GET /runs/{run_id}` — the run receipt: status, turns, input/output/cached/reasoning tokens,
cost, duration, retries, model, and (for a resumed run) `final_text`. 404 if unknown.

### Permissions

A mutating-and-unrecognised tool call pauses the run and emits a permission-request event on the
bus (a client streaming the run sees it). Answer it:

`GET /permissions` — the prompts currently waiting.

`POST /permissions/{call_id}` — body: `allowed` (**required**, `true`/`false`; 400 if missing),
`duration` (optional, default `"once"`), `scope` (optional). `duration` is one of `once`
(this call only), `session` (the rest of this session) or `always` (across sessions); an
unrecognised value falls back to `once`. `scope` narrows a remembered *yes* to a folder, so
approving for the session need not mean approving everywhere. Returns
`{call_id, allowed, duration}`.

### Memories

`GET /memories` — query: `q` (if present, recall notes matching it; otherwise list recent),
`session_id` (scope), `limit` (default 5).

`POST /memories` — body: `text` (the note), `kind` (default `"fact"`), `session_id`
(default `""` = unscoped).

## Run limits

Any endpoint that starts a run (`/messages`, `/resume`) reads an optional limits object from the
body. Supply none and the server uses the agent's defaults; supply any subset and it builds a
`Limits` from just those keys: `max_turns`, `max_cost_usd`, `max_total_tokens`,
`reasoning_effort`, `plan_mode`. Setting `plan_mode` to true puts the run behind the read-only
plan-mode gate (every mutating tool call is denied) — the HTTP counterpart of the CLI's plan
mode.

## Errors

A handler raises `HTTPError(status, message)` to return a clean status rather than a 500 — 404
for an unknown session or run, 400 for a malformed body (e.g. a permission answer without
`allowed`). Everything else is JSON.

## Testing it offline

The app is a plain ASGI callable and the handlers only touch the injected `AgentService`, so the
whole surface is exercised without a socket, an HTTP client, or a key: see
`packages/api/tests/test_api_e2e.py`, which drives `create_app(service)` directly. That is the
same property the core relies on throughout — the transport is a thin shell over a service that
was designed to be driven from anywhere.
