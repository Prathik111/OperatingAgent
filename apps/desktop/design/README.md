# OperatingAgent — Desktop Design

> Tauri 2 + Webview frontend for the autonomous AI operating agent.
> This folder is the **design source of truth** — IA, system, and interactive prototype.

---

## 1. What this app is

Not a chatbot. A **mission control** for an agent that *acts* on the local machine through sandboxed tools.

The core is headless (`AgentService` over loopback HTTP + SSE). The desktop is a peer client — same as CLI and benchmark runner. That shapes the UI: **the transcript is the state**, streaming is first-class, and every tool call is inspectable.

```
User goal → Session → Agent Loop (think → tool → observe → repeat) → Receipt
                ↕ SSE (numbered, replayable)                    ↕ Postgres
           Permission prompts                              Run history
```

---

## 2. Information Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Titlebar (Tauri native)  — traffic lights, window drag │
├──────────┬──────────────────────────────────────────────┤
│          │  Session header                              │
│ Sidebar  │   working_directory · agent · model · budget │
│          ├──────────────────────────────────────────────┤
│ Sessions │  Transcript (streaming)                      │
│ Search   │   User messages                              │
│          │   Assistant text (delta) + reasoning toggle  │
│ Sessions │   Tool calls (grouped by turn, parallel/     │
│  · active│    serial, states: pending→running→done/     │
│  · recent│    error/denied)                             │
│  · runs  │   Permission prompts (inline, blocking)      │
│          ├──────────────────────────────────────────────┤
│  ─────── │  Composer                                     │
│ New      │   prompt + plan_mode toggle + limits + send  │
│ Settings │                                              │
├──────────┴──────────────────────────────────────────────┤
│  Status bar — sandbox on/off · cost · tokens · turns · model latency │
└─────────────────────────────────────────────────────────┘
```

### Views

| View | Route / trigger | Purpose |
|------|----------------|---------|
| **Session** | default | Streaming transcript, the primary surface |
| **Permission sheet** | `permission.requested` event | Blocking decision — shows real tool + args + preview diff/command, not description |
| **Run receipt** | `RUN_FINISHED` | Turns, tokens (in/out/cached/reasoning), cost, duration, model, retries, stop_reason |
| **Sessions list** | sidebar | Search, filter by working_directory, fork/delete, resume |
| **Run history** | session detail / global | Per-session and cross-session, with totals |
| **Checkpoints** | toolbar | List snapshots, rewind (filesystem snapshot) |
| **Skills** | prompt context | Catalog in prompt, `invoke_skill` loads body |
| **Settings** | gear | Provider, model, fallback_models, sandbox, database |

---

## 3. Design Principles

1. **Transcript over chrome.** The conversation *is* the app. Chrome stays quiet; messages and tool cards carry the weight.
2. **Show the real operation.** Permission prompts display `tool_name + arguments + preview` — never a planner's description. Diffs for edits, command lines for shell.
3. **Streaming is expected, interruption is normal.** `text.delta` / `reasoning.delta` / `tool.state_changed` animate in; cancel returns a partial RunRecord — never a blank error.
4. **Budgets are visible.** Cost, tokens, wall-clock, turn count, and ceiling (`max_cost / max_tokens / max_turns`) are always one glance away.
5. **Every run is replayable.** Events are numbered; `Last-Event-ID` resume, fork (copy messages, new event stream), and receipt rebuild are exposed, not hidden.

---

## 4. Visual Direction

**Reference:** Linear + Raycast + VS Code — dark, precise, monospace where it matters, no illustration.

- **Dark by default** (Tauri desktop, long sessions). Light theme tokens included.
- **Type:** Inter for UI, JetBrains Mono for code/tool I/O/diffs. Small caps for labels.
- **Density:** compact but breathable — 12px UI, 13px transcript, 12px tool cards.
- **Motion:** 120–180ms ease-out for card expansion, streaming caret, permission slide-in. No bounce.
- **Color:** neutral base (zinc), single accent (violet) for agent actions, semantic tokens for tool states (amber=ask, red=deny/error, green=success, blue=running).

---

## 5. Files in this folder

| File | What it is |
|------|-----------|
| `README.md` | this brief |
| `tokens.css` | CSS custom properties — color, type, radius, shadow, motion |
| `index.html` | **Interactive prototype** — open in browser, no build step. Sidebar, streaming transcript, permission sheet, tool cards, composer, receipts. Uses Tailwind CDN + tokens.css. |
| `components.html` | Isolated component gallery (tool cards, permission states, receipts, empty states) |

Open `index.html` directly:

```powershell
start apps/desktop/design/index.html
# or
npx serve apps/desktop/design
```

---

## 6. Mapping to the API

| UI action | API call |
|-----------|---------|
| New session | `POST /sessions {agent, title, working_directory}` |
| Send prompt | `POST /sessions/{id}/messages {message, limits}` → SSE stream |
| Resume dropped stream | `GET /sessions/{id}/events?from=N&stream=1` (`Last-Event-ID`) |
| Resume interrupted run | `POST /sessions/{id}/resume` |
| Fork | `POST /sessions/{id}/fork {title}` |
| Permission | `GET /permissions` + `POST /permissions/{call_id} {allowed, duration, scope}` |
| History | `GET /sessions`, `GET /sessions/{id}`, `GET /runs/{run_id}` |
| Memories | `GET /memories?q=` / `POST /memories` |

SSE frame: `id: <seq>\nevent: <type>\ndata: <json>\n\n` — `content-type: text/event-stream`, `x-accel-buffering: no`.

---

## 7. Next steps

- [ ] Wire prototype to real `packages/api` (loopback port from Tauri sidecar stdout)
- [ ] Tauri scaffolding (`apps/desktop/src-tauri/`, `src/`)
- [ ] Component extraction to React + Tailwind (or keep vanilla for sidecar simplicity)
- [ ] A11y pass (keyboard nav for permission sheet, `aria-live` for streaming)
- [ ] Light theme polish + high-contrast
