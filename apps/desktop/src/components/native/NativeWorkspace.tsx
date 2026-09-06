import { useCallback, useEffect, useRef, useState } from "react";
import { nativeApi } from "../../lib/api";
import type { EventResponse, PermissionResponse, RunResponse, SessionResponse } from "../../lib/types";
import { Card, Label } from "../layout/Shell";

function useNativeHealth() {
  const [data, setData] = useState<{ status: string; database: string; agents: string[]; models: string[]; langfuse_enabled: boolean } | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    nativeApi
      .health()
      .then(setData)
      .catch((e: Error) => setErr(e.message));
  }, []);
  return { data, err };
}

export function NativeWorkspace() {
  const health = useNativeHealth();
  const [sessions, setSessions] = useState<SessionResponse[]>([]);
  const [sessionsErr, setSessionsErr] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [detail, setDetail] = useState<{ runs: RunResponse[]; message_count: number } | null>(null);
  const [conversation, setConversation] = useState<{ messages: Array<{ id: string; role: string; parts: unknown[]; model: string }> } | null>(null);
  const [events, setEvents] = useState<EventResponse[]>([]);
  const [runs, setRuns] = useState<RunResponse[]>([]);
  const [permissions, setPermissions] = useState<PermissionResponse[]>([]);
  const [composer, setComposer] = useState("");
  const [limits, setLimits] = useState({ max_turns: 10, max_cost_usd: 0.05, plan_mode: false, reasoning_effort: "" });
  const [sending, setSending] = useState(false);
  const [streamLog, setStreamLog] = useState<string[]>([]);
  const [newTitle, setNewTitle] = useState("");
  const [newWorkspace, setNewWorkspace] = useState(".");
  const [newAgent, setNewAgent] = useState("build");
  const [forkTitle, setForkTitle] = useState("");
  const [activeTab, setActiveTab] = useState<"conversation" | "events" | "runs" | "permissions">("conversation");
  const esRef = useRef<EventSource | null>(null);

  const refreshSessions = useCallback(async () => {
    try {
      const list = await nativeApi.listSessions({ limit: 100 });
      setSessions(list);
      setSessionsErr(null);
      if (!selected && list[0]) setSelected(list[0].id);
    } catch (e) {
      setSessionsErr((e as Error).message);
    }
  }, [selected]);

  const refreshDetail = useCallback(async (id: string) => {
    try {
      const d = await nativeApi.getSession(id);
      setDetail({ runs: d.runs, message_count: d.message_count });
      const conv = await nativeApi.getConversation(id).catch(() => ({ session_id: id, messages: [] as Array<{ id: string; role: string; parts: unknown[]; model: string }> }));
      setConversation(conv as unknown as { messages: Array<{ id: string; role: string; parts: unknown[]; model: string }> });
      const ev = await nativeApi.getEvents(id, 0).catch(() => [] as EventResponse[]);
      setEvents(ev);
      const r = await nativeApi.listRuns(id).catch(() => [] as RunResponse[]);
      setRuns(r);
      const perms = await nativeApi.listPermissions(id).catch(() => [] as PermissionResponse[]);
      setPermissions(perms);
    } catch {
      // ignore
    }
  }, []);

  useEffect(() => {
    refreshSessions();
    const t = setInterval(() => {
      if (selected) refreshDetail(selected);
      nativeApi.listPermissions(selected || undefined).then(setPermissions).catch(() => {});
    }, 3000);
    return () => clearInterval(t);
  }, [refreshSessions, refreshDetail, selected]);

  useEffect(() => {
    if (selected) refreshDetail(selected);
  }, [selected, refreshDetail]);

  // poll permissions globally too
  useEffect(() => {
    const t = setInterval(() => nativeApi.listPermissions().then(setPermissions).catch(() => {}), 2500);
    return () => clearInterval(t);
  }, []);

  const onCreate = async () => {
    try {
      const s = await nativeApi.createSession({ agent: newAgent, title: newTitle, workspace: newWorkspace });
      setSessions((prev) => [s, ...prev]);
      setSelected(s.id);
      setNewTitle("");
    } catch (e) {
      alert((e as Error).message);
    }
  };

  const onDelete = async () => {
    if (!selected || !confirm("Delete session and all runs/events?")) return;
    try {
      await nativeApi.deleteSession(selected);
      setSessions((prev) => prev.filter((s) => s.id !== selected));
      setSelected(null);
    } catch (e) {
      alert((e as Error).message);
    }
  };

  const onFork = async () => {
    if (!selected) return;
    try {
      const f = await nativeApi.forkSession(selected, forkTitle || `${sessions.find((s) => s.id === selected)?.title || selected} (fork)`);
      setSessions((prev) => [f, ...prev]);
      setSelected(f.id);
      setForkTitle("");
    } catch (e) {
      alert((e as Error).message);
    }
  };

  const onSend = async () => {
    if (!selected || !composer.trim() || sending) return;
    setSending(true);
    setStreamLog([]);
    const url = `${nativeApi.sendMessageUrl(selected)}`;
    // We POST via fetch but the endpoint returns SSE; easiest: POST with fetch + stream manually
    // Here we do POST that returns SSE via fetch streaming
    try {
      const body = JSON.stringify({
        message: composer,
        limits: {
          max_turns: limits.max_turns || undefined,
          max_cost_usd: limits.max_cost_usd || undefined,
          plan_mode: limits.plan_mode || undefined,
          reasoning_effort: limits.reasoning_effort || undefined,
        },
      });
      const res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body });
      if (!res.ok || !res.body) {
        const t = await res.text().catch(() => "");
        throw new Error(`${res.status} ${res.statusText} ${t.slice(0, 300)}`);
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      const onChunk = (chunk: string) => {
        buf += chunk;
        const frames = buf.split("\n\n");
        buf = frames.pop() || "";
        for (const f of frames) {
          if (f.trim()) setStreamLog((prev) => [...prev.slice(-200), f.trim()]);
        }
      };
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        onChunk(decoder.decode(value, { stream: true }));
      }
      if (buf.trim()) setStreamLog((prev) => [...prev, buf.trim()]);
      setComposer("");
      if (selected) refreshDetail(selected);
    } catch (e) {
      setStreamLog((prev) => [...prev, `error: ${(e as Error).message}`]);
    } finally {
      setSending(false);
      if (esRef.current) {
        esRef.current.close();
        esRef.current = null;
      }
    }
  };

  const onResume = async () => {
    if (!selected) return;
    try {
      const r = await nativeApi.resumeRun(selected);
      setStreamLog((prev) => [...prev, `resume → ${r.status} ${r.run_id} turns=${r.turns}`]);
      refreshDetail(selected);
    } catch (e) {
      alert((e as Error).message);
    }
  };

  const onCancel = async () => {
    if (!selected) return;
    try {
      const r = await nativeApi.cancelRun(selected);
      setStreamLog((prev) => [...prev, `cancel → ${JSON.stringify(r)}`]);
    } catch (e) {
      alert((e as Error).message);
    }
  };

  const onResolve = async (callId: string, allowed: boolean, duration: string, scope: string) => {
    try {
      await nativeApi.resolvePermission(callId, { allowed, duration, scope });
      setPermissions((prev) => prev.filter((p) => p.call_id !== callId));
    } catch (e) {
      alert((e as Error).message);
    }
  };

  const selectedMeta = sessions.find((s) => s.id === selected);

  return (
    <div className="flex flex-1 min-h-0">
      {/* left: sessions */}
      <div className="w-[320px] shrink-0 border-r flex flex-col" style={{ borderColor: "var(--bg-4)", background: "var(--bg-1)" }}>
        <div className="p-3 border-b space-y-3" style={{ borderColor: "var(--bg-4)" }}>
          <div className="flex items-center gap-2">
            <Label>Native sessions</Label>
            <span className="ml-auto text-[10px] font-mono px-1.5 py-0.5 rounded" style={{ background: "var(--bg-2)", border: "1px solid var(--bg-4)", color: "var(--fg-2)" }}>
              {sessions.length}
            </span>
          </div>
          <div className="grid gap-2">
            <input value={newTitle} onChange={(e) => setNewTitle(e.target.value)} placeholder="Title (optional)" className="h-8 px-2 rounded-lg text-[12px] outline-none" style={{ background: "var(--bg-2)", border: "1px solid var(--bg-4)", color: "var(--fg-0)" }} />
            <div className="flex gap-2">
              <input value={newWorkspace} onChange={(e) => setNewWorkspace(e.target.value)} placeholder="workspace" className="flex-1 h-8 px-2 rounded-lg text-[12px] font-mono outline-none" style={{ background: "var(--bg-2)", border: "1px solid var(--bg-4)", color: "var(--fg-0)" }} />
              <input value={newAgent} onChange={(e) => setNewAgent(e.target.value)} placeholder="agent" className="w-20 h-8 px-2 rounded-lg text-[12px] outline-none" style={{ background: "var(--bg-2)", border: "1px solid var(--bg-4)", color: "var(--fg-0)" }} />
            </div>
            <button onClick={onCreate} className="btn-grad h-8 rounded-lg text-[12px] font-medium" style={{ color: "white", border: "1px solid transparent" }}>
              POST /native/sessions
            </button>
          </div>
          {sessionsErr && <div className="text-[11px] font-mono p-2 rounded" style={{ background: "var(--danger-soft)", border: "1px solid var(--danger-soft)", color: "var(--danger)" }}>{sessionsErr} — is API at {import.meta.env.VITE_API_URL || "http://127.0.0.1:8000"} running?</div>}
          <div className="text-[11px] font-mono p-2 rounded" style={{ background: "var(--bg-2)", border: "1px solid var(--bg-4)", color: "var(--fg-2)" }}>
            <div>GET /native/health</div>
            <div style={{ color: "var(--fg-1)" }}>{health.data ? `${health.data.status} · ${health.data.database} · ${health.data.models.slice(0, 2).join(", ") || "no models"}` : health.err || "…"}</div>
            <div className="mt-1 flex gap-1.5 flex-wrap">
              <span className="px-1.5 py-0.5 rounded" style={{ background: "var(--bg-3)", border: "1px solid var(--bg-4)" }}>agents: {health.data?.agents.join(", ") || "—"}</span>
              <span className="px-1.5 py-0.5 rounded" style={{ background: "var(--bg-3)", border: "1px solid var(--bg-4)" }}>langfuse: {health.data?.langfuse_enabled ? "on" : "off"}</span>
            </div>
          </div>
        </div>
        <div className="flex-1 overflow-auto p-2 space-y-1">
          {sessions.map((s) => (
            <button
              key={s.id}
              onClick={() => setSelected(s.id)}
              className="w-full text-left p-2.5 rounded-xl"
              style={{
                background: selected === s.id ? "var(--bg-2)" : "transparent",
                border: `1px solid ${selected === s.id ? "var(--accent-ring)" : "transparent"}`,
              }}
            >
              <div className="text-[12px] font-medium truncate" style={{ color: "var(--fg-0)" }}>{s.title || s.id}</div>
              <div className="text-[11px] font-mono truncate" style={{ color: "var(--fg-2)" }}>{s.id} · {s.agent} · {s.workspace}</div>
            </button>
          ))}
          {sessions.length === 0 && <div className="text-[11px] p-3" style={{ color: "var(--fg-3)" }}>No sessions — create one above (POST /native/sessions)</div>}
        </div>
      </div>

      {/* right: detail */}
      <div className="flex-1 flex flex-col min-w-0">
        {!selected ? (
          <div className="flex-1 grid place-items-center p-8 text-center">
            <div className="max-w-[420px] rounded-2xl p-6" style={{ background: "var(--bg-1)", border: "1px solid var(--bg-4)" }}>
              <div className="text-[13px] font-medium">Select a session</div>
              <div className="text-[12px] mt-1" style={{ color: "var(--fg-2)" }}>Create or pick a session to drive all Native endpoints.</div>
            </div>
          </div>
        ) : (
          <>
            <div className="px-4 py-3 border-b flex flex-wrap gap-2 items-center" style={{ borderColor: "var(--bg-4)", background: "var(--bg-0)" }}>
              <div className="min-w-0">
                <div className="text-[13px] font-semibold truncate">{selectedMeta?.title || selected}</div>
                <div className="text-[11px] font-mono truncate" style={{ color: "var(--fg-2)" }}>{selected} · {selectedMeta?.workspace} · {selectedMeta?.agent} · {detail?.message_count ?? 0} msgs</div>
              </div>
              <div className="ml-auto flex flex-wrap gap-1.5">
                <button onClick={onDelete} className="h-7 px-2.5 rounded-lg text-[11px] font-medium" style={{ background: "var(--bg-2)", border: "1px solid var(--bg-4)", color: "var(--danger)" }}>DELETE /sessions/{"{id}"}</button>
                <div className="flex gap-1">
                  <input value={forkTitle} onChange={(e) => setForkTitle(e.target.value)} placeholder="fork title" className="h-7 w-28 px-2 rounded-lg text-[11px] outline-none" style={{ background: "var(--bg-2)", border: "1px solid var(--bg-4)" }} />
                  <button onClick={onFork} className="h-7 px-2.5 rounded-lg text-[11px] font-medium" style={{ background: "var(--bg-2)", border: "1px solid var(--bg-4)" }}>POST fork</button>
                </div>
                <button onClick={onResume} className="h-7 px-2.5 rounded-lg text-[11px] font-medium" style={{ background: "var(--bg-2)", border: "1px solid var(--bg-4)" }}>POST resume</button>
                <button onClick={onCancel} className="h-7 px-2.5 rounded-lg text-[11px] font-medium" style={{ background: "var(--bg-2)", border: "1px solid var(--bg-4)" }}>POST cancel</button>
              </div>
            </div>

            <div className="flex gap-1 px-3 py-2 border-b" style={{ borderColor: "var(--bg-4)", background: "var(--bg-1)" }}>
              {(["conversation", "events", "runs", "permissions"] as const).map((t) => (
                <button
                  key={t}
                  onClick={() => setActiveTab(t)}
                  className="h-7 px-3 rounded-full text-[11px] font-medium capitalize btn-quiet"
                  style={{
                    background: activeTab === t ? "var(--accent-grad)" : "var(--bg-2)",
                    color: activeTab === t ? "white" : "var(--fg-2)",
                    border: "1px solid var(--bg-4)",
                    boxShadow: activeTab === t ? "var(--accent-glow)" : "none",
                  }}
                >
                  {t} {t === "permissions" && permissions.length ? `(${permissions.length})` : ""}
                </button>
              ))}
              <span className="ml-auto text-[10px] font-mono px-2 py-1 rounded" style={{ background: "var(--bg-2)", border: "1px solid var(--bg-4)", color: "var(--fg-3)" }}>
                GET /native/sessions/{"{id}"}/… + SSE
              </span>
            </div>

            <div className="flex-1 overflow-auto p-4 space-y-4">
              {activeTab === "conversation" && (
                <div className="space-y-3">
                  <Label>GET /native/sessions/{"{id}"}/conversation</Label>
                  <Card className="p-3 max-h-[420px] overflow-auto">
                    {conversation ? (
                      <div className="space-y-3">
                        {conversation.messages.map((m) => (
                          <div key={m.id} className="rounded-xl p-3" style={{ background: m.role === "user" ? "var(--bg-0)" : "var(--bg-2)", border: "1px solid var(--bg-4)" }}>
                            <div className="text-[11px] font-mono" style={{ color: "var(--fg-2)" }}>{m.role} · {m.id.slice(0, 8)} · {m.model || "—"}</div>
                            <div className="mt-1 space-y-1">
                              {m.parts.map((p: unknown, i: number) => {
                                const part = p as Record<string, unknown>;
                                return (
                                  <pre key={i} className="text-[11px] font-mono whitespace-pre-wrap break-words p-2 rounded" style={{ background: "var(--bg-1)", border: "1px solid var(--bg-4)" }}>
                                    {JSON.stringify(part, null, 2).slice(0, 800)}
                                  </pre>
                                );
                              })}
                            </div>
                          </div>
                        ))}
                        {conversation.messages.length === 0 && <div className="text-[11px]" style={{ color: "var(--fg-3)" }}>No messages yet.</div>}
                      </div>
                    ) : (
                      <div className="text-[11px]" style={{ color: "var(--fg-3)" }}>—</div>
                    )}
                  </Card>
                  <Label>GET /native/runs/{"{run_id}"} + GET /native/sessions/{"{id}"}/runs</Label>
                  <div className="grid gap-2">
                    {(runs.length ? runs : detail?.runs || []).slice(0, 6).map((r) => (
                      <div key={r.run_id} className="rounded-xl p-3 flex gap-3 items-start" style={{ background: "var(--bg-1)", border: "1px solid var(--bg-4)" }}>
                        <span className="text-[10px] px-1.5 py-0.5 rounded font-mono" style={{ background: r.status === "finished" ? "var(--success-soft)" : "var(--bg-2)", border: "1px solid var(--bg-4)", color: r.status === "finished" ? "var(--success)" : "var(--fg-2)" }}>{r.status}</span>
                        <div className="min-w-0 flex-1">
                          <div className="text-[11px] font-mono truncate">{r.run_id} · {r.model} · {r.turns} turns · {r.duration_seconds.toFixed(2)}s · ${r.cost_usd.toFixed(4)}</div>
                          <div className="text-[11px] truncate" style={{ color: "var(--fg-2)" }}>{r.final_message || r.final_text || r.error || "—"}</div>
                          <div className="text-[10px] font-mono" style={{ color: "var(--fg-3)" }}>in {r.input_tokens} · out {r.output_tokens} · cached {r.cached_tokens} · reasoning {r.reasoning_tokens} · {r.stop_reason || "—"}</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {activeTab === "events" && (
                <div className="space-y-3">
                  <Label>GET /native/sessions/{"{id}"}/events?from=&stream= (SSE)</Label>
                  <div className="flex gap-2">
                    <button
                      onClick={async () => {
                        const ev = await nativeApi.getEvents(selected, 0).catch(() => [] as EventResponse[]);
                        setEvents(ev);
                      }}
                      className="h-7 px-3 rounded-lg text-[11px] font-medium"
                      style={{ background: "var(--bg-2)", border: "1px solid var(--bg-4)" }}
                    >
                      Reload JSON (from=0)
                    </button>
                    <a
                      href={nativeApi.getEventsSSEUrl(selected, events.length ? events[events.length - 1].sequence : 0, 1)}
                      target="_blank"
                      rel="noreferrer"
                      className="h-7 px-3 rounded-lg text-[11px] font-medium inline-flex items-center"
                      style={{ background: "var(--bg-2)", border: "1px solid var(--bg-4)", color: "var(--fg-1)" }}
                    >
                      Open SSE stream ↗
                    </a>
                  </div>
                  <Card className="p-3 max-h-[520px] overflow-auto">
                    <div className="space-y-1">
                      {events.map((e) => (
                        <div key={e.sequence} className="flex gap-2 text-[11px] font-mono p-1.5 rounded" style={{ background: "var(--bg-0)", border: "1px solid var(--bg-4)" }}>
                          <span style={{ color: "var(--fg-2)" }}>{e.sequence}</span>
                          <span style={{ color: "var(--accent)" }}>{e.type}</span>
                          <span className="truncate" style={{ color: "var(--fg-1)" }}>{JSON.stringify(e.data).slice(0, 180)}</span>
                        </div>
                      ))}
                      {events.length === 0 && <div className="text-[11px]" style={{ color: "var(--fg-3)" }}>No events.</div>}
                    </div>
                  </Card>
                </div>
              )}

              {activeTab === "runs" && (
                <div className="space-y-3">
                  <Label>Runs for this session</Label>
                  <div className="grid gap-2">
                    {runs.map((r) => (
                      <Card key={r.run_id} className="p-3">
                        <div className="text-[12px] font-mono">{r.run_id} — {r.status} · {r.turns} turns</div>
                        <div className="text-[11px] font-mono" style={{ color: "var(--fg-2)" }}>{r.model} · {r.input_tokens}/{r.output_tokens} · ${r.cost_usd.toFixed(4)} · {r.duration_seconds.toFixed(1)}s</div>
                        <a href={r.trace_url || "#"} target="_blank" rel="noreferrer" className="text-[11px] underline" style={{ color: "var(--accent)" }}>{r.trace_id ? `trace ${r.trace_id.slice(0, 8)}` : "no trace"}</a>
                      </Card>
                    ))}
                    {runs.length === 0 && <div className="text-[11px]" style={{ color: "var(--fg-3)" }}>No runs.</div>}
                  </div>
                </div>
              )}

              {activeTab === "permissions" && (
                <div className="space-y-3">
                  <Label>GET /native/permissions + POST /native/permissions/{"{call_id}"}</Label>
                  {permissions.length === 0 && <div className="text-[11px] p-3 rounded-xl" style={{ background: "var(--bg-1)", border: "1px solid var(--bg-4)", color: "var(--fg-3)" }}>No pending permissions.</div>}
                  {permissions.map((p) => (
                    <Card key={p.call_id} className="p-3 space-y-2">
                      <div className="text-[12px] font-mono font-medium">{p.tool} · {p.call_id.slice(0, 8)}</div>
                      <div className="text-[11px]" style={{ color: "var(--fg-2)" }}>{p.reason || "—"}</div>
                      <pre className="text-[11px] font-mono p-2 rounded overflow-auto" style={{ background: "var(--bg-0)", border: "1px solid var(--bg-4)" }}>{p.preview || JSON.stringify(p.arguments, null, 2)}</pre>
                      <div className="flex gap-2">
                        <button onClick={() => onResolve(p.call_id, true, "once", "")} className="btn-grad h-7 px-3 rounded-lg text-[11px] font-medium" style={{ color: "white", border: "1px solid transparent" }}>Allow once</button>
                        <button onClick={() => onResolve(p.call_id, true, "session", "")} className="h-7 px-3 rounded-lg text-[11px] font-medium" style={{ background: "var(--bg-2)", border: "1px solid var(--bg-4)" }}>Allow session</button>
                        <button onClick={() => onResolve(p.call_id, true, "always", "")} className="h-7 px-3 rounded-lg text-[11px] font-medium" style={{ background: "var(--bg-2)", border: "1px solid var(--bg-4)" }}>Allow always</button>
                        <button onClick={() => onResolve(p.call_id, false, "once", "")} className="ml-auto h-7 px-3 rounded-lg text-[11px] font-medium" style={{ background: "var(--bg-2)", border: "1px solid var(--bg-4)", color: "var(--danger)" }}>Deny</button>
                      </div>
                    </Card>
                  ))}
                </div>
              )}
            </div>

            {/* composer */}
            <div className="p-3 border-t space-y-2" style={{ borderColor: "var(--bg-4)", background: "var(--bg-1)" }}>
              <div className="flex gap-2 items-center text-[11px] font-mono">
                <label className="flex items-center gap-1.5"><input type="number" value={limits.max_turns} onChange={(e) => setLimits((s) => ({ ...s, max_turns: Number(e.target.value) }))} className="w-16 h-7 px-2 rounded-lg outline-none" style={{ background: "var(--bg-2)", border: "1px solid var(--bg-4)" }} /> turns</label>
                <label className="flex items-center gap-1.5"><input type="number" step="0.01" value={limits.max_cost_usd} onChange={(e) => setLimits((s) => ({ ...s, max_cost_usd: Number(e.target.value) }))} className="w-20 h-7 px-2 rounded-lg outline-none" style={{ background: "var(--bg-2)", border: "1px solid var(--bg-4)" }} /> $ cap</label>
                <label className="flex items-center gap-1.5"><input type="checkbox" checked={limits.plan_mode} onChange={(e) => setLimits((s) => ({ ...s, plan_mode: e.target.checked }))} /> plan_mode</label>
                <select value={limits.reasoning_effort} onChange={(e) => setLimits((s) => ({ ...s, reasoning_effort: e.target.value }))} className="h-7 px-2 rounded-lg text-[11px] outline-none" style={{ background: "var(--bg-2)", border: "1px solid var(--bg-4)" }}>
                  <option value="">reasoning default</option>
                  <option value="low">low</option>
                  <option value="medium">medium</option>
                  <option value="high">high</option>
                </select>
              </div>
              <div className="flex gap-2">
                <textarea value={composer} onChange={(e) => setComposer(e.target.value)} onKeyDown={(e) => e.key === "Enter" && (e.metaKey || e.ctrlKey) && onSend()} placeholder="POST /native/sessions/{id}/messages — describe the goal… (⌘Enter to send)" rows={3} className="flex-1 rounded-xl px-3 py-2 text-[13px] outline-none resize-none" style={{ background: "var(--bg-2)", border: "1px solid var(--bg-4)", color: "var(--fg-0)" }} />
              </div>
              <div className="flex gap-2">
                <button onClick={onSend} disabled={sending} className="btn-grad h-8 px-4 rounded-xl text-[13px] font-semibold" style={{ color: "white", border: "1px solid transparent" }}>{sending ? "Streaming…" : "Send → SSE"}</button>
                <span className="text-[11px] font-mono self-center" style={{ color: "var(--fg-3)" }}>POST /native/sessions/{"{id}"}/messages · resume · cancel</span>
              </div>
              {streamLog.length > 0 && (
                <pre className="max-h-40 overflow-auto p-2 rounded-xl text-[11px] font-mono whitespace-pre-wrap" style={{ background: "var(--bg-0)", border: "1px solid var(--bg-4)", color: "var(--fg-1)" }}>{streamLog.join("\n")}</pre>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
