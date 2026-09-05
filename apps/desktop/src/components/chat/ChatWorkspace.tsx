import { useCallback, useEffect, useRef, useState } from "react";
import { nativeApi, taskApi } from "../../lib/api";
import type { EventResponse, PermissionResponse, SessionResponse, ThreadResponse } from "../../lib/types";
import { AlertDialog, ConfirmDialog, PromptDialog } from "../Modal";
import { AnalyticsView } from "./AnalyticsView";
import { EventTimeline } from "./EventTimeline";

type ChatItem =
  | { kind: "native"; id: string; title: string; subtitle: string; updatedAt: string }
  | { kind: "langgraph"; id: string; title: string; subtitle: string; updatedAt: string };

type ChatMessage = {
  id: string;
  role: "user" | "assistant" | "tool";
  text: string;
  parts?: unknown[];
  model?: string;
  time?: string;
  toolCalls?: Array<{ name: string; status: string; preview?: string }>;
  thinking?: string;
  streaming?: boolean;
};

type DialogState =
  | { kind: "rename"; current: string }
  | { kind: "delete"; id: string; chatKind: "native" | "langgraph"; title: string }
  | { kind: "fork" }
  | { kind: "error"; message: string }
  | null;

function ThinkingBlock({ text, live }: { text: string; live: boolean }) {
  const [open, setOpen] = useState(live);
  useEffect(() => {
    if (live) setOpen(true);
  }, [live]);
  if (!text && !live) return null;
  return (
    <div className="mb-2 rounded-lg overflow-hidden" style={{ background: "var(--bg-0)", border: "1px solid var(--bg-4)", borderLeft: "2px solid var(--accent)" }}>
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center gap-1.5 px-2.5 py-1.5 text-left btn-quiet"
      >
        {live ? (
          <span className="w-1.5 h-1.5 rounded-full anim-pulse-dot shrink-0" style={{ background: "var(--accent)" }} />
        ) : (
          <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="shrink-0" style={{ color: "var(--fg-3)", transform: open ? "rotate(180deg)" : undefined, transition: "transform var(--dur-1) ease" }}>
            <path d="m6 9 6 6 6-6" />
          </svg>
        )}
        <span className="text-[11px] font-medium" style={{ color: "var(--fg-2)" }}>
          {live ? "Thinking…" : "Thought process"}
        </span>
      </button>
      {open && (
        <div className={`px-2.5 pb-2 text-[12px] leading-relaxed whitespace-pre-wrap break-words anim-fade-in ${live ? "stream-caret" : ""}`} style={{ color: "var(--fg-2)" }}>
          {text}
        </div>
      )}
    </div>
  );
}

function timeAgo(iso: string): string {
  const ms = Date.now() - new Date(iso).getTime();
  if (Number.isNaN(ms)) return "";
  const mins = Math.floor(ms / 60000);
  if (mins < 1) return "now";
  if (mins < 60) return `${mins}m`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h`;
  const days = Math.floor(hrs / 24);
  if (days < 7) return `${days}d`;
  return new Date(iso).toLocaleDateString();
}

function workspaceLabel(workspace: string): string {
  if (!workspace || workspace === ".") return "";
  const parts = workspace.split(/[/\\]/).filter(Boolean);
  return parts.length ? parts[parts.length - 1] : workspace;
}

// Frontend-only chat titles (localStorage). The backend has no rename
// endpoint, so a rename overwrites the displayed title on this machine only.
const TITLES_KEY = "operating-agent:chat-titles";

function loadTitleOverrides(): Record<string, string> {
  try {
    const raw = JSON.parse(localStorage.getItem(TITLES_KEY) || "{}");
    return raw && typeof raw === "object" ? (raw as Record<string, string>) : {};
  } catch {
    return {};
  }
}

function saveTitleOverrides(overrides: Record<string, string>) {
  try {
    localStorage.setItem(TITLES_KEY, JSON.stringify(overrides));
  } catch {
    // ignore — storage may be blocked in webview
  }
}

export function ChatWorkspace({
  track,
  onOpenSettings,
  onBackToPicker,
}: {
  track: "native" | "langgraph";
  onOpenSettings: () => void;
  onBackToPicker: () => void;
}) {
  const [chats, setChats] = useState<ChatItem[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [events, setEvents] = useState<EventResponse[]>([]);
  const [permissions, setPermissions] = useState<PermissionResponse[]>([]);
  const [pendingApprovals, setPendingApprovals] = useState<Array<{ id: string; tool_name: string; risk_level: string }>>([]);
  const [composer, setComposer] = useState("");
  const [sending, setSending] = useState(false);
  const [showAnalytics, setShowAnalytics] = useState(false);
  const [dialog, setDialog] = useState<DialogState>(null);
  const [pendingTaskId, setPendingTaskId] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [workspace, setWorkspace] = useState(".");
  const [titleOverrides, setTitleOverrides] = useState<Record<string, string>>(loadTitleOverrides);
  const listRef = useRef<HTMLDivElement>(null);
  const endRef = useRef<HTMLDivElement>(null);

  const scrollToEnd = useCallback(() => endRef.current?.scrollIntoView({ behavior: "smooth" }), []);

  // ——— load chats ———
  const refreshChats = useCallback(async () => {
    const overrides = loadTitleOverrides();
    setTitleOverrides(overrides);
    if (track === "native") {
      const sessions = await nativeApi.listSessions({ limit: 100 }).catch(() => [] as SessionResponse[]);
      const items: ChatItem[] = sessions.map((s) => ({
        kind: "native" as const,
        id: s.id,
        title: overrides[`native:${s.id}`] || s.title || s.id.slice(0, 12),
        subtitle: workspaceLabel(s.workspace),
        updatedAt: new Date().toISOString(),
      }));
      setChats(items);
      if (!selected && items[0]) setSelected(items[0].id);
    } else {
      const threads = await taskApi.listThreads({ limit: 100 }).catch(() => [] as ThreadResponse[]);
      const items: ChatItem[] = threads.map((t) => ({
        kind: "langgraph" as const,
        id: t.id,
        title: overrides[`langgraph:${t.id}`] || t.title || t.id.slice(0, 12),
        subtitle: `${t.task_count} tasks`,
        updatedAt: t.updated_at,
      }));
      setChats(items);
      if (!selected && items[0]) setSelected(items[0].id);
    }
  }, [track, selected]);

  const refreshConversation = useCallback(
    async (id: string) => {
      if (track === "native") {
        try {
          const conv = await nativeApi.getConversation(id);
          const msgs: ChatMessage[] = conv.messages.map((m) => {
            const parts = m.parts as Array<Record<string, unknown>>;
            const firstText = parts.find((p) =>
              p.text && p.part_type !== "reasoning" && p.part_type !== "compaction" && p.hidden !== true,
            )?.text as string | undefined;
            const thinking = parts.find((p) => p.part_type === "reasoning" && p.text)?.text as string | undefined;
            const toolParts = parts.filter((p) => p.name);
            return {
              id: m.id,
              role: m.role as ChatMessage["role"],
              text: firstText || (m.parts.length ? JSON.stringify(m.parts[0]).slice(0, 200) : ""),
              parts: m.parts,
              model: m.model,
              time: m.created_at || undefined,
              toolCalls: toolParts.map((p) => ({ name: String(p.name), status: String(p.status || "completed"), preview: String(p.output || p.error || "").slice(0, 120) })),
              thinking: thinking || undefined,
            };
          });
          setMessages(msgs);
          const allEvents = await nativeApi.getEvents(id, 0).catch(() => [] as EventResponse[]);
          // Activity covers the current response only: keep events from the
          // latest top-level run (helper runs carry "/" in their run id).
          let currentRun = "";
          for (let i = allEvents.length - 1; i >= 0; i--) {
            const rid = allEvents[i].run_id || "";
            if (!rid || rid.includes("/")) continue;
            currentRun = rid;
            break;
          }
          setEvents(currentRun ? allEvents.filter((e) => e.run_id === currentRun) : allEvents.slice(-30));
          const perms = await nativeApi.listPermissions(id).catch(() => [] as PermissionResponse[]);
          setPermissions(perms);
        } catch {
          setMessages([]);
        }
      } else {
        try {
          const tasks = await taskApi.listThreadTasks(id, { limit: 100 }).catch(() => []);
          const msgs: ChatMessage[] = tasks
            .slice()
            .reverse()
            .flatMap((t) => [
              { id: `${t.id}-user`, role: "user" as const, text: t.goal, time: t.created_at },
              ...(t.output || t.final_message
                ? [{ id: `${t.id}-assistant`, role: "assistant" as const, text: String(t.output || t.final_message), time: t.created_at }]
                : t.error
                  ? [{ id: `${t.id}-error`, role: "assistant" as const, text: `Run failed: ${t.error}`, time: t.created_at }]
                  : []),
            ]);
          setMessages(msgs);
          setPendingTaskId((cur) => {
            if (!cur) return cur;
            const finished = tasks.some((t) => t.id === cur && (t.output || t.final_message || t.error));
            return finished ? null : cur;
          });
          // Activity covers the current response only: keep events from the
          // latest task in this thread.
          const latestTaskId = tasks.length > 0 ? tasks[0].id : "";
          const ev = (await taskApi.listThreadEvents(id).catch(() => []))
            .filter((e) => !latestTaskId || e.task_id === latestTaskId);
          setEvents(
            ev.slice(-30).map((e, i) => ({
              sequence: i + 1,
              type: e.type,
              session_id: id,
              run_id: e.task_id,
              data: e.payload,
              time: null,
            })),
          );
          const approvals = await taskApi.listApprovals().catch(() => []);
          setPendingApprovals(approvals.filter((a) => tasks.some((t) => t.id === a.task_id)));
        } catch {
          setMessages([]);
        }
      }
    },
    [track],
  );

  useEffect(() => {
    refreshChats();
  }, [refreshChats]);

  useEffect(() => {
    if (selected) refreshConversation(selected);
  }, [selected, refreshConversation]);

  useEffect(() => {
    if (!selected || track !== "langgraph") return;
    const timer = setInterval(() => refreshConversation(selected), 2000);
    return () => clearInterval(timer);
  }, [selected, track, refreshConversation]);

  // poll permissions / approvals
  useEffect(() => {
    const t = setInterval(async () => {
      if (track === "native" && selected) {
        const perms = await nativeApi.listPermissions(selected).catch(() => [] as PermissionResponse[]);
        setPermissions(perms);
      } else if (track === "langgraph") {
        const approvals = await taskApi.listApprovals().catch(() => []);
        setPendingApprovals(approvals);
      }
    }, 2500);
    return () => clearInterval(t);
  }, [track, selected]);

  // ——— actions ———
  const onNewChat = async () => {
    const title = `Chat ${new Date().toLocaleTimeString()}`;
    if (track === "native") {
      try {
        const s = await nativeApi.createSession({ title, workspace, agent: "build" });
        setChats((prev) => [{ kind: "native", id: s.id, title: s.title || s.id, subtitle: workspaceLabel(s.workspace), updatedAt: new Date().toISOString() }, ...prev]);
        setSelected(s.id);
        setMessages([]);
      } catch (e) {
        setDialog({ kind: "error", message: (e as Error).message });
      }
    } else {
      try {
        const thread = await taskApi.createThread(title);
        setChats((prev) => [{ kind: "langgraph", id: thread.id, title: thread.title || thread.id, subtitle: "0 tasks", updatedAt: thread.updated_at }, ...prev]);
        setSelected(thread.id);
        setMessages([]);
        setEvents([]);
      } catch (e) {
        setDialog({ kind: "error", message: (e as Error).message });
      }
    }
  };

  const applyRename = (title: string) => {
    if (!selected) return;
    // Frontend-only: the backend has no rename endpoint, so the override is
    // kept in localStorage and applied whenever the list is rebuilt.
    setTitleOverrides((prev) => {
      const updated = { ...prev, [`${track}:${selected}`]: title };
      saveTitleOverrides(updated);
      return updated;
    });
    setChats((prev) => prev.map((c) => (c.id === selected ? { ...c, title } : c)));
    setDialog(null);
  };

  const doForkChat = async () => {
    if (track !== "native" || !selected) return;
    setDialog(null);
    try {
      const f = await nativeApi.forkSession(selected, `${selectedMeta?.title || selected} (fork)`);
      setChats((prev) => [{ kind: "native", id: f.id, title: f.title || f.id, subtitle: workspaceLabel(f.workspace), updatedAt: new Date().toISOString() }, ...prev]);
      setSelected(f.id);
    } catch (e) {
      setDialog({ kind: "error", message: (e as Error).message });
    }
  };

  const onDeleteChat = async (id: string, kind: "native" | "langgraph") => {
    setDialog(null);
    try {
      if (kind === "native") {
        await nativeApi.deleteSession(id);
      } else {
        await taskApi.deleteThread(id);
      }
      setChats((prev) => prev.filter((c) => c.id !== id));
      setTitleOverrides((prev) => {
        if (!(`${track}:${id}` in prev)) return prev;
        const updated = { ...prev };
        delete updated[`${track}:${id}`];
        saveTitleOverrides(updated);
        return updated;
      });
      if (selected === id) {
        setSelected(null);
        setMessages([]);
        setEvents([]);
      }
    } catch (e) {
      setDialog({ kind: "error", message: (e as Error).message });
    }
  };

  const patchLiveMessage = (id: string, patch: Partial<ChatMessage>) => {
    setMessages((prev) => prev.map((m) => (m.id === id ? { ...m, ...patch } : m)));
    setTimeout(scrollToEnd, 30);
  };

  const onSend = async () => {
    const text = composer.trim();
    if (!text || sending) return;
    setSending(true);
    const userMsg: ChatMessage = { id: `local-${Date.now()}`, role: "user", text };
    setMessages((prev) => [...prev, userMsg]);
    setEvents([]);
    setComposer("");
    setTimeout(scrollToEnd, 50);

    if (track === "native") {
      // ensure we have a session
      let sid = selected;
      if (!sid) {
        try {
          const s = await nativeApi.createSession({ title: text.slice(0, 40), workspace, agent: "build" });
          sid = s.id;
          setChats((prev) => [{ kind: "native", id: s.id, title: s.title || s.id, subtitle: workspaceLabel(s.workspace), updatedAt: new Date().toISOString() }, ...prev]);
          setSelected(s.id);
        } catch (e) {
          setMessages((prev) => [...prev, { id: `err-${Date.now()}`, role: "assistant", text: `Failed to create session: ${(e as Error).message}` }]);
          setSending(false);
          return;
        }
      }
      // Live assistant bubble: deltas grow this message in place, so the
      // answer streams instead of popping in as one full block at the end.
      const liveId = `live-${Date.now()}`;
      setMessages((prev) => [...prev, { id: liveId, role: "assistant", text: "", thinking: "", streaming: true }]);
      try {
        const url = nativeApi.sendMessageUrl(sid);
        const res = await fetch(url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: text, limits: { max_turns: 10, max_cost_usd: 0.05 } }),
        });
        if (!res.ok || !res.body) throw new Error(`${res.status} ${res.statusText}`);
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buf = "";
        let acc = "";
        let thinkAcc = "";
        let finalAnswer = "";
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buf += decoder.decode(value, { stream: true });
          const frames = buf.split("\n\n");
          buf = frames.pop() || "";
          for (const f of frames) {
            const eventLine = f.split("\n").find((l) => l.startsWith("event:"));
            const dataLine = f.split("\n").find((l) => l.startsWith("data:"));
            if (dataLine) {
              try {
                const payload = JSON.parse(dataLine.slice(5).trim());
                const eventType = eventLine?.slice(6).trim() || payload?.type || "";
                const eventData = payload?.data || payload;
                if (eventType === "assistant_delta") {
                  const t = eventData?.text || "";
                  if (t) {
                    acc += t;
                    patchLiveMessage(liveId, { text: acc });
                  }
                } else if (eventType === "reasoning_delta") {
                  const t = eventData?.text || "";
                  if (t) {
                    thinkAcc += t;
                    patchLiveMessage(liveId, { thinking: thinkAcc });
                  }
                } else if (eventType === "run_receipt") {
                  finalAnswer = eventData?.final_message || eventData?.final_text || "";
                }
                // also surface events
                if (payload?.type || eventType) setEvents((prev) => [...prev.slice(-29), payload as EventResponse]);
              } catch {
                // raw frame
                setEvents((prev) => [...prev.slice(-29), { sequence: prev.length + 1, type: "sse", session_id: sid, run_id: "", data: { raw: f.slice(0, 200) }, time: null }]);
              }
            }
          }
        }
        // Settle the live bubble with the receipt's final text when present;
        // it is the same content that streamed, so there is no pop-in.
        patchLiveMessage(liveId, { text: finalAnswer || acc, thinking: thinkAcc || undefined, streaming: false });
        if (sid) refreshConversation(sid);
      } catch (e) {
        patchLiveMessage(liveId, { text: `Error: ${(e as Error).message}`, thinking: undefined, streaming: false });
      } finally {
        setSending(false);
      }
    } else {
      // langgraph
      try {
        const body = { goal: text, track: "langgraph" as const, workspace, metadata: {} };
        const task = selected ? await taskApi.createThreadTask(selected, body) : await taskApi.createTask(body);
        if (!selected) {
          setSelected(task.thread_id);
          setChats((prev) => [{ kind: "langgraph", id: task.thread_id, title: task.thread_id.slice(0, 12), subtitle: "1 tasks", updatedAt: new Date().toISOString() }, ...prev]);
        }
        setPendingTaskId(task.id);
        // stream via SSE for a bit
        const url = taskApi.streamEventsUrl(task.id);
        const es = new EventSource(url);
        es.onmessage = (e) => setEvents((prev) => [...prev.slice(-29), { sequence: prev.length + 1, type: "sse", session_id: task.thread_id, run_id: task.id, data: { data: e.data.slice(0, 200) }, time: null }]);
        setTimeout(() => es.close(), 15000);
        await refreshConversation(task.thread_id);
        await refreshChats();
      } catch (e) {
        setMessages((prev) => [...prev, { id: `err-${Date.now()}`, role: "assistant", text: `Error: ${(e as Error).message}` }]);
      } finally {
        setSending(false);
      }
    }
  };

  const filtered = chats.filter((c) => !search || c.title.toLowerCase().includes(search.toLowerCase()) || c.id.toLowerCase().includes(search.toLowerCase()));
  const selectedMeta = chats.find((c) => c.id === selected);

  return (
    <div className="flex flex-1 min-h-0">
      {/* Sidebar */}
      <div className="w-[300px] shrink-0 flex flex-col border-r" style={{ borderColor: "var(--bg-4)", background: "var(--bg-1)" }}>
        <div className="p-3 pb-2 space-y-2.5">
          <button onClick={onNewChat} className="btn-grad w-full h-9 rounded-xl text-[12px] font-semibold flex items-center justify-center gap-2" style={{ color: "white", border: "1px solid transparent" }}>
            <span className="text-[14px] leading-none">＋</span> New chat
          </button>
          <div className="relative">
            <span className="absolute left-2.5 top-1/2 -translate-y-1/2 text-[11px]" style={{ color: "var(--fg-3)" }}>⌕</span>
            <input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search chats…" className="field !h-8 !text-[11px] !pl-7" />
          </div>
          {/* Segmented view switch */}
          <div className="grid grid-cols-2 gap-1 p-1 rounded-xl" style={{ background: "var(--bg-0)", border: "1px solid var(--bg-4)" }}>
            {(["chats", "analytics"] as const).map((view) => {
              const active = showAnalytics === (view === "analytics");
              return (
                <button
                  key={view}
                  onClick={() => setShowAnalytics(view === "analytics")}
                  className="h-7 rounded-lg text-[11px] font-semibold capitalize"
                  style={{
                    background: active ? "var(--accent-grad)" : "transparent",
                    color: active ? "white" : "var(--fg-2)",
                    boxShadow: active ? "var(--accent-glow)" : "none",
                    transition: "all var(--dur-2) var(--ease-out)",
                  }}
                >
                  {view}
                </button>
              );
            })}
          </div>
          <label className="block">
            <span className="block text-[10px] font-semibold uppercase tracking-wider mb-1" style={{ color: "var(--fg-3)" }}>Working directory</span>
            <input value={workspace} onChange={(e) => setWorkspace(e.target.value)} placeholder="." className="field !h-8 !text-[11px] mono" />
          </label>
        </div>

        <div ref={listRef} className="flex-1 overflow-auto px-2 pb-2">
          {showAnalytics ? (
            <AnalyticsView track={track} />
          ) : (
            <>
              <div className="flex items-center px-1.5 pt-1 pb-1.5">
                <span className="text-[10px] font-semibold uppercase tracking-wider" style={{ color: "var(--fg-3)" }}>Chats</span>
                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded-md" style={{ background: "var(--bg-2)", border: "1px solid var(--bg-4)", color: "var(--fg-2)" }}>{filtered.length}</span>
                <button
                  onClick={() => {
                    const current = chats.find((c) => c.id === selected);
                    if (current) setDialog({ kind: "rename", current: current.title });
                  }}
                  disabled={!selected}
                  title="Rename selected chat"
                  className="btn-quiet ml-auto h-6 px-2 rounded-md text-[10px] font-medium disabled:opacity-40 disabled:cursor-not-allowed"
                  style={{ background: "var(--bg-2)", border: "1px solid var(--bg-4)", color: "var(--fg-1)" }}
                >
                  Rename
                </button>
              </div>
              <div className="space-y-1">
              {filtered.map((c) => (
                <div
                  key={c.id}
                  role="button"
                  tabIndex={0}
                  onClick={() => setSelected(c.id)}
                  onKeyDown={(e) => e.key === "Enter" && setSelected(c.id)}
                  className="group w-full text-left p-2.5 rounded-xl flex gap-2.5 btn-quiet cursor-pointer"
                  style={{ background: selected === c.id ? "var(--bg-2)" : "transparent", border: `1px solid ${selected === c.id ? "var(--accent-ring)" : "transparent"}`, boxShadow: selected === c.id ? "0 0 0 1px var(--accent-ring), 0 2px 12px rgba(34,211,238,0.10)" : "none" }}
                >
                  <span className="w-7 h-7 rounded-lg grid place-items-center text-[11px] font-bold shrink-0" style={{ background: "var(--bg-2)", border: "1px solid var(--bg-4)", color: "var(--fg-2)" }}>{c.kind === "native" ? "◈" : "⬢"}</span>
                  <span className="min-w-0 flex-1">
                    <span className="flex items-baseline gap-2">
                      <span className="block text-[12px] font-medium truncate" style={{ color: "var(--fg-0)" }}>{c.title}</span>
                      <span className="ml-auto text-[10px] shrink-0" style={{ color: "var(--fg-3)" }}>{timeAgo(c.updatedAt)}</span>
                    </span>
                    {c.subtitle ? (
                      <span className="block text-[11px] truncate" style={{ color: "var(--fg-2)" }}>{c.subtitle}</span>
                    ) : null}
                  </span>
                  <button
                    onClick={(e) => { e.stopPropagation(); setDialog({ kind: "delete", id: c.id, chatKind: c.kind, title: c.title }); }}
                    title="Delete chat"
                    className="w-6 h-6 rounded-md grid place-items-center text-[12px] shrink-0 self-center opacity-0 group-hover:opacity-100 group-focus-within:opacity-100"
                    style={{ background: "var(--bg-3)", border: "1px solid var(--bg-4)", color: "var(--fg-2)", transition: "opacity var(--dur-1) ease, color var(--dur-1) ease" }}
                    onMouseEnter={(e) => (e.currentTarget.style.color = "var(--danger)")}
                    onMouseLeave={(e) => (e.currentTarget.style.color = "var(--fg-2)")}
                  >
                    ×
                  </button>
                </div>
              ))}
              {filtered.length === 0 && <div className="text-[11px] p-3" style={{ color: "var(--fg-3)" }}>{chats.length === 0 ? "No chats — create one above." : "No matches."}</div>}
              </div>
            </>
          )}
        </div>

        <div className="p-2.5 border-t space-y-2" style={{ borderColor: "var(--bg-4)" }}>
          {track === "native" && selected && (
            <button
              onClick={() => setDialog({ kind: "fork" })}
              className="btn-quiet w-full h-8 rounded-lg text-[11px] font-medium flex items-center justify-center gap-1.5"
              style={{ background: "var(--bg-2)", border: "1px solid var(--bg-4)", color: "var(--fg-1)" }}
            >
              ⑂ Fork chat
            </button>
          )}
          <div className="grid grid-cols-2 gap-2">
            <button
              onClick={onBackToPicker}
              className="btn-quiet h-8 rounded-lg text-[11px] font-medium flex items-center justify-center gap-1.5"
              style={{ background: "var(--bg-2)", border: "1px solid var(--bg-4)", color: "var(--fg-1)" }}
            >
              ⇄ Switch track
            </button>
            <button
              onClick={onOpenSettings}
              className="btn-quiet h-8 rounded-lg text-[11px] font-medium flex items-center justify-center gap-1.5"
              style={{ background: "var(--bg-2)", border: "1px solid var(--bg-4)", color: "var(--fg-1)" }}
            >
              ⚙ Settings
            </button>
          </div>
        </div>
      </div>

      {/* Main chat */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* header */}
        <div className="min-h-11 shrink-0 flex items-center gap-3 px-4 py-2 border-b" style={{ borderColor: "var(--bg-4)", background: "var(--bg-0)" }}>
          <span className="w-8 h-8 rounded-xl grid place-items-center text-[13px] font-bold shrink-0" style={{ background: "var(--accent-grad-soft)", border: "1px solid var(--accent-ring)", color: "var(--accent)" }}>{track === "native" ? "◈" : "⬢"}</span>
          <div className="min-w-0">
            <div className="text-[13px] font-semibold truncate font-display">{selectedMeta?.title || "New chat"}</div>
            <div className="text-[11px] truncate" style={{ color: "var(--fg-2)" }}>{selected ? `${selectedMeta?.subtitle || ""}${messages.length ? ` · ${messages.length} messages` : ""}` : "No chat selected — send a message to create one."}</div>
          </div>
          <div className="ml-auto flex gap-1.5">
            <span className="hidden sm:inline text-[10px] font-medium px-2 py-1 rounded-full capitalize" style={{ background: "var(--bg-2)", border: "1px solid var(--bg-4)", color: "var(--fg-2)" }}>
              {track}
            </span>
          </div>
        </div>

        {/* permissions strip */}
        {(track === "native" ? permissions.length > 0 : pendingApprovals.length > 0) && (
          <div className="px-3 py-2 border-b flex gap-2 items-center overflow-auto anim-slide-down" style={{ borderColor: "var(--bg-4)", background: "var(--warning-soft)" }}>
            <span className="text-[10px] font-semibold uppercase tracking-wider shrink-0" style={{ color: "var(--warning)" }}>Approval needed</span>
            {(track === "native" ? permissions : pendingApprovals.map((a) => ({ call_id: a.id, tool: a.tool_name, preview: a.risk_level, reason: a.id } as unknown as PermissionResponse))).slice(0, 3).map((p) => (
              <div key={p.call_id} className="flex items-center gap-2 px-2.5 py-1.5 rounded-full text-[11px] font-mono shrink-0" style={{ background: "var(--bg-1)", border: "1px solid var(--warning-soft)", color: "var(--fg-1)" }}>
                <span className="anim-pulse-dot" style={{ color: "var(--warning)" }}>⚠</span> {p.tool} · {p.call_id.slice(0, 6)}
                <button onClick={async () => { if (track === "native") { await nativeApi.resolvePermission(p.call_id, { allowed: true, duration: "once" }).catch(() => {}); setPermissions((prev) => prev.filter((x) => x.call_id !== p.call_id)); } else { const { taskApi: t } = await import("../../lib/api"); await t.resolveApproval(p.call_id, { approved: true }).catch(() => {}); } }} className="btn-grad ml-1 px-2 py-0.5 rounded-full text-[10px] font-medium" style={{ color: "white", border: "1px solid transparent" }}>Allow</button>
                <button onClick={async () => { if (track === "native") { await nativeApi.resolvePermission(p.call_id, { allowed: false }).catch(() => {}); setPermissions((prev) => prev.filter((x) => x.call_id !== p.call_id)); } }} className="btn-quiet px-2 py-0.5 rounded-full text-[10px] font-medium" style={{ background: "var(--bg-2)", border: "1px solid var(--bg-4)" }}>Deny</button>
              </div>
            ))}
          </div>
        )}

        {/* messages */}
        <div className="flex-1 overflow-auto p-4 sm:p-6">
          <div className="mx-auto w-full max-w-[760px] space-y-4">
            {messages.length === 0 && !sending ? (
              <div className="rounded-2xl p-8 text-center hero-glow anim-fade-up" style={{ background: "var(--bg-1)", border: "1px solid var(--bg-4)" }}>
                <div className="w-11 h-11 rounded-2xl mx-auto grid place-items-center text-[18px] font-bold" style={{ background: "var(--accent-grad)", color: "#fff", boxShadow: "var(--accent-glow)" }}>{track === "native" ? "◈" : "⬢"}</div>
                <div className="mt-3 text-[15px] font-semibold font-display">Start a <span className="grad-text">new chat</span></div>
                <div className="mt-1 text-[12px] leading-relaxed" style={{ color: "var(--fg-2)" }}>
                  {track === "native" ? "Describe a goal — the agent plans, uses tools, and streams the answer back." : "Describe a goal — it runs as a task and streams progress back."}
                </div>
                <div className="mt-4 flex flex-wrap justify-center gap-2">
                  {["Refactor auth middleware", "Add session forking", "Run tests and fix failures"].map((s) => (
                    <button key={s} onClick={() => setComposer(s)} className="btn-quiet px-3 py-1.5 rounded-full text-[11px] font-medium" style={{ background: "var(--bg-2)", border: "1px solid var(--bg-4)", color: "var(--fg-1)" }}>{s}</button>
                  ))}
                </div>
              </div>
            ) : (
              <>
                {messages.map((m) => (
                  <div key={m.id} className={`flex gap-3 ${m.streaming ? "" : "anim-fade-up"} ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                    {m.role !== "user" && <span className="w-7 h-7 rounded-lg grid place-items-center text-[11px] font-bold shrink-0 mt-0.5" style={{ background: "var(--accent-grad)", color: "white", boxShadow: "0 2px 10px rgba(34,211,238,0.2)" }}>◈</span>}
                    <div className={`max-w-[78%] rounded-2xl px-3.5 py-2.5 ${m.role === "user" ? "rounded-br-sm" : "rounded-bl-sm"}`} style={{ background: m.role === "user" ? "var(--accent-grad)" : "var(--bg-1)", color: m.role === "user" ? "white" : "var(--fg-0)", border: `1px solid ${m.role === "user" ? "transparent" : m.streaming ? "var(--accent-ring)" : "var(--bg-4)"}`, boxShadow: m.role === "user" ? "0 2px 14px rgba(34,211,238,0.2)" : "none" }}>
                      {m.role !== "user" && (m.thinking || m.streaming) && (
                        <ThinkingBlock text={m.thinking || ""} live={!!m.streaming} />
                      )}
                      {m.text ? (
                        <div className={`text-[13px] leading-relaxed whitespace-pre-wrap break-words ${m.streaming ? "stream-caret" : ""}`}>{m.text}</div>
                      ) : m.streaming ? (
                        <div className="text-[13px] leading-relaxed stream-caret" style={{ color: "var(--fg-2)" }}> </div>
                      ) : null}
                      {m.toolCalls && m.toolCalls.length > 0 && (
                        <div className="mt-2 space-y-1">
                          {m.toolCalls.map((t, i) => (
                            <div key={i} className="text-[11px] font-mono px-2 py-1 rounded-lg" style={{ background: m.role === "user" ? "rgba(255,255,255,0.15)" : "var(--bg-2)", border: "1px solid var(--bg-4)", color: m.role === "user" ? "white" : "var(--fg-2)" }}>{t.name} · {t.status} {t.preview ? `— ${t.preview}` : ""}</div>
                          ))}
                        </div>
                      )}
                      {m.time && !m.streaming && <div className="mt-1 text-[10px] font-mono" style={{ color: m.role === "user" ? "rgba(255,255,255,0.7)" : "var(--fg-3)" }}>{new Date(m.time).toLocaleTimeString()}</div>}
                    </div>
                    {m.role === "user" && <span className="w-7 h-7 rounded-full grid place-items-center text-[11px] font-semibold shrink-0 mt-0.5" style={{ background: "var(--bg-2)", border: "1px solid var(--bg-4)", color: "var(--fg-2)" }}>you</span>}
                  </div>
                ))}
                {pendingTaskId && (
                  <div className="flex gap-3 anim-fade-up">
                    <span className="w-7 h-7 rounded-lg grid place-items-center text-[11px] font-bold shrink-0" style={{ background: "var(--accent-grad)", color: "white", boxShadow: "0 2px 10px rgba(34,211,238,0.2)" }}>⬢</span>
                    <div className="max-w-[78%] rounded-2xl rounded-bl-sm px-3.5 py-2.5" style={{ background: "var(--bg-1)", border: "1px solid var(--accent-ring)" }}>
                      <div className="text-[13px] leading-relaxed flex items-center gap-2" style={{ color: "var(--fg-2)" }}>
                        <span className="w-1.5 h-1.5 rounded-full anim-pulse-dot" style={{ background: "var(--accent)" }} />
                        Working on your task…
                      </div>
                    </div>
                  </div>
                )}
                <EventTimeline events={events} live={sending} />
                <div ref={endRef} />
              </>
            )}
          </div>
        </div>

        {/* composer — floating card, no attached bar */}
        <div className="px-4 sm:px-6 pt-1 pb-4">
          <div
            className="mx-auto max-w-[760px] rounded-2xl p-2 pl-4 flex gap-2 items-end anim-fade-up"
            style={{
              background: "var(--bg-1)",
              border: "1px solid var(--bg-4)",
              boxShadow: "0 12px 32px rgba(2, 8, 20, 0.55), 0 0 0 1px rgba(6, 182, 212, 0.06)",
            }}
          >
            <textarea
              value={composer}
              onChange={(e) => setComposer(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  onSend();
                }
              }}
              placeholder={selected ? "Message the agent…" : "Message the agent to start a new chat…"}
              rows={composer.includes("\n") ? 3 : 1}
              className="flex-1 min-h-[44px] max-h-28 py-2.5 text-[13px] leading-relaxed outline-none resize-none bg-transparent"
              style={{ color: "var(--fg-0)" }}
            />
            <button onClick={onSend} disabled={sending || !composer.trim()} className="btn-grad h-10 px-5 rounded-xl text-[13px] font-semibold shrink-0" style={{ color: "white", border: "1px solid transparent" }}>
              {sending ? "…" : "Send →"}
            </button>
          </div>
        </div>
      </div>
      {dialog?.kind === "rename" && (
        <PromptDialog
          title="Rename chat"
          subtitle="Stored on this machine and applied whenever the list reloads."
          initialValue={dialog.current}
          placeholder="Chat name"
          confirmLabel="Rename"
          onSubmit={applyRename}
          onClose={() => setDialog(null)}
        />
      )}
      {dialog?.kind === "delete" && (
        <ConfirmDialog
          title="Delete chat?"
          message={`"${dialog.title}" and its full history will be permanently removed.`}
          confirmLabel="Delete"
          danger
          onConfirm={() => onDeleteChat(dialog.id, dialog.chatKind)}
          onClose={() => setDialog(null)}
        />
      )}
      {dialog?.kind === "fork" && (
        <ConfirmDialog
          title="Fork this chat?"
          message="The full conversation is copied into a new chat with its own history. The original stays untouched."
          confirmLabel="Fork chat"
          onConfirm={doForkChat}
          onClose={() => setDialog(null)}
        />
      )}
      {dialog?.kind === "error" && (
        <AlertDialog message={dialog.message} onClose={() => setDialog(null)} />
      )}
    </div>
  );
}
