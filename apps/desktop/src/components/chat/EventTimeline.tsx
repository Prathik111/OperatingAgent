import { useEffect, useRef, useState } from "react";
import type { EventResponse } from "../../lib/types";

interface TypeMeta {
  label: string;
  color: string;
}

const META: Record<string, TypeMeta> = {
  message_added: { label: "Message", color: "var(--fg-2)" },
  turn_started: { label: "Turn", color: "var(--accent)" },
  assistant_delta: { label: "Streaming", color: "var(--accent)" },
  reasoning_delta: { label: "Reasoning", color: "var(--fg-3)" },
  tool_started: { label: "Tool started", color: "var(--info)" },
  tool_finished: { label: "Tool finished", color: "var(--success)" },
  permission_requested: { label: "Approval needed", color: "var(--warning)" },
  permission_resolved: { label: "Approval resolved", color: "var(--success)" },
  run_started: { label: "Run started", color: "var(--accent)" },
  run_finished: { label: "Response", color: "var(--success)" },
  run_receipt: { label: "Receipt", color: "var(--fg-2)" },
  state: { label: "State", color: "var(--accent)" },
  finished: { label: "Response", color: "var(--success)" },
  error: { label: "Error", color: "var(--danger)" },
  sse: { label: "Stream", color: "var(--fg-3)" },
};

function metaFor(type: string): TypeMeta {
  return META[type] || { label: type.replace(/_/g, " "), color: "var(--fg-2)" };
}

function summaryOf(event: EventResponse): string {
  const data = event.data || {};
  const pick = (...keys: string[]) => {
    for (const key of keys) {
      const value = data[key];
      if (typeof value === "string" && value.trim()) return value.trim();
    }
    return "";
  };
  const text =
    pick("final_message", "final_text", "output", "text", "goal", "role", "tool_name", "tool", "error", "reason");
  if (text) return text.length > 90 ? `${text.slice(0, 90)}…` : text;
  if (typeof data.turn === "number") return `turn ${data.turn}`;
  return "";
}

function EventRow({ event }: { event: EventResponse }) {
  const [open, setOpen] = useState(false);
  const meta = metaFor(event.type);
  const summary = summaryOf(event);
  return (
    <div
      className="rounded-lg overflow-hidden"
      style={{ background: "var(--bg-0)", border: "1px solid var(--bg-4)" }}
    >
      <button
        onClick={() => setOpen((v) => !v)}
        className="btn-quiet w-full flex items-center gap-2 px-2.5 py-2 text-left"
      >
        <span
          className="w-1.5 h-1.5 rounded-full shrink-0"
          style={{ background: meta.color, boxShadow: `0 0 6px ${meta.color}` }}
        />
        <span className="text-[11px] font-medium capitalize shrink-0" style={{ color: "var(--fg-1)" }}>
          {meta.label}
        </span>
        {summary && (
          <span className="text-[11px] truncate flex-1" style={{ color: "var(--fg-2)" }}>
            {summary}
          </span>
        )}
        <span
          className="text-[10px] font-mono shrink-0"
          style={{ color: "var(--fg-3)" }}
        >
          #{event.sequence}
        </span>
        <svg
          width="12"
          height="12"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          className="shrink-0"
          style={{ color: "var(--fg-3)", transform: open ? "rotate(180deg)" : undefined, transition: "transform var(--dur-1) ease" }}
        >
          <path d="m6 9 6 6 6-6" />
        </svg>
      </button>
      {open && (
        <pre
          className="mx-2 mb-2 p-2 rounded-lg text-[10px] font-mono whitespace-pre-wrap break-words overflow-auto max-h-40 anim-fade-in"
          style={{ background: "var(--bg-1)", border: "1px solid var(--bg-4)", color: "var(--fg-2)" }}
        >
          {JSON.stringify(event.data, null, 2)}
        </pre>
      )}
    </div>
  );
}

export function EventTimeline({ events, live }: { events: EventResponse[]; live?: boolean }) {
  const [open, setOpen] = useState(false);
  const wasLive = useRef(false);

  // Auto-expand when a new run starts streaming; the user can collapse it.
  useEffect(() => {
    if (live && !wasLive.current) setOpen(true);
    wasLive.current = !!live;
  }, [live]);

  if (events.length === 0 && !live) return null;

  return (
    <div className="rounded-xl overflow-hidden" style={{ background: "var(--bg-1)", border: "1px solid var(--bg-4)" }}>
      <button
        onClick={() => setOpen((v) => !v)}
        className="btn-quiet w-full flex items-center gap-2 px-3 py-2 text-left"
      >
        <span
          className={`w-1.5 h-1.5 rounded-full shrink-0 ${live ? "anim-pulse-dot" : ""}`}
          style={{ background: live ? "var(--accent)" : "var(--fg-3)" }}
        />
        <span className="text-[11px] font-semibold" style={{ color: "var(--fg-1)" }}>
          Activity
        </span>
        <span className="text-[10px] font-mono" style={{ color: "var(--fg-3)" }}>
          {events.length} {events.length === 1 ? "event" : "events"} · this response
        </span>
        <svg
          width="12"
          height="12"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          className="ml-auto shrink-0"
          style={{ color: "var(--fg-3)", transform: open ? "rotate(180deg)" : undefined, transition: "transform var(--dur-1) ease" }}
        >
          <path d="m6 9 6 6 6-6" />
        </svg>
      </button>
      {open && (
        <div className="px-2 pb-2 space-y-1.5 max-h-64 overflow-auto anim-fade-in">
          {events.map((e) => (
            <EventRow key={e.sequence} event={e} />
          ))}
          {events.length === 0 && (
            <div className="text-[11px] px-1 py-2" style={{ color: "var(--fg-3)" }}>
              Waiting for the first events…
            </div>
          )}
        </div>
      )}
    </div>
  );
}
