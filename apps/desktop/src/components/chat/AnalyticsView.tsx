import { useEffect, useState } from "react";
import { nativeApi, taskApi } from "../../lib/api";
import type { RunResponse, TaskResponse } from "../../lib/types";

export function AnalyticsView({ track }: { track: "native" | "langgraph" }) {
  const [nativeRuns, setNativeRuns] = useState<RunResponse[]>([]);
  const [tasks, setTasks] = useState<TaskResponse[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setLoading(true);
      try {
        if (track === "native") {
          // fetch a few sessions' runs to aggregate
          const sessions = await nativeApi.listSessions({ limit: 20 }).catch(() => []);
          const allRuns: RunResponse[] = [];
          for (const s of sessions.slice(0, 6)) {
            const runs = await nativeApi.listRuns(s.id).catch(() => [] as RunResponse[]);
            allRuns.push(...runs);
          }
          if (!cancelled) setNativeRuns(allRuns);
        } else {
          const threads = await taskApi.listThreads({ limit: 20 }).catch(() => []);
          const allTasks: TaskResponse[] = [];
          for (const th of threads.slice(0, 6)) {
            const ts = await taskApi.listThreadTasks(th.id, { limit: 50 }).catch(() => [] as TaskResponse[]);
            allTasks.push(...ts);
          }
          if (!cancelled) setTasks(allTasks);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [track]);

  if (loading) {
    return (
      <div className="p-4 space-y-3 anim-fade-in">
        <div className="grid grid-cols-2 gap-2">
          {[0, 1, 2, 3].map((i) => (
            <div key={i} className="rounded-xl p-3 skeleton" style={{ border: "1px solid var(--bg-4)", height: 76 }} />
          ))}
        </div>
        <div className="rounded-xl p-3 skeleton" style={{ border: "1px solid var(--bg-4)", height: 120 }} />
      </div>
    );
  }

  if (track === "native") {
    const totalCost = nativeRuns.reduce((s, r) => s + (r.cost_usd || 0), 0);
    const totalTokens = nativeRuns.reduce((s, r) => s + (r.input_tokens || 0) + (r.output_tokens || 0), 0);
    const totalTurns = nativeRuns.reduce((s, r) => s + (r.turns || 0), 0);
    const finished = nativeRuns.filter((r) => r.status === "finished").length;
    const byStatus: Record<string, number> = {};
    nativeRuns.forEach((r) => (byStatus[r.status] = (byStatus[r.status] || 0) + 1));

    return (
      <div className="p-4 space-y-4">
        <h3 className="text-[13px] font-semibold font-display">Usage</h3>

        <div className="grid grid-cols-2 gap-2">
          <Stat label="Runs" value={String(nativeRuns.length)} sub={`${finished} finished`} />
          <Stat label="Cost" value={`$${totalCost.toFixed(4)}`} sub={`${totalTokens.toLocaleString()} tok`} />
          <Stat label="Turns" value={String(totalTurns)} sub={`avg ${(nativeRuns.length ? (totalTurns / nativeRuns.length).toFixed(1) : "—")}`} />
          <Stat label="Tokens" value={totalTokens.toLocaleString()} sub={`${nativeRuns.reduce((s, r) => s + (r.cached_tokens || 0), 0).toLocaleString()} cached`} />
        </div>

        <div className="rounded-xl p-3" style={{ background: "var(--bg-1)", border: "1px solid var(--bg-4)" }}>
          <div className="text-[11px] font-semibold tracking-wide uppercase" style={{ color: "var(--fg-2)" }}>
            By status
          </div>
          <div className="mt-2 space-y-1.5">
            {Object.entries(byStatus).map(([k, v]) => (
              <div key={k} className="flex items-center gap-2 text-[11px] font-mono">
                <span className="w-20 truncate" style={{ color: "var(--fg-2)" }}>{k}</span>
                <span className="flex-1 h-1.5 rounded-full overflow-hidden" style={{ background: "var(--bg-3)" }}>
                  <span className="block h-full" style={{ width: `${(v / nativeRuns.length) * 100}%`, background: k === "finished" ? "var(--success)" : k === "limit_reached" ? "var(--warning)" : "var(--bg-4)" }} />
                </span>
                <span style={{ color: "var(--fg-1)" }}>{v}</span>
              </div>
            ))}
            {nativeRuns.length === 0 && <div className="text-[11px]" style={{ color: "var(--fg-3)" }}>No runs yet — send a task to see metrics.</div>}
          </div>
        </div>

        <div className="rounded-xl p-3 max-h-[320px] overflow-auto" style={{ background: "var(--bg-1)", border: "1px solid var(--bg-4)" }}>
          <div className="text-[11px] font-semibold tracking-wide uppercase" style={{ color: "var(--fg-2)" }}>Recent runs</div>
          <div className="mt-2 space-y-1.5">
            {nativeRuns.slice(0, 12).map((r) => (
              <div key={r.run_id} className="flex gap-2 text-[11px] font-mono p-1.5 rounded" style={{ background: "var(--bg-0)", border: "1px solid var(--bg-4)" }}>
                <span style={{ color: r.status === "finished" ? "var(--success)" : "var(--fg-2)" }}>{r.status}</span>
                <span className="truncate flex-1" style={{ color: "var(--fg-1)" }}>{r.turns} turns · {r.duration_seconds.toFixed(1)}s · ${r.cost_usd.toFixed(4)}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  const totalTasks = tasks.length;
  const byStatus: Record<string, number> = {};
  tasks.forEach((t) => {
    const s = t.status || "pending";
    byStatus[s] = (byStatus[s] || 0) + 1;
  });
  const byTrack: Record<string, number> = {};
  tasks.forEach((t) => (byTrack[t.track] = (byTrack[t.track] || 0) + 1));

  return (
      <div className="p-4 space-y-4">
        <h3 className="text-[13px] font-semibold font-display">Usage</h3>

        <div className="grid grid-cols-2 gap-2">
        <Stat label="Tasks" value={String(totalTasks)} sub={`${byStatus["completed"] || 0} completed`} />
        <Stat label="Threads" value={String(new Set(tasks.map((t) => t.thread_id)).size)} sub={`${Object.keys(byTrack).length} tracks`} />
        <Stat label="Native" value={String(byTrack["native"] || 0)} sub="track=native" />
        <Stat label="LangGraph" value={String(byTrack["langgraph"] || 0)} sub="track=langgraph" />
      </div>

      <div className="rounded-xl p-3" style={{ background: "var(--bg-1)", border: "1px solid var(--bg-4)" }}>
        <div className="text-[11px] font-semibold tracking-wide uppercase" style={{ color: "var(--fg-2)" }}>By status</div>
        <div className="mt-2 space-y-1.5">
          {Object.entries(byStatus).map(([k, v]) => (
            <div key={k} className="flex items-center gap-2 text-[11px] font-mono">
              <span className="w-20 truncate" style={{ color: "var(--fg-2)" }}>{k}</span>
              <span className="flex-1 h-1.5 rounded-full overflow-hidden" style={{ background: "var(--bg-3)" }}>
                <span className="block h-full" style={{ width: `${(v / totalTasks) * 100}%`, background: k === "completed" ? "var(--success)" : "var(--bg-4)" }} />
              </span>
              <span style={{ color: "var(--fg-1)" }}>{v}</span>
            </div>
          ))}
          {totalTasks === 0 && <div className="text-[11px]" style={{ color: "var(--fg-3)" }}>No tasks yet.</div>}
        </div>
      </div>

      <div className="rounded-xl p-3 max-h-[320px] overflow-auto" style={{ background: "var(--bg-1)", border: "1px solid var(--bg-4)" }}>
        <div className="text-[11px] font-semibold tracking-wide uppercase" style={{ color: "var(--fg-2)" }}>Recent tasks</div>
        <div className="mt-2 space-y-1.5">
            {tasks.slice(0, 12).map((t) => (
              <div key={t.id} className="flex gap-2 text-[11px] font-mono p-1.5 rounded" style={{ background: "var(--bg-0)", border: "1px solid var(--bg-4)" }}>
                <span style={{ color: t.status === "completed" ? "var(--success)" : "var(--fg-2)" }}>{t.status || "pending"}</span>
                <span className="truncate flex-1" style={{ color: "var(--fg-1)" }}>{t.goal.slice(0, 60)}</span>
              </div>
            ))}
        </div>
      </div>
    </div>
  );
}

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="rounded-xl p-3 anim-fade-up card-lift" style={{ background: "var(--bg-1)", border: "1px solid var(--bg-4)" }}>
      <div className="text-[10px] font-semibold tracking-wide uppercase" style={{ color: "var(--fg-3)" }}>{label}</div>
      <div className="text-[16px] font-semibold grad-text inline-block">{value}</div>
      {sub && <div className="text-[11px] font-mono" style={{ color: "var(--fg-2)" }}>{sub}</div>}
    </div>
  );
}
