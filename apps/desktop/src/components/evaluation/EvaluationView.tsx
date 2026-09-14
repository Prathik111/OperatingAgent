import { useEffect, useState } from "react";
import * as XLSX from "xlsx";
import { taskApi } from "../../lib/api";
import type { EvaluationDashboard, EvaluationExecution } from "../../lib/types";
import { folderName, isTauri, pickDirectory } from "../../lib/pickFolder";
import { MarkdownText } from "../MarkdownText";

function pct(value: number | null) {
  return value == null ? "—" : `${(value * 100).toFixed(1)}%`;
}

function score(value: number | null) {
  return value == null ? "—" : value.toFixed(2);
}

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="rounded-xl p-3" style={{ background: "var(--bg-1)", border: "1px solid var(--bg-4)" }}>
      <div className="text-[10px] font-semibold uppercase tracking-wide" style={{ color: "var(--fg-3)" }}>{label}</div>
      <div className="text-[20px] font-semibold grad-text">{value}</div>
      {sub && <div className="text-[11px] font-mono" style={{ color: "var(--fg-2)" }}>{sub}</div>}
    </div>
  );
}

const EVALUATION_WORKSPACE_KEY = "operating-agent:evaluation-workspace";

function loadEvaluationWorkspace(): string {
  try {
    return localStorage.getItem(EVALUATION_WORKSPACE_KEY) || ".";
  } catch {
    return ".";
  }
}

function saveEvaluationWorkspace(workspace: string) {
  try {
    localStorage.setItem(EVALUATION_WORKSPACE_KEY, workspace);
  } catch {
    // Storage can be unavailable in embedded webviews.
  }
}

function AgentReport({
  track,
  run,
  executions,
}: {
  track: "native" | "langgraph";
  run: EvaluationDashboard["comparison"][number] | undefined;
  executions: EvaluationExecution[];
}) {
  const latency = run?.avg_latency_ms == null ? "â€”" : `${Math.round(run.avg_latency_ms)} ms`;
  const tokens = run?.total_tokens == null ? "â€”" : run.total_tokens.toLocaleString();
  const cost = run?.total_cost == null ? "â€”" : `$${run.total_cost.toFixed(4)}`;
  return (
    <section className="rounded-xl p-4 space-y-4" style={{ background: "var(--bg-1)", border: "1px solid var(--bg-4)" }}>
      <div className="flex items-start justify-between gap-3"><div><div className="text-[10px] font-semibold uppercase tracking-wide" style={{ color: "var(--accent)" }}>{track} agent</div><h2 className="text-[16px] font-semibold capitalize">{track} evaluation report</h2><p className="text-[11px]" style={{ color: "var(--fg-3)" }}>Latest completed suite run and its execution transcripts.</p></div><span className="rounded-full px-2 py-1 font-mono text-[10px]" style={{ background: "var(--bg-3)", color: "var(--fg-2)" }}>{run?.status || "no runs"}</span></div>
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
        <Stat label="Pass rate" value={pct(run?.pass_rate ?? null)} sub={`${run?.result_count ?? 0} cases`} />
        <Stat label="Score" value={score(run?.average_score ?? null)} sub="correctness" />
        <Stat label="Latency" value={latency} sub="average" />
        <Stat label="Tokens" value={tokens} sub="reported usage" />
        <Stat label="Cost" value={cost} sub="reported cost" />
        <Stat label="Tools" value={pct(run?.tool_success_rate ?? null)} sub="success rate" />
      </div>
      <div className="border-t pt-3" style={{ borderColor: "var(--bg-4)" }}><div className="mb-2 flex items-center justify-between"><h3 className="text-[11px] font-semibold">Execution chats</h3><span className="font-mono text-[10px]" style={{ color: "var(--fg-3)" }}>{executions.length} completed</span></div><div className="space-y-2">{executions.map((execution) => <details key={execution.id} className="rounded-lg p-3" style={{ background: "var(--bg-0)", border: "1px solid var(--bg-4)" }}><summary className="cursor-pointer list-none flex gap-2 text-[11px]"><span className="font-mono" style={{ color: execution.success ? "var(--success)" : "var(--danger)" }}>{execution.success ? "passed" : execution.status}</span><span className="truncate flex-1">{execution.case_id}: {execution.goal}</span></summary><div className="mt-3 space-y-3 text-[12px]"><div><div className="mb-1 text-[10px] uppercase tracking-wide" style={{ color: "var(--fg-3)" }}>Prompt</div><MarkdownText>{execution.goal}</MarkdownText></div><div><div className="mb-1 text-[10px] uppercase tracking-wide" style={{ color: "var(--fg-3)" }}>Response</div><MarkdownText>{execution.output || execution.error || "No final response recorded."}</MarkdownText></div></div></details>)}{executions.length === 0 && <div className="py-3 text-[11px]" style={{ color: "var(--fg-3)" }}>No completed execution chats yet.</div>}</div></div>
    </section>
  );
}

export function EvaluationView() {
  const [data, setData] = useState<EvaluationDashboard | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [tracks, setTracks] = useState<Array<"native" | "langgraph">>(["native", "langgraph"]);
  const [reportTrack, setReportTrack] = useState<"native" | "langgraph">("native");
  const [suiteName, setSuiteName] = useState("desktop-suite");
  const [suiteVersion, setSuiteVersion] = useState("1");
  const [workspace, setWorkspace] = useState(loadEvaluationWorkspace);
  const canBrowse = isTauri();
  const [cases, setCases] = useState<Array<{ id: string; goal: string; working_directory?: string; expected_output_contains?: string; metadata?: Record<string, unknown> }>>([
    { id: "greeting", goal: "Reply with a short friendly greeting.", expected_output_contains: "hello" },
    { id: "capabilities", goal: "Summarize what you can do in one sentence." },
    { id: "evidence", goal: "Explain in two sentences why checking evidence matters before claiming a task is complete." },
  ]);

  const loadDashboard = () => {
    setLoading(true);
    return taskApi.evaluationDashboard()
      .then((next) => { setData(next); setError(null); })
      .catch((cause: Error) => setError(cause.message))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    let cancelled = false;
    taskApi.evaluationDashboard()
      .then((next) => { if (!cancelled) { setData(next); setError(null); } })
      .catch((cause: Error) => { if (!cancelled) setError(cause.message); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, []);

  const runDefaultSuite = async () => {
    setRunning(true);
    try {
      const started = await taskApi.startEvaluation({
        name: suiteName.trim() || "desktop-suite",
        version: suiteVersion.trim() || "1",
        tracks,
        cases: cases.map((item) => ({
          ...item,
          working_directory: item.working_directory?.trim() || workspace.trim() || ".",
        })),
      });
      // Poll the same dashboard endpoint until every selected track has a
      // finished evaluation run. This removes the old refresh-only behavior.
      for (let attempt = 0; attempt < 120; attempt += 1) {
        const next = await taskApi.evaluationDashboard();
        setData(next);
        setError(null);
        const completed = started.evaluation_run_ids.every((id) => next.run_breakdown.some((run) => run.id === id && !!run.finished_at));
        if (completed) break;
        await new Promise((resolve) => setTimeout(resolve, 1000));
      }
    } catch (cause) {
      setError((cause as Error).message);
    } finally {
      setRunning(false);
    }
  };

  const importCases = async (file: File) => {
    try {
      const lowerName = file.name.toLowerCase();
      let parsed: unknown;
      if (lowerName.endsWith(".json")) {
        parsed = JSON.parse(await file.text());
      } else {
        const workbook = XLSX.read(await file.arrayBuffer(), { type: "array" });
        const sheet = workbook.Sheets[workbook.SheetNames[0]];
        parsed = sheet ? XLSX.utils.sheet_to_json<Record<string, unknown>>(sheet, { defval: "" }) : [];
      }
      const rows = Array.isArray(parsed) ? parsed : (parsed && typeof parsed === "object" && Array.isArray((parsed as { cases?: unknown }).cases) ? (parsed as { cases: unknown[] }).cases : []);
      const value = (row: Record<string, unknown>, ...keys: string[]) => {
        const normalized = Object.fromEntries(Object.entries(row).map(([key, item]) => [key.toLowerCase().replace(/[\s-]+/g, "_"), item]));
        return keys.map((key) => normalized[key]).find((item) => item !== undefined && String(item).trim() !== "");
      };
      const next = rows.filter((item): item is Record<string, unknown> => !!item && typeof item === "object").map((item) => {
        const id = value(item, "id", "case_id");
        const goal = value(item, "goal", "prompt", "input");
        const directory = value(item, "working_directory", "workspace", "directory", "dir");
        const expected = value(item, "expected_output_contains", "expected", "expected_text");
        const metadataValue = value(item, "metadata");
        let metadata: Record<string, unknown> | undefined;
        if (metadataValue && typeof metadataValue === "object") metadata = metadataValue as Record<string, unknown>;
        else if (metadataValue) {
          try { metadata = JSON.parse(String(metadataValue)); } catch { metadata = undefined; }
        }
        return { id: id ? String(id).trim() : "", goal: goal ? String(goal).trim() : "", working_directory: directory ? String(directory).trim() : "", expected_output_contains: expected ? String(expected).trim() : "", metadata };
      }).filter((item) => item.id && item.goal);
      if (next.length) setCases(next); else setError("Import needs rows with id and goal fields.");
    } catch { setError("Could not parse the file. Use JSON, CSV/TSV, or XLSX with id, goal, working_directory, expected_output_contains, and optional metadata columns."); }
  };

  const chooseWorkspace = async () => {
    const selected = await pickDirectory(workspace);
    if (!selected) return;
    setWorkspace(selected);
    saveEvaluationWorkspace(selected);
  };

  return (
    <div className="flex-1 min-h-0 overflow-auto" style={{ background: "var(--bg-0)" }}>
      <div className="mx-auto w-full max-w-[1100px] p-5 sm:p-8 space-y-5">
        <div className="flex items-start gap-3">
          <div>
            <div className="text-[11px] font-semibold uppercase tracking-wider" style={{ color: "var(--accent)" }}>Evaluation</div>
            <h1 className="text-[22px] font-semibold font-display">Quality and benchmark results</h1>
            <p className="text-[12px] mt-1" style={{ color: "var(--fg-2)" }}>Normalized suites, cases, scores, and run outcomes from the evaluation tables.</p>
          </div>
          <div className="ml-auto flex gap-2">
            <div className="flex items-center gap-2 h-8 px-2 rounded-lg text-[11px]" style={{ background: "var(--bg-1)", border: "1px solid var(--bg-4)" }}>
              {(["native", "langgraph"] as const).map((item) => <label key={item} className="flex items-center gap-1"><input type="checkbox" checked={tracks.includes(item)} onChange={() => setTracks((current) => current.includes(item) ? current.filter((track) => track !== item) : [...current, item])} />{item}</label>)}
            </div>
            <button onClick={runDefaultSuite} disabled={running || tracks.length === 0} className="btn-grad h-8 px-3 rounded-lg text-[11px] font-medium disabled:opacity-50" style={{ color: "white" }}>{running ? "Running…" : "Compare agents"}</button>
            <button onClick={loadDashboard} className="btn-quiet h-8 px-3 rounded-lg text-[11px]" style={{ background: "var(--bg-1)", border: "1px solid var(--bg-4)" }}>Refresh</button>
          </div>
        </div>

        {loading ? (
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">{[0, 1, 2, 3].map((i) => <div key={i} className="rounded-xl skeleton" style={{ height: 92 }} />)}</div>
        ) : error ? (
          <div className="rounded-xl p-4 text-[12px] font-mono" style={{ color: "var(--danger)", background: "var(--danger-soft)", border: "1px solid var(--danger-soft)" }}>{error}</div>
        ) : data ? (
          <>
            <section className="rounded-xl p-4 space-y-3" style={{ background: "var(--bg-1)", border: "1px solid var(--bg-4)" }}>
              <div className="flex items-center justify-between gap-3"><h2 className="text-[13px] font-semibold">Evaluation suite</h2><label className="text-[11px] cursor-pointer" style={{ color: "var(--accent)" }}>Import JSON/CSV/TSV/XLSX<input type="file" accept=".json,.csv,.tsv,.txt,.xlsx,.xls" className="hidden" onChange={(e) => e.target.files?.[0] && void importCases(e.target.files[0])} /></label></div>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-2"><input value={suiteName} onChange={(e) => setSuiteName(e.target.value)} placeholder="Suite name" className="field" /><input value={suiteVersion} onChange={(e) => setSuiteVersion(e.target.value)} placeholder="Version" className="field" /><button type="button" onClick={chooseWorkspace} disabled={!canBrowse} className="field flex items-center justify-between gap-2 text-left disabled:opacity-60"><span className="truncate"><span className="mr-2" style={{ color: "var(--accent)" }}>Folder</span>{folderName(workspace)}</span><span className="text-[10px]" style={{ color: "var(--fg-3)" }}>{canBrowse ? "Chooseâ€¦" : "Desktop only"}</span></button></div>
              <div className="text-[10px]" style={{ color: "var(--fg-3)" }}>{canBrowse ? "Choose the folder once; both agents run every case in that same workspace." : "Folder selection is available in the Tauri desktop app."}</div>
              <div className="text-[10px]" style={{ color: "var(--fg-3)" }}>Both agents receive every case below. Sheet columns: id, goal, working_directory (optional), expected_output_contains, metadata (optional JSON). Blank directories use the selected evaluation folder.</div>
              <div className="space-y-2">{cases.map((item, index) => <div key={`${item.id}-${index}`} className="grid grid-cols-[110px_1fr_180px_auto] gap-2"><input value={item.id} onChange={(e) => setCases((all) => all.map((row, i) => i === index ? { ...row, id: e.target.value } : row))} placeholder="case id" className="field mono" /><input value={item.goal} onChange={(e) => setCases((all) => all.map((row, i) => i === index ? { ...row, goal: e.target.value } : row))} placeholder="Prompt / goal" className="field" /><input value={item.expected_output_contains || ""} onChange={(e) => setCases((all) => all.map((row, i) => i === index ? { ...row, expected_output_contains: e.target.value } : row))} placeholder="Expected text" className="field" /><button type="button" onClick={() => setCases((all) => all.filter((_, i) => i !== index))} className="btn-quiet px-2" aria-label="Remove case">x</button></div>)}</div>
              <button type="button" onClick={() => setCases((all) => [...all, { id: `case-${all.length + 1}`, goal: "", expected_output_contains: "" }])} className="btn-quiet h-7 px-2 text-[11px]">+ Add case</button>
            </section>
            <div className="hidden text-[10px] font-semibold uppercase tracking-wide" style={{ color: "var(--fg-3)" }}>Combined benchmark summary</div>
            <div className="hidden grid grid-cols-2 lg:grid-cols-4 gap-3">
              <Stat label="Suites" value={String(data.suites)} sub={`${data.cases} cases`} />
              <Stat label="Evaluation runs" value={String(data.runs)} sub={`${data.results} scored results`} />
              <Stat label="Pass rate" value={pct(data.pass_rate)} sub="successful cases" />
              <Stat label="Average score" value={score(data.average_score)} sub="across evaluation scores" />
              <Stat label="Avg latency" value={data.avg_latency_ms == null ? "—" : `${Math.round(data.avg_latency_ms)} ms`} sub="evaluated runs" />
              <Stat label="Tokens" value={data.total_tokens == null ? "—" : data.total_tokens.toLocaleString()} sub={data.total_cost == null ? "cost unavailable" : `$${data.total_cost.toFixed(4)} total`} />
              <Stat label="Tool success" value={pct(data.tool_success_rate)} sub="successful tool calls" />
            </div>

            {!data.available && (
              <div className="rounded-xl p-4" style={{ background: "var(--warning-soft)", border: "1px solid var(--warning-soft)" }}>
                <div className="text-[12px] font-semibold" style={{ color: "var(--warning)" }}>Evaluation data is not available</div>
                <div className="text-[11px] mt-1" style={{ color: "var(--fg-2)" }}>{data.reason || "Use PostgreSQL with the evaluation schema to populate this page."}</div>
              </div>
            )}

            <section className="space-y-3">
              <div className="inline-flex rounded-xl p-1" style={{ background: "var(--bg-1)", border: "1px solid var(--bg-4)" }}>
                {(["native", "langgraph"] as const).map((track) => <button key={track} type="button" onClick={() => setReportTrack(track)} className="h-8 rounded-lg px-4 text-[11px] font-semibold capitalize" style={{ background: reportTrack === track ? "var(--accent-grad)" : "transparent", color: reportTrack === track ? "white" : "var(--fg-2)" }}>{track} report</button>)}
              </div>
              <AgentReport track={reportTrack} run={data.comparison.find((run) => run.track === reportTrack)} executions={data.executions.filter((execution) => execution.track === reportTrack)} />
            </section>

            <section className="hidden rounded-xl p-4" style={{ background: "var(--bg-1)", border: "1px solid var(--bg-4)" }}>
              <div className="flex items-baseline justify-between gap-3"><h2 className="text-[13px] font-semibold">Metric breakdown</h2><span className="text-[10px] font-mono" style={{ color: "var(--fg-3)" }}>{data.sources.database ? "PostgreSQL evaluation_scores" : "No score rows"}</span></div>
              <div className="mt-3 space-y-2">
                {data.metrics.map((metric) => (
                  <div key={metric.metric} className="flex items-center gap-3 text-[11px]">
                    <span className="w-40 truncate font-mono" style={{ color: "var(--fg-1)" }}>{metric.metric}</span>
                    <div className="flex-1 h-2 rounded-full overflow-hidden" style={{ background: "var(--bg-3)" }}><div className="h-full" style={{ width: `${Math.max(0, Math.min(100, (metric.average || 0) * 100))}%`, background: "var(--accent-grad)" }} /></div>
                    <span className="w-12 text-right font-mono" style={{ color: "var(--fg-1)" }}>{score(metric.average)}</span>
                    <span className="w-16 text-right font-mono" style={{ color: "var(--fg-3)" }}>{metric.count} samples</span>
                  </div>
                ))}
                {data.metrics.length === 0 && <div className="text-[11px]" style={{ color: "var(--fg-3)" }}>No evaluation scores have been recorded.</div>}
              </div>
            </section>

            <section className="hidden rounded-xl p-4" style={{ background: "var(--bg-1)", border: "1px solid var(--bg-4)" }}>
              <h2 className="text-[13px] font-semibold">Per-agent results</h2>
              <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-3">
                {(["native", "langgraph"] as const).map((item) => {
                  const run = data.comparison.find((candidate) => candidate.track === item);
                  return <div key={item} className="rounded-lg p-3" style={{ background: "var(--bg-0)", border: "1px solid var(--bg-4)" }}><div className="text-[11px] font-semibold capitalize" style={{ color: "var(--accent)" }}>{item}</div><div className="mt-2 grid grid-cols-3 gap-2 text-[11px] font-mono"><span>Pass<br /><b>{run ? pct(run.pass_rate) : "—"}</b></span><span>Score<br /><b>{run ? score(run.average_score) : "—"}</b></span><span>Cases<br /><b>{run?.result_count ?? "—"}</b></span></div></div>;
                })}
              </div>
            </section>

            <section className="hidden rounded-xl p-4" style={{ background: "var(--bg-1)", border: "1px solid var(--bg-4)" }}>
              <div className="flex items-baseline justify-between gap-3"><h2 className="text-[13px] font-semibold">Recent evaluation runs</h2><span className="text-[10px] font-mono" style={{ color: "var(--fg-3)" }}>latest 100</span></div>
              <div className="mt-3 overflow-x-auto"><table className="w-full text-[11px] font-mono"><thead><tr className="text-left" style={{ color: "var(--fg-3)" }}><th className="pb-2">Suite</th><th className="pb-2">Track</th><th className="pb-2">Cases</th><th className="pb-2">Pass</th><th className="pb-2">Score</th><th className="pb-2">Started</th></tr></thead><tbody>
                {data.run_breakdown.map((run) => <tr key={run.id} style={{ borderTop: "1px solid var(--bg-4)" }}><td className="py-2 pr-3" style={{ color: "var(--fg-1)" }}>{run.suite}</td><td className="py-2 pr-3" style={{ color: "var(--accent)" }}>{run.track}</td><td className="py-2 pr-3">{run.result_count}</td><td className="py-2 pr-3">{pct(run.pass_rate)}</td><td className="py-2 pr-3">{score(run.average_score)}</td><td className="py-2" style={{ color: "var(--fg-3)" }}>{new Date(run.started_at).toLocaleString()}</td></tr>)}
                {data.run_breakdown.length === 0 && <tr><td colSpan={6} className="py-4" style={{ color: "var(--fg-3)" }}>No evaluation runs have been recorded.</td></tr>}
              </tbody></table></div>
            </section>

            <section className="hidden rounded-xl p-4" style={{ background: "var(--bg-1)", border: "1px solid var(--bg-4)" }}>
              <div className="flex items-baseline justify-between gap-3"><h2 className="text-[13px] font-semibold">Evaluation execution chats</h2><span className="text-[10px] font-mono" style={{ color: "var(--fg-3)" }}>latest 100</span></div>
              <p className="mt-1 text-[11px]" style={{ color: "var(--fg-3)" }}>Benchmark transcripts stay separate from ordinary Workspace chats. Open a run to inspect its prompt and final response.</p>
              <div className="mt-3 grid grid-cols-1 xl:grid-cols-2 gap-3">
                {(["native", "langgraph"] as const).map((track) => (
                  <div key={track} className="rounded-lg p-3 space-y-2" style={{ background: "var(--bg-0)", border: "1px solid var(--bg-4)" }}>
                    <div className="flex items-center justify-between"><h3 className="text-[11px] font-semibold capitalize" style={{ color: "var(--accent)" }}>{track} execution chats</h3><span className="font-mono text-[10px]" style={{ color: "var(--fg-3)" }}>{data.executions.filter((execution) => execution.track === track).length} runs</span></div>
                    {data.executions.filter((execution) => execution.track === track).map((execution) => (
                  <details key={execution.id} className="rounded-lg p-3" style={{ background: "var(--bg-0)", border: "1px solid var(--bg-4)" }}>
                    <summary className="cursor-pointer list-none flex items-center gap-3 text-[11px]">
                      <span className="font-semibold capitalize" style={{ color: "var(--accent)" }}>{execution.track}</span>
                      <span className="font-mono" style={{ color: execution.success ? "var(--success)" : "var(--danger)" }}>{execution.success ? "passed" : execution.status}</span>
                      <span className="truncate flex-1" style={{ color: "var(--fg-1)" }}>{execution.case_id}: {execution.goal}</span>
                      <span className="hidden md:inline font-mono" style={{ color: "var(--fg-3)" }}>{new Date(execution.created_at).toLocaleString()}</span>
                    </summary>
                    <div className="mt-3 space-y-3 text-[12px]">
                      <div className="rounded-lg p-3" style={{ background: "var(--bg-1)" }}><div className="mb-1 text-[10px] font-semibold uppercase tracking-wide" style={{ color: "var(--fg-3)" }}>User / evaluation prompt</div><MarkdownText>{execution.goal}</MarkdownText></div>
                      <div className="rounded-lg p-3" style={{ background: "var(--bg-1)" }}><div className="mb-1 text-[10px] font-semibold uppercase tracking-wide" style={{ color: "var(--fg-3)" }}>Assistant</div><MarkdownText>{execution.output || execution.error || "Execution has not produced a final response yet."}</MarkdownText></div>
                      <div className="font-mono text-[10px]" style={{ color: "var(--fg-3)" }}>workspace: {execution.workspace ? folderName(execution.workspace) : "—"} · thread: {execution.thread_id}</div>
                    </div>
                  </details>
                    ))}
                    {data.executions.filter((execution) => execution.track === track).length === 0 && <div className="text-[11px]" style={{ color: "var(--fg-3)" }}>No completed {track} executions yet.</div>}
                  </div>
                ))}
              </div>
            </section>

            <div className="text-[11px]" style={{ color: "var(--fg-3)" }}>Database metrics come from evaluation tables and derived run data. Langfuse tracing is {data.sources.langfuse ? "enabled for trace enrichment" : "not configured"}; trace-level observations and model-judge scores require an ingestion path into evaluation_scores.</div>
          </>
        ) : null}
      </div>
    </div>
  );
}
