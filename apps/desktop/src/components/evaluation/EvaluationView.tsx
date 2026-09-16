import { useEffect, useRef, useState } from "react";
import * as XLSX from "xlsx";
import { taskApi, judgeApi } from "../../lib/api";
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

// Named deterministic checks for a generated UI bundle. These mirror
// `ui_artifact_checks` in packages/evaluation/src/evaluation/suite.py so a
// spreadsheet cell ("ui_bundle", "code+info+pacing+script") expands into the
// same checks the CLI harness uses.
const UI_CHECK_PRESETS: Record<string, Record<string, unknown>> = {
  code: { name: "component_code", kind: "labeled_code_block", labels: ["component code"] },
  info: { name: "component_info", kind: "labeled_contains", labels: ["component info"], terms: ["input", "state", "accessib"] },
  pacing: { name: "pacing", kind: "labeled_regex", labels: ["pacing"], pattern: "\\d+\\s*ms" },
  script: { name: "script", kind: "labeled_code_block", labels: ["script"] },
};

function parseChecksColumn(cell: unknown): Array<Record<string, unknown>> | undefined {
  if (cell == null || String(cell).trim() === "") return undefined;
  if (Array.isArray(cell)) return cell.filter((item): item is Record<string, unknown> => !!item && typeof item === "object");
  const text = String(cell).trim();
  if (text.startsWith("[")) {
    try {
      const parsed = JSON.parse(text);
      if (Array.isArray(parsed)) return parsed.filter((item): item is Record<string, unknown> => !!item && typeof item === "object");
    } catch { /* fall through to preset names */ }
  }
  const shortNames = text === "ui_bundle" || text === "full"
    ? ["code", "info", "pacing", "script"]
    : text.split("+").map((part) => part.trim().toLowerCase()).filter(Boolean);
  if (!shortNames.length) return undefined;
  const checks = shortNames.map((short) => UI_CHECK_PRESETS[short]).filter(Boolean);
  return checks.length ? checks : undefined;
}

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
  const loadingCount = executions.filter((execution) => execution.loading).length;
  const completedCount = executions.length - loadingCount;
  const judgeActive = (run?.judge_judged ?? 0) > 0 || (run?.judge_errors ?? 0) > 0;
  const excludedCount = run?.excluded_count ?? 0;
  return (
    <section className="rounded-xl p-4 space-y-4" style={{ background: "var(--bg-1)", border: "1px solid var(--bg-4)" }}>
      <div className="flex items-start justify-between gap-3"><div><div className="text-[10px] font-semibold uppercase tracking-wide" style={{ color: "var(--accent)" }}>{track} agent</div><h2 className="text-[16px] font-semibold capitalize">{track} evaluation report</h2><p className="text-[11px]" style={{ color: "var(--fg-3)" }}>Latest suite run and its execution transcripts.</p></div><span className="rounded-full px-2 py-1 font-mono text-[10px]" style={{ background: "var(--bg-3)", color: "var(--fg-2)" }}>{run?.status || "no runs"}</span></div>
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-4">
        <Stat label="Pass rate" value={pct(run?.pass_rate ?? null)} sub={`${run?.result_count ?? 0} scored${excludedCount ? ` · ${excludedCount} excluded` : ""}`} />
        <Stat label="Score" value={score(run?.average_score ?? null)} sub="correctness" />
        <Stat label="LLM judge" value={judgeActive ? score(run?.judge_average ?? null) : "-"} sub={judgeActive ? `${run?.judge_judged ?? 0} judged` : run?.status === "running" ? "in progress" : "not enabled"} />
        <Stat label="Latency" value={run?.avg_latency_ms == null ? "-" : `${Math.round(run.avg_latency_ms)} ms`} sub="average" />
        <Stat label="Tokens" value={run?.total_tokens == null ? "-" : run.total_tokens.toLocaleString()} sub="reported usage" />
        <Stat label="Cost" value={run?.total_cost == null ? "-" : `$${run.total_cost.toFixed(4)}`} sub="reported cost" />
        <Stat label="Tools" value={pct(run?.tool_success_rate ?? null)} sub="success rate" />
      </div>
      <div className="border-t pt-3" style={{ borderColor: "var(--bg-4)" }}><div className="mb-2 flex items-center justify-between"><h3 className="text-[11px] font-semibold">Execution chats</h3><span className="font-mono text-[10px]" style={{ color: "var(--fg-3)" }}>{loadingCount > 0 ? `${loadingCount} loading / ` : ""}{completedCount} completed</span></div><div className="space-y-2">{executions.map((execution) => {
        const excluded = !!execution.excluded;
        const label = execution.loading ? "... running" : execution.outcome === "rate_limited" ? "rate limited" : execution.outcome === "provider_unavailable" ? "provider unavailable" : execution.success ? "passed" : "failed";
        const labelColor = execution.loading || excluded ? "var(--warning)" : execution.success ? "var(--success)" : "var(--danger)";
        return <details key={execution.id} open={execution.loading} className="rounded-lg p-3" style={{ background: "var(--bg-0)", border: "1px solid var(--bg-4)" }}><summary className="cursor-pointer list-none flex gap-2 text-[11px]"><span className="font-mono shrink-0" style={{ color: labelColor }}>{label}</span><span className="font-mono shrink-0" style={{ color: "var(--fg-3)" }}>{new Date(execution.created_at).toLocaleString()}</span><span className="truncate flex-1">{execution.case_id}: {execution.goal}</span>{excluded && <span className="font-mono shrink-0" style={{ color: "var(--warning)" }}>not scored</span>}{execution.judge_score != null && <span className="font-mono shrink-0 rounded px-1.5" style={{ color: "white", background: execution.judge_score >= 0.75 ? "var(--success)" : execution.judge_score >= 0.5 ? "var(--warning)" : "var(--danger)" }}>judge {execution.judge_score.toFixed(2)}</span>}</summary><div className="mt-3 space-y-3 text-[12px]"><div><div className="mb-1 text-[10px] uppercase tracking-wide" style={{ color: "var(--fg-3)" }}>Prompt</div><MarkdownText>{execution.goal}</MarkdownText></div><div><div className="mb-1 text-[10px] uppercase tracking-wide" style={{ color: "var(--fg-3)" }}>Response</div><MarkdownText>{execution.loading && !execution.output ? "Agent is executing this test..." : execution.output || execution.error || "No final response recorded."}</MarkdownText></div>{excluded && <div className="rounded-lg p-3 text-[11px]" style={{ color: "var(--warning)", background: "var(--warning-soft)", border: "1px solid var(--warning-soft)" }}>This execution was excluded from evaluation metrics after provider retries were exhausted.</div>}{(execution.judge_score != null || execution.judge_error) && <div className="rounded-lg p-3" style={{ background: "var(--bg-1)", border: "1px solid var(--bg-4)" }}><div className="mb-1 text-[10px] font-semibold uppercase tracking-wide" style={{ color: "var(--fg-3)" }}>LLM judge</div><div className="font-mono text-[11px]" style={{ color: execution.judge_score != null ? "var(--fg-1)" : "var(--danger)" }}>{execution.judge_score != null ? `score ${execution.judge_score.toFixed(2)}` : "not scored"}</div>{execution.judge_comment && <div className="text-[11px] mt-1" style={{ color: "var(--fg-2)" }}>{execution.judge_comment}</div>}{execution.judge_error && <div className="text-[11px] mt-1 font-mono" style={{ color: "var(--danger)" }}>{execution.judge_error}</div>}</div>}</div></details>;
      })}{executions.length === 0 && <div className="py-3 text-[11px]" style={{ color: "var(--fg-3)" }}>No execution chats yet.</div>}</div></div>
    </section>
  );
}

export function EvaluationView() {
  const [data, setData] = useState<EvaluationDashboard | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [degraded, setDegraded] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [tracks, setTracks] = useState<Array<"native" | "langgraph">>(["native", "langgraph"]);
  const [reportTrack, setReportTrack] = useState<"native" | "langgraph">("native");
  const [suiteName, setSuiteName] = useState("desktop-suite");
  const [suiteVersion, setSuiteVersion] = useState("1");
  const [judgeModel, setJudgeModel] = useState("");
  const [judgeProvider, setJudgeProvider] = useState("");
  const [judgeProviders, setJudgeProviders] = useState<string[]>(["ollama", "groq", "openai", "anthropic"]);
  const [judgeDefaultModel, setJudgeDefaultModel] = useState("");
  const [stallNote, setStallNote] = useState<string | null>(null);
  const [runProgress, setRunProgress] = useState<Array<{ id: string; track: string; status?: string; result_count: number; passed_count: number; judge_judged?: number; judge_errors?: number; finished_at: string | null }>>([]);
const [workspace, setWorkspace] = useState(loadEvaluationWorkspace);
  const canBrowse = isTauri();
  // Blocks the run-wait polling loop after an unmount so a navigation away
  // cannot keep issuing dashboard requests (or set state) in the background.
  const mountedRef = useRef(true);
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);
  const [cases, setCases] = useState<Array<{ id: string; goal: string; working_directory?: string; expected_output_contains?: string; checks?: Array<Record<string, unknown>>; metadata?: Record<string, unknown> }>>([
    { id: "greeting", goal: "Reply with a short friendly greeting.", expected_output_contains: "hello" },
    { id: "capabilities", goal: "Summarize what you can do in one sentence." },
    { id: "evidence", goal: "Explain in two sentences why checking evidence matters before claiming a task is complete." },
  ]);

  const loadHealth = () => taskApi.health()
    .then((health) => setDegraded(health.degraded || []))
    .catch(() => undefined);

  const loadDashboard = () => {
    setLoading(true);
    void loadHealth();
    return taskApi.evaluationDashboard()
      .then((next) => { setData(next); setError(null); })
      .catch((cause: Error) => setError(cause.message))
      .finally(() => setLoading(false));
  };

useEffect(() => {
    let cancelled = false;
    void taskApi.health()
      .then((health) => { if (!cancelled) setDegraded(health.degraded || []); })
      .catch(() => undefined);
    taskApi.evaluationDashboard()
      .then((next) => { if (!cancelled) { setData(next); setError(null); } })
      .catch((cause: Error) => { if (!cancelled) setError(cause.message); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, []);

  // Seed the judge picker from the active backend setting (a provider can be
  // selected even before any model name is typed).
  useEffect(() => {
    let cancelled = false;
    judgeApi.getSettings()
      .then((settings) => {
        if (cancelled) return;
        if (settings.providers?.length) setJudgeProviders(settings.providers);
        if (settings.provider) setJudgeProvider(settings.provider);
        if (settings.default_model) setJudgeDefaultModel(settings.default_model);
      })
      .catch(() => undefined);
    return () => { cancelled = true; };
  }, []);

  const runDefaultSuite = async () => {
    setRunning(true);
    setRunProgress([]);
    setStallNote(null);
    setError(null);
    let startFailed = false;
    try {
      const started = await taskApi.startEvaluation({
        name: suiteName.trim() || "desktop-suite",
        version: suiteVersion.trim() || "1",
        tracks,
        judge_model: judgeModel.trim() || undefined,
        judge_provider: judgeProvider.trim() || undefined,
        cases: cases.map((item) => ({
          ...item,
          working_directory: item.working_directory?.trim() || workspace.trim() || ".",
        })),
      });
      // Hold the button until every started run — including any LLM judging
      // that happens after the last agent turn — is finished. The run record
      // is only marked finished after judging completes, so `finished_at` is
      // the single completion signal. Two safety valves: a one-hour hard cap,
      // and stall detection — if no run's results/judge counts move for five
      // minutes (e.g. the API process was restarted and the background runs
      // died silently), stop waiting and say so instead of hanging forever.
      const ids = new Set(started.evaluation_run_ids);
      const deadline = Date.now() + 60 * 60 * 1000;
      const STALL_AFTER_MS = 5 * 60 * 1000;
      let lastSignature = "";
      let lastProgressAt = Date.now();
      while (Date.now() < deadline && mountedRef.current) {
        try {
          const next = await taskApi.evaluationDashboard();
          setData(next);
          setError(null);
          const mine = next.run_breakdown.filter((run) => ids.has(run.id));
          setRunProgress(mine);
          const activeExecutions = next.executions.filter((execution) => ids.has(execution.evaluation_run_id) && execution.loading);
          const signature = mine
            .map((run) => `${run.id}:${run.result_count}:${run.passed_count}:${run.judge_judged ?? 0}`)
            .sort()
            .join("|") + `|active:${activeExecutions.length}`;
          if (signature !== lastSignature) {
            lastSignature = signature;
            lastProgressAt = Date.now();
          }
          const done = [...ids].every((id) =>
            next.run_breakdown.some((run) => run.id === id && !!run.finished_at),
          );
          if (done) break;
          if (!activeExecutions.length && Date.now() - lastProgressAt > STALL_AFTER_MS) {
            setStallNote("No evaluation progress for 5 minutes — the runs may have stopped (an API restart kills in-flight evaluations). Showing the latest data; press Refresh later or run again.");
            break;
          }
        } catch {
          // Transient dashboard failure — keep waiting (stall timer still applies).
          if (Date.now() - lastProgressAt > STALL_AFTER_MS) {
            setStallNote("The dashboard has been unreachable for 5 minutes while waiting for the runs to finish. Showing the latest data.");
            break;
          }
        }
        await new Promise((resolve) => setTimeout(resolve, 2000));
      }
    } catch (cause) {
      startFailed = true;
      setError((cause as Error).message);
} finally {
      if (!mountedRef.current) return;
      // Fetch one final no-store snapshot so completed results and judge
      // verdicts appear immediately after the background task settles.
      if (startFailed) {
        await taskApi.evaluationDashboard().then(setData).catch(() => undefined);
      } else {
        await loadDashboard().catch(() => undefined);
      }
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
        return { id: id ? String(id).trim() : "", goal: goal ? String(goal).trim() : "", working_directory: directory ? String(directory).trim() : "", expected_output_contains: expected ? String(expected).trim() : "", checks: parseChecksColumn(value(item, "checks")), metadata };
      }).filter((item) => item.id && item.goal);
      if (next.length) setCases(next); else setError("Import needs rows with id and goal fields.");
    } catch { setError("Could not parse the file. Use JSON, CSV/TSV, or XLSX with id, goal, working_directory, expected_output_contains, checks, and optional metadata columns."); }
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

        {error && (
          <div className="rounded-xl p-4 text-[12px] font-mono" style={{ color: "var(--danger)", background: "var(--danger-soft)", border: "1px solid var(--danger-soft)" }}>{error}</div>
        )}

        {degraded.length > 0 && (
          <div role="status" className="rounded-xl p-4" style={{ color: "var(--warning)", background: "var(--warning-soft)", border: "1px solid var(--warning-soft)" }}>
            <div className="text-[12px] font-semibold">API running in degraded mode</div>
            <div className="mt-1 space-y-1 text-[11px]" style={{ color: "var(--fg-2)" }}>{degraded.map((message) => <div key={message}>{message}</div>)}</div>
          </div>
        )}

        {stallNote && (
          <div className="rounded-xl p-4" style={{ background: "var(--warning-soft)", border: "1px solid var(--warning-soft)" }}>
            <div className="text-[12px] font-semibold" style={{ color: "var(--warning)" }}>Evaluation wait stopped</div>
            <div className="text-[11px] mt-1" style={{ color: "var(--fg-2)" }}>{stallNote}</div>
          </div>
        )}

        {loading && !data ? (
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">{[0, 1, 2, 3].map((i) => <div key={i} className="rounded-xl skeleton" style={{ height: 92 }} />)}</div>
        ) : data ? (
          <>
            <section className="rounded-xl p-4 space-y-3" style={{ background: "var(--bg-1)", border: "1px solid var(--bg-4)" }}>
              <div className="flex items-center justify-between gap-3"><h2 className="text-[13px] font-semibold">Evaluation suite</h2><label className="text-[11px] cursor-pointer" style={{ color: "var(--accent)" }}>Import JSON/CSV/TSV/XLSX<input type="file" accept=".json,.csv,.tsv,.txt,.xlsx,.xls" className="hidden" onChange={(e) => { const file = e.target.files?.[0]; if (file) void importCases(file); e.currentTarget.value = ""; }} /></label></div>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-2"><input value={suiteName} onChange={(e) => setSuiteName(e.target.value)} placeholder="Suite name" className="field" /><input value={suiteVersion} onChange={(e) => setSuiteVersion(e.target.value)} placeholder="Version" className="field" /><div className="field flex items-center gap-2 p-0 pl-2" title="Optional LLM judge: pick a provider and a model on it. One shared blind judge scores every track on the same provider; results appear beside the deterministic pass rate. The API key is configured in Settings and never sent from the browser."><select value={judgeProvider} onChange={(e) => { const next = e.target.value; setJudgeProvider(next); setJudgeModel(""); setJudgeDefaultModel(""); if (next) { void judgeApi.listModels(next).then(({ default_model }) => setJudgeDefaultModel(default_model || "")).catch(() => undefined); } else { void judgeApi.getSettings().then((settings) => { if (settings.provider) setJudgeProvider(settings.provider); if (settings.default_model) setJudgeDefaultModel(settings.default_model); }).catch(() => undefined); } }} className="w-24 bg-transparent outline-none" aria-label="Judge provider"><option value="">default</option>{judgeProviders.map((provider) => <option key={provider} value={provider}>{provider}</option>)}</select><input value={judgeModel} onChange={(e) => setJudgeModel(e.target.value)} placeholder={judgeDefaultModel ? `Judge model (e.g. ${judgeDefaultModel})` : "Judge model (blank = none)"} className="mono flex-1 bg-transparent outline-none" /></div>{canBrowse ? <button type="button" onClick={chooseWorkspace} className="field flex items-center justify-between gap-2 text-left"><span className="truncate"><span className="mr-2" style={{ color: "var(--accent)" }}>Folder</span>{folderName(workspace)}</span><span className="text-[10px]" style={{ color: "var(--fg-3)" }}>Choose…</span></button> : <input value={workspace} onChange={(e) => { setWorkspace(e.target.value); saveEvaluationWorkspace(e.target.value.trim()); }} placeholder="Evaluation folder path" className="field mono" title="Workspace folder every case runs in (filled into blank working_directory cells)" />}</div>
              <div className="text-[10px]" style={{ color: "var(--fg-3)" }}>{canBrowse ? "Choose the folder once; both agents run every case in that same workspace." : "Type a folder path; both agents run every case in that same workspace. Folder picking requires the Tauri desktop app."} Judge: pick a provider and model to have one shared, track-blind LLM judge score every case. The judge provider may differ from the agents'; its scores are reported separately from the deterministic pass rate.</div>
              <div className="text-[10px]" style={{ color: "var(--fg-3)" }}>Both agents receive every case below. Sheet columns: id, goal, working_directory (optional), expected_output_contains, checks (optional — "ui_bundle", "code+info+pacing+script", or check JSON), metadata (optional JSON). Blank directories use the selected evaluation folder.</div>
              <div className="space-y-2">{cases.map((item, index) => <div key={`${item.id}-${index}`} className="grid grid-cols-[110px_1fr_180px_auto] gap-2"><input value={item.id} onChange={(e) => setCases((all) => all.map((row, i) => i === index ? { ...row, id: e.target.value } : row))} placeholder="case id" className="field mono" /><input value={item.goal} onChange={(e) => setCases((all) => all.map((row, i) => i === index ? { ...row, goal: e.target.value } : row))} placeholder="Prompt / goal" className="field" /><input value={item.expected_output_contains || ""} onChange={(e) => setCases((all) => all.map((row, i) => i === index ? { ...row, expected_output_contains: e.target.value } : row))} placeholder={item.checks?.length ? `${item.checks.length} checks attached` : "Expected text"} className="field" /><button type="button" onClick={() => setCases((all) => all.filter((_, i) => i !== index))} className="btn-quiet px-2" aria-label="Remove case">x</button></div>)}</div>
              <button type="button" onClick={() => setCases((all) => [...all, { id: `case-${all.length + 1}`, goal: "", expected_output_contains: "" }])} className="btn-quiet h-7 px-2 text-[11px]">+ Add case</button>
              {running && runProgress.length > 0 && (
                <div className="rounded-lg p-3 space-y-1.5 font-mono text-[10px]" style={{ background: "var(--bg-0)", border: "1px solid var(--bg-4)" }}>
                  <div className="uppercase tracking-wide" style={{ color: "var(--fg-3)" }}>Run progress — tracks run sequentially; judging finishes before a run is marked done</div>
                  {runProgress.map((run) => {
                    const judging = !!judgeModel.trim();
                    const status = run.finished_at ? "done" : run.status || "running";
                    return (
                      <div key={run.id} className="flex items-center justify-between gap-2">
                        <span className="capitalize" style={{ color: status === "done" ? "var(--success)" : "var(--fg-2)" }}>{run.track}: {status}</span>
                        <span style={{ color: "var(--fg-3)" }}>
                          {run.passed_count}/{run.result_count} cases
                          {judging ? ` · judge ${run.judge_judged ?? 0}${(run.judge_errors ?? 0) > 0 ? ` (${run.judge_errors} err)` : ""}` : ""}
                        </span>
                      </div>
                    );
                  })}
                </div>
              )}
            </section>
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
              {(() => {
                const run = data.comparison.find((candidate) => candidate.track === reportTrack);
                return <AgentReport track={reportTrack} run={run} executions={data.executions.filter((execution) => execution.track === reportTrack && (!run || execution.evaluation_run_id === run.id))} />;
              })()}
            </section>

            <div className="text-[11px]" style={{ color: "var(--fg-3)" }}>Database metrics come from evaluation tables and derived run data. Langfuse tracing is {data.sources.langfuse ? "enabled for trace enrichment" : "not configured"}; trace-level observations and model-judge scores require an ingestion path into evaluation_scores.</div>
          </>
        ) : null}
      </div>
    </div>
  );
}
