import type {
  ApprovalResponse,
  CreateSessionRequest,
  CreateTaskRequest,
  EventResponse,
  HealthResponse,
  NativeHealthResponse,
  PermissionResponse,
  RunResponse,
  SandboxStatusResponse,
  SessionResponse,
  SessionWithRunsResponse,
  TaskResponse,
  ThreadEventResponse,
  ThreadResponse,
} from "./types";

const DEFAULT_API_BASE = (import.meta.env.VITE_API_URL as string) || "http://127.0.0.1:8000";

function apiBase() {
  try {
    return JSON.parse(localStorage.getItem("operating-agent:settings") || "{}").apiUrl || DEFAULT_API_BASE;
  } catch {
    return DEFAULT_API_BASE;
  }
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${apiBase()}${path}`, {
      headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
      ...init,
    });
  } catch {
    throw new Error(`API unavailable at ${apiBase()}. Start the desktop app with Tauri or run 'uv run --package api api'.`);
  }
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}${text ? `: ${text.slice(0, 400)}` : ""}`);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

// ——— Native ———
export const nativeApi = {
  health: () => req<NativeHealthResponse>("/native/health"),

  listSessions: (params?: { workspace?: string; limit?: number; offset?: number }) => {
    const q = new URLSearchParams();
    if (params?.workspace) q.set("workspace", params.workspace);
    q.set("limit", String(params?.limit ?? 100));
    q.set("offset", String(params?.offset ?? 0));
    return req<SessionResponse[]>(`/native/sessions?${q.toString()}`);
  },
  createSession: (body: CreateSessionRequest) =>
    req<SessionResponse>("/native/sessions", { method: "POST", body: JSON.stringify(body) }),
  getSession: (id: string) => req<SessionWithRunsResponse>(`/native/sessions/${encodeURIComponent(id)}`),
  deleteSession: (id: string) => req<void>(`/native/sessions/${encodeURIComponent(id)}`, { method: "DELETE" }),
  forkSession: (id: string, title?: string) =>
    req<SessionResponse>(`/native/sessions/${encodeURIComponent(id)}/fork`, {
      method: "POST",
      body: JSON.stringify({ title: title || "" }),
    }),
  getConversation: (id: string) =>
    req<{ session_id: string; messages: Array<{ id: string; role: string; parts: unknown[]; model: string; created_at: string | null }> }>(
      `/native/sessions/${encodeURIComponent(id)}/conversation`,
    ),

  // SSE: caller must handle EventSourceResponse streaming
  sendMessageUrl: (sessionId: string) => `${apiBase()}/native/sessions/${encodeURIComponent(sessionId)}/messages`,
  resumeRun: (sessionId: string, limits?: Record<string, unknown>) =>
    req<RunResponse>(`/native/sessions/${encodeURIComponent(sessionId)}/resume`, {
      method: "POST",
      body: JSON.stringify({ limits: limits || null }),
    }),
  cancelRun: (sessionId: string) =>
    req<{ session_id: string; cancelled: boolean; reason?: string }>(`/native/sessions/${encodeURIComponent(sessionId)}/cancel`, {
      method: "POST",
    }),

  getEvents: (sessionId: string, from = 0) =>
    req<EventResponse[]>(`/native/sessions/${encodeURIComponent(sessionId)}/events?from=${from}`),
  getEventsSSEUrl: (sessionId: string, from = 0, stream = 1) =>
    `${apiBase()}/native/sessions/${encodeURIComponent(sessionId)}/events?from=${from}&stream=${stream}`,

  getSandbox: () => req<SandboxStatusResponse>("/native/sandbox"),

  listRuns: (sessionId: string) => req<RunResponse[]>(`/native/sessions/${encodeURIComponent(sessionId)}/runs`),
  getRun: (runId: string) => req<RunResponse>(`/native/runs/${encodeURIComponent(runId)}`),
  getSettings: () => req<Record<string, unknown>>("/native/settings"),
  updateSettings: (body: Record<string, unknown>) =>
    req<Record<string, unknown>>("/native/settings", { method: "PATCH", body: JSON.stringify(body) }),
  listModels: (provider: string, baseUrl?: string) => {
    const q = new URLSearchParams({ provider });
    if (baseUrl?.trim()) q.set("base_url", baseUrl.trim());
    return req<{ provider: string; models: string[]; default_model: string }>(`/native/settings/models?${q}`);
  },

  listPermissions: (sessionId?: string) => {
    const q = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : "";
    return req<PermissionResponse[]>(`/native/permissions${q}`);
  },
  getPermission: (callId: string) => req<PermissionResponse>(`/native/permissions/${encodeURIComponent(callId)}`),
  resolvePermission: (callId: string, body: { allowed: boolean; duration?: string; scope?: string }) =>
    req<{ call_id: string; allowed: boolean; duration: string; scope: string }>(
      `/native/permissions/${encodeURIComponent(callId)}`,
      { method: "POST", body: JSON.stringify(body) },
    ),
};

// ——— Task / LangGraph ———
export const taskApi = {
  health: () => req<HealthResponse>("/health"),

  createTask: (body: CreateTaskRequest) =>
    req<TaskResponse>("/tasks", { method: "POST", body: JSON.stringify(body) }),
  getTask: (threadId: string, taskId: string) =>
    req<TaskResponse>(`/threads/${encodeURIComponent(threadId)}/tasks/${encodeURIComponent(taskId)}`),
  resumeTask: (threadId: string, taskId: string, body: { resume_value?: unknown; checkpoint_id?: string }) =>
    req<TaskResponse>(`/threads/${encodeURIComponent(threadId)}/tasks/${encodeURIComponent(taskId)}/resume`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  createThreadTask: (threadId: string, body: CreateTaskRequest) =>
    req<TaskResponse>(`/threads/${encodeURIComponent(threadId)}/tasks`, { method: "POST", body: JSON.stringify(body) }),
  createThread: (title?: string) =>
    req<ThreadResponse>("/threads", { method: "POST", body: JSON.stringify({ title: title || null }) }),
  deleteThread: (threadId: string) =>
    req<void>(`/threads/${encodeURIComponent(threadId)}`, { method: "DELETE" }),
  getSettings: () => req<Record<string, unknown>>("/settings/langgraph"),
  updateSettings: (body: Record<string, unknown>) =>
    req<Record<string, unknown>>("/settings/langgraph", { method: "PATCH", body: JSON.stringify(body) }),
  listModels: (provider: string, baseUrl?: string) => {
    const q = new URLSearchParams({ provider });
    if (baseUrl?.trim()) q.set("base_url", baseUrl.trim());
    return req<{ provider: string; models: string[]; default_model: string }>(`/settings/langgraph/models?${q}`);
  },

  listThreads: (params?: { limit?: number; offset?: number }) => {
    const q = new URLSearchParams();
    q.set("limit", String(params?.limit ?? 100));
    q.set("offset", String(params?.offset ?? 0));
    return req<ThreadResponse[]>(`/threads?${q.toString()}`);
  },
  listThreadTasks: (threadId: string, params?: { limit?: number; offset?: number }) => {
    const q = new URLSearchParams();
    q.set("limit", String(params?.limit ?? 100));
    q.set("offset", String(params?.offset ?? 0));
    return req<TaskResponse[]>(`/threads/${encodeURIComponent(threadId)}/tasks?${q.toString()}`);
  },
  listThreadEvents: (threadId: string) => req<ThreadEventResponse[]>(`/threads/${encodeURIComponent(threadId)}/events`),

  streamEventsUrl: (threadId: string, taskId: string) =>
    `${apiBase()}/threads/${encodeURIComponent(threadId)}/tasks/${encodeURIComponent(taskId)}/events`,
  wsStreamUrl: (threadId: string, taskId: string) =>
    `${apiBase().replace(/^http/, "ws")}/ws/threads/${encodeURIComponent(threadId)}/tasks/${encodeURIComponent(taskId)}`,

  listApprovals: () => req<ApprovalResponse[]>("/approvals"),
  getApproval: (id: string) => req<ApprovalResponse>(`/approvals/${encodeURIComponent(id)}`),
  resolveApproval: (id: string, body: { approved: boolean; note?: string }) =>
    req<{ id: string; approved: boolean }>(`/approvals/${encodeURIComponent(id)}/resolve`, {
      method: "POST",
      body: JSON.stringify(body),
    }),
};

export interface SSEMessage {
  id: string;
  event: string;
  data: string;
}

const SSE_EVENT_TYPES = [
  "state",
  "finished",
  "error",
  "assistant_delta",
  "reasoning_delta",
  "message_added",
  "tool_started",
  "tool_finished",
  "permission_requested",
  "permission_resolved",
  "turn_started",
  "run_finished",
  "run_receipt",
  "model_fallback",
  // Persisted task-event names (see TaskService._run.on_event): subscribed so a
  // contracted event is never silently dropped by EventSource routing.
  "llm_call",
  "tool_call",
  "phase_entered",
  "phase_exited",
  "plan_created",
  "finding_recorded",
  "verification_recorded",
  "trace_ref",
  "approval_requested",
  "approval_resolved",
] as const;

// ——— Activity (shared by ChatWorkspace + EventTimeline) ———
// Token-level deltas drive the streaming bubble, not the timeline: one row per
// chunk would flood Activity with hundreds of fragments for a single answer.

/** Event types that never get an Activity row. */
export const ACTIVITY_HIDDEN_TYPES: ReadonlySet<string> = new Set([
  "assistant_delta",
  "reasoning_delta",
]);

/** Whether an event of this type belongs in the Activity timeline. */
export function isActivityEvent(type: string): boolean {
  return !ACTIVITY_HIDDEN_TYPES.has(type);
}

const ACTIVITY_SUMMARY_KEYS = [
  "final_message",
  "final_text",
  "output",
  "text",
  "goal",
  "role",
  "tool_name",
  "tool",
  "preview",
  "name",
  "error",
  "reason",
  "status",
] as const;

/** One-line summary for an Activity row. Never throws on odd payloads. */
export function summarizeEventData(type: string, data: unknown): string {
  if (!data || typeof data !== "object") return "";
  const record = data as Record<string, unknown>;
  for (const key of ACTIVITY_SUMMARY_KEYS) {
    const value = record[key];
    if (typeof value === "string" && value.trim()) {
      const text = value.trim();
      return text.length > 90 ? `${text.slice(0, 90)}…` : text;
    }
  }
  if (typeof record.turn === "number") return `turn ${record.turn}`;
  return "";
}

/** Stable signature for one activity row, for keying across refresh + live. */
export function activitySignature(type: string, data: unknown): string {
  let rendered: string;
  try {
    rendered = JSON.stringify(data) ?? "null";
  } catch {
    rendered = String(data);
  }
  return `${type}|${rendered}`;
}

function parseSSEFrame(frame: string): SSEMessage | null {
  let id = "";
  let event = "message";
  const data: string[] = [];
  for (const rawLine of frame.split(/\r?\n/)) {
    if (!rawLine || rawLine.startsWith(":")) continue;
    const separator = rawLine.indexOf(":");
    const field = separator === -1 ? rawLine : rawLine.slice(0, separator);
    let value = separator === -1 ? "" : rawLine.slice(separator + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "id") id = value;
    else if (field === "event") event = value || "message";
    else if (field === "data") data.push(value);
  }
  return data.length ? { id, event, data: data.join("\n") } : null;
}

export async function readSSEStream(
  stream: ReadableStream<Uint8Array>,
  onEvent: (event: SSEMessage) => void,
): Promise<void> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const drain = (flush: boolean) => {
    while (true) {
      const boundary = /\r?\n\r?\n/.exec(buffer);
      if (!boundary) break;
      const frame = buffer.slice(0, boundary.index);
      buffer = buffer.slice(boundary.index + boundary[0].length);
      const parsed = parseSSEFrame(frame);
      if (parsed) onEvent(parsed);
    }
    if (flush && buffer.trim()) {
      const parsed = parseSSEFrame(buffer);
      if (parsed) onEvent(parsed);
      buffer = "";
    }
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    drain(false);
  }
  buffer += decoder.decode();
  drain(true);
}

export function sseSubscribe(url: string, onEvent: (ev: SSEMessage) => void, onError?: (e: Event) => void) {
  const es = new EventSource(url);
  es.onmessage = (e) => onEvent({ id: e.lastEventId || "", event: "message", data: e.data });
  const handler = (event: Event) => {
    const e = event as MessageEvent;
    // EventSource also uses the `error` type for transport failures. Those
    // events have no data and are handled by `onerror` below.
    if (typeof e.data !== "string") return;
    onEvent({ id: e.lastEventId || "", event: e.type || "message", data: e.data });
  };
  for (const type of SSE_EVENT_TYPES) es.addEventListener(type, handler as EventListener);
  es.onerror = (e) => onError?.(e as Event);
  return () => es.close();
}

export { DEFAULT_API_BASE };
