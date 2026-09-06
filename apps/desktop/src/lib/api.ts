import type {
  ApprovalResponse,
  CreateSessionRequest,
  CreateTaskRequest,
  EventResponse,
  HealthResponse,
  NativeHealthResponse,
  PermissionResponse,
  RunResponse,
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
  getTask: (taskId: string) => req<TaskResponse>(`/tasks/${encodeURIComponent(taskId)}`),
  resumeTask: (taskId: string, body: { resume_value?: unknown; checkpoint_id?: string }) =>
    req<TaskResponse>(`/tasks/${encodeURIComponent(taskId)}/resume`, { method: "POST", body: JSON.stringify(body) }),

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
  getThreadTask: (threadId: string, taskId: string) =>
    req<TaskResponse>(`/threads/${encodeURIComponent(threadId)}/tasks/${encodeURIComponent(taskId)}`),
  resumeThreadTask: (threadId: string, taskId: string, body: { resume_value?: unknown; checkpoint_id?: string }) =>
    req<TaskResponse>(`/threads/${encodeURIComponent(threadId)}/tasks/${encodeURIComponent(taskId)}/resume`, {
      method: "POST",
      body: JSON.stringify(body),
    }),
  listThreadEvents: (threadId: string) => req<ThreadEventResponse[]>(`/threads/${encodeURIComponent(threadId)}/events`),

  streamEventsUrl: (taskId: string) => `${apiBase()}/tasks/${encodeURIComponent(taskId)}/events`,
  wsStreamUrl: (taskId: string) => `${apiBase().replace(/^http/, "ws")}/ws/tasks/${encodeURIComponent(taskId)}`,

  listApprovals: () => req<ApprovalResponse[]>("/approvals"),
  getApproval: (id: string) => req<ApprovalResponse>(`/approvals/${encodeURIComponent(id)}`),
  resolveApproval: (id: string, body: { approved: boolean; note?: string }) =>
    req<{ id: string; approved: boolean }>(`/approvals/${encodeURIComponent(id)}/resolve`, {
      method: "POST",
      body: JSON.stringify(body),
    }),
};

export function sseSubscribe(url: string, onEvent: (ev: { id: string; event: string; data: string }) => void, onError?: (e: Event) => void) {
  const es = new EventSource(url);
  es.onmessage = (e) => onEvent({ id: (e as MessageEvent).lastEventId || "", event: "message", data: (e as MessageEvent).data });
  // also listen for named events (EventSource dispatches by event type)
  const handler = (e: MessageEvent) => onEvent({ id: e.lastEventId || "", event: (e as unknown as { type: string }).type || "message", data: e.data });
  // generic: native events use typed events; we capture all via addEventListener with wildcard workaround: listen for common types
  for (const t of ["message", "run_finished", "error", "tool_call", "text_delta", "run_receipt", "event"]) {
    try {
      es.addEventListener(t, handler as EventListener);
    } catch {
      // ignore
    }
  }
  es.onerror = (e) => onError?.(e as Event);
  return () => es.close();
}

export { DEFAULT_API_BASE };
