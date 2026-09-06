// Mirrors packages/api/src/api/native/schemas.py + packages/api/src/api/schemas.py

export type AgentTrack = "native" | "langgraph";

export interface SessionResponse {
  id: string;
  agent: string;
  title: string;
  workspace: string;
}

export interface RunResponse {
  run_id: string;
  session_id: string;
  status: string;
  turns: number;
  final_text: string;
  final_message: string;
  error: string;
  duration_seconds: number;
  cost_usd: number;
  model: string;
  retries: number;
  fallbacks: number;
  stop_reason: string;
  input_tokens: number;
  output_tokens: number;
  cached_tokens: number;
  reasoning_tokens: number;
  trace_id: string;
  trace_url: string;
}

export interface SessionWithRunsResponse extends SessionResponse {
  runs: RunResponse[];
  message_count: number;
}

export interface EventResponse {
  sequence: number;
  type: string;
  session_id: string;
  run_id: string;
  data: Record<string, unknown>;
  time: string | null;
}

export interface PermissionResponse {
  call_id: string;
  tool: string;
  arguments: Record<string, unknown>;
  preview: string;
  reason: string;
}

export interface NativeHealthResponse {
  status: string;
  database: string;
  agents: string[];
  models: string[];
  langfuse_enabled: boolean;
}

// LangGraph / Task API
export interface HealthResponse {
  status: string;
  repository: string;
  tracks: string[];
}

export interface TaskResponse {
  id: string;
  goal: string;
  thread_id: string;
  workspace: string | null;
  track: AgentTrack;
  status: string | null;
  output: string | null;
  final_message: string | null;
  error: string | null;
  run_id: string | null;
  trace_id: string | null;
  metadata: Record<string, unknown>;
  created_at: string;
}

export interface ThreadResponse {
  id: string;
  title: string | null;
  task_count: number;
  created_at: string;
  updated_at: string;
}

export interface ThreadEventResponse {
  task_id: string;
  type: string;
  payload: Record<string, unknown>;
}

export interface ApprovalResponse {
  id: string;
  task_id: string;
  tool_name: string;
  arguments: Record<string, unknown>;
  risk_level: string;
}

export interface CreateSessionRequest {
  agent?: string;
  title?: string;
  workspace?: string;
  working_directory?: string;
}

export interface SendMessageRequest {
  message?: string;
  text?: string;
  media?: Array<{ data?: string; data_base64?: string; mime_type?: string; mimeType?: string; detail?: string }>;
  limits?: LimitsRequest;
}

export interface LimitsRequest {
  max_turns?: number;
  wall_clock_seconds?: number;
  max_cost_usd?: number;
  max_total_tokens?: number;
  max_retries?: number;
  max_parallel_tools?: number;
  helper_max_turns?: number;
  reasoning_effort?: string;
  plan_mode?: boolean;
}

export interface CreateTaskRequest {
  goal: string;
  track?: AgentTrack;
  workspace?: string;
  metadata?: Record<string, unknown>;
}

export interface ConversationMessage {
  id: string;
  role: string;
  parts: Array<Record<string, unknown>>;
  model: string;
  created_at: string | null;
}
