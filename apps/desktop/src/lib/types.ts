// Mirrors packages/api/src/api/native/schemas.py + packages/api/src/api/schemas.py

export type AgentTrack = "native" | "langgraph";

export interface SessionResponse {
  id: string;
  agent: string;
  title: string;
  workspace: string;
  created_at: string | null;
  updated_at: string | null;
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

export interface SandboxStatusResponse {
  available: boolean;
  image: string;
  status: string;
  reason: string;
  containers?: SandboxContainerResponse[];
}

export interface SandboxContainerResponse {
  session_id: string;
  workspace: string;
  container_id: string;
  image: string;
  status: string;
}

export interface EnvVariableResponse {
  name: string;
  value: string | null;
  secret: boolean;
  set: boolean;
}

export interface EnvironmentResponse {
  variables: EnvVariableResponse[];
}

// LangGraph / Task API
export interface HealthResponse {
  status: string;
  repository: string;
  tracks: string[];
}

export interface EvaluationMetricSummary {
  metric: string;
  average: number | null;
  count: number;
}

export interface EvaluationRunSummary {
  id: string;
  suite: string;
  track: AgentTrack;
  started_at: string;
  finished_at: string | null;
  result_count: number;
  passed_count: number;
  pass_rate: number | null;
  average_score: number | null;
  avg_latency_ms?: number | null;
  total_tokens?: number | null;
  total_cost?: number | null;
  tool_success_rate?: number | null;
  status?: string;
}

export interface EvaluationExecution {
  id: string;
  evaluation_run_id: string;
  agent_run_id: string;
  task_id: string;
  thread_id: string;
  suite: string;
  track: AgentTrack;
  case_id: string;
  goal: string;
  workspace: string;
  status: string;
  output: string | null;
  error: string | null;
  success: boolean;
  created_at: string;
}

export interface EvaluationDashboard {
  available: boolean;
  reason: string | null;
  suites: number;
  cases: number;
  runs: number;
  results: number;
  pass_rate: number | null;
  average_score: number | null;
  avg_latency_ms: number | null;
  total_tokens: number | null;
  total_cost: number | null;
  tool_success_rate: number | null;
  metrics: EvaluationMetricSummary[];
  run_breakdown: EvaluationRunSummary[];
  comparison: Array<EvaluationRunSummary & { avg_latency_ms?: number | null; total_tokens?: number | null; total_cost?: number | null; tool_success_rate?: number | null; status?: string }>;
  executions: EvaluationExecution[];
  sources: { database: boolean; langfuse: boolean };
}

export interface StartEvaluationResponse {
  evaluation_run_ids: string[];
  status: string;
}

export interface TaskResponse {
  id: string;
  goal: string;
  thread_id: string;
  workspace: string | null;
  track: AgentTrack;
  status: string | null;
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
