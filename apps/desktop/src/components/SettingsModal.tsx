import { useEffect, useState } from "react";
import type { ReactNode } from "react";
import { nativeApi, taskApi } from "../lib/api";
import type { SandboxStatusResponse } from "../lib/types";

export interface DesktopSettings {
  apiUrl: string;
  provider: "ollama" | "groq" | "openai" | "anthropic";
  model: string;
  baseUrl: string;
  temperature: string;
  topP: string;
  maxTokens: string;
  workspace: string;
  maxTurns: string;
  maxCost: string;
  autoApproveAll: boolean;
}

export const SETTINGS_KEY = "operating-agent:settings";

export const DEFAULT_SETTINGS: DesktopSettings = {
  apiUrl: "http://127.0.0.1:8000",
  provider: "groq",
  model: "llama-3.3-70b-versatile",
  baseUrl: "",
  temperature: "0",
  topP: "1",
  maxTokens: "",
  workspace: ".",
  maxTurns: "10",
  maxCost: "0.05",
  autoApproveAll: false,
};

export function loadSettings(): DesktopSettings {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    return raw ? { ...DEFAULT_SETTINGS, ...JSON.parse(raw) } : DEFAULT_SETTINGS;
  } catch {
    return DEFAULT_SETTINGS;
  }
}

export function saveSettings(settings: DesktopSettings) {
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
  window.dispatchEvent(new CustomEvent("operating-agent:settings", { detail: settings }));
}

export function SettingsModal({
  track,
  onClose,
  onSaved,
}: {
  track: "native" | "langgraph";
  onClose: () => void;
  onSaved: (settings: DesktopSettings) => void;
}) {
  const [settings, setSettings] = useState<DesktopSettings>(loadSettings);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState("");
  const [models, setModels] = useState<string[]>([]);
  const [defaultModel, setDefaultModel] = useState("");
  const [modelsLoading, setModelsLoading] = useState(false);
  const [modelsError, setModelsError] = useState("");
  const [sandbox, setSandbox] = useState<SandboxStatusResponse | null>(null);

  // The native track only serves ollama/groq: a provider persisted from the
  // LangGraph track (openai/anthropic) is shown and saved as the native
  // default instead of leaking an unsupported value into the native API.
  const provider =
    track === "native" && (settings.provider === "openai" || settings.provider === "anthropic")
      ? "groq"
      : settings.provider;

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => event.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      setModelsLoading(true);
      setModelsError("");
      try {
        const api = track === "native" ? nativeApi : taskApi;
        const current = (await api.getSettings()) as Record<string, unknown>;
        if (cancelled) return;
        setSettings((prev) => ({
          ...prev,
          provider: typeof current.provider === "string" && current.provider
            ? current.provider as DesktopSettings["provider"]
            : prev.provider,
          model: typeof current.model === "string" ? current.model : prev.model,
          baseUrl: typeof current.base_url === "string" ? current.base_url : "",
          temperature: current.temperature == null ? prev.temperature : String(current.temperature),
          topP: current.top_p == null ? prev.topP : String(current.top_p),
          maxTokens: current.max_tokens == null ? "" : String(current.max_tokens),
          autoApproveAll: typeof current.auto_approve_all === "boolean" ? current.auto_approve_all : prev.autoApproveAll,
        }));
        if (Array.isArray(current.models)) {
          setModels((current.models as unknown[]).map(String));
        }
        if (typeof current.default_model === "string") {
          setDefaultModel(current.default_model);
        }
      } catch (cause) {
        if (!cancelled) setModelsError((cause as Error).message);
      } finally {
        if (!cancelled) setModelsLoading(false);
      }
    };
    load();
    return () => {
      cancelled = true;
    };
  }, [track]);

  useEffect(() => {
    let cancelled = false;
    const timer = window.setTimeout(async () => {
      setModelsLoading(true);
      setModelsError("");
      try {
        const api = track === "native" ? nativeApi : taskApi;
        try {
          const data = await api.listModels(provider, settings.baseUrl);
          if (cancelled) return;
          setModels(Array.isArray(data.models) ? data.models.map(String) : []);
          setDefaultModel(typeof data.default_model === "string" ? data.default_model : "");
        } catch (listCause) {
          // Older API without the /models routes: fall back to the model list
          // on GET settings so the picker still works after an API restart is
          // pending. Anything else is a real failure.
          if (!String((listCause as Error).message || "").startsWith("404")) throw listCause;
          const current = (await api.getSettings()) as Record<string, unknown>;
          if (cancelled) return;
          setModels(Array.isArray(current.models) ? (current.models as unknown[]).map(String) : []);
          if (typeof current.default_model === "string") setDefaultModel(current.default_model);
          setModelsError("Model list is from a stale API — restart the API for live discovery.");
        }
      } catch (cause) {
        if (!cancelled) {
          setModels([]);
          setModelsError((cause as Error).message);
        }
      } finally {
        if (!cancelled) setModelsLoading(false);
      }
    }, 350);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [track, provider, settings.baseUrl]);

  // Live Docker probe while the modal is open: starting Docker on the machine
  // flips this to connected on the next poll with no restart, and new shell
  // commands then run in containers automatically.
  useEffect(() => {
    if (track !== "native") return;
    let cancelled = false;
    const load = async () => {
      try {
        const status = await nativeApi.getSandbox();
        if (!cancelled) setSandbox(status);
      } catch {
        if (!cancelled) setSandbox(null);
      }
    };
    void load();
    const timer = window.setInterval(load, 4000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [track]);

  const set = <K extends keyof DesktopSettings>(key: K, value: DesktopSettings[K]) =>
    setSettings((current) => ({ ...current, [key]: value }));

  const handleSave = async () => {
    setError("");
    const previous = loadSettings();
    const normalized = { ...settings, provider };
    setSettings(normalized);
    saveSettings(normalized);
    try {
      const body = {
        provider: normalized.provider,
        model: settings.model,
        base_url: settings.baseUrl || null,
        temperature: Number(settings.temperature || 0),
        top_p: Number(settings.topP || 1),
        max_tokens: settings.maxTokens ? Number(settings.maxTokens) : null,
        timeout_seconds: 60,
        auto_approve_all: settings.autoApproveAll,
      };
      if (track === "native") await nativeApi.updateSettings(body);
      else await taskApi.updateSettings(body);
      onSaved(settings);
      setSaved(true);
      window.setTimeout(onClose, 700);
    } catch (cause) {
      saveSettings(previous);
      setError((cause as Error).message);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 anim-fade-in" style={{ background: "rgba(2,6,14,0.7)", backdropFilter: "blur(6px)" }} onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="w-full max-w-[680px] max-h-[90vh] overflow-auto rounded-2xl anim-scale-in" style={{ background: "var(--bg-1)", border: "1px solid var(--accent-ring)", boxShadow: "0 24px 64px rgba(2,8,20,0.7), var(--accent-glow)" }}>
        <div className="px-5 py-4 flex items-start gap-3 border-b titlebar-glow" style={{ borderColor: "var(--bg-4)" }}>
          <div>
            <div className="text-[15px] font-semibold">Settings <span className="ml-1 text-[10px] font-mono font-medium px-1.5 py-0.5 rounded" style={{ background: "var(--accent-soft)", border: "1px solid var(--accent-ring)", color: "var(--accent)" }}>{track}</span></div>
            <div className="text-[11px] mt-1" style={{ color: "var(--fg-2)" }}>Configure the API connection and default agent runtime.</div>
          </div>
          <button onClick={onClose} className="btn-quiet ml-auto w-7 h-7 rounded-lg" style={{ background: "var(--bg-2)", border: "1px solid var(--bg-4)", color: "var(--fg-2)" }}>×</button>
        </div>

        <div className="p-5 space-y-5">
          <section className="space-y-3">
            <SectionTitle>Connection</SectionTitle>
            <Field label="API URL" hint="The FastAPI sidecar address">
              <input value={settings.apiUrl} onChange={(e) => set("apiUrl", e.target.value)} placeholder="http://127.0.0.1:8000" className="field" />
            </Field>
            <div className="grid sm:grid-cols-2 gap-3">
              <Field label="Workspace" hint="Must be an existing directory">
                <input value={settings.workspace} onChange={(e) => set("workspace", e.target.value)} placeholder="." className="field mono" />
              </Field>
              <Field label="Terminal isolation">
                {track === "native" ? (
                  <div className="min-h-9 px-3 py-2 rounded-lg text-[12px] space-y-1" style={{ background: "var(--bg-2)", border: "1px solid var(--bg-4)", color: "var(--fg-2)" }}>
                    <span
                      className={`inline-flex items-center gap-1.5 text-[11px] font-semibold px-2 py-0.5 rounded-full ${sandbox?.available ? "" : "anim-pulse-dot"}`}
                      style={
                        sandbox?.available
                          ? { background: "var(--success-soft)", border: "1px solid rgba(34,197,94,0.35)", color: "var(--success)" }
                          : { background: "var(--warning-soft)", border: "1px solid rgba(245,158,11,0.4)", color: "var(--warning)" }
                      }
                    >
                      {sandbox ? (sandbox.available ? "● Docker connected" : "○ Docker disconnected") : "○ Checking Docker…"}
                    </span>
                    <div className="font-mono text-[11px]">
                      {sandbox ? (sandbox.image || "no image") : "probing…"}
                    </div>
                    {sandbox && !sandbox.available && sandbox.reason && (
                      <div className="text-[11px]">{sandbox.reason}</div>
                    )}
                    <div className="text-[10px]" style={{ color: "var(--fg-3)" }}>
                      Workspace mounted read/write at /workspace. Start Docker anytime — new shell commands connect automatically.
                    </div>
                  </div>
                ) : (
                  <div className="min-h-9 px-3 py-2 rounded-lg text-[12px]" style={{ background: "var(--bg-2)", border: "1px solid var(--bg-4)", color: "var(--fg-2)" }}>
                    Docker required; workspace mounted read/write at /workspace.
                  </div>
                )}
              </Field>
            </div>
          </section>

          <section className="space-y-3">
            <SectionTitle>Model Provider</SectionTitle>
            <div className="grid sm:grid-cols-2 gap-3">
              <Field label="Provider">
                <select value={provider} onChange={(e) => setSettings((current) => ({ ...current, provider: e.target.value as DesktopSettings["provider"], model: "", baseUrl: "" }))} className="field">
                  <option value="ollama">Ollama · local</option>
                  <option value="groq">Groq · cloud</option>
                  {track === "langgraph" && <option value="openai">OpenAI · cloud</option>}
                  {track === "langgraph" && <option value="anthropic">Anthropic · cloud</option>}
                </select>
              </Field>
              <Field label="Model" hint={defaultModel ? `Empty uses default: ${defaultModel}` : "Empty uses provider default"}>
                <input
                  value={settings.model}
                  onChange={(e) => set("model", e.target.value)}
                  placeholder={defaultModel ? `Default: ${defaultModel}` : "Default model"}
                  className="field mono"
                />
              </Field>
            </div>
            <Field label="Base URL" hint="Optional. For Ollama use http://localhost:11434">
              <input value={settings.baseUrl} onChange={(e) => set("baseUrl", e.target.value)} placeholder="Provider default" className="field mono" />
            </Field>
            <div className="rounded-lg px-3 py-2 space-y-2" style={{ background: "var(--bg-2)", border: "1px solid var(--bg-4)" }}>
              <div className="flex items-center gap-2 text-[11px] font-medium" style={{ color: "var(--fg-1)" }}>
                <span>{provider === "ollama" ? `Downloaded Ollama models (${models.length})` : `Known ${provider} models (${models.length})`}</span>
                {modelsLoading && <span className="font-normal" style={{ color: "var(--fg-3)" }}>· loading…</span>}
                {settings.model.trim() && (
                  <button
                    onClick={() => set("model", "")}
                    className="ml-auto h-6 px-2 rounded-md text-[10px]"
                    style={{ background: "var(--bg-1)", border: "1px solid var(--bg-4)", color: "var(--fg-2)" }}
                  >
                    Clear · use default
                  </button>
                )}
              </div>
              {modelsError ? (
                <div className="text-[11px]" style={{ color: "var(--danger)" }}>{modelsError}</div>
              ) : models.length === 0 && !modelsLoading ? (
                <div className="text-[11px]" style={{ color: "var(--fg-3)" }}>
                  {provider === "ollama" ? "No downloaded Ollama models found at this base URL." : "No known models for this provider."}
                </div>
              ) : (
                <div className="flex flex-wrap gap-1.5">
                  {models.map((name) => {
                    const active = settings.model.trim() === name;
                    return (
                      <button
                        key={name}
                        onClick={() => set("model", active ? "" : name)}
                        title={active ? "Click again to clear and use default" : `Use ${name}`}
                        className="h-7 px-2.5 rounded-full text-[11px] font-mono btn-quiet"
                        style={{
                          background: active ? "var(--accent-grad)" : "var(--bg-1)",
                          border: "1px solid var(--bg-4)",
                          color: active ? "white" : "var(--fg-1)",
                          boxShadow: active ? "var(--accent-glow)" : "none",
                        }}
                      >
                        {name}
                      </button>
                    );
                  })}
                </div>
              )}
              <div className="text-[10px]" style={{ color: "var(--fg-3)" }}>
                {settings.model.trim()
                  ? `Selected: ${settings.model.trim()} — saved on Apply.`
                  : defaultModel
                    ? `Empty — will use default: ${defaultModel}.`
                    : "Empty — will use provider default."}
              </div>
            </div>
            <div className="rounded-lg px-3 py-2 text-[11px] leading-relaxed" style={{ background: "var(--accent-soft)", border: "1px solid var(--accent-ring)", color: "var(--fg-1)" }}>
              Applying to the <b>{track}</b> track only. Active runs keep their current model; new runs use these settings immediately without restarting.
            </div>
            <div className="grid sm:grid-cols-3 gap-3">
              <Field label="Temperature"><input value={settings.temperature} onChange={(e) => set("temperature", e.target.value)} placeholder="0" className="field mono" /></Field>
              <Field label="Top P"><input value={settings.topP} onChange={(e) => set("topP", e.target.value)} placeholder="1" className="field mono" /></Field>
              <Field label="Max output tokens"><input value={settings.maxTokens} onChange={(e) => set("maxTokens", e.target.value)} placeholder="Provider default" className="field mono" /></Field>
            </div>
          </section>

          {track === "native" && (
            <section className="space-y-3">
              <SectionTitle>Run Defaults</SectionTitle>
              <div className="grid sm:grid-cols-2 gap-3">
                <Field label="Maximum turns"><input value={settings.maxTurns} onChange={(e) => set("maxTurns", e.target.value)} className="field mono" /></Field>
                <Field label="Maximum cost (USD)"><input value={settings.maxCost} onChange={(e) => set("maxCost", e.target.value)} className="field mono" /></Field>
              </div>
              <label
                className="flex items-start gap-2.5 rounded-lg px-3 py-2.5 cursor-pointer"
                style={{ background: settings.autoApproveAll ? "var(--warning-soft)" : "var(--bg-2)", border: "1px solid var(--bg-4)" }}
              >
                <input
                  type="checkbox"
                  checked={settings.autoApproveAll}
                  onChange={(e) => set("autoApproveAll", e.target.checked)}
                  className="mt-0.5 h-4 w-4 shrink-0"
                />
                <span>
                  <span className="block text-[12px] font-medium" style={{ color: "var(--fg-1)" }}>
                    Allow all tool permissions
                  </span>
                  <span className="block text-[11px] leading-relaxed" style={{ color: "var(--fg-2)" }}>
                    Skip approval prompts — the agent runs every tool without asking.
                    Safety denials (workspace escapes, plan-mode blocks) still apply.
                  </span>
                </span>
              </label>
            </section>
          )}
        </div>

        <div className="px-5 py-3 flex items-center gap-2 border-t" style={{ borderColor: "var(--bg-4)", background: "var(--bg-2)" }}>
          <button onClick={() => { setSettings(DEFAULT_SETTINGS); saveSettings(DEFAULT_SETTINGS); }} className="btn-quiet h-8 px-3 rounded-lg text-[11px]" style={{ background: "var(--bg-1)", border: "1px solid var(--bg-4)", color: "var(--fg-2)" }}>Reset defaults</button>
          {error && <span className="ml-auto text-[11px] truncate max-w-[300px]" style={{ color: "var(--danger)" }}>{error}</span>}
          {!error && <span className="ml-auto text-[11px]" style={{ color: saved ? "var(--success)" : "var(--fg-3)" }}>{saved ? `Applied to ${track}` : "Esc to close"}</span>}
          <button onClick={onClose} className="btn-quiet h-8 px-3 rounded-lg text-[11px]" style={{ background: "var(--bg-1)", border: "1px solid var(--bg-4)", color: "var(--fg-1)" }}>Cancel</button>
          <button onClick={handleSave} className="btn-grad h-8 px-4 rounded-lg text-[11px] font-semibold" style={{ color: "white", border: "1px solid transparent" }}>Save settings</button>
        </div>
      </div>
    </div>
  );
}

function SectionTitle({ children }: { children: string }) {
  return <div className="text-[11px] font-semibold tracking-wide uppercase" style={{ color: "var(--fg-2)" }}>{children}</div>;
}

function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return <label className="block space-y-1"><span className="flex items-center gap-2 text-[11px] font-medium" style={{ color: "var(--fg-1)" }}>{label}{hint && <span className="font-normal" style={{ color: "var(--fg-3)" }}>· {hint}</span>}</span>{children}</label>;
}
