type AgentTrack = "native" | "langgraph";

interface AgentOption {
  id: AgentTrack;
  name: string;
  subtitle: string;
  badge: string;
  badgeTone: "violet" | "zinc";
  status: string;
  statusTone: "success" | "warning";
  description: string;
  features: string[];
  metrics: { label: string; value: string }[];
  icon: string;
}

const OPTIONS: AgentOption[] = [
  {
    id: "native",
    name: "Native Agent",
    subtitle: "Plan-and-Execute + ReAct — hand-written loop",
    badge: "Ready",
    badgeTone: "violet",
    status: "● Ready",
    statusTone: "success",
    description:
      "The thesis agent. One transcript, one loop, every failure is an observation. Built to measure what it costs to hand-build what a framework gives for free.",
    features: [
      "Transcript is the state — single append-only conversation",
      "Policy is a hook — Risk, Permission, Audit as a chain",
      "Budgets are real — cost / tokens / turns / wall-clock",
      "Sandboxed tools via MCP gateway (filesystem · git · terminal)",
      "Checkpoints, skills, plan-mode, subagent fan-out",
    ],
    metrics: [
      { label: "Tools", value: "14 via gateway" },
      { label: "Model", value: "llama-3.3-70b · Groq" },
      { label: "Tests", value: "272 passing" },
    ],
    icon: "◈",
  },
  {
    id: "langgraph",
    name: "LangGraph Agent",
    subtitle: "StateGraph — framework-built track",
    badge: "Implemented",
    badgeTone: "violet",
    status: "● Ready when model is configured",
    statusTone: "success",
    description:
      "The comparison track is implemented as a real StateGraph. It uses the same tool layer and task suite as Native, while keeping LangGraph's planner, executor, verifier, checkpoints, and interrupt flow.",
    features: [
      "create_react_agent + Pregel execution engine",
      "Checkpointer for durable state & resume",
      "Human-in-the-loop via interrupt",
      "Streaming via graph event stream",
      "Measured through identical Track interface",
    ],
    metrics: [
      { label: "Tools", value: "same gateway" },
      { label: "State", value: "AgentState ↔ graph" },
      { label: "Harness", value: "evaluation compare" },
    ],
    icon: "⬢",
  },
];

export function AgentSelect({
  selected,
  onSelect,
  onContinue,
}: {
  selected: AgentTrack | null;
  onSelect: (id: AgentTrack) => void;
  onContinue: () => void;
}) {
  return (
    <div className="min-h-screen flex flex-col" style={{ background: "var(--bg-0)" }}>
      {/* Center */}
      <div className="flex-1 flex flex-col items-center justify-center px-6 py-10 hero-glow">
        {/* Header */}
        <div className="text-center max-w-[640px] w-full anim-fade-up">
          <div
            className="w-11 h-11 rounded-2xl mx-auto grid place-items-center text-[18px] font-bold"
            style={{ background: "var(--accent-grad)", color: "#fff", boxShadow: "var(--accent-glow)" }}
          >
            ◈
          </div>
          <h1 className="mt-4 text-[24px] font-semibold tracking-tight font-display" style={{ color: "var(--fg-0)" }}>
            Choose your <span className="grad-text">agent</span>
          </h1>
          <p className="mt-2 text-[13px] leading-relaxed" style={{ color: "var(--fg-2)" }}>
            Pick the track to drive — you can switch anytime.
          </p>
        </div>

        {/* Cards */}
        <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-4 w-full max-w-[760px]">
          {OPTIONS.map((opt, i) => {
            const isSelected = selected === opt.id;
            const isNative = opt.id === "native";
            return (
              <button
                key={opt.id}
                onClick={() => onSelect(opt.id)}
                className="group text-left rounded-2xl p-5 flex flex-col gap-4 relative overflow-hidden card-lift anim-fade-up"
                style={{
                  background: isSelected ? "var(--bg-2)" : "var(--bg-1)",
                  border: `1px solid ${isSelected ? "var(--accent-ring)" : "var(--bg-4)"}`,
                  boxShadow: isSelected ? "var(--accent-glow), 0 0 0 1px var(--accent-ring)" : "0 1px 2px rgba(0,0,0,0.3)",
                  transform: isSelected ? "translateY(-1px)" : undefined,
                  animationDelay: `${i * 90}ms`,
                }}
              >
                {/* selected check */}
                <span
                  className="absolute top-3 right-3 w-6 h-6 rounded-full grid place-items-center text-[11px]"
                  style={{
                    background: isSelected ? "var(--accent-grad)" : "transparent",
                    border: `1.5px solid ${isSelected ? "transparent" : "var(--bg-4)"}`,
                    color: isSelected ? "white" : "transparent",
                    boxShadow: isSelected ? "var(--accent-glow)" : "none",
                    transition: "all var(--dur-2) var(--ease-out)",
                  }}
                >
                  ✓
                </span>

                {/* header */}
                <div className="flex gap-3 pr-6">
                  <span
                    className="w-9 h-9 rounded-xl grid place-items-center text-[15px] font-bold shrink-0"
                    style={{
                      background: isNative ? "var(--accent-soft)" : "var(--bg-3)",
                      border: `1px solid ${isNative ? "var(--accent-ring)" : "var(--bg-4)"}`,
                      color: isNative ? "var(--accent)" : "var(--fg-1)",
                    }}
                  >
                    {opt.icon}
                  </span>
                  <div className="min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-[13px] font-semibold" style={{ color: "var(--fg-0)" }}>
                        {opt.name}
                      </span>
                      <span
                        className="text-[10px] font-semibold tracking-wide uppercase px-1.5 py-0.5 rounded"
                        style={
                          opt.badgeTone === "violet"
                            ? { background: "var(--accent-soft)", color: "var(--accent)", border: "1px solid var(--accent-ring)" }
                            : { background: "var(--bg-3)", color: "var(--fg-2)", border: "1px solid var(--bg-4)" }
                        }
                      >
                        {opt.badge}
                      </span>
                    </div>
                    <div className="text-[11px] font-medium leading-tight mt-0.5" style={{ color: "var(--fg-2)" }}>
                      {opt.subtitle}
                    </div>
                    <div
                      className="inline-flex items-center gap-1 mt-1.5 text-[11px] font-mono px-2 py-0.5 rounded-full"
                      style={{
                        background: opt.statusTone === "success" ? "var(--success-soft)" : "var(--warning-soft)",
                        border: `1px solid ${opt.statusTone === "success" ? "rgba(34,197,94,0.35)" : "rgba(245,158,11,0.4)"}`,
                        color: opt.statusTone === "success" ? "var(--success)" : "var(--warning)",
                      }}
                    >
                      {opt.status}
                    </div>
                  </div>
                </div>

                <p className="text-[12px] leading-relaxed" style={{ color: "var(--fg-1)" }}>
                  {opt.description}
                </p>

                <ul className="space-y-1.5">
                  {opt.features.map((f) => (
                    <li key={f} className="flex gap-2 text-[11px] leading-relaxed" style={{ color: "var(--fg-2)" }}>
                      <span className="mt-[6px] w-1 h-1 rounded-full shrink-0" style={{ background: isSelected ? "var(--accent)" : "var(--bg-4)" }} />
                      <span>{f}</span>
                    </li>
                  ))}
                </ul>

                <div className="mt-auto pt-3 flex gap-2 border-t" style={{ borderColor: "var(--bg-4)" }}>
                  {opt.metrics.map((m) => (
                    <span key={m.label} className="flex-1 text-center px-2 py-1.5 rounded-lg" style={{ background: "var(--bg-0)", border: "1px solid var(--bg-4)" }}>
                      <span className="block text-[10px] font-semibold tracking-wide uppercase" style={{ color: "var(--fg-3)" }}>
                        {m.label}
                      </span>
                      <span className="block text-[11px] font-mono font-medium mt-0.5" style={{ color: "var(--fg-1)" }}>
                        {m.value}
                      </span>
                    </span>
                  ))}
                </div>

                {/* keyboard hint */}
                <span
                  className="absolute bottom-2 right-3 text-[10px] font-mono px-1.5 py-0.5 rounded hidden sm:block"
                  style={{ background: "var(--bg-3)", border: "1px solid var(--bg-4)", color: "var(--fg-3)" }}
                >
                  {opt.id === "native" ? "1" : "2"}
                </span>
              </button>
            );
          })}
        </div>

        {/* Action */}
        <div className="mt-8 flex flex-col items-center gap-3 w-full max-w-[760px]">
          <button
            onClick={onContinue}
            disabled={!selected}
            className={`h-10 px-7 rounded-xl text-[13px] font-semibold inline-flex items-center gap-2 disabled:cursor-not-allowed ${selected ? "btn-grad" : ""}`}
            style={
              selected
                ? { color: "white", border: "1px solid transparent" }
                : {
                    background: "var(--bg-2)",
                    color: "var(--fg-3)",
                    border: "1px solid var(--bg-4)",
                    opacity: 0.7,
                  }
            }
          >
            Continue with {selected ? (selected === "native" ? "Native" : "LangGraph") : "…"}
            <span className="text-[12px]">→</span>
            <span className="hidden sm:inline font-mono text-[11px] opacity-70 ml-1">↵</span>
          </button>
        </div>
      </div>
    </div>
  );
}
