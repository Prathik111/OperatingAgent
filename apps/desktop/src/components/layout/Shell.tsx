import type { ReactNode } from "react";

export function Shell({
  sidebar,
  header,
  children,
  statusBar,
}: {
  sidebar: ReactNode;
  header: ReactNode;
  children: ReactNode;
  statusBar?: ReactNode;
}) {
  return (
    <div className="h-[100vh] flex flex-col overflow-hidden" style={{ background: "var(--bg-0)", color: "var(--fg-0)" }}>
      <div
        data-tauri-drag-region
        className="h-7 shrink-0 flex items-center px-3 border-b"
        style={{ background: "var(--bg-1)", borderColor: "var(--bg-4)" }}
      >
        <div className="flex gap-1.5">
          <span className="w-3 h-3 rounded-full" style={{ background: "#ff5f57", border: "1px solid #e0443e" }} />
          <span className="w-3 h-3 rounded-full" style={{ background: "#ffbd2e", border: "1px solid #dea123" }} />
          <span className="w-3 h-3 rounded-full" style={{ background: "#28c940", border: "1px solid #1aab29" }} />
        </div>
        <span className="ml-3 text-[11px] font-medium tracking-wide" style={{ color: "var(--fg-2)" }}>
          OperatingAgent
        </span>
        <span
          className="ml-auto text-[10px] font-mono px-1.5 py-0.5 rounded hidden sm:inline"
          style={{ background: "var(--bg-2)", border: "1px solid var(--bg-4)", color: "var(--fg-2)" }}
        >
          Tauri · SSE · {import.meta.env.VITE_API_URL || "http://127.0.0.1:8000"}
        </span>
      </div>
      <div className="flex flex-1 min-h-0">
        <aside
          className="w-[280px] shrink-0 flex flex-col border-r hidden md:flex"
          style={{ background: "var(--bg-1)", borderColor: "var(--bg-4)" }}
        >
          {sidebar}
        </aside>
        <main className="flex-1 flex flex-col min-w-0" style={{ background: "var(--bg-0)" }}>
          <div className="h-12 shrink-0 flex items-center gap-3 px-4 border-b" style={{ borderColor: "var(--bg-4)" }}>
            {header}
          </div>
          <div className="flex-1 min-h-0 overflow-auto">{children}</div>
          {statusBar && (
            <div
              className="h-6 shrink-0 flex items-center gap-3 px-3 text-[11px] font-mono border-t"
              style={{ background: "var(--bg-1)", borderColor: "var(--bg-4)", color: "var(--fg-3)" }}
            >
              {statusBar}
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

export function Label({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <span className={`text-[11px] font-semibold tracking-wide uppercase ${className}`} style={{ color: "var(--fg-2)" }}>
      {children}
    </span>
  );
}

export function Card({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <div className={`rounded-xl ${className}`} style={{ background: "var(--bg-1)", border: "1px solid var(--bg-4)" }}>
      {children}
    </div>
  );
}
