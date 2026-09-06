import { useEffect, useState } from "react";
import { AgentSelect } from "./components/AgentSelect";
import { ChatWorkspace } from "./components/chat/ChatWorkspace";
import { SettingsModal, loadSettings, type DesktopSettings } from "./components/SettingsModal";

type AgentTrack = "native" | "langgraph";
const STORAGE_KEY = "operating-agent:track";

function usePersistedTrack() {
  const [track, setTrack] = useState<AgentTrack | null>(() => {
    try {
      const v = localStorage.getItem(STORAGE_KEY);
      return v === "native" || v === "langgraph" ? (v as AgentTrack) : null;
    } catch {
      return null;
    }
  });

  const save = (next: AgentTrack) => {
    try {
      localStorage.setItem(STORAGE_KEY, next);
    } catch {
      // ignore
    }
    setTrack(next);
    try {
      document.documentElement.setAttribute("data-track", next);
      window.dispatchEvent(new CustomEvent("operating-agent:track", { detail: next }));
      const w = window as unknown as { __TAURI__?: { invoke: (cmd: string, args?: unknown) => Promise<unknown> } };
      w.__TAURI__?.invoke("set_track", { track: next }).catch(() => {});
    } catch {
      // no-op
    }
  };

  const clear = () => {
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch {
      // ignore
    }
    setTrack(null);
  };

  return { track, save, clear };
}

export default function App() {
  const { track: persisted, save } = usePersistedTrack();
  const [selected, setSelected] = useState<AgentTrack | null>(persisted);
  const [confirmed, setConfirmed] = useState<AgentTrack | null>(persisted);
  const [track, setTrack] = useState<AgentTrack>(persisted || "native");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settings, setSettings] = useState<DesktopSettings>(loadSettings);

  useEffect(() => {
    if (persisted) {
      if (!selected) setSelected(persisted);
      setConfirmed(persisted);
      setTrack(persisted);
    }
  }, [persisted]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
      if (e.key === "1" && !confirmed) setSelected("native");
      if (e.key === "2" && !confirmed) setSelected("langgraph");
      if (e.key === "Enter" && selected && !confirmed) handleContinue();
      if (e.key === "Escape" && confirmed) setConfirmed(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selected, confirmed]);

  const handleContinue = () => {
    if (!selected) return;
    save(selected);
    setConfirmed(selected);
    setTrack(selected);
  };

  const handleBackToPicker = () => setConfirmed(null);

  if (!confirmed) {
    return <AgentSelect selected={selected} onSelect={setSelected} onContinue={handleContinue} />;
  }

  return (
    <div className="h-[100vh] flex flex-col overflow-hidden" style={{ background: "var(--bg-0)", color: "var(--fg-0)" }}>
      <ChatWorkspace
        track={track}
        onOpenSettings={() => setSettingsOpen(true)}
        onBackToPicker={handleBackToPicker}
      />
      {settingsOpen && <SettingsModal track={track} onClose={() => setSettingsOpen(false)} onSaved={(next) => setSettings(next)} />}
    </div>
  );
}
