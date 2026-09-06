import { useEffect, useRef, useState } from "react";

function Shell({
  title,
  subtitle,
  onClose,
  children,
  footer,
}: {
  title: string;
  subtitle?: string;
  onClose: () => void;
  children: React.ReactNode;
  footer: React.ReactNode;
}) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 anim-fade-in"
      style={{ background: "rgba(2, 6, 14, 0.7)", backdropFilter: "blur(6px)" }}
      onMouseDown={(e) => e.target === e.currentTarget && onClose()}
    >
      <div
        className="w-full max-w-[420px] rounded-2xl anim-scale-in"
        style={{
          background: "var(--bg-1)",
          border: "1px solid var(--bg-4)",
          boxShadow: "0 24px 64px rgba(2, 8, 20, 0.7), var(--accent-glow)",
        }}
      >
        <div className="px-5 pt-4 pb-3 border-b" style={{ borderColor: "var(--bg-4)" }}>
          <div className="text-[14px] font-semibold font-display">{title}</div>
          {subtitle && (
            <div className="text-[12px] mt-0.5" style={{ color: "var(--fg-2)" }}>
              {subtitle}
            </div>
          )}
        </div>
        <div className="px-5 py-4">{children}</div>
        <div
          className="px-5 py-3 flex items-center justify-end gap-2 border-t"
          style={{ borderColor: "var(--bg-4)", background: "var(--bg-2)", borderRadius: "0 0 16px 16px" }}
        >
          {footer}
        </div>
      </div>
    </div>
  );
}

export function PromptDialog({
  title,
  subtitle,
  initialValue,
  placeholder,
  confirmLabel,
  onSubmit,
  onClose,
}: {
  title: string;
  subtitle?: string;
  initialValue: string;
  placeholder?: string;
  confirmLabel: string;
  onSubmit: (value: string) => void;
  onClose: () => void;
}) {
  const [value, setValue] = useState(initialValue);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    inputRef.current?.focus();
    inputRef.current?.select();
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const submit = () => {
    const v = value.trim();
    if (v) onSubmit(v);
  };

  return (
    <Shell
      title={title}
      subtitle={subtitle}
      onClose={onClose}
      footer={
        <>
          <button
            onClick={onClose}
            className="btn-quiet h-8 px-4 rounded-lg text-[12px] font-medium"
            style={{ background: "var(--bg-1)", border: "1px solid var(--bg-4)", color: "var(--fg-1)" }}
          >
            Cancel
          </button>
          <button
            onClick={submit}
            disabled={!value.trim()}
            className="btn-grad h-8 px-4 rounded-lg text-[12px] font-semibold"
            style={{ color: "white", border: "1px solid transparent" }}
          >
            {confirmLabel}
          </button>
        </>
      }
    >
      <input
        ref={inputRef}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && submit()}
        placeholder={placeholder}
        className="field"
      />
    </Shell>
  );
}

export function ConfirmDialog({
  title,
  message,
  confirmLabel,
  danger,
  onConfirm,
  onClose,
}: {
  title: string;
  message: string;
  confirmLabel: string;
  danger?: boolean;
  onConfirm: () => void;
  onClose: () => void;
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      if (e.key === "Enter") onConfirm();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose, onConfirm]);

  return (
    <Shell
      title={title}
      onClose={onClose}
      footer={
        <>
          <button
            onClick={onClose}
            className="btn-quiet h-8 px-4 rounded-lg text-[12px] font-medium"
            style={{ background: "var(--bg-1)", border: "1px solid var(--bg-4)", color: "var(--fg-1)" }}
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            className={danger ? "h-8 px-4 rounded-lg text-[12px] font-semibold btn-quiet" : "btn-grad h-8 px-4 rounded-lg text-[12px] font-semibold"}
            style={
              danger
                ? { background: "var(--danger-soft)", border: "1px solid var(--danger)", color: "var(--danger)" }
                : { color: "white", border: "1px solid transparent" }
            }
          >
            {confirmLabel}
          </button>
        </>
      }
    >
      <div className="flex gap-3 items-start">
        <span
          className="w-8 h-8 rounded-lg grid place-items-center text-[14px] shrink-0"
          style={{
            background: danger ? "var(--danger-soft)" : "var(--accent-soft)",
            border: `1px solid ${danger ? "var(--danger)" : "var(--accent-ring)"}`,
            color: danger ? "var(--danger)" : "var(--accent)",
          }}
        >
          {danger ? "!" : "?"}
        </span>
        <p className="text-[13px] leading-relaxed pt-1" style={{ color: "var(--fg-1)" }}>
          {message}
        </p>
      </div>
    </Shell>
  );
}

export function AlertDialog({ message, onClose }: { message: string; onClose: () => void }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" || e.key === "Enter") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <Shell
      title="Something went wrong"
      onClose={onClose}
      footer={
        <button
          onClick={onClose}
          className="btn-grad h-8 px-5 rounded-lg text-[12px] font-semibold"
          style={{ color: "white", border: "1px solid transparent" }}
        >
          OK
        </button>
      }
    >
      <div className="flex gap-3 items-start">
        <span
          className="w-8 h-8 rounded-lg grid place-items-center text-[14px] shrink-0"
          style={{ background: "var(--danger-soft)", border: "1px solid var(--danger)", color: "var(--danger)" }}
        >
          !
        </span>
        <p className="text-[13px] leading-relaxed pt-1 font-mono break-words" style={{ color: "var(--fg-1)" }}>
          {message}
        </p>
      </div>
    </Shell>
  );
}
