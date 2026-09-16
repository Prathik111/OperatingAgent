// OS folder picker for the Tauri shell, with a silent fallback.
//
// Inside the Tauri window this opens the native file-explorer directory
// dialog (via @tauri-apps/plugin-dialog, registered in src-tauri). In a plain
// browser (vite dev) there is no backing shell, so callers keep their typed
// input and the UI hides the browse button via `isTauri()`.

/** Whether the OS folder dialog is available (Tauri shell, not a browser). */
export function isTauri(): boolean {
  if (typeof window === "undefined") return false;
  const value = window as unknown as { __TAURI_INTERNALS__?: unknown; __TAURI__?: unknown };
  return Boolean(value.__TAURI_INTERNALS__ || value.__TAURI__ || navigator.userAgent.includes("Tauri"));
}

/** Open the native directory picker. Null when cancelled or unavailable. */
export async function pickDirectory(startPath?: string): Promise<string | null> {
  if (!isTauri()) return null;
  try {
    const { open } = await import("@tauri-apps/plugin-dialog");
    const selected = await open({
      directory: true,
      multiple: false,
      defaultPath: startPath?.trim() || undefined,
      title: "Select working directory",
    });
    return typeof selected === "string" ? selected : null;
  } catch (error) {
    console.warn("Unable to open the native folder picker", error);
    return null;
  }
}

/** Display-safe folder label; callers keep the original path for API requests. */
export function folderName(path: string): string {
  const value = path.trim();
  if (!value || value === ".") return "Choose workspace folder";
  const parts = value.split(/[/\\\\]/).filter(Boolean);
  return parts[parts.length - 1] || "Choose workspace folder";
}
