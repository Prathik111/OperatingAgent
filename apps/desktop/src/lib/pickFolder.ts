// OS folder picker for the Tauri shell, with a silent fallback.
//
// Inside the Tauri window this opens the native file-explorer directory
// dialog (via @tauri-apps/plugin-dialog, registered in src-tauri). In a plain
// browser (vite dev) there is no backing shell, so callers keep their typed
// input and the UI hides the browse button via `isTauri()`.

/** Whether the OS folder dialog is available (Tauri shell, not a browser). */
export function isTauri(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
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
  } catch {
    return null;
  }
}
