import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { resolve } from "node:path";
import { setTimeout as sleep } from "node:timers/promises";

const root = resolve(fileURLToPath(new URL("../../../", import.meta.url)));
const desktop = resolve(fileURLToPath(new URL("../", import.meta.url)));
const isWindows = process.platform === "win32";
const apiCommand = isWindows ? "cmd.exe" : "uv";
const apiArgs = isWindows
  ? ["/d", "/s", "/c", "uv.exe run --package api api"]
  : ["run", "--package", "api", "api"];
const viteCommand = isWindows ? "cmd.exe" : "npm";
const viteArgs = isWindows ? ["/d", "/s", "/c", "npm.cmd run dev"] : ["run", "dev"];

let api = null;
let vite = null;
let ownsApi = false;

function launch(command, args, label, cwd) {
  const child = spawn(command, args, {
    cwd,
    env: process.env,
    stdio: "inherit",
    shell: false,
    windowsHide: false,
  });
  child.on("error", (error) => console.error(`[${label}] ${error.message}`));
  return child;
}

async function apiIsReady() {
  try {
    const [health, openapi] = await Promise.all([
      fetch("http://127.0.0.1:8000/health"),
      fetch("http://127.0.0.1:8000/openapi.json"),
    ]);
    if (!health.ok || !openapi.ok) return false;
    const spec = await openapi.json();
    return Boolean(
      spec.paths?.["/native/settings"]
      && spec.paths?.["/native/settings/models"]
      && spec.paths?.["/settings/langgraph"]
      && spec.paths?.["/settings/langgraph/models"]
      && spec.paths?.["/threads"]?.post,
    );
  } catch {
    return false;
  }
}

async function stopStaleApi() {
  if (!isWindows) return;
  const result = spawn("powershell.exe", [
    "-NoProfile", "-NonInteractive", "-Command",
    "$c=Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue; if($c){ Stop-Process -Id $c.OwningProcess -Force }",
  ], { stdio: "ignore", windowsHide: true });
  await new Promise((resolveExit) => result.on("exit", resolveExit));
}

if (await apiIsReady()) {
  console.log("[api] reusing http://127.0.0.1:8000");
} else {
  await stopStaleApi();
  api = launch(apiCommand, apiArgs, "api", root);
  ownsApi = true;
}

function shutdown() {
  for (const child of [vite, ownsApi ? api : null]) {
    if (child && !child.killed) child.kill();
  }
}

// Do not expose the window until the API sidecar is reachable. This avoids the
// frontend's first session request racing API startup on a clean checkout.
let ready = false;
for (let attempt = 0; attempt < 120; attempt += 1) {
  if (await apiIsReady()) {
    ready = true;
    break;
  }
  await sleep(250);
}
if (!ready) {
  console.error("[api] did not become ready at http://127.0.0.1:8000 within 30 seconds");
  shutdown();
  process.exit(1);
}

// Tauri waits for the Vite URL; the API remains alongside it for the lifetime
// of the dev process.
vite = launch(viteCommand, viteArgs, "vite", desktop);

process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);
process.on("exit", shutdown);

await new Promise((resolveExit) => {
  vite.on("exit", resolveExit);
  api?.on("exit", (code) => {
    if (code && !vite?.killed) {
      console.error(`[api] exited with code ${code}`);
      vite?.kill();
    }
  });
});

shutdown();
