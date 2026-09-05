#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum AgentTrack {
    Native,
    Langgraph,
}

#[tauri::command]
fn set_track(track: AgentTrack) -> Result<String, String> {
    // Persisted by the frontend in localStorage as `operating-agent:track`.
    // This command is the hook where Rust will:
    //   1. spawn / reconfigure the Python sidecar (packages/api) for the chosen track,
    //   2. hold the bearer token and OS-assigned loopback port,
    //   3. kill the whole tree on quit.
    // For now it just acknowledges — the thesis comparison stays honest because
    // both tracks go through the same MCP gateway regardless of this choice.
    println!("[tauri] set_track: {:?}", track);
    Ok(format!("track set to {:?}", track))
}

#[tauri::command]
fn get_track() -> Option<AgentTrack> {
    // Frontend is the source of truth until the sidecar owns the store.
    None
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![set_track, get_track])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
