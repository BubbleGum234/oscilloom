// Oscilloom Desktop Shell
// Launches the PyInstaller-bundled backend server as a sidecar process,
// waits for it to be ready, then opens the frontend in a native window.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use tauri::Manager;
use tauri_plugin_shell::ShellExt;

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .setup(|app| {
            // Launch the Python backend as a sidecar process
            let shell = app.shell();
            let sidecar = shell
                .sidecar("oscilloom-server")
                .expect("failed to create sidecar command");

            let (_rx, _child) = sidecar
                .spawn()
                .expect("failed to spawn oscilloom-server sidecar");

            // The frontend will poll /status until the server is ready
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running Oscilloom");
}
