mod grants;
mod laya;
mod tools;
mod ug;
#[cfg_attr(debug_assertions, allow(dead_code))]
mod ui_server;

use tauri::Manager;

/// The test-harness commands below ship in every build (the release e2e needs
/// them), so they only work when the harness launched the app with
/// `ANDAI_SMOKE=1`. The webview can't set the environment (AGENTS.md §9).
fn smoke_enabled() -> Result<(), String> {
    match std::env::var("ANDAI_SMOKE").as_deref() {
        Ok("1") => Ok(()),
        _ => Err("test harness commands are disabled".into()),
    }
}

/// Lets the webview print to the terminal — used by the `VITE_SMOKE` harness.
#[tauri::command]
fn dev_log(line: String) -> Result<(), String> {
    smoke_enabled()?;
    println!("[webview] {line}");
    Ok(())
}

/// Ends the process; the smoke harness calls it once it has reported.
#[tauri::command]
fn dev_exit(app: tauri::AppHandle, code: i32) -> Result<(), String> {
    smoke_enabled()?;
    app.exit(code);
    Ok(())
}

/// Release builds serve the UI from http://localhost instead of `tauri://`:
/// WebKit only grants cross-origin isolation (SharedArrayBuffer, so
/// multi-threaded wllama) to http(s) origins, even when `tauri://` sends
/// COOP/COEP. Dev already gets this from the Vite server. The port is fixed:
/// webview storage (chats, the model cache) is keyed by origin.
#[cfg_attr(debug_assertions, allow(dead_code))]
const UI_PORT: u16 = 14230;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            grants::grant_from_env(&app.state::<grants::FileGrants>());
            let mut config = app.config().app.windows[0].clone();

            #[cfg(not(debug_assertions))]
            let app_url: tauri::Url = {
                // Own the port before any window can load from it (ui_server.rs).
                match ui_server::bind(UI_PORT) {
                    Ok(listeners) => ui_server::serve(listeners, UI_PORT, app.asset_resolver()),
                    Err(msg) => {
                        use tauri_plugin_dialog::{DialogExt, MessageDialogKind};
                        let handle = app.handle().clone();
                        app.dialog()
                            .message(msg)
                            .title("Andai can't start")
                            .kind(MessageDialogKind::Error)
                            .show(move |_| handle.exit(1));
                        return Ok(());
                    }
                }
                let url: tauri::Url = format!("http://localhost:{UI_PORT}").parse().expect("valid url");
                config.url = tauri::WebviewUrl::External(url.clone());
                url
            };
            #[cfg(debug_assertions)]
            let app_url = app.config().build.dev_url.clone().expect("devUrl is set in tauri.conf.json");

            config.create = true;
            let window = tauri::WebviewWindowBuilder::from_config(app.handle(), &config)?
                .on_navigation(move |url| ui_server::is_app_url(url, &app_url))
                .on_new_window(|_, _| tauri::webview::NewWindowResponse::Deny)
                .build()?;
            // A drop is the user choosing files: grant them before the webview
            // hears about it (grants.rs). Runs on the main thread, as does
            // the sync kb_add_files that consumes the grant.
            let handle = app.handle().clone();
            window.on_window_event(move |event| {
                if let tauri::WindowEvent::DragDrop(tauri::DragDropEvent::Drop { paths, .. }) = event {
                    handle.state::<grants::FileGrants>().grant(paths);
                }
            });
            Ok(())
        })
        .manage(ug::Indexing::default())
        .manage(grants::FileGrants::default())
        .manage(laya::Laya::default())
        .invoke_handler(tauri::generate_handler![
            dev_log,
            dev_exit,
            ug::ug_status,
            ug::kb_list,
            ug::kb_create,
            ug::kb_add_files,
            grants::kb_pick_files,
            ug::kb_remove_source,
            ug::kb_delete,
            ug::kb_index,
            ug::kb_search,
            ug::kb_set_kind,
            tools::kb_tool,
            laya::laya_status,
            laya::laya_write_chunk,
            laya::laya_finish,
            laya::laya_remove,
            laya::laya_load,
            laya::laya_unload,
            laya::laya_decide,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Andai");
}
