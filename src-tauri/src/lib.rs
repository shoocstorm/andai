mod ug;

/// Lets the webview print to the terminal — used by the `VITE_SMOKE` harness.
#[tauri::command]
fn dev_log(line: String) {
    println!("[webview] {line}");
}

/// Ends the process; the smoke harness calls it once it has reported.
#[tauri::command]
fn dev_exit(app: tauri::AppHandle, code: i32) {
    app.exit(code);
}

/// Release builds serve the UI from http://localhost instead of `tauri://`:
/// WebKit only grants cross-origin isolation (SharedArrayBuffer, so
/// multi-threaded wllama) to http(s) origins, even when `tauri://` sends
/// COOP/COEP. Dev already gets this from the Vite server.
#[cfg(not(debug_assertions))]
const UI_PORT: u16 = 14230;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let builder = tauri::Builder::default();

    #[cfg(not(debug_assertions))]
    let builder = builder.plugin(
        tauri_plugin_localhost::Builder::new(UI_PORT)
            .on_request(|_req, res| {
                res.add_header("Cross-Origin-Opener-Policy", "same-origin");
                res.add_header("Cross-Origin-Embedder-Policy", "require-corp");
                res.add_header("Cross-Origin-Resource-Policy", "cross-origin");
            })
            .build(),
    );

    builder
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let mut config = app.config().app.windows[0].clone();
            #[cfg(not(debug_assertions))]
            {
                let url = format!("http://localhost:{UI_PORT}").parse().expect("valid url");
                config.url = tauri::WebviewUrl::External(url);
            }
            config.create = true;
            tauri::WebviewWindowBuilder::from_config(app.handle(), &config)?.build()?;
            Ok(())
        })
        .manage(ug::Indexing::default())
        .invoke_handler(tauri::generate_handler![
            dev_log,
            dev_exit,
            ug::ug_status,
            ug::kb_list,
            ug::kb_create,
            ug::kb_add_files,
            ug::kb_remove_source,
            ug::kb_delete,
            ug::kb_index,
            ug::kb_search,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Andai");
}
