// App commands are declared here so they get ACL permissions: release builds
// load the UI from http://localhost (see lib.rs), a "remote" origin that may
// only call commands its capability explicitly allows.
const COMMANDS: &[&str] = &[
    "dev_log",
    "dev_exit",
    "ug_status",
    "kb_list",
    "kb_create",
    "kb_add_files",
    "kb_pick_files",
    "kb_remove_source",
    "kb_delete",
    "kb_index",
    "kb_search",
    "kb_set_kind",
    "kb_tool",
    "laya_status",
    "laya_write_chunk",
    "laya_finish",
    "laya_remove",
    "laya_load",
    "laya_unload",
    "laya_decide",
];

fn main() {
    // `cfg(laya)`: the Laya decision model (MLX) is built in. MLX exists only
    // for Apple Silicon; everywhere else src/laya/ reports "not supported".
    println!("cargo::rustc-check-cfg=cfg(laya)");
    let target = |k: &str| std::env::var(k).unwrap_or_default();
    if target("CARGO_CFG_TARGET_OS") == "macos" && target("CARGO_CFG_TARGET_ARCH") == "aarch64" {
        println!("cargo::rustc-cfg=laya");
    }
    tauri_build::try_build(
        tauri_build::Attributes::new().app_manifest(tauri_build::AppManifest::new().commands(COMMANDS)),
    )
    .expect("failed to run tauri-build");
}
