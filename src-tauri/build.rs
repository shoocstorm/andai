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
    "kb_add_sample",
    "kb_tool",
    "laya_status",
    "laya_write_chunk",
    "laya_finish",
    "laya_remove",
    "laya_load",
    "laya_unload",
    "laya_decide",
    "laya_relevance",
    "llm_status",
    "llm_write_chunk",
    "llm_finish",
    "llm_remove",
    "llm_load",
    "llm_unload",
    "llm_generate",
    "llm_cancel",
    "llm_add_custom",
];

fn main() {
    // `cfg(mlx)`: MLX is built in, with the models that run on it (the Laya
    // decision model, src/laya/, and the native chat models, src/llm/). MLX
    // exists only for Apple Silicon; everywhere else those report "not supported".
    println!("cargo::rustc-check-cfg=cfg(mlx)");
    let target = |k: &str| std::env::var(k).unwrap_or_default();
    if target("CARGO_CFG_TARGET_OS") == "macos" && target("CARGO_CFG_TARGET_ARCH") == "aarch64" {
        println!("cargo::rustc-cfg=mlx");
    }
    tauri_build::try_build(
        tauri_build::Attributes::new().app_manifest(tauri_build::AppManifest::new().commands(COMMANDS)),
    )
    .expect("failed to run tauri-build");
}
