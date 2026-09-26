//! The Laya decision model: a small bidirectional encoder that scores a
//! choice between options in one forward pass (~10–20 ms, versus ~0.7 s for a
//! letter readout on the chat model). It runs on MLX, so only on Apple Silicon
//! (`cfg(laya)`, set by build.rs); elsewhere these commands say so and the
//! webview keeps deciding with wllama (llm/decide.ts).
//!
//! The webview is untrusted (AGENTS.md §9): checkpoints and files are named
//! from the closed catalog, downloads only land after their sha256 matches
//! (store.rs), and decision inputs are bounded here.

#[cfg_attr(not(laya), allow(dead_code))]
mod catalog;
#[cfg_attr(not(laya), allow(dead_code))]
mod store;

#[cfg(laya)]
mod engine;
#[cfg(laya)]
mod model;
#[cfg(laya)]
mod prompt;
#[cfg(laya)]
mod worker;

use serde::{Deserialize, Serialize};
#[cfg(laya)]
use std::sync::{Arc, Mutex};
use tauri::AppHandle;
#[cfg(laya)]
use tauri::Manager;

#[cfg_attr(laya, allow(dead_code))]
const UNSUPPORTED: &str = "The Laya decision model needs an Apple Silicon Mac.";

/// Loaded-model state; the MLX thread starts on first use.
#[derive(Default)]
pub struct Laya {
    #[cfg(laya)]
    worker: Mutex<Option<Arc<worker::Worker>>>,
}

#[cfg(laya)]
impl Laya {
    fn worker(&self, app: &AppHandle) -> Arc<worker::Worker> {
        let mut w = self.worker.lock().unwrap();
        w.get_or_insert_with(|| {
            // Release bundles ship mlx.metallib as a resource (tauri.conf.json).
            let metallib = app.path().resource_dir().ok().map(|d| d.join("mlx.metallib"));
            Arc::new(worker::Worker::spawn(metallib))
        })
        .clone()
    }

    fn loaded(&self) -> Option<String> {
        self.worker.lock().unwrap().as_ref().and_then(|w| w.loaded())
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CheckpointStatus {
    id: &'static str,
    repo: &'static str,
    commit: &'static str,
    bytes: u64,
    files: Vec<FileInfo>,
    downloaded: bool,
}

#[derive(Debug, Serialize)]
pub struct FileInfo {
    path: &'static str,
    bytes: u64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LayaStatus {
    supported: bool,
    loaded: Option<String>,
    checkpoints: Vec<CheckpointStatus>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LayaOption {
    id: String,
    text: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LayaChoice {
    /// Calibrated probability per option, in the order given.
    probabilities: Vec<f64>,
    input_tokens: usize,
    truncated: bool,
    ms: f64,
    model: String,
}

pub const MAX_STATE: usize = 64 * 1024;
pub const MAX_QUESTION: usize = 2 * 1024;
pub const MAX_OPTION_TEXT: usize = 1024;
pub const MAX_OPTIONS: usize = 16;
/// One IPC chunk of a checkpoint download.
pub const MAX_CHUNK: usize = 16 * 1024 * 1024;

/// Bounds on what the webview may ask the model to score.
pub fn validate(state: &str, question: &str, options: &[LayaOption]) -> Result<(), String> {
    if state.len() > MAX_STATE {
        return Err(format!("decision state is over {MAX_STATE} bytes"));
    }
    if question.is_empty() || question.len() > MAX_QUESTION {
        return Err(format!("decision question must be 1–{MAX_QUESTION} bytes"));
    }
    if !(2..=MAX_OPTIONS).contains(&options.len()) {
        return Err(format!("a decision needs 2–{MAX_OPTIONS} options, got {}", options.len()));
    }
    let mut seen = std::collections::HashSet::new();
    for o in options {
        let ok_id = !o.id.is_empty() && o.id.len() <= 64 && o.id.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_');
        if !ok_id {
            return Err(format!("invalid option id: {:?}", o.id));
        }
        if !seen.insert(o.id.as_str()) {
            return Err(format!("duplicate option id: {}", o.id));
        }
        if o.text.len() > MAX_OPTION_TEXT {
            return Err(format!("option {} text is over {MAX_OPTION_TEXT} bytes", o.id));
        }
    }
    Ok(())
}

#[cfg(laya)]
fn root(app: &AppHandle) -> Result<std::path::PathBuf, String> {
    crate::ug::data_dir(app, &["models", "laya"])
}

#[cfg(laya)]
async fn blocking<T: Send + 'static>(f: impl FnOnce() -> Result<T, String> + Send + 'static) -> Result<T, String> {
    tauri::async_runtime::spawn_blocking(f).await.map_err(|e| e.to_string())?
}

#[tauri::command]
pub async fn laya_status(app: AppHandle, laya: tauri::State<'_, Laya>) -> Result<LayaStatus, String> {
    #[cfg(laya)]
    {
        let root = root(&app)?;
        let checkpoints = catalog::CHECKPOINTS
            .iter()
            .map(|c| CheckpointStatus {
                id: c.id,
                repo: c.repo,
                commit: c.commit,
                bytes: c.bytes(),
                files: c.files.iter().map(|f| FileInfo { path: f.path, bytes: f.bytes }).collect(),
                downloaded: store::is_downloaded(&root, c),
            })
            .collect();
        Ok(LayaStatus { supported: true, loaded: laya.loaded(), checkpoints })
    }
    #[cfg(not(laya))]
    {
        let _ = (app, laya);
        Ok(LayaStatus { supported: false, loaded: None, checkpoints: vec![] })
    }
}

/// One chunk of a download, as a raw IPC body. Headers name it:
/// `x-laya-checkpoint`, `x-laya-file` (catalog path) and `x-laya-offset`.
#[tauri::command]
pub async fn laya_write_chunk(app: AppHandle, request: tauri::ipc::Request<'_>) -> Result<u64, String> {
    let header = |k: &str| request.headers().get(k).and_then(|v| v.to_str().ok()).map(str::to_string).ok_or_else(|| format!("missing {k} header"));
    let (id, file, offset) = (header("x-laya-checkpoint")?, header("x-laya-file")?, header("x-laya-offset")?);
    let offset: u64 = offset.parse().map_err(|_| "x-laya-offset must be a byte offset".to_string())?;
    let tauri::ipc::InvokeBody::Raw(bytes) = request.body() else {
        return Err("a chunk must be sent as raw bytes".into());
    };
    if bytes.len() > MAX_CHUNK {
        return Err(format!("a chunk is at most {MAX_CHUNK} bytes"));
    }
    #[cfg(laya)]
    {
        let c = catalog::checkpoint(&id)?;
        let root = root(&app)?;
        let bytes = bytes.clone();
        blocking(move || store::write_chunk(&root, c, &file, offset, &bytes)).await
    }
    #[cfg(not(laya))]
    {
        let _ = (app, id, file, offset);
        Err(UNSUPPORTED.into())
    }
}

/// Verifies a finished download and moves it into place (store.rs).
#[tauri::command]
pub async fn laya_finish(app: AppHandle, checkpoint: String) -> Result<(), String> {
    #[cfg(laya)]
    {
        let c = catalog::checkpoint(&checkpoint)?;
        let root = root(&app)?;
        blocking(move || store::finish(&root, c)).await
    }
    #[cfg(not(laya))]
    {
        let _ = (app, checkpoint);
        Err(UNSUPPORTED.into())
    }
}

/// Deletes a downloaded checkpoint (the UI confirms first), unloading it if loaded.
#[tauri::command]
pub async fn laya_remove(app: AppHandle, laya: tauri::State<'_, Laya>, checkpoint: String) -> Result<(), String> {
    #[cfg(laya)]
    {
        let c = catalog::checkpoint(&checkpoint)?;
        if laya.loaded().as_deref() == Some(c.id) {
            let w = laya.worker(&app);
            blocking(move || w.unload()).await?;
        }
        let root = root(&app)?;
        blocking(move || store::remove(&root, c)).await
    }
    #[cfg(not(laya))]
    {
        let _ = (app, laya, checkpoint);
        Err(UNSUPPORTED.into())
    }
}

/// Loads a downloaded checkpoint; returns how long it took, in ms.
#[tauri::command]
pub async fn laya_load(app: AppHandle, laya: tauri::State<'_, Laya>, checkpoint: String) -> Result<f64, String> {
    #[cfg(laya)]
    {
        let c = catalog::checkpoint(&checkpoint)?;
        let dir = store::resolve(&root(&app)?, c)?;
        let w = laya.worker(&app);
        blocking(move || w.load(c.id, dir)).await
    }
    #[cfg(not(laya))]
    {
        let _ = (app, laya, checkpoint);
        Err(UNSUPPORTED.into())
    }
}

#[tauri::command]
pub async fn laya_unload(app: AppHandle, laya: tauri::State<'_, Laya>) -> Result<(), String> {
    #[cfg(laya)]
    {
        let w = laya.worker(&app);
        blocking(move || w.unload()).await
    }
    #[cfg(not(laya))]
    {
        let _ = (app, laya);
        Err(UNSUPPORTED.into())
    }
}

/// Scores `options` for `question` given `state` on the loaded checkpoint.
#[tauri::command]
pub async fn laya_decide(app: AppHandle, laya: tauri::State<'_, Laya>, state: String, question: String, options: Vec<LayaOption>) -> Result<LayaChoice, String> {
    validate(&state, &question, &options)?;
    #[cfg(laya)]
    {
        let model = laya.loaded().ok_or("No Laya model is loaded.")?;
        let w = laya.worker(&app);
        let pairs = options.into_iter().map(|o| (o.id, o.text)).collect();
        let c = blocking(move || w.choose(state, question, pairs)).await?;
        Ok(LayaChoice { probabilities: c.probabilities, input_tokens: c.input_tokens, truncated: c.truncated, ms: c.ms, model })
    }
    #[cfg(not(laya))]
    {
        let _ = (app, laya, state, question, options);
        Err(UNSUPPORTED.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opt(id: &str, text: &str) -> LayaOption {
        LayaOption { id: id.into(), text: text.into() }
    }

    #[test]
    fn accepts_a_normal_decision() {
        assert!(validate("state", "q?", &[opt("answer_now", "Answer"), opt("kb_search", "Search")]).is_ok());
    }

    #[test]
    fn bounds_every_input() {
        let two = [opt("a", "x"), opt("b", "y")];
        assert!(validate(&"s".repeat(MAX_STATE + 1), "q", &two).unwrap_err().contains("state"));
        assert!(validate("s", "", &two).is_err());
        assert!(validate("s", &"q".repeat(MAX_QUESTION + 1), &two).is_err());
        assert!(validate("s", "q", &two[..1]).unwrap_err().contains("2–16"));
        let many: Vec<_> = (0..17).map(|i| opt(&format!("o{i}"), "t")).collect();
        assert!(validate("s", "q", &many).unwrap_err().contains("2–16"));
        assert!(validate("s", "q", &[opt("a", "x"), opt("b", &"t".repeat(MAX_OPTION_TEXT + 1))]).is_err());
    }

    #[test]
    fn option_ids_are_plain_slugs_and_unique() {
        for bad in ["", "Answer", "a-b", "../x", "a b", "<mask>", &"a".repeat(65)] {
            assert!(validate("s", "q", &[opt(bad, "x"), opt("b", "y")]).is_err(), "{bad:?}");
        }
        assert!(validate("s", "q", &[opt("a", "x"), opt("a", "y")]).unwrap_err().contains("duplicate"));
    }

    #[cfg(not(laya))]
    #[test]
    fn says_it_is_unsupported_off_apple_silicon() {
        assert!(UNSUPPORTED.contains("Apple Silicon"));
    }
}
