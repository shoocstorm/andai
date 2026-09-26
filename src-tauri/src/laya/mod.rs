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

/// `choice`: pick one of `options`. `noul`: does the statement in `question`
/// hold, given the state? (no options; the answer is `[P(false), P(true)]`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LayaKind {
    Choice,
    Noul,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LayaQuestion {
    id: String,
    kind: LayaKind,
    question: String,
    #[serde(default)]
    options: Vec<LayaOption>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LayaAnswer {
    id: String,
    /// Calibrated probability per option, in the order given; `[P(false), P(true)]` for a noul.
    probabilities: Vec<f64>,
    input_tokens: usize,
    truncated: bool,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LayaAnswers {
    answers: Vec<LayaAnswer>,
    /// Model time for the whole batch (Rust), ms.
    ms: f64,
    model: String,
}

pub const MAX_STATE: usize = 64 * 1024;
pub const MAX_QUESTION: usize = 2 * 1024;
pub const MAX_OPTION_TEXT: usize = 1024;
pub const MAX_OPTIONS: usize = 16;
/// Relevance check: passages per call, and bounds on each.
pub const MAX_PASSAGES: usize = 24;
pub const MAX_PASSAGE: usize = 16 * 1024;
pub const MAX_SOURCE: usize = 512;
pub const MAX_REQUEST: usize = 4 * 1024;

/// The relevance question, asked of each retrieved passage (a `noul`). Of two
/// phrasings probed on the eval fixtures (47 passages × 29 questions), this
/// separated passages holding the expected fact best: AUC 0.76 multilingual,
/// 0.85 English (AGENTS.md §2).
#[cfg_attr(not(laya), allow(dead_code))]
pub const RELEVANT: &str = "This passage contains information that helps answer the user's request.";
/// Questions per call: each is its own row in the batch.
pub const MAX_QUESTIONS: usize = 4;
/// One IPC chunk of a checkpoint download.
pub const MAX_CHUNK: usize = 16 * 1024 * 1024;

fn slug(id: &str) -> bool {
    !id.is_empty() && id.len() <= 64 && id.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
}

#[derive(Debug, Clone, Deserialize)]
pub struct LayaPassage {
    /// Where it's from (file and lines), shown to the model with the text.
    source: String,
    text: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LayaRelevance {
    /// P(the passage helps answer the request), per passage in order.
    scores: Vec<f64>,
    /// Model time for all passages (Rust), ms.
    ms: f64,
    model: String,
}

pub fn validate_passages(request: &str, passages: &[LayaPassage]) -> Result<(), String> {
    if request.trim().is_empty() || request.len() > MAX_REQUEST {
        return Err(format!("the request must be 1–{MAX_REQUEST} bytes"));
    }
    if !(1..=MAX_PASSAGES).contains(&passages.len()) {
        return Err(format!("check 1–{MAX_PASSAGES} passages at a time, got {}", passages.len()));
    }
    for (i, p) in passages.iter().enumerate() {
        if p.source.len() > MAX_SOURCE || p.text.len() > MAX_PASSAGE {
            return Err(format!("passage {} is over {MAX_PASSAGE} bytes (or its source over {MAX_SOURCE})", i + 1));
        }
    }
    Ok(())
}

/// The state one passage is judged in: the request, then the passage.
#[cfg_attr(not(laya), allow(dead_code))]
pub fn passage_state(request: &str, p: &LayaPassage) -> String {
    format!("User request:\n{}\n\nPassage from {}:\n{}", request.trim(), p.source, p.text.trim())
}

/// Bounds on what the webview may ask the model to score.
pub fn validate(state: &str, questions: &[LayaQuestion]) -> Result<(), String> {
    if state.len() > MAX_STATE {
        return Err(format!("decision state is over {MAX_STATE} bytes"));
    }
    if !(1..=MAX_QUESTIONS).contains(&questions.len()) {
        return Err(format!("ask 1–{MAX_QUESTIONS} questions at a time, got {}", questions.len()));
    }
    let mut ids = std::collections::HashSet::new();
    for q in questions {
        if !slug(&q.id) || !ids.insert(q.id.as_str()) {
            return Err(format!("invalid or duplicate question id: {:?}", q.id));
        }
        if q.question.is_empty() || q.question.len() > MAX_QUESTION {
            return Err(format!("question {} must be 1–{MAX_QUESTION} bytes", q.id));
        }
        match q.kind {
            LayaKind::Noul if !q.options.is_empty() => return Err(format!("yes/no question {} takes no options", q.id)),
            LayaKind::Noul => {}
            LayaKind::Choice => {
                if !(2..=MAX_OPTIONS).contains(&q.options.len()) {
                    return Err(format!("a decision needs 2–{MAX_OPTIONS} options, got {}", q.options.len()));
                }
                let mut seen = std::collections::HashSet::new();
                for o in &q.options {
                    if !slug(&o.id) {
                        return Err(format!("invalid option id: {:?}", o.id));
                    }
                    if !seen.insert(o.id.as_str()) {
                        return Err(format!("duplicate option id: {}", o.id));
                    }
                    if o.text.len() > MAX_OPTION_TEXT {
                        return Err(format!("option {} text is over {MAX_OPTION_TEXT} bytes", o.id));
                    }
                }
            }
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

/// Scores every question about `state` on the loaded checkpoint, in one batch.
#[tauri::command]
pub async fn laya_decide(app: AppHandle, laya: tauri::State<'_, Laya>, state: String, questions: Vec<LayaQuestion>) -> Result<LayaAnswers, String> {
    validate(&state, &questions)?;
    #[cfg(laya)]
    {
        let model = laya.loaded().ok_or("No Laya model is loaded.")?;
        let w = laya.worker(&app);
        let ids: Vec<String> = questions.iter().map(|q| q.id.clone()).collect();
        let qs = questions
            .into_iter()
            .map(|q| engine::Question {
                kind: match q.kind {
                    LayaKind::Choice => prompt::Kind::Choice,
                    LayaKind::Noul => prompt::Kind::Noul,
                },
                instructions: q.question,
                options: q.options.into_iter().map(|o| (o.id, o.text)).collect(),
            })
            .collect();
        let asked = blocking(move || w.questions(state, qs)).await?;
        let answers = ids
            .into_iter()
            .zip(asked.answers)
            .map(|(id, a)| LayaAnswer { id, probabilities: a.probabilities, input_tokens: a.input_tokens, truncated: a.truncated })
            .collect();
        Ok(LayaAnswers { answers, ms: asked.ms, model })
    }
    #[cfg(not(laya))]
    {
        let _ = (app, laya, state, questions);
        Err(UNSUPPORTED.into())
    }
}

/// Scores how likely each retrieved passage helps answer `request`, one
/// `noul` row per passage, batched. The webview decides what to drop.
#[tauri::command]
pub async fn laya_relevance(app: AppHandle, laya: tauri::State<'_, Laya>, request: String, passages: Vec<LayaPassage>) -> Result<LayaRelevance, String> {
    validate_passages(&request, &passages)?;
    #[cfg(laya)]
    {
        let model = laya.loaded().ok_or("No Laya model is loaded.")?;
        let w = laya.worker(&app);
        let rows = passages
            .iter()
            .map(|p| (passage_state(&request, p), engine::Question { kind: prompt::Kind::Noul, instructions: RELEVANT.into(), options: vec![] }))
            .collect();
        let asked = blocking(move || w.rows(rows)).await?;
        Ok(LayaRelevance { scores: asked.answers.iter().map(|a| a.probabilities[1]).collect(), ms: asked.ms, model })
    }
    #[cfg(not(laya))]
    {
        let _ = (app, laya, request, passages);
        Err(UNSUPPORTED.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opt(id: &str, text: &str) -> LayaOption {
        LayaOption { id: id.into(), text: text.into() }
    }
    fn choice(id: &str, options: Vec<LayaOption>) -> LayaQuestion {
        LayaQuestion { id: id.into(), kind: LayaKind::Choice, question: "q?".into(), options }
    }
    fn noul(id: &str) -> LayaQuestion {
        LayaQuestion { id: id.into(), kind: LayaKind::Noul, question: "It holds.".into(), options: vec![] }
    }
    fn two() -> Vec<LayaOption> {
        vec![opt("a", "x"), opt("b", "y")]
    }

    #[test]
    fn accepts_a_choice_with_a_yes_no_question() {
        assert!(validate("state", &[choice("next", vec![opt("answer_now", "Answer"), opt("kb_search", "Search")]), noul("stop")]).is_ok());
    }

    #[test]
    fn bounds_every_input() {
        assert!(validate(&"s".repeat(MAX_STATE + 1), &[choice("c", two())]).unwrap_err().contains("state"));
        assert!(validate("s", &[]).unwrap_err().contains("1–4"));
        assert!(validate("s", &(0..5).map(|i| noul(&format!("n{i}"))).collect::<Vec<_>>()).unwrap_err().contains("1–4"));
        let mut empty = noul("n");
        empty.question.clear();
        assert!(validate("s", &[empty]).is_err());
        let mut long = noul("n");
        long.question = "q".repeat(MAX_QUESTION + 1);
        assert!(validate("s", &[long]).is_err());
        assert!(validate("s", &[choice("c", two()[..1].to_vec())]).unwrap_err().contains("2–16"));
        assert!(validate("s", &[choice("c", (0..17).map(|i| opt(&format!("o{i}"), "t")).collect())]).unwrap_err().contains("2–16"));
        assert!(validate("s", &[choice("c", vec![opt("a", "x"), opt("b", &"t".repeat(MAX_OPTION_TEXT + 1))])]).is_err());
        let mut with_options = noul("n");
        with_options.options = two();
        assert!(validate("s", &[with_options]).unwrap_err().contains("takes no options"));
    }

    #[test]
    fn ids_are_plain_slugs_and_unique() {
        for bad in ["", "Answer", "a-b", "../x", "a b", "<mask>", &"a".repeat(65)] {
            assert!(validate("s", &[choice("c", vec![opt(bad, "x"), opt("b", "y")])]).is_err(), "{bad:?}");
            assert!(validate("s", &[noul(bad)]).is_err(), "{bad:?}");
        }
        assert!(validate("s", &[choice("c", vec![opt("a", "x"), opt("a", "y")])]).unwrap_err().contains("duplicate"));
        assert!(validate("s", &[noul("q"), noul("q")]).unwrap_err().contains("duplicate"));
    }

    #[test]
    fn questions_parse_from_the_webview_shape() {
        let q: Vec<LayaQuestion> = serde_json::from_str(r#"[{"id":"next","kind":"choice","question":"q?","options":[{"id":"a","text":"x"},{"id":"b","text":"y"}]},{"id":"stop","kind":"noul","question":"It holds."}]"#).unwrap();
        assert_eq!((q[0].kind, q[1].kind, q[1].options.len()), (LayaKind::Choice, LayaKind::Noul, 0));
        assert!(serde_json::from_str::<Vec<LayaQuestion>>(r#"[{"id":"x","kind":"score","question":"q"}]"#).is_err(), "only choice and noul");
    }

    fn passage(source: &str, text: &str) -> LayaPassage {
        LayaPassage { source: source.into(), text: text.into() }
    }

    #[test]
    fn bounds_the_relevance_check() {
        assert!(validate_passages("q?", &[passage("a.md:1-3", "text")]).is_ok());
        assert!(validate_passages("  ", &[passage("a", "t")]).is_err());
        assert!(validate_passages(&"q".repeat(MAX_REQUEST + 1), &[passage("a", "t")]).is_err());
        assert!(validate_passages("q", &[]).unwrap_err().contains("1–24"));
        assert!(validate_passages("q", &vec![passage("a", "t"); MAX_PASSAGES + 1]).is_err());
        assert!(validate_passages("q", &[passage("a", &"t".repeat(MAX_PASSAGE + 1))]).is_err());
        assert!(validate_passages("q", &[passage(&"s".repeat(MAX_SOURCE + 1), "t")]).is_err());
    }

    #[test]
    fn a_passage_is_judged_after_the_request() {
        assert_eq!(passage_state(" q? ", &passage("a.md:1-3", " body ")), "User request:\nq?\n\nPassage from a.md:1-3:\nbody");
    }

    #[cfg(not(laya))]
    #[test]
    fn says_it_is_unsupported_off_apple_silicon() {
        assert!(UNSUPPORTED.contains("Apple Silicon"));
    }
}
