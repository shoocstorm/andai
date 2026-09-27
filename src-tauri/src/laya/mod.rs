//! The Laya decision model: a small bidirectional encoder that scores a
//! choice between options in one forward pass (~10–20 ms, versus ~0.7 s for a
//! letter readout on the chat model). It runs on MLX, so only on Apple Silicon
//! (`cfg(mlx)`, set by build.rs); elsewhere these commands say so and the
//! webview keeps deciding with wllama (llm/decide.ts). It shares the MLX
//! thread (mlx.rs) with the native chat models (llm/).
//!
//! The webview is untrusted (AGENTS.md §9): checkpoints and files are named
//! from the closed catalog, downloads only land after their sha256 matches
//! (store.rs), and decision inputs are bounded here.

// The catalog and store are shared with the native chat models (llm/).
#[cfg_attr(not(mlx), allow(dead_code))]
pub(crate) mod catalog;
#[cfg_attr(not(mlx), allow(dead_code))]
pub(crate) mod store;

#[cfg(mlx)]
pub(crate) mod engine;
#[cfg(mlx)]
mod model;
#[cfg(mlx)]
mod prompt;

use crate::mlx::Mlx;
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, State};

#[cfg_attr(mlx, allow(dead_code))]
const UNSUPPORTED: &str = "The Laya decision model needs an Apple Silicon Mac.";

/// A catalog checkpoint and whether it's downloaded (Laya and llm/ share it).
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CheckpointStatus {
    pub id: String,
    pub repo: String,
    pub commit: String,
    pub bytes: u64,
    /// The files the webview downloads (a custom model's small JSON files came with it).
    pub files: Vec<FileInfo>,
    pub downloaded: bool,
}

#[derive(Debug, Serialize)]
pub struct FileInfo {
    path: String,
    bytes: u64,
}

impl CheckpointStatus {
    #[cfg_attr(not(mlx), allow(dead_code))]
    pub(crate) fn of(root: &std::path::Path, c: &catalog::Checkpoint) -> Self {
        CheckpointStatus {
            id: c.id.to_string(),
            repo: c.repo.to_string(),
            commit: c.commit.to_string(),
            bytes: c.bytes(),
            files: c.files.iter().map(|f| FileInfo { path: f.path.to_string(), bytes: f.bytes }).collect(),
            downloaded: store::is_downloaded(root, c),
        }
    }
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
#[cfg_attr(not(mlx), allow(dead_code))]
pub const RELEVANT: &str = "This passage contains information that helps answer the user's request.";
/// The claim check, asked of each cited sentence against the passage it cites
/// (a `noul`). Probed on the item 10 eval answers (60 cited sentences, each
/// with its own passage and one cited for another question): AUC 0.67
/// multilingual, 0.82 English (AGENTS.md §2).
#[cfg_attr(not(mlx), allow(dead_code))]
pub const SUPPORTS: &str = "The passage supports this statement.";
/// Claim check: claims per call, and the bound on each statement.
pub const MAX_CLAIMS: usize = 24;
pub const MAX_STATEMENT: usize = 2 * 1024;
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
#[cfg_attr(not(mlx), allow(dead_code))]
pub fn passage_state(request: &str, p: &LayaPassage) -> String {
    format!("User request:\n{}\n\nPassage from {}:\n{}", request.trim(), p.source, p.text.trim())
}

#[derive(Debug, Clone, Deserialize)]
pub struct LayaClaim {
    /// One sentence of the answer.
    statement: String,
    /// The passage it cites: where it's from, and its text.
    source: String,
    text: String,
}

pub fn validate_claims(claims: &[LayaClaim]) -> Result<(), String> {
    if !(1..=MAX_CLAIMS).contains(&claims.len()) {
        return Err(format!("check 1–{MAX_CLAIMS} claims at a time, got {}", claims.len()));
    }
    for (i, c) in claims.iter().enumerate() {
        if c.statement.trim().is_empty() || c.statement.len() > MAX_STATEMENT {
            return Err(format!("claim {} must be 1–{MAX_STATEMENT} bytes", i + 1));
        }
        if c.source.len() > MAX_SOURCE || c.text.len() > MAX_PASSAGE {
            return Err(format!("claim {}'s passage is over {MAX_PASSAGE} bytes (or its source over {MAX_SOURCE})", i + 1));
        }
    }
    Ok(())
}

/// The state one claim is judged in: the statement, then the passage it cites.
#[cfg_attr(not(mlx), allow(dead_code))]
pub fn claim_state(c: &LayaClaim) -> String {
    format!("Statement:\n{}\n\nPassage from {}:\n{}", c.statement.trim(), c.source, c.text.trim())
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

#[cfg(mlx)]
fn root(app: &AppHandle) -> Result<std::path::PathBuf, String> {
    crate::ug::data_dir(app, &["models", "laya"])
}

#[cfg(mlx)]
const NOT_LOADED: &str = "No Laya model is loaded.";

#[cfg(mlx)]
fn loaded(m: &mut crate::mlx::Models) -> Result<&mut engine::Engine, String> {
    m.laya.as_mut().map(|(_, e)| e).ok_or_else(|| NOT_LOADED.to_string())
}

#[cfg(mlx)]
pub(crate) async fn blocking<T: Send + 'static>(f: impl FnOnce() -> Result<T, String> + Send + 'static) -> Result<T, String> {
    tauri::async_runtime::spawn_blocking(f).await.map_err(|e| e.to_string())?
}

#[tauri::command]
pub async fn laya_status(app: AppHandle, mlx: State<'_, Mlx>) -> Result<LayaStatus, String> {
    #[cfg(mlx)]
    {
        let root = root(&app)?;
        let checkpoints = catalog::CHECKPOINTS.iter().map(|c| CheckpointStatus::of(&root, c)).collect();
        Ok(LayaStatus { supported: true, loaded: mlx.loaded().laya, checkpoints })
    }
    #[cfg(not(mlx))]
    {
        let _ = (app, mlx);
        Ok(LayaStatus { supported: false, loaded: None, checkpoints: vec![] })
    }
}

/// One chunk of a checkpoint download, as a raw IPC body. Headers name it:
/// `x-checkpoint`, `x-file` (catalog path) and `x-offset`. Shared by the Laya
/// and native chat model downloads.
pub(crate) fn chunk_request(request: &tauri::ipc::Request<'_>) -> Result<(String, String, u64, Vec<u8>), String> {
    let header = |k: &str| request.headers().get(k).and_then(|v| v.to_str().ok()).map(str::to_string).ok_or_else(|| format!("missing {k} header"));
    let (id, file, offset) = (header("x-checkpoint")?, header("x-file")?, header("x-offset")?);
    let offset: u64 = offset.parse().map_err(|_| "x-offset must be a byte offset".to_string())?;
    let tauri::ipc::InvokeBody::Raw(bytes) = request.body() else {
        return Err("a chunk must be sent as raw bytes".into());
    };
    if bytes.len() > MAX_CHUNK {
        return Err(format!("a chunk is at most {MAX_CHUNK} bytes"));
    }
    Ok((id, file, offset, bytes.clone()))
}

#[tauri::command]
pub async fn laya_write_chunk(app: AppHandle, request: tauri::ipc::Request<'_>) -> Result<u64, String> {
    let (id, file, offset, bytes) = chunk_request(&request)?;
    #[cfg(mlx)]
    {
        let c = catalog::checkpoint(&id)?;
        let root = root(&app)?;
        blocking(move || store::write_chunk(&root, c, &file, offset, &bytes)).await
    }
    #[cfg(not(mlx))]
    {
        let _ = (app, id, file, offset, bytes);
        Err(UNSUPPORTED.into())
    }
}

/// Verifies a finished download and moves it into place (store.rs).
#[tauri::command]
pub async fn laya_finish(app: AppHandle, checkpoint: String) -> Result<(), String> {
    #[cfg(mlx)]
    {
        let c = catalog::checkpoint(&checkpoint)?;
        let root = root(&app)?;
        blocking(move || store::finish(&root, c)).await
    }
    #[cfg(not(mlx))]
    {
        let _ = (app, checkpoint);
        Err(UNSUPPORTED.into())
    }
}

/// Deletes a downloaded checkpoint (the UI confirms first), unloading it if loaded.
#[tauri::command]
pub async fn laya_remove(app: AppHandle, mlx: State<'_, Mlx>, checkpoint: String) -> Result<(), String> {
    #[cfg(mlx)]
    {
        let c = catalog::checkpoint(&checkpoint)?;
        if mlx.loaded().laya.as_deref() == Some(&*c.id) {
            let t = mlx.thread(&app);
            blocking(move || {
                t.run(|m| {
                    m.laya = None;
                    Ok(())
                })
            })
            .await?;
        }
        let root = root(&app)?;
        blocking(move || store::remove(&root, c)).await
    }
    #[cfg(not(mlx))]
    {
        let _ = (app, mlx, checkpoint);
        Err(UNSUPPORTED.into())
    }
}

/// Loads a downloaded checkpoint; returns how long it took, in ms.
#[tauri::command]
pub async fn laya_load(app: AppHandle, mlx: State<'_, Mlx>, checkpoint: String) -> Result<f64, String> {
    #[cfg(mlx)]
    {
        let c = catalog::checkpoint(&checkpoint)?;
        let dir = store::resolve(&root(&app)?, c)?;
        let t = mlx.thread(&app);
        blocking(move || {
            t.run(move |m| {
                let started = std::time::Instant::now();
                m.laya = None;
                m.laya = Some((c.id.to_string(), engine::Engine::load(&dir)?));
                Ok(started.elapsed().as_secs_f64() * 1e3)
            })
        })
        .await
    }
    #[cfg(not(mlx))]
    {
        let _ = (app, mlx, checkpoint);
        Err(UNSUPPORTED.into())
    }
}

#[tauri::command]
pub async fn laya_unload(app: AppHandle, mlx: State<'_, Mlx>) -> Result<(), String> {
    #[cfg(mlx)]
    {
        let t = mlx.thread(&app);
        blocking(move || {
            t.run(|m| {
                m.laya = None;
                Ok(())
            })
        })
        .await
    }
    #[cfg(not(mlx))]
    {
        let _ = (app, mlx);
        Err(UNSUPPORTED.into())
    }
}

/// Scores every question about `state` on the loaded checkpoint, in one batch.
#[tauri::command]
pub async fn laya_decide(app: AppHandle, mlx: State<'_, Mlx>, state: String, questions: Vec<LayaQuestion>) -> Result<LayaAnswers, String> {
    validate(&state, &questions)?;
    #[cfg(mlx)]
    {
        let model = mlx.loaded().laya.ok_or(NOT_LOADED)?;
        let t = mlx.thread(&app);
        let ids: Vec<String> = questions.iter().map(|q| q.id.clone()).collect();
        let qs: Vec<engine::Question> = questions
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
        let asked = blocking(move || t.run(move |m| loaded(m)?.ask(&state, &qs))).await?;
        let answers = ids
            .into_iter()
            .zip(asked.answers)
            .map(|(id, a)| LayaAnswer { id, probabilities: a.probabilities, input_tokens: a.input_tokens, truncated: a.truncated })
            .collect();
        Ok(LayaAnswers { answers, ms: asked.ms, model })
    }
    #[cfg(not(mlx))]
    {
        let _ = (app, mlx, state, questions);
        Err(UNSUPPORTED.into())
    }
}

/// Scores how likely each retrieved passage helps answer `request`, one
/// `noul` row per passage, batched. The webview decides what to drop.
#[tauri::command]
pub async fn laya_relevance(app: AppHandle, mlx: State<'_, Mlx>, request: String, passages: Vec<LayaPassage>) -> Result<LayaRelevance, String> {
    validate_passages(&request, &passages)?;
    #[cfg(mlx)]
    {
        let model = mlx.loaded().laya.ok_or(NOT_LOADED)?;
        let t = mlx.thread(&app);
        let rows: Vec<(String, engine::Question)> = passages
            .iter()
            .map(|p| (passage_state(&request, p), engine::Question { kind: prompt::Kind::Noul, instructions: RELEVANT.into(), options: vec![] }))
            .collect();
        let asked = blocking(move || {
            t.run(move |m| {
                let rows: Vec<(&str, &engine::Question)> = rows.iter().map(|(s, q)| (s.as_str(), q)).collect();
                loaded(m)?.ask_rows(&rows)
            })
        })
        .await?;
        Ok(LayaRelevance { scores: asked.answers.iter().map(|a| a.probabilities[1]).collect(), ms: asked.ms, model })
    }
    #[cfg(not(mlx))]
    {
        let _ = (app, mlx, request, passages);
        Err(UNSUPPORTED.into())
    }
}

/// Scores how likely each cited passage supports the sentence that cites it,
/// one `noul` row per claim, batched. The webview decides what to flag.
#[tauri::command]
pub async fn laya_support(app: AppHandle, mlx: State<'_, Mlx>, claims: Vec<LayaClaim>) -> Result<LayaRelevance, String> {
    validate_claims(&claims)?;
    #[cfg(mlx)]
    {
        let model = mlx.loaded().laya.ok_or(NOT_LOADED)?;
        let t = mlx.thread(&app);
        let rows: Vec<(String, engine::Question)> = claims
            .iter()
            .map(|c| (claim_state(c), engine::Question { kind: prompt::Kind::Noul, instructions: SUPPORTS.into(), options: vec![] }))
            .collect();
        let asked = blocking(move || {
            t.run(move |m| {
                let rows: Vec<(&str, &engine::Question)> = rows.iter().map(|(s, q)| (s.as_str(), q)).collect();
                loaded(m)?.ask_rows(&rows)
            })
        })
        .await?;
        Ok(LayaRelevance { scores: asked.answers.iter().map(|a| a.probabilities[1]).collect(), ms: asked.ms, model })
    }
    #[cfg(not(mlx))]
    {
        let _ = (app, mlx, claims);
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

    fn claim(statement: &str, text: &str) -> LayaClaim {
        LayaClaim { statement: statement.into(), source: "a.md:1-3".into(), text: text.into() }
    }

    #[test]
    fn bounds_the_claim_check() {
        assert!(validate_claims(&[claim("It is 48 hours.", "text")]).is_ok());
        assert!(validate_claims(&[]).unwrap_err().contains("1–24"));
        assert!(validate_claims(&vec![claim("s", "t"); MAX_CLAIMS + 1]).is_err());
        assert!(validate_claims(&[claim("  ", "t")]).is_err());
        assert!(validate_claims(&[claim(&"s".repeat(MAX_STATEMENT + 1), "t")]).is_err());
        assert!(validate_claims(&[claim("s", &"t".repeat(MAX_PASSAGE + 1))]).is_err());
        let mut long_source = claim("s", "t");
        long_source.source = "s".repeat(MAX_SOURCE + 1);
        assert!(validate_claims(&[long_source]).is_err());
    }

    #[test]
    fn a_claim_is_judged_before_its_passage() {
        assert_eq!(claim_state(&claim(" It is 48. ", " body ")), "Statement:\nIt is 48.\n\nPassage from a.md:1-3:\nbody");
    }

    #[cfg(not(mlx))]
    #[test]
    fn says_it_is_unsupported_off_apple_silicon() {
        assert!(UNSUPPORTED.contains("Apple Silicon"));
    }
}
