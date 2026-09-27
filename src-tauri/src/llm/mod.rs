//! Native chat models: Qwen3 on MLX, in-process on the shared MLX thread
//! (mlx.rs). On an M5 Max, Qwen3 1.7B writes about 350 tok/s here, against
//! 30–65 tok/s for the same model in wllama on WebGPU (docs/performance.md).
//! Apple Silicon only (`cfg(mlx)`); everywhere else the webview keeps wllama.
//!
//! Two slots, so a small model can decide while a larger one answers, as
//! with wllama (llm/engine.ts): `chat` and `decider`.
//!
//! The webview is untrusted (AGENTS.md §9): checkpoints and files come from
//! the closed catalog, downloads only land after their sha256 matches
//! (laya/store.rs), and every generation input is bounded here.

#[cfg_attr(not(mlx), allow(dead_code))]
pub(crate) mod catalog;
#[cfg_attr(not(mlx), allow(dead_code))]
pub(crate) mod config;
#[cfg_attr(not(mlx), allow(dead_code))]
pub(crate) mod custom;
#[cfg_attr(not(mlx), allow(dead_code))]
mod grammar;
#[cfg_attr(not(mlx), allow(dead_code))]
pub(crate) mod template;

#[cfg(mlx)]
pub(crate) mod engine;
#[cfg(mlx)]
mod model;

use crate::laya::CheckpointStatus;
use crate::mlx::Mlx;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use tauri::ipc::Channel;
use tauri::{AppHandle, State};
use template::Message;

#[cfg_attr(mlx, allow(dead_code))]
const UNSUPPORTED: &str = "MLX models need an Apple Silicon Mac.";

pub const MAX_MESSAGES: usize = 512;
/// All message text together; the context window runs out well before this.
pub const MAX_PROMPT_BYTES: usize = 512 * 1024;
pub const MAX_NEW_TOKENS: u32 = 8192;
pub const MAX_TOP_K: u32 = 1000;
pub const MAX_TOP_LOGPROBS: u32 = 20;
pub const MAX_GRAMMAR: usize = 8 * 1024;
/// Context window a model is loaded with; the KV cache grows as it's used.
pub const CTX_RANGE: std::ops::RangeInclusive<u32> = 512..=32_768;

/// Which loaded model a request runs on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Slot {
    Chat,
    Decider,
}

impl Slot {
    #[cfg_attr(not(mlx), allow(dead_code))]
    fn index(self) -> usize {
        match self {
            Slot::Chat => 0,
            Slot::Decider => 1,
        }
    }
}

/// Cancellation: the webview names each generation with a random id, and
/// `llm_cancel` marks one; the loop checks after every token.
#[derive(Default)]
pub struct Llm {
    cancelled: Arc<AtomicU32>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenParams {
    pub max_tokens: u32,
    pub temperature: f32,
    #[serde(default)]
    pub top_k: u32,
    #[serde(default = "one")]
    pub top_p: f32,
    #[serde(default)]
    pub seed: Option<u64>,
    /// Qwen3's reasoning block (`enable_thinking`).
    #[serde(default)]
    pub thinking: bool,
    #[serde(default)]
    pub top_logprobs: u32,
    /// GBNF without recursion (grammar.rs): a choice between literals, or a JSON shape.
    #[serde(default)]
    pub grammar: Option<String>,
    /// Reuse the KV cache of the previous prompt's shared prefix.
    #[serde(default = "yes")]
    pub cache_prompt: bool,
}

fn one() -> f32 {
    1.0
}
fn yes() -> bool {
    true
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LlmStatus {
    supported: bool,
    chat: Option<String>,
    decider: Option<String>,
    checkpoints: Vec<LlmCheckpoint>,
}

/// A catalog model, or one the user added (`custom`, with what Rust read from it).
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LlmCheckpoint {
    #[serde(flatten)]
    status: CheckpointStatus,
    custom: Option<CustomInfo>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CustomInfo {
    thinking: bool,
    layers: usize,
    bits: i32,
    added_at: u64,
}

#[cfg(mlx)]
fn custom_status(root: &std::path::Path, m: &custom::Manifest) -> LlmCheckpoint {
    let mut status = CheckpointStatus::of(root, &m.checkpoint());
    status.bytes = m.bytes();
    status.downloaded = custom::is_complete(root, m);
    LlmCheckpoint { status, custom: Some(CustomInfo { thinking: m.thinking, layers: m.layers, bits: m.bits, added_at: m.added_at }) }
}

/// A model by id: the closed catalog, else a model the user added (its manifest re-validated).
#[cfg(mlx)]
fn lookup(root: &std::path::Path, id: &str) -> Result<(crate::laya::catalog::Checkpoint, Option<custom::Manifest>), String> {
    match catalog::checkpoint(id) {
        Ok(c) => Ok((c.clone(), None)),
        Err(_) => {
            let m = custom::get(root, id)?;
            Ok((m.checkpoint(), Some(m)))
        }
    }
}

/// Bounds on what the webview may ask the model to generate.
pub fn validate(messages: &[Message], p: &GenParams) -> Result<(), String> {
    if messages.is_empty() || messages.len() > MAX_MESSAGES {
        return Err(format!("send 1–{MAX_MESSAGES} messages, got {}", messages.len()));
    }
    let bytes: usize = messages.iter().map(|m| m.content.len()).sum();
    if bytes > MAX_PROMPT_BYTES {
        return Err(format!("the messages are over {MAX_PROMPT_BYTES} bytes"));
    }
    if messages.iter().any(|m| m.content.contains('\0')) {
        return Err("a message contains a NUL byte".into());
    }
    if !(1..=MAX_NEW_TOKENS).contains(&p.max_tokens) {
        return Err(format!("max_tokens must be 1–{MAX_NEW_TOKENS}"));
    }
    if !(p.temperature.is_finite() && (0.0..=2.0).contains(&p.temperature)) {
        return Err("temperature must be 0–2".into());
    }
    if p.top_k > MAX_TOP_K {
        return Err(format!("top_k must be 0–{MAX_TOP_K}"));
    }
    if !(p.top_p.is_finite() && p.top_p > 0.0 && p.top_p <= 1.0) {
        return Err("top_p must be in (0, 1]".into());
    }
    if p.top_logprobs > MAX_TOP_LOGPROBS {
        return Err(format!("top_logprobs must be 0–{MAX_TOP_LOGPROBS}"));
    }
    if p.grammar.as_ref().is_some_and(|g| g.len() > MAX_GRAMMAR) {
        return Err(format!("the grammar is over {MAX_GRAMMAR} bytes"));
    }
    Ok(())
}

#[cfg(mlx)]
fn root(app: &AppHandle) -> Result<std::path::PathBuf, String> {
    crate::ug::data_dir(app, &["models", "llm"])
}

#[cfg(mlx)]
use crate::laya::{blocking, store};

#[tauri::command]
pub async fn llm_status(app: AppHandle, mlx: State<'_, Mlx>) -> Result<LlmStatus, String> {
    #[cfg(mlx)]
    {
        let root = root(&app)?;
        let r = root.clone();
        let mut checkpoints: Vec<LlmCheckpoint> = catalog::CHECKPOINTS.iter().map(|c| LlmCheckpoint { status: CheckpointStatus::of(&root, c), custom: None }).collect();
        // Hashing a custom model's small files is cheap, but its folder is on disk: off the async runtime.
        checkpoints.extend(blocking(move || Ok(custom::list(&r).iter().map(|m| custom_status(&r, m)).collect::<Vec<_>>())).await?);
        let [chat, decider] = mlx.loaded().llm;
        Ok(LlmStatus { supported: true, chat, decider, checkpoints })
    }
    #[cfg(not(mlx))]
    {
        let _ = (app, mlx);
        Ok(LlmStatus { supported: false, chat: None, decider: None, checkpoints: vec![] })
    }
}

#[tauri::command]
pub async fn llm_write_chunk(app: AppHandle, request: tauri::ipc::Request<'_>) -> Result<u64, String> {
    let (id, file, offset, bytes) = crate::laya::chunk_request(&request)?;
    #[cfg(mlx)]
    {
        let root = root(&app)?;
        blocking(move || {
            let (c, _) = lookup(&root, &id)?;
            store::write_chunk(&root, &c, &file, offset, &bytes)
        })
        .await
    }
    #[cfg(not(mlx))]
    {
        let _ = (app, id, file, offset, bytes);
        Err(UNSUPPORTED.into())
    }
}

/// Verifies a finished download and moves it into place (laya/store.rs).
#[tauri::command]
pub async fn llm_finish(app: AppHandle, checkpoint: String) -> Result<(), String> {
    #[cfg(mlx)]
    {
        let root = root(&app)?;
        blocking(move || {
            let (c, _) = lookup(&root, &checkpoint)?;
            store::finish(&root, &c)
        })
        .await
    }
    #[cfg(not(mlx))]
    {
        let _ = (app, checkpoint);
        Err(UNSUPPORTED.into())
    }
}

/// Deletes a downloaded model (the UI confirms first), unloading it from any
/// slot. A model the user added is forgotten too.
#[tauri::command]
pub async fn llm_remove(app: AppHandle, mlx: State<'_, Mlx>, checkpoint: String) -> Result<(), String> {
    #[cfg(mlx)]
    {
        let root = root(&app)?;
        let (c, custom) = lookup(&root, &checkpoint)?;
        if mlx.loaded().llm.iter().any(|l| l.as_deref() == Some(&*c.id)) {
            let t = mlx.thread(&app);
            let id = c.id.to_string();
            blocking(move || {
                t.run(move |m| {
                    for s in &mut m.llm {
                        if s.as_ref().is_some_and(|(loaded, _)| *loaded == id) {
                            *s = None;
                        }
                    }
                    Ok(())
                })
            })
            .await?;
        }
        blocking(move || match custom {
            Some(m) => custom::remove(&root, &m.id),
            None => store::remove(&root, &c),
        })
        .await
    }
    #[cfg(not(mlx))]
    {
        let _ = (app, mlx, checkpoint);
        Err(UNSUPPORTED.into())
    }
}

/// Adds a model from Hugging Face (custom.rs checks everything first); the
/// webview then downloads its `files` like a catalog model's.
#[tauri::command]
pub async fn llm_add_custom(app: AppHandle, spec: custom::Spec) -> Result<LlmCheckpoint, String> {
    #[cfg(mlx)]
    {
        let root = root(&app)?;
        blocking(move || {
            let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
            let m = custom::add(&root, &spec, now)?;
            Ok(custom_status(&root, &m))
        })
        .await
    }
    #[cfg(not(mlx))]
    {
        let _ = (app, spec);
        Err(UNSUPPORTED.into())
    }
}

/// What a load reports: its time, and facts about the model for Settings.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Loaded {
    ms: f64,
    layers: usize,
    /// Weight bits, e.g. 4 or 8.
    bits: i32,
    n_ctx: u32,
}

/// Loads a downloaded model into `slot`.
#[tauri::command]
pub async fn llm_load(app: AppHandle, mlx: State<'_, Mlx>, slot: Slot, checkpoint: String, n_ctx: u32) -> Result<Loaded, String> {
    if !CTX_RANGE.contains(&n_ctx) {
        return Err(format!("the context must be {}–{} tokens", CTX_RANGE.start(), CTX_RANGE.end()));
    }
    #[cfg(mlx)]
    {
        let root = root(&app)?;
        let (c, custom) = lookup(&root, &checkpoint)?;
        let dir = store::resolve(&root, &c)?;
        if custom.as_ref().is_some_and(|m| !custom::is_complete(&root, m)) {
            return Err(format!("{} isn't completely downloaded; download it again in Settings → Models.", c.repo));
        }
        let t = mlx.thread(&app);
        blocking(move || {
            t.run(move |m| {
                let started = std::time::Instant::now();
                m.llm[slot.index()] = None;
                let engine = engine::Engine::load(&dir, n_ctx as usize)?;
                let (layers, bits) = engine.shape();
                m.llm[slot.index()] = Some((c.id.to_string(), engine));
                Ok(Loaded { ms: started.elapsed().as_secs_f64() * 1e3, layers, bits, n_ctx })
            })
        })
        .await
    }
    #[cfg(not(mlx))]
    {
        let _ = (app, mlx, slot, checkpoint, Loaded { ms: 0.0, layers: 0, bits: 0, n_ctx });
        Err(UNSUPPORTED.into())
    }
}

#[tauri::command]
pub async fn llm_unload(app: AppHandle, mlx: State<'_, Mlx>, slot: Slot) -> Result<(), String> {
    #[cfg(mlx)]
    {
        let t = mlx.thread(&app);
        blocking(move || {
            t.run(move |m| {
                m.llm[slot.index()] = None;
                Ok(())
            })
        })
        .await
    }
    #[cfg(not(mlx))]
    {
        let _ = (app, mlx, slot);
        Err(UNSUPPORTED.into())
    }
}

/// Generates a reply to `messages` on the model in `slot`, streaming text to
/// `on_text`; the result carries the whole text, token counts and timings.
#[allow(clippy::too_many_arguments)]
#[tauri::command]
pub async fn llm_generate(
    app: AppHandle,
    mlx: State<'_, Mlx>,
    llm: State<'_, Llm>,
    slot: Slot,
    request: u32,
    messages: Vec<Message>,
    params: GenParams,
    on_text: Channel<String>,
) -> Result<serde_json::Value, String> {
    validate(&messages, &params)?;
    #[cfg(mlx)]
    {
        let p = engine::Params {
            max_tokens: params.max_tokens as usize,
            temperature: params.temperature,
            top_k: params.top_k as usize,
            top_p: params.top_p,
            seed: params.seed,
            top_logprobs: params.top_logprobs as usize,
            choices: params.grammar.as_deref().and_then(grammar::choices),
            // Any other grammar is followed through its DFA.
            grammar: params.grammar.clone().filter(|g| grammar::choices(g).is_none()),
            reuse_prefix: params.cache_prompt,
        };
        let cancelled = llm.cancelled.clone();
        let t = mlx.thread(&app);
        let out = blocking(move || {
            t.run(move |m| {
                let (_, engine) = m.llm[slot.index()].as_mut().ok_or("No MLX model is loaded; open Settings → Models to load one.")?;
                // A template without Qwen3's thinking switch gets no empty think block:
                // rendered as with thinking on, which adds none.
                let text = template::render(&messages, params.thinking || !engine.think_switch);
                let mut send = |piece: &str| {
                    let _ = on_text.send(piece.to_string());
                };
                engine.generate(&text, &p, &mut send, &|| cancelled.load(Ordering::Relaxed) == request)
            })
        })
        .await?;
        serde_json::to_value(out).map_err(|e| e.to_string())
    }
    #[cfg(not(mlx))]
    {
        let _ = (app, mlx, llm, slot, request, on_text);
        Err(UNSUPPORTED.into())
    }
}

/// Stops the generation named `request` after its current token.
#[tauri::command]
pub fn llm_cancel(llm: State<'_, Llm>, request: u32) {
    llm.cancelled.store(request, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;
    use template::Role;

    fn msg(content: &str) -> Message {
        Message { role: Role::User, content: content.into() }
    }
    fn params() -> GenParams {
        serde_json::from_str(r#"{"maxTokens":64,"temperature":0.7}"#).unwrap()
    }

    #[test]
    fn params_parse_from_the_webview_shape_with_defaults() {
        let p = params();
        assert_eq!((p.top_k, p.top_p, p.top_logprobs, p.thinking, p.cache_prompt), (0, 1.0, 0, false, true));
        assert!(p.grammar.is_none() && p.seed.is_none());
        let s: Slot = serde_json::from_str(r#""decider""#).unwrap();
        assert_eq!(s, Slot::Decider);
        assert!(serde_json::from_str::<Slot>(r#""other""#).is_err());
        assert!(serde_json::from_str::<Vec<Message>>(r#"[{"role":"tool","content":"x"}]"#).is_err(), "only system, user, assistant");
    }

    #[test]
    fn bounds_every_input() {
        assert!(validate(&[msg("hi")], &params()).is_ok());
        assert!(validate(&[], &params()).is_err());
        assert!(validate(&vec![msg("x"); MAX_MESSAGES + 1], &params()).is_err());
        assert!(validate(&[msg(&"x".repeat(MAX_PROMPT_BYTES + 1))], &params()).is_err());
        assert!(validate(&[msg("a\0b")], &params()).is_err());
        let bad = |f: fn(&mut GenParams)| {
            let mut p = params();
            f(&mut p);
            validate(&[msg("hi")], &p).is_err()
        };
        assert!(bad(|p| p.max_tokens = 0));
        assert!(bad(|p| p.max_tokens = MAX_NEW_TOKENS + 1));
        assert!(bad(|p| p.temperature = -0.1));
        assert!(bad(|p| p.temperature = f32::NAN));
        assert!(bad(|p| p.temperature = 2.5));
        assert!(bad(|p| p.top_k = MAX_TOP_K + 1));
        assert!(bad(|p| p.top_p = 0.0));
        assert!(bad(|p| p.top_p = 1.5));
        assert!(bad(|p| p.top_logprobs = MAX_TOP_LOGPROBS + 1));
        assert!(bad(|p| p.grammar = Some("x".repeat(MAX_GRAMMAR + 1))));
    }

    #[cfg(not(mlx))]
    #[test]
    fn says_it_is_unsupported_off_apple_silicon() {
        assert!(UNSUPPORTED.contains("Apple Silicon"));
    }
}
