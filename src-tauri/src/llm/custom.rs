//! MLX models the user adds from Hugging Face (Settings → Models → Add from
//! Hugging Face), beside the closed catalog (catalog.rs).
//!
//! The webview is untrusted (AGENTS.md §9) and Rust has no HTTP client, so the
//! webview looks the model up and sends what it found; Rust decides what it
//! accepts, before any weights download:
//! - a public repo pinned to a 40-hex commit;
//! - a closed set of file names (config, tokenizer, chat template, weights as
//!   safetensors), with sizes and caps; nothing that can run code;
//! - the large files (weights, tokenizer) by their sha256 (Hugging Face's LFS
//!   oid), verified by the store after download like the catalog's;
//! - the small JSON files sent inline, parsed here, and a model the native
//!   engine can run (config.rs: Qwen3, MLX-quantized, a ChatML template).
//!
//! Rust writes the manifest itself, under `<data>/models/llm/custom/<id>.json`,
//! with an id it derives from the repo and commit; loading re-reads and
//! re-validates it. Pinning and hashing prove the files are the ones
//! Hugging Face served for that commit, not that the model is any good: the
//! UI says so.

use super::config::{self, Config};
use crate::laya::catalog::{Checkpoint, CheckpointFile};
use crate::laya::store;
use crate::ug::{create_private_dir, write_private};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::borrow::Cow;
use std::fs;
use std::path::{Path, PathBuf};

/// How many models a user may add.
pub const MAX_CUSTOM: usize = 32;
const GIB: u64 = 1 << 30;
pub const MAX_FILE_BYTES: u64 = 32 * GIB;
pub const MAX_TOTAL_BYTES: u64 = 64 * GIB;
/// A file sent inline (JSON, the template, or a tokenizer that isn't in LFS).
pub const MAX_INLINE_BYTES: usize = 16 << 20;

/// A downloadable file: path, size and sha256 as Hugging Face lists them.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SpecFile {
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
}

/// A small file the webview fetched at the pinned commit and sends as text.
#[derive(Debug, Clone, Deserialize)]
pub struct InlineFile {
    pub path: String,
    pub content: String,
}

/// What the webview sends to add a model.
#[derive(Debug, Clone, Deserialize)]
pub struct Spec {
    pub repo: String,
    pub commit: String,
    pub files: Vec<SpecFile>,
    pub inline: Vec<InlineFile>,
}

/// What Rust keeps: written here, never by the webview.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Manifest {
    pub id: String,
    pub repo: String,
    pub commit: String,
    /// Downloaded by the webview and verified by the store.
    pub files: Vec<SpecFile>,
    /// Written by Rust when the model was added; checked on every load.
    pub inline: Vec<SpecFile>,
    /// The chat template has Qwen3's `enable_thinking` switch.
    pub thinking: bool,
    pub layers: usize,
    pub bits: i32,
    pub added_at: u64,
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

pub fn valid_repo(repo: &str) -> bool {
    let part = |p: &str| {
        !p.is_empty() && p.len() <= 96 && p != "." && p != ".." && p.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
    };
    matches!(repo.split_once('/'), Some((owner, name)) if part(owner) && part(name) && !name.contains('/'))
}

fn valid_commit(commit: &str) -> bool {
    commit.len() == 40 && commit.chars().all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c))
}

fn valid_sha256(h: &str) -> bool {
    h.len() == 64 && h.chars().all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c))
}

/// `model-00001-of-00004.safetensors`: a shard of sharded weights.
pub fn is_shard_name(name: &str) -> bool {
    let Some(rest) = name.strip_prefix("model-").and_then(|r| r.strip_suffix(".safetensors")) else { return false };
    let Some((i, n)) = rest.split_once("-of-") else { return false };
    i.len() == 5 && n.len() == 5 && i.chars().chain(n.chars()).all(|c| c.is_ascii_digit()) && i <= n && i != "00000"
}

/// Files that may be downloaded (by sha256).
fn downloadable(path: &str) -> bool {
    path == "model.safetensors" || path == "tokenizer.json" || is_shard_name(path)
}

/// Files that may come inline.
fn inline_ok(path: &str) -> bool {
    matches!(path, "config.json" | "tokenizer_config.json" | "tokenizer.json" | "chat_template.jinja" | "model.safetensors.index.json")
}

/// The id Rust gives a repo at a commit: `hf-<owner>--<name>-<commit[..8]>`, lowercase.
pub fn id_for(repo: &str, commit: &str) -> String {
    let slug: String = repo
        .to_ascii_lowercase()
        .replace('/', "--")
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '.' { c } else { '-' })
        .collect();
    format!("hf-{}-{}", &slug[..slug.len().min(100)], &commit[..8])
}

fn valid_id(id: &str) -> bool {
    id.starts_with("hf-") && id.len() <= 128 && !id.contains("..") && id.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '.')
}

/// Checks everything the webview sent; returns the manifest to keep.
pub fn validate(spec: &Spec, now: u64) -> Result<Manifest, String> {
    if !valid_repo(&spec.repo) {
        return Err(format!("{:?} isn't a Hugging Face repository name", spec.repo));
    }
    if !valid_commit(&spec.commit) {
        return Err("the model must be pinned to a full commit".into());
    }
    let mut seen = std::collections::HashSet::new();
    for path in spec.files.iter().map(|f| &f.path).chain(spec.inline.iter().map(|f| &f.path)) {
        if !seen.insert(path.as_str()) {
            return Err(format!("{path} is listed twice"));
        }
    }
    let mut total = 0u64;
    for f in &spec.files {
        if !downloadable(&f.path) {
            return Err(format!("{} isn't a file the native engine downloads", f.path));
        }
        if f.bytes == 0 || f.bytes > MAX_FILE_BYTES {
            return Err(format!("{} is {} bytes; files may be up to {} GB", f.path, f.bytes, MAX_FILE_BYTES / GIB));
        }
        if !valid_sha256(&f.sha256) {
            return Err(format!("{} has no sha256 to verify it by", f.path));
        }
        total += f.bytes;
    }
    let mut inline = Vec::new();
    for f in &spec.inline {
        if !inline_ok(&f.path) {
            return Err(format!("{} can't be sent inline", f.path));
        }
        if f.content.len() > MAX_INLINE_BYTES {
            return Err(format!("{} is too large to send inline", f.path));
        }
        if f.path.ends_with(".json") {
            serde_json::from_str::<Value>(&f.content).map_err(|err| format!("{} isn't valid JSON: {err}", f.path))?;
        }
        total += f.content.len() as u64;
        inline.push(SpecFile { path: f.path.clone(), bytes: f.content.len() as u64, sha256: sha256_hex(f.content.as_bytes()) });
    }
    if total > MAX_TOTAL_BYTES {
        return Err(format!("the model is {} GB; up to {} GB is supported", total / GIB, MAX_TOTAL_BYTES / GIB));
    }
    let text = |p: &str| spec.inline.iter().find(|f| f.path == p).map(|f| f.content.as_str());
    let json = |p: &str| -> Result<Value, String> { serde_json::from_str(text(p).ok_or_else(|| format!("{p} is missing"))?).map_err(|err| err.to_string()) };

    let cfg = Config::from_json(&json("config.json")?).map_err(|why| format!("The native engine can't run this model: {why}."))?;
    let tcfg = json("tokenizer_config.json")?;
    config::eos_token(&tcfg).ok_or("tokenizer_config.json names no end-of-turn token")?;
    let template = config::chat_template(&tcfg, text("chat_template.jinja"));
    let thinking = config::check_template(template.as_deref()).map_err(|why| format!("The native engine can't prompt this model: {why}."))?;
    if !seen.contains("tokenizer.json") {
        return Err("tokenizer.json is missing".into());
    }
    // Weights: one file, or an index and exactly the shards it names.
    let shards: Vec<&str> = spec.files.iter().map(|f| f.path.as_str()).filter(|p| is_shard_name(p)).collect();
    match (seen.contains("model.safetensors"), text("model.safetensors.index.json")) {
        (true, None) if shards.is_empty() => {}
        (false, Some(index)) => {
            let v: Value = serde_json::from_str(index).map_err(|err| err.to_string())?;
            let mut named: Vec<&str> = v["weight_map"].as_object().ok_or("the weight index has no weight_map")?.values().filter_map(Value::as_str).collect();
            named.sort_unstable();
            named.dedup();
            let mut listed = shards.clone();
            listed.sort_unstable();
            if named.is_empty() || named != listed {
                return Err("the weight files don't match the weight index".into());
            }
        }
        _ => return Err("the weights must be model.safetensors, or an index with its shards".into()),
    }
    Ok(Manifest {
        id: id_for(&spec.repo, &spec.commit),
        repo: spec.repo.clone(),
        commit: spec.commit.clone(),
        files: spec.files.clone(),
        inline,
        thinking,
        layers: cfg.layers,
        bits: cfg.bits,
        added_at: now,
    })
}

impl Manifest {
    /// The downloadable part, for the store (write_chunk, finish, is_downloaded).
    pub fn checkpoint(&self) -> Checkpoint {
        Checkpoint {
            id: Cow::Owned(self.id.clone()),
            repo: Cow::Owned(self.repo.clone()),
            commit: Cow::Owned(self.commit.clone()),
            files: Cow::Owned(
                self.files
                    .iter()
                    .map(|f| CheckpointFile { path: Cow::Owned(f.path.clone()), bytes: f.bytes, sha256: Cow::Owned(f.sha256.clone()) })
                    .collect(),
            ),
        }
    }

    pub fn bytes(&self) -> u64 {
        self.files.iter().chain(&self.inline).map(|f| f.bytes).sum()
    }
}

fn manifests(root: &Path) -> PathBuf {
    root.join("custom")
}

fn manifest_path(root: &Path, id: &str) -> Result<PathBuf, String> {
    if !valid_id(id) {
        return Err(format!("unknown model: {id}"));
    }
    Ok(manifests(root).join(format!("{id}.json")))
}

/// A kept manifest, re-validated (a hand-edited or damaged one isn't loaded).
pub fn get(root: &Path, id: &str) -> Result<Manifest, String> {
    let text = fs::read_to_string(manifest_path(root, id)?).map_err(|_| format!("unknown model: {id}"))?;
    let m: Manifest = serde_json::from_str(&text).map_err(|err| format!("{id}: damaged manifest ({err})"))?;
    let sane = m.id == id
        && m.id == id_for(&m.repo, &m.commit)
        && valid_repo(&m.repo)
        && valid_commit(&m.commit)
        && m.files.iter().all(|f| downloadable(&f.path) && valid_sha256(&f.sha256) && f.bytes > 0 && f.bytes <= MAX_FILE_BYTES)
        && m.inline.iter().all(|f| inline_ok(&f.path) && valid_sha256(&f.sha256) && f.bytes as usize <= MAX_INLINE_BYTES);
    if !sane {
        return Err(format!("{id}: damaged manifest"));
    }
    Ok(m)
}

/// Every kept manifest that still validates, oldest first.
pub fn list(root: &Path) -> Vec<Manifest> {
    let mut out: Vec<Manifest> = fs::read_dir(manifests(root))
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| e.file_name().to_str()?.strip_suffix(".json").map(str::to_string))
        .filter_map(|id| get(root, &id).ok())
        .collect();
    out.sort_by_key(|m| m.added_at);
    out
}

/// Keeps a validated model: writes its inline files into its folder and the
/// manifest. Adding the same repo and commit again is fine (it rewrites them).
pub fn add(root: &Path, spec: &Spec, now: u64) -> Result<Manifest, String> {
    let m = validate(spec, now)?;
    let existing = list(root);
    let again = existing.iter().find(|e| e.id == m.id);
    if again.is_none() && existing.len() >= MAX_CUSTOM {
        return Err(format!("You can add up to {MAX_CUSTOM} models; remove one in Settings first."));
    }
    let m = Manifest { added_at: again.map_or(now, |e| e.added_at), ..m };
    let dir = store::dir(root, &m.checkpoint());
    create_private_dir(&dir).map_err(|err| err.to_string())?;
    for f in &spec.inline {
        write_private(&dir.join(&f.path), f.content.as_bytes()).map_err(|err| format!("{}: {err}", f.path))?;
    }
    create_private_dir(&manifests(root)).map_err(|err| err.to_string())?;
    let text = serde_json::to_string_pretty(&m).map_err(|err| err.to_string())?;
    write_private(&manifest_path(root, &m.id)?, text.as_bytes()).map_err(|err| err.to_string())?;
    Ok(m)
}

/// Downloaded, verified, and its inline files still as Rust wrote them.
pub fn is_complete(root: &Path, m: &Manifest) -> bool {
    let c = m.checkpoint();
    let dir = store::dir(root, &c);
    store::is_downloaded(root, &c) && m.inline.iter().all(|f| fs::read(dir.join(&f.path)).is_ok_and(|b| b.len() as u64 == f.bytes && sha256_hex(&b) == f.sha256))
}

/// Forgets a model: its files and its manifest.
pub fn remove(root: &Path, id: &str) -> Result<(), String> {
    let m = get(root, id)?;
    store::remove(root, &m.checkpoint())?;
    match fs::remove_file(manifest_path(root, id)?) {
        Err(err) if err.kind() != std::io::ErrorKind::NotFound => Err(err.to_string()),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const SHA: &str = "0e86d9677e519323849eac1bc272caae88567a481ff188c431f70be543d9995f";
    const COMMIT: &str = "3b1b1768f8f8cf8351c712464f906e86c2b8269e";

    fn config() -> String {
        json!({
            "model_type": "qwen3", "vocab_size": 151936, "hidden_size": 2048, "intermediate_size": 6144,
            "num_hidden_layers": 28, "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128,
            "tie_word_embeddings": true, "quantization": {"group_size": 64, "bits": 4}
        })
        .to_string()
    }

    fn inline(path: &str, content: &str) -> InlineFile {
        InlineFile { path: path.into(), content: content.into() }
    }

    fn file(path: &str, bytes: u64) -> SpecFile {
        SpecFile { path: path.into(), bytes, sha256: SHA.into() }
    }

    fn spec() -> Spec {
        Spec {
            repo: "mlx-community/Qwen3-1.7B-4bit".into(),
            commit: COMMIT.into(),
            files: vec![file("model.safetensors", 968_080_210), file("tokenizer.json", 11_422_654)],
            inline: vec![
                inline("config.json", &config()),
                inline("tokenizer_config.json", &json!({"eos_token": "<|im_end|>", "chat_template": "<|im_start|>{% if enable_thinking %}"}).to_string()),
            ],
        }
    }

    #[test]
    fn accepts_a_runnable_pinned_model_and_derives_its_id() {
        let m = validate(&spec(), 7).unwrap();
        assert_eq!(m.id, "hf-mlx-community--qwen3-1.7b-4bit-3b1b1768");
        assert!(valid_id(&m.id));
        assert!(m.thinking);
        assert_eq!((m.layers, m.bits, m.inline.len(), m.files.len()), (28, 4, 2, 2));
        assert_eq!(m.inline[0].sha256, sha256_hex(config().as_bytes()));
        assert_eq!(m.checkpoint().files.len(), 2, "only the downloads go to the store");
    }

    #[test]
    fn refuses_anything_off_the_closed_list_and_says_why() {
        type Break = Box<dyn Fn(&mut Spec)>;
        let cases: Vec<(Break, &str)> = vec![
            (Box::new(|s| s.repo = "../etc".into()), "repository"),
            (Box::new(|s| s.repo = "a/b/c".into()), "repository"),
            (Box::new(|s| s.commit = "main".into()), "commit"),
            (Box::new(|s| s.files.push(file("pytorch_model.bin", 5))), "pytorch_model.bin"),
            (Box::new(|s| s.files.push(file("../model.safetensors", 5))), "../model.safetensors"),
            (Box::new(|s| s.inline.push(inline("modeling_qwen.py", "import os"))), "modeling_qwen.py"),
            (Box::new(|s| s.files[0].sha256 = "abc".into()), "sha256"),
            (Box::new(|s| s.files[0].bytes = 0), "bytes"),
            (Box::new(|s| s.files[0].bytes = MAX_FILE_BYTES + 1), "bytes"),
            (Box::new(|s| s.files.push(file("tokenizer.json", 1))), "twice"),
            (Box::new(|s| s.inline[0].content = "{".into()), "valid JSON"),
            (Box::new(|s| s.inline[0].content = config().replace("qwen3", "llama")), "can't run"),
            (Box::new(|s| s.inline[1].content = json!({"eos_token": "<|im_end|>", "chat_template": "[INST]"}).to_string()), "ChatML"),
            (Box::new(|s| s.inline[1].content = json!({"chat_template": "<|im_start|>"}).to_string()), "end-of-turn"),
            (Box::new(|s| drop(s.inline.remove(0))), "config.json"),
            (Box::new(|s| s.files.retain(|f| f.path != "tokenizer.json")), "tokenizer.json"),
            (Box::new(|s| s.files.retain(|f| f.path != "model.safetensors")), "weights"),
            (Box::new(|s| s.files.push(file("model-00001-of-00002.safetensors", 5))), "weights"),
        ];
        for (break_it, why) in cases {
            let mut s = spec();
            break_it(&mut s);
            let err = validate(&s, 0).unwrap_err();
            assert!(err.contains(why), "{why}: {err}");
        }
    }

    #[test]
    fn sharded_weights_must_match_their_index() {
        let mut s = spec();
        s.files.retain(|f| f.path != "model.safetensors");
        s.files.push(file("model-00001-of-00002.safetensors", 5));
        s.files.push(file("model-00002-of-00002.safetensors", 5));
        let index = |names: &[&str]| json!({"weight_map": names.iter().enumerate().map(|(i, n)| (format!("w{i}"), json!(n))).collect::<serde_json::Map<_, _>>()}).to_string();
        s.inline.push(inline("model.safetensors.index.json", &index(&["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"])));
        assert!(validate(&s, 0).is_ok());
        s.inline.last_mut().unwrap().content = index(&["model-00001-of-00002.safetensors"]);
        assert!(validate(&s, 0).unwrap_err().contains("don't match"));
        for bad in ["model-1-of-2.safetensors", "model-00003-of-00002.safetensors", "model-00000-of-00002.safetensors", "../model-00001-of-00002.safetensors"] {
            assert!(!is_shard_name(bad), "{bad}");
        }
    }

    #[test]
    fn keeps_lists_completes_and_removes() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let m = add(root, &spec(), 5).unwrap();
        assert_eq!(list(root), vec![m.clone()]);
        assert_eq!(get(root, &m.id).unwrap(), m);
        assert!(!is_complete(root, &m), "nothing downloaded yet");
        let again = add(root, &spec(), 9).unwrap();
        assert_eq!((list(root).len(), again.added_at), (1, 5), "adding again keeps one entry");
        // A hand-edited manifest (a file off the list) isn't loaded.
        let path = manifest_path(root, &m.id).unwrap();
        let tampered = fs::read_to_string(&path).unwrap().replace("tokenizer.json", "evil.py");
        fs::write(&path, tampered).unwrap();
        assert!(get(root, &m.id).is_err());
        assert!(list(root).is_empty());
        add(root, &spec(), 5).unwrap();
        remove(root, &m.id).unwrap();
        assert!(list(root).is_empty() && !store::dir(root, &m.checkpoint()).exists());
        assert!(get(root, "../../etc/passwd").is_err());
    }

    #[test]
    fn caps_how_many_models_can_be_added() {
        let tmp = tempfile::tempdir().unwrap();
        for i in 0..MAX_CUSTOM {
            let mut s = spec();
            s.repo = format!("someone/model-{i}");
            add(tmp.path(), &s, i as u64).unwrap();
        }
        let mut s = spec();
        s.commit = "f".repeat(40);
        assert!(add(tmp.path(), &s, 99).unwrap_err().contains("up to 32"));
    }
}
