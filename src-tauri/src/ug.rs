//! Knowledge bases, backed by the `ug` CLI.
//!
//! Each knowledge base is a folder under `<app data>/kb/<slug>/`:
//!
//!   kb.json   — name, sources and index status (owned by Andai)
//!   docs/     — the normalized files `ug gen` indexes as project `andai-<slug>`
//!
//! Every ug call shells out to the CLI with `--json` where it exists. GUI apps
//! on macOS don't inherit the login shell's PATH, so `ug_path()` also probes
//! the usual install locations (per platform: `ug_candidates`).

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};
use tauri::{AppHandle, Emitter, Manager, State};

use crate::grants::FileGrants;

pub(crate) const PROJECT_PREFIX: &str = "andai-";
/// Longest search query passed to ug; anything larger is not a question.
const MAX_QUERY_BYTES: usize = 2048;
/// Largest file a knowledge base accepts, so one drop can't fill the disk.
const MAX_SOURCE_BYTES: u64 = 100 * 1024 * 1024;

/// Slugs with a `ug gen` in flight; a second index request for one is refused.
#[derive(Default)]
pub struct Indexing(pub Mutex<HashSet<String>>);

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Source {
    /// File name inside `docs/`.
    pub file: String,
    /// Absolute path the user added it from.
    pub original: String,
    /// Display type: PDF, MD, TXT, CSV, CODE.
    pub kind: String,
    pub bytes: u64,
    /// ~4 chars per token; `None` for binary formats (PDF).
    pub approx_tokens: Option<u64>,
    pub added_at: u64,
    /// pending | indexed | failed
    pub status: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct KbMeta {
    pub slug: String,
    pub name: String,
    pub created_at: u64,
    pub sources: Vec<Source>,
    pub last_indexed_at: Option<u64>,
    pub last_error: Option<String>,
    /// The user's choice of kind; `None` derives it from the sources.
    #[serde(default)]
    pub kind_override: Option<KbKind>,
}

/// What a knowledge base holds. It decides which agent tools apply: code
/// navigation (symbols, callers) only makes sense over source code.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum KbKind {
    Document,
    Code,
    Mixed,
}

/// Documents only → document; code only → code; both → mixed. An empty KB
/// counts as documents.
fn derive_kind(sources: &[Source]) -> KbKind {
    let code = sources.iter().filter(|s| s.kind == "CODE").count();
    match code {
        0 => KbKind::Document,
        n if n == sources.len() => KbKind::Code,
        _ => KbKind::Mixed,
    }
}

#[derive(Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct KbInfo {
    #[serde(flatten)]
    pub meta: KbMeta,
    pub dir: String,
    /// empty | pending | indexing | ready | failed
    pub status: String,
    /// `kind_override` if set, else derived from the sources.
    pub kind: KbKind,
    pub nodes: u64,
    pub edges: u64,
    pub size_bytes: u64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UgStatus {
    pub found: bool,
    pub path: Option<String>,
    pub version: Option<String>,
}

#[derive(Serialize, Clone)]
struct Progress<'a> {
    slug: &'a str,
    line: String,
}

// ── helpers ──────────────────────────────────────────────────────────────

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// Relative PATH entries (`.`, `bin`) resolve against whatever the cwd is, so
/// they could pick up a planted `ug`; only absolute directories are probed.
fn absolute_dirs(dirs: impl IntoIterator<Item = PathBuf>) -> Vec<PathBuf> {
    dirs.into_iter().filter(|d| d.is_absolute()).collect()
}

#[cfg(windows)]
const UG_EXE: &str = "ug.exe";
#[cfg(not(windows))]
const UG_EXE: &str = "ug";

/// Where the user's home is: `HOME` on macOS, `USERPROFILE` on Windows (which
/// usually has no `HOME`).
fn home_dir() -> Option<PathBuf> {
    #[cfg(windows)]
    let var = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME"));
    #[cfg(not(windows))]
    let var = std::env::var_os("HOME");
    var.map(PathBuf::from).filter(|h| h.is_absolute())
}

/// Every place `ug` may live, in order: the absolute PATH entries, then the
/// per-user install dirs, then (macOS) Homebrew and /usr/local.
fn ug_candidates(path_var: Option<&std::ffi::OsStr>, home: Option<&Path>) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = path_var
        .map(|p| absolute_dirs(std::env::split_paths(p)).into_iter().map(|d| d.join(UG_EXE)).collect())
        .unwrap_or_default();
    if let Some(h) = home {
        out.extend([".local", ".cargo", ".ug"].map(|d| h.join(d).join("bin").join(UG_EXE)));
    }
    #[cfg(not(windows))]
    out.extend(["/opt/homebrew/bin/ug", "/usr/local/bin/ug"].map(PathBuf::from));
    out
}

pub(crate) fn ug_path() -> Option<PathBuf> {
    ug_candidates(std::env::var_os("PATH").as_deref(), home_dir().as_deref()).into_iter().find(|p| p.is_file())
}

fn ug() -> Result<Command, String> {
    let path = ug_path().ok_or("The `ug` CLI was not found. Install it, then restart Andai.")?;
    let mut cmd = Command::new(path);
    cmd.env("NO_COLOR", "1").env("CLICOLOR", "0").stdin(Stdio::null());
    Ok(cmd)
}

pub(crate) fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            if chars.peek() == Some(&'[') {
                chars.next();
                for c in chars.by_ref() {
                    if c.is_ascii_alphabetic() {
                        break;
                    }
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn run_json(mut cmd: Command) -> Result<Value, String> {
    let out = cmd.output().map_err(|e| format!("failed to run ug: {e}"))?;
    if !out.status.success() {
        let err = strip_ansi(&String::from_utf8_lossy(&out.stderr));
        return Err(err.trim().lines().last().unwrap_or("ug failed").to_string());
    }
    serde_json::from_slice(&out.stdout).map_err(|e| format!("bad ug JSON: {e}"))
}

/// Knowledge-base folders hold the user's documents: owner-only (0700).
/// Existing folders keep their permissions; only new ones are created private.
fn create_private_dir(dir: &Path) -> std::io::Result<()> {
    let mut builder = fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    std::os::unix::fs::DirBuilderExt::mode(&mut builder, 0o700);
    builder.create(dir)
}

/// Opens `path` for writing, creating it owner-only (0600) if it is new.
fn private_file(path: &Path) -> std::io::Result<fs::File> {
    let mut opts = fs::OpenOptions::new();
    opts.write(true).create(true).truncate(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut opts, 0o600);
    opts.open(path)
}

fn write_private(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    private_file(path)?.write_all(bytes)
}

/// `ANDAI_DATA_DIR` relocates all knowledge-base files — the e2e runner points
/// it at a temp dir so tests never touch a user's real knowledge bases.
fn kb_root(app: &AppHandle) -> Result<PathBuf, String> {
    let base = match std::env::var_os("ANDAI_DATA_DIR") {
        Some(dir) => PathBuf::from(dir),
        None => app.path().app_data_dir().map_err(|e| e.to_string())?,
    };
    let dir = base.join("kb");
    create_private_dir(&dir).map_err(|e| e.to_string())?;
    Ok(dir)
}

/// Slugs become directory and ug project names, so only `[a-z0-9-]` gets through
/// (no `..`, `/`, or anything else that could escape `kb/`).
fn valid_slug(slug: &str) -> bool {
    !slug.is_empty() && slug.len() <= 64 && slug.chars().all(|c| c.is_ascii_alphanumeric() || c == '-')
}

pub(crate) fn kb_dir(app: &AppHandle, slug: &str) -> Result<PathBuf, String> {
    if !valid_slug(slug) {
        return Err(format!("invalid knowledge base id: {slug}"));
    }
    Ok(kb_root(app)?.join(slug))
}

fn read_meta(dir: &Path) -> Result<KbMeta, String> {
    let raw = fs::read_to_string(dir.join("kb.json")).map_err(|e| e.to_string())?;
    serde_json::from_str(&raw).map_err(|e| e.to_string())
}

fn write_meta(dir: &Path, meta: &KbMeta) -> Result<(), String> {
    let raw = serde_json::to_string_pretty(meta).map_err(|e| e.to_string())?;
    write_private(&dir.join("kb.json"), raw.as_bytes()).map_err(|e| e.to_string())
}

fn slugify(name: &str) -> String {
    let mut slug = String::new();
    for c in name.trim().chars() {
        if c.is_ascii_alphanumeric() {
            slug.push(c.to_ascii_lowercase());
        } else if !slug.ends_with('-') && !slug.is_empty() {
            slug.push('-');
        }
    }
    let slug = slug.trim_end_matches('-').to_string();
    if slug.is_empty() { "kb".into() } else { slug }
}

/// ug project stats keyed by project name.
fn ug_projects() -> Vec<Value> {
    let Ok(mut cmd) = ug() else { return vec![] };
    cmd.args(["list", "--json", "--quick"]);
    run_json(cmd)
        .ok()
        .and_then(|v| v.get("projects").and_then(|p| p.as_array()).cloned())
        .unwrap_or_default()
}

fn info(dir: &Path, meta: KbMeta, projects: &[Value], indexing: &HashSet<String>) -> KbInfo {
    let project = projects
        .iter()
        .find(|p| p.get("name").and_then(|n| n.as_str()) == Some(&format!("{PROJECT_PREFIX}{}", meta.slug)));
    let num = |k: &str| project.and_then(|p| p.get(k)).and_then(|v| v.as_u64()).unwrap_or(0);
    let status = if indexing.contains(&meta.slug) {
        "indexing"
    } else if meta.sources.is_empty() {
        "empty"
    } else if meta.last_error.is_some() {
        "failed"
    } else if meta.sources.iter().any(|s| s.status == "pending") {
        "pending"
    } else {
        "ready"
    };
    KbInfo {
        dir: dir.to_string_lossy().into(),
        status: status.into(),
        kind: meta.kind_override.unwrap_or_else(|| derive_kind(&meta.sources)),
        nodes: num("nodes"),
        edges: num("edges"),
        size_bytes: num("sizeBytes"),
        meta,
    }
}

fn load_info(app: &AppHandle, slug: &str, indexing: &Indexing) -> Result<KbInfo, String> {
    let dir = kb_dir(app, slug)?;
    let meta = read_meta(&dir)?;
    let busy = indexing.0.lock().unwrap().clone();
    Ok(info(&dir, meta, &ug_projects(), &busy))
}

/// A source is addressed by its bare file name inside `docs/`.
fn valid_source_name(file: &str) -> bool {
    !file.is_empty() && !file.contains('/') && !file.contains('\\') && !file.contains("..")
}

/// The query reaches ug as a positional argument. ug has no `--` separator,
/// so a query starting with `-` would be parsed as a flag (`--base-url` would
/// send it to a remote embedder). A leading space keeps it positional and
/// doesn't change the search (probed against ug 0.1.21, AGENTS.md §9).
pub(crate) fn search_query(query: &str) -> Result<String, String> {
    if query.len() > MAX_QUERY_BYTES {
        return Err(format!("Search query is too long (max {MAX_QUERY_BYTES} bytes)."));
    }
    if query.contains('\0') {
        return Err("Search query contains a NUL byte.".into());
    }
    Ok(if query.starts_with('-') { format!(" {query}") } else { query.to_string() })
}

/// Result count and context budget, clamped to what the prompt can use.
fn search_limits(k: u32, max_chars: u32) -> (u32, u32) {
    (k.clamp(1, 50), max_chars.clamp(100, 8000))
}

/// Copy one file into `docs/`, converting formats ug can't parse into Markdown.
pub(crate) fn ingest_file(docs: &Path, src: &Path) -> Result<Source, String> {
    let original = src.to_string_lossy().to_string();
    let stem = src.file_stem().and_then(|s| s.to_str()).unwrap_or("file").to_string();
    let ext = src.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    let file_name = src.file_name().and_then(|s| s.to_str()).unwrap_or("file").to_string();
    let size = fs::metadata(src).map_err(|e| format!("{file_name}: {e}"))?.len();
    if size > MAX_SOURCE_BYTES {
        return Err(format!("{file_name}: too large ({} MB; the limit is 100 MB)", size / (1024 * 1024)));
    }

    let code = ["ts", "tsx", "js", "jsx", "mjs", "cjs", "py", "java", "rs"];
    let (kind, target_ext, body): (&str, String, Option<String>) = match ext.as_str() {
        "md" | "markdown" | "mdx" => ("MD", "md".into(), None),
        "pdf" => ("PDF", "pdf".into(), None),
        "txt" | "text" | "log" | "rst" => {
            let text = fs::read_to_string(src).map_err(|e| format!("{file_name}: {e}"))?;
            ("TXT", "md".into(), Some(format!("# {file_name}\n\n{text}\n")))
        }
        "csv" | "tsv" => {
            let text = fs::read_to_string(src).map_err(|e| format!("{file_name}: {e}"))?;
            ("CSV", "md".into(), Some(format!("# {file_name}\n\n```{ext}\n{text}\n```\n")))
        }
        e if code.contains(&e) => ("CODE", e.to_string(), None),
        _ => return Err(format!("{file_name}: unsupported type (use PDF, Markdown, TXT, CSV or source code)")),
    };

    let mut target = docs.join(format!("{stem}.{target_ext}"));
    let mut n = 2;
    while target.exists() {
        target = docs.join(format!("{stem}-{n}.{target_ext}"));
        n += 1;
    }
    // Written through private_file, not fs::copy, which would carry over the
    // source's (often world-readable) permissions.
    match &body {
        Some(text) => write_private(&target, text.as_bytes()),
        None => fs::File::open(src).and_then(|mut f| std::io::copy(&mut f, &mut private_file(&target)?)).map(|_| ()),
    }
    .map_err(|e| format!("{file_name}: {e}"))?;

    let bytes = fs::metadata(&target).map(|m| m.len()).unwrap_or(0);
    Ok(Source {
        file: target.file_name().unwrap().to_string_lossy().into(),
        original,
        kind: kind.into(),
        bytes,
        approx_tokens: if kind == "PDF" { None } else { Some(bytes / 4) },
        added_at: now(),
        status: "pending".into(),
    })
}

// ── commands ─────────────────────────────────────────────────────────────

#[tauri::command]
pub fn ug_status() -> UgStatus {
    let path = ug_path();
    let version = path.as_ref().and_then(|p| {
        let out = Command::new(p).arg("-v").output().ok()?;
        Some(strip_ansi(&String::from_utf8_lossy(&out.stdout)).trim().to_string())
    });
    UgStatus { found: path.is_some(), path: path.map(|p| p.to_string_lossy().into()), version }
}

#[tauri::command]
pub async fn kb_list(app: AppHandle, indexing: State<'_, Indexing>) -> Result<Vec<KbInfo>, String> {
    let root = kb_root(&app)?;
    let busy = indexing.0.lock().unwrap().clone();
    tauri::async_runtime::spawn_blocking(move || {
        let projects = ug_projects();
        let mut out: Vec<KbInfo> = fs::read_dir(&root)
            .map_err(|e| e.to_string())?
            .flatten()
            .filter_map(|entry| {
                let dir = entry.path();
                read_meta(&dir).ok().map(|meta| info(&dir, meta, &projects, &busy))
            })
            .collect();
        out.sort_by_key(|k| k.meta.created_at);
        Ok(out)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
pub fn kb_create(app: AppHandle, name: String, indexing: State<'_, Indexing>) -> Result<KbInfo, String> {
    let name = name.trim();
    if name.is_empty() {
        return Err("Give the knowledge base a name.".into());
    }
    let base = slugify(name);
    let mut slug = base.clone();
    let mut n = 2;
    while kb_dir(&app, &slug)?.exists() {
        slug = format!("{base}-{n}");
        n += 1;
    }
    let dir = kb_dir(&app, &slug)?;
    create_private_dir(&dir.join("docs")).map_err(|e| e.to_string())?;
    let meta = KbMeta {
        slug: slug.clone(),
        name: name.into(),
        created_at: now(),
        sources: vec![],
        last_indexed_at: None,
        last_error: None,
        kind_override: None,
    };
    write_meta(&dir, &meta)?;
    load_info(&app, &slug, &indexing)
}

/// Ingests each path the user granted (see grants.rs) into `docs/`; returns
/// one actionable message per file that was skipped.
fn add_sources(docs: &Path, meta: &mut KbMeta, paths: &[String], grants: &FileGrants) -> Vec<String> {
    let mut errors = vec![];
    for p in paths {
        if Path::new(p).is_dir() {
            errors.push(format!("{p}: folders aren't supported yet — drop the files inside it"));
            continue;
        }
        match grants.take(Path::new(p)).and_then(|src| ingest_file(docs, &src)) {
            Ok(source) => meta.sources.push(source),
            Err(e) => errors.push(e),
        }
    }
    errors
}

/// Copies files in; returns the updated KB plus per-file errors. Call `kb_index` after.
/// Sync on purpose: it runs on the main thread, after the drag-drop window
/// event that granted the dropped paths (lib.rs) has finished.
#[tauri::command]
pub fn kb_add_files(
    app: AppHandle,
    slug: String,
    paths: Vec<String>,
    indexing: State<'_, Indexing>,
    grants: State<'_, FileGrants>,
) -> Result<(KbInfo, Vec<String>), String> {
    let dir = kb_dir(&app, &slug)?;
    let docs = dir.join("docs");
    create_private_dir(&docs).map_err(|e| e.to_string())?;
    let mut meta = read_meta(&dir)?;
    let errors = add_sources(&docs, &mut meta, &paths, &grants);
    write_meta(&dir, &meta)?;
    Ok((load_info(&app, &slug, &indexing)?, errors))
}

#[tauri::command]
pub fn kb_remove_source(
    app: AppHandle,
    slug: String,
    file: String,
    indexing: State<'_, Indexing>,
) -> Result<KbInfo, String> {
    let dir = kb_dir(&app, &slug)?;
    let mut meta = read_meta(&dir)?;
    if !valid_source_name(&file) {
        return Err("invalid file name".into());
    }
    let _ = fs::remove_file(dir.join("docs").join(&file));
    meta.sources.retain(|s| s.file != file);
    write_meta(&dir, &meta)?;
    load_info(&app, &slug, &indexing)
}

/// Sets or clears (`None`) the user's override of the derived kind.
#[tauri::command]
pub fn kb_set_kind(
    app: AppHandle,
    slug: String,
    kind: Option<KbKind>,
    indexing: State<'_, Indexing>,
) -> Result<KbInfo, String> {
    let dir = kb_dir(&app, &slug)?;
    let mut meta = read_meta(&dir)?;
    meta.kind_override = kind;
    write_meta(&dir, &meta)?;
    load_info(&app, &slug, &indexing)
}

#[tauri::command]
pub async fn kb_delete(app: AppHandle, slug: String) -> Result<(), String> {
    let dir = kb_dir(&app, &slug)?;
    tauri::async_runtime::spawn_blocking(move || {
        if let Ok(mut cmd) = ug() {
            let _ = cmd.args(["remove", &format!("{PROJECT_PREFIX}{slug}"), "-y"]).output();
        }
        fs::remove_dir_all(&dir).map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| e.to_string())?
}

/// Runs `ug gen --with-embed` over `docs/`, streaming its progress lines as
/// `kb-progress` events. Resolves with the updated KB once the run finishes.
#[tauri::command]
pub async fn kb_index(app: AppHandle, slug: String, indexing: State<'_, Indexing>) -> Result<KbInfo, String> {
    let dir = kb_dir(&app, &slug)?;
    if !indexing.0.lock().unwrap().insert(slug.clone()) {
        return Err("This knowledge base is already indexing.".into());
    }
    let _ = app.emit("kb-progress", Progress { slug: &slug, line: "Starting ug gen…".into() });

    let (app2, slug2, dir2) = (app.clone(), slug.clone(), dir.clone());
    let result = tauri::async_runtime::spawn_blocking(move || -> Result<(), String> {
        let docs = dir2.join("docs");
        let project = format!("{PROJECT_PREFIX}{slug2}");
        let empty = fs::read_dir(&docs).map(|mut d| d.next().is_none()).unwrap_or(true);
        if empty {
            // Nothing left to index: drop the ug project so search can't return stale hits.
            let _ = ug()?.args(["remove", &project, "-y"]).output();
            return Ok(());
        }
        let mut child = ug()?
            .arg("gen")
            .arg(&docs)
            .args(["-n", &project, "--with-embed"])
            .current_dir(&docs)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("failed to start ug: {e}"))?;

        // ug redraws progress with \r, so split on both line endings.
        let stderr = child.stderr.take().unwrap();
        let (app3, slug3) = (app2.clone(), slug2.clone());
        let err_thread = std::thread::spawn(move || {
            let mut buf = String::new();
            let _ = BufReader::new(stderr).read_to_string(&mut buf);
            let text = strip_ansi(&buf);
            for line in text.split(['\r', '\n']).map(str::trim).filter(|l| !l.is_empty()) {
                let _ = app3.emit("kb-progress", Progress { slug: &slug3, line: line.into() });
            }
            text
        });
        let mut last = String::new();
        for chunk in BufReader::new(child.stdout.take().unwrap()).split(b'\n') {
            let Ok(chunk) = chunk else { break };
            for line in strip_ansi(&String::from_utf8_lossy(&chunk)).split('\r') {
                let line = line.trim();
                if !line.is_empty() {
                    last = line.to_string();
                    let _ = app2.emit("kb-progress", Progress { slug: &slug2, line: last.clone() });
                }
            }
        }
        let status = child.wait().map_err(|e| e.to_string())?;
        let err_text = err_thread.join().unwrap_or_default();
        if status.success() {
            Ok(())
        } else {
            let msg = err_text.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or(&last);
            Err(format!("ug gen failed: {}", msg.trim()))
        }
    })
    .await
    .map_err(|e| e.to_string())
    .and_then(|r| r);

    indexing.0.lock().unwrap().remove(&slug);
    let mut meta = read_meta(&dir)?;
    match &result {
        Ok(()) => {
            meta.last_error = None;
            meta.last_indexed_at = Some(now());
            for s in &mut meta.sources {
                s.status = "indexed".into();
            }
        }
        Err(e) => {
            meta.last_error = Some(e.clone());
            for s in meta.sources.iter_mut().filter(|s| s.status == "pending") {
                s.status = "failed".into();
            }
        }
    }
    write_meta(&dir, &meta)?;
    let _ = app.emit(
        "kb-progress",
        Progress { slug: &slug, line: result.clone().map(|_| "Index ready.".into()).unwrap_or_else(|e| e) },
    );
    load_info(&app, &slug, &indexing)
}

/// GraphRAG search over one knowledge base: `ug search … --snippets --json`.
#[tauri::command]
pub async fn kb_search(
    app: AppHandle,
    slug: String,
    query: String,
    k: u32,
    max_chars: u32,
) -> Result<Value, String> {
    let dir = kb_dir(&app, &slug)?;
    let query = search_query(&query)?;
    let (k, max_chars) = search_limits(k, max_chars);
    tauri::async_runtime::spawn_blocking(move || {
        let mut cmd = ug()?;
        cmd.arg("search")
            .arg(&query)
            .args(["-n", &format!("{PROJECT_PREFIX}{slug}")])
            .args(["-k", &k.to_string(), "--max-chars", &max_chars.to_string()])
            .arg("--snippets")
            .arg("--repo-root")
            .arg(dir.join("docs"))
            .arg("--json")
            .current_dir(dir.join("docs"));
        run_json(cmd)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn meta(sources: Vec<Source>, last_error: Option<&str>) -> KbMeta {
        KbMeta {
            slug: "docs".into(),
            name: "Docs".into(),
            created_at: 1,
            sources,
            last_indexed_at: None,
            last_error: last_error.map(Into::into),
            kind_override: None,
        }
    }

    fn source(status: &str) -> Source {
        source_of("MD", status)
    }

    fn source_of(kind: &str, status: &str) -> Source {
        Source {
            file: "a.md".into(),
            original: "/x/a.md".into(),
            kind: kind.into(),
            bytes: 4,
            approx_tokens: Some(1),
            added_at: 1,
            status: status.into(),
        }
    }

    #[test]
    fn slugify_makes_safe_ids() {
        assert_eq!(slugify("Product Specs 2026!"), "product-specs-2026");
        assert_eq!(slugify("  --Hello__World--  "), "hello-world");
        assert_eq!(slugify("日本語"), "kb");
        assert_eq!(slugify(""), "kb");
        assert!(valid_slug(&slugify("../../etc/passwd")));
    }

    #[test]
    fn slug_validation_blocks_path_escapes() {
        assert!(valid_slug("product-specs-2"));
        for bad in ["", "..", "../x", "a/b", "a b", "a.b", &"x".repeat(65)] {
            assert!(!valid_slug(bad), "{bad:?} should be rejected");
        }
    }

    #[test]
    fn source_names_must_be_bare_file_names() {
        assert!(valid_source_name("README.md"));
        for bad in ["", "../kb.json", "docs/a.md", "a\\b.md", ".."] {
            assert!(!valid_source_name(bad), "{bad:?} should be rejected");
        }
    }

    #[test]
    fn search_queries_cannot_become_ug_flags() {
        // ug has no `--` separator; a leading space keeps the query positional
        // (probed against ug 0.1.21, see AGENTS.md §9).
        assert_eq!(search_query("--base-url http://evil").unwrap(), " --base-url http://evil");
        assert_eq!(search_query("-k").unwrap(), " -k");
        assert_eq!(search_query("what is wllama").unwrap(), "what is wllama");
        assert!(search_query("a\0b").is_err(), "NUL bytes can't reach argv");
        assert!(search_query(&"x".repeat(MAX_QUERY_BYTES + 1)).is_err());
    }

    #[test]
    fn search_budgets_are_clamped() {
        assert_eq!(search_limits(0, 0), (1, 100));
        assert_eq!(search_limits(8, 6000), (8, 6000));
        assert_eq!(search_limits(u32::MAX, u32::MAX), (50, 8000));
    }

    #[test]
    fn relative_path_entries_are_not_trusted_for_ug() {
        let abs = std::env::temp_dir();
        let dirs = [PathBuf::from("."), PathBuf::from("bin"), abs.clone()];
        assert_eq!(absolute_dirs(dirs), vec![abs]);
    }

    #[test]
    fn ug_is_looked_up_on_path_then_in_the_home_install_dirs() {
        let a = tempfile::tempdir().unwrap();
        let home = tempfile::tempdir().unwrap();
        let path = std::env::join_paths([PathBuf::from("rel"), a.path().to_path_buf()]).unwrap();
        let c = ug_candidates(Some(&path), Some(home.path()));
        assert_eq!(c[0], a.path().join(UG_EXE), "PATH first, relative entries skipped");
        assert_eq!(c[1], home.path().join(".local").join("bin").join(UG_EXE));
        assert!(c.contains(&home.path().join(".cargo").join("bin").join(UG_EXE)));
        assert!(c.iter().all(|p| p.is_absolute()));
        #[cfg(windows)]
        assert!(c.iter().all(|p| p.extension().is_some_and(|e| e == "exe")));
    }

    #[test]
    fn only_granted_files_are_added() {
        let src = tempfile::tempdir().unwrap();
        let docs = tempfile::tempdir().unwrap();
        let (mine, secret) = (src.path().join("mine.md"), src.path().join("secret.md"));
        fs::write(&mine, "ok").unwrap();
        fs::write(&secret, "key").unwrap();
        let grants = FileGrants::default();
        grants.grant([&mine]);

        let mut m = meta(vec![], None);
        let paths = [mine, secret].map(|p| p.to_string_lossy().to_string());
        let errors = add_sources(docs.path(), &mut m, &paths, &grants);
        assert_eq!(m.sources.iter().map(|s| s.file.as_str()).collect::<Vec<_>>(), ["mine.md"]);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("secret.md") && errors[0].contains("not added by you"), "{errors:?}");
        assert!(!docs.path().join("secret.md").exists());
    }

    #[test]
    fn ingest_rejects_oversized_files() {
        let src = tempfile::tempdir().unwrap();
        let docs = tempfile::tempdir().unwrap();
        let big = src.path().join("big.md");
        fs::File::create(&big).unwrap().set_len(MAX_SOURCE_BYTES + 1).unwrap();
        let err = ingest_file(docs.path(), &big).unwrap_err();
        assert!(err.contains("too large"), "{err}");
        assert_eq!(fs::read_dir(docs.path()).unwrap().count(), 0);
    }

    #[cfg(unix)]
    #[test]
    fn kb_files_are_private_to_the_user() {
        use std::os::unix::fs::PermissionsExt;
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("kb").join("docs");
        create_private_dir(&dir).unwrap();
        assert_eq!(fs::metadata(&dir).unwrap().permissions().mode() & 0o777, 0o700);
        assert_eq!(fs::metadata(root.path().join("kb")).unwrap().permissions().mode() & 0o777, 0o700);
        write_private(&dir.join("kb.json"), b"{}").unwrap();
        assert_eq!(fs::metadata(dir.join("kb.json")).unwrap().permissions().mode() & 0o777, 0o600);

        let src = tempfile::tempdir().unwrap();
        fs::write(src.path().join("n.md"), "x").unwrap();
        ingest_file(&dir, &src.path().join("n.md")).unwrap();
        assert_eq!(fs::metadata(dir.join("n.md")).unwrap().permissions().mode() & 0o777, 0o600);
    }

    #[test]
    fn strip_ansi_removes_color_codes() {
        assert_eq!(strip_ansi("\u{1b}[32m✓ done\u{1b}[0m in \u{1b}[1m2s\u{1b}[0m"), "✓ done in 2s");
        assert_eq!(strip_ansi("plain"), "plain");
    }

    #[test]
    fn ingest_copies_markdown_and_pdf_as_is() {
        let src = tempfile::tempdir().unwrap();
        let docs = tempfile::tempdir().unwrap();
        fs::write(src.path().join("notes.md"), "# Notes\nbody").unwrap();
        fs::write(src.path().join("paper.pdf"), b"%PDF-1.4 fake").unwrap();

        let md = ingest_file(docs.path(), &src.path().join("notes.md")).unwrap();
        assert_eq!((md.file.as_str(), md.kind.as_str(), md.status.as_str()), ("notes.md", "MD", "pending"));
        assert_eq!(md.approx_tokens, Some(md.bytes / 4));
        assert_eq!(fs::read_to_string(docs.path().join("notes.md")).unwrap(), "# Notes\nbody");

        let pdf = ingest_file(docs.path(), &src.path().join("paper.pdf")).unwrap();
        assert_eq!((pdf.file.as_str(), pdf.kind.as_str()), ("paper.pdf", "PDF"));
        assert_eq!(pdf.approx_tokens, None, "binary formats have no token estimate");
    }

    #[test]
    fn ingest_converts_text_and_csv_to_markdown() {
        let src = tempfile::tempdir().unwrap();
        let docs = tempfile::tempdir().unwrap();
        fs::write(src.path().join("log.txt"), "line one").unwrap();
        fs::write(src.path().join("data.csv"), "a,b\n1,2").unwrap();

        let txt = ingest_file(docs.path(), &src.path().join("log.txt")).unwrap();
        assert_eq!((txt.file.as_str(), txt.kind.as_str()), ("log.md", "TXT"));
        assert_eq!(fs::read_to_string(docs.path().join("log.md")).unwrap(), "# log.txt\n\nline one\n");

        let csv = ingest_file(docs.path(), &src.path().join("data.csv")).unwrap();
        assert_eq!((csv.file.as_str(), csv.kind.as_str()), ("data.md", "CSV"));
        assert_eq!(fs::read_to_string(docs.path().join("data.md")).unwrap(), "# data.csv\n\n```csv\na,b\n1,2\n```\n");
    }

    #[test]
    fn ingest_never_overwrites_an_existing_source() {
        let src = tempfile::tempdir().unwrap();
        let docs = tempfile::tempdir().unwrap();
        fs::write(src.path().join("a.md"), "one").unwrap();
        let first = ingest_file(docs.path(), &src.path().join("a.md")).unwrap();
        fs::write(src.path().join("a.md"), "two").unwrap();
        let second = ingest_file(docs.path(), &src.path().join("a.md")).unwrap();
        assert_eq!((first.file.as_str(), second.file.as_str()), ("a.md", "a-2.md"));
        assert_eq!(fs::read_to_string(docs.path().join("a.md")).unwrap(), "one");
    }

    #[test]
    fn ingest_rejects_unsupported_and_missing_files() {
        let src = tempfile::tempdir().unwrap();
        let docs = tempfile::tempdir().unwrap();
        fs::write(src.path().join("pic.png"), b"png").unwrap();
        let err = ingest_file(docs.path(), &src.path().join("pic.png")).unwrap_err();
        assert!(err.contains("unsupported type"), "{err}");
        assert!(ingest_file(docs.path(), &src.path().join("gone.txt")).is_err());
        assert_eq!(fs::read_dir(docs.path()).unwrap().count(), 0, "nothing written on failure");
    }

    #[test]
    fn status_is_derived_from_sources_errors_and_indexing() {
        let dir = Path::new("/tmp/kb/docs");
        let idle = HashSet::new();
        let busy: HashSet<String> = ["docs".to_string()].into();
        let status = |m: KbMeta, i: &HashSet<String>| info(dir, m, &[], i).status;

        assert_eq!(status(meta(vec![], None), &idle), "empty");
        assert_eq!(status(meta(vec![source("pending")], None), &idle), "pending");
        assert_eq!(status(meta(vec![source("indexed")], None), &idle), "ready");
        assert_eq!(status(meta(vec![source("indexed")], Some("boom")), &idle), "failed");
        assert_eq!(status(meta(vec![source("indexed")], Some("boom")), &busy), "indexing", "indexing wins");
    }

    #[test]
    fn kind_is_derived_from_sources_unless_overridden() {
        let kind = |m: KbMeta| info(Path::new("/tmp"), m, &[], &HashSet::new()).kind;
        assert_eq!(kind(meta(vec![], None)), KbKind::Document);
        assert_eq!(kind(meta(vec![source("indexed"), source_of("PDF", "indexed")], None)), KbKind::Document);
        assert_eq!(kind(meta(vec![source_of("CODE", "indexed")], None)), KbKind::Code);
        assert_eq!(kind(meta(vec![source("indexed"), source_of("CODE", "indexed")], None)), KbKind::Mixed);
        let mut m = meta(vec![source_of("CODE", "indexed")], None);
        m.kind_override = Some(KbKind::Document);
        assert_eq!(kind(m), KbKind::Document, "the user's choice wins");
    }

    #[test]
    fn kb_json_without_a_kind_override_still_loads() {
        let raw = r#"{"slug":"a","name":"A","createdAt":1,"sources":[],"lastIndexedAt":null,"lastError":null}"#;
        assert_eq!(serde_json::from_str::<KbMeta>(raw).unwrap().kind_override, None);
        let bad = r#"{"slug":"a","name":"A","createdAt":1,"sources":[],"lastIndexedAt":null,"lastError":null,"kindOverride":"evil"}"#;
        assert!(serde_json::from_str::<KbMeta>(bad).is_err());
    }

    #[test]
    fn graph_stats_come_from_the_matching_ug_project() {
        let projects = vec![
            json!({ "name": "andai-other", "nodes": 99, "edges": 99, "sizeBytes": 99 }),
            json!({ "name": "andai-docs", "nodes": 9, "edges": 8, "sizeBytes": 1234 }),
        ];
        let kb = info(Path::new("/tmp"), meta(vec![], None), &projects, &HashSet::new());
        assert_eq!((kb.nodes, kb.edges, kb.size_bytes), (9, 8, 1234));
        let none = info(Path::new("/tmp"), meta(vec![], None), &[], &HashSet::new());
        assert_eq!((none.nodes, none.edges), (0, 0));
    }

    #[test]
    fn kb_info_serializes_flat_camel_case_for_the_frontend() {
        let kb = info(Path::new("/tmp"), meta(vec![source("indexed")], None), &[], &HashSet::new());
        let v = serde_json::to_value(&kb).unwrap();
        for key in ["slug", "name", "createdAt", "sources", "lastIndexedAt", "lastError", "kindOverride", "dir", "status", "kind", "nodes", "sizeBytes"] {
            assert!(v.get(key).is_some(), "missing {key} in {v}");
        }
        assert!(v["sources"][0].get("approxTokens").is_some());
    }

    /// Real ug round trip: gen --with-embed → search → remove.
    /// Needs the `ug` CLI and its embedding model: `cargo test -- --ignored`.
    #[test]
    #[ignore = "requires the ug CLI"]
    fn ug_indexes_and_searches_documents() {
        assert!(ug_path().is_some(), "ug not found on PATH or fallbacks");
        let docs = tempfile::tempdir().unwrap();
        fs::write(
            docs.path().join("notes.md"),
            "# Deployment\n\n## Isolation\n\nwllama needs Cross-Origin-Opener-Policy and Cross-Origin-Embedder-Policy headers.\n",
        )
        .unwrap();
        let project = format!("andai-test-{}", std::process::id());

        let gen = ug().unwrap().arg("gen").arg(docs.path()).args(["-n", &project, "--with-embed"]).output().unwrap();
        let cleanup = || {
            let _ = ug().unwrap().args(["remove", &project, "-y"]).output();
        };
        if !gen.status.success() {
            cleanup();
            panic!("ug gen failed: {}", String::from_utf8_lossy(&gen.stderr));
        }

        let mut search = ug().unwrap();
        search
            .args(["search", "which headers does wllama need", "-n", &project, "-k", "4", "--snippets", "--json"])
            .arg("--repo-root")
            .arg(docs.path());
        let result = run_json(search);
        let listed = ug_projects().iter().any(|p| p["name"] == project.as_str());
        cleanup();

        let items = result.expect("search JSON")["items"].as_array().cloned().unwrap_or_default();
        assert!(listed, "project should appear in `ug list --json`");
        assert!(!items.is_empty(), "search returned no items");
        let text = serde_json::to_string(&items).unwrap();
        assert!(text.contains("Cross-Origin-Embedder-Policy"), "snippet should carry the passage: {text}");
    }
}
