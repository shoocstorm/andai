//! Knowledge bases, backed by the `ug` CLI. A knowledge base *is* a ug
//! project: the list comes from `ug list`, so every project the user indexed
//! with ug shows up in Andai, and every one Andai makes shows up in ug.
//!
//! Andai's own are named `andai-<slug>` and index the copies Andai keeps in
//! `<app data>/kb/<slug>/docs/` (files can be added and removed there). A
//! folder that ug has no project for yet (never indexed, or its graph was
//! removed) is listed too, so the user's files never go missing from view.
//! Everything else Andai shows comes from ug or the files themselves; there
//! is no metadata file. Other projects index their own folder (`repoRoot`),
//! which Andai only reads.
//!
//! Every ug call shells out to the CLI with `--json` where it exists. GUI apps
//! on macOS don't inherit the login shell's PATH, so `ug_path()` also probes
//! the usual install locations (per platform: `ug_candidates`).

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
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

/// Knowledge bases with a `ug gen` in flight (a second run for one is
/// refused), and the last indexing error per knowledge base: kept until its
/// next run, not across restarts.
#[derive(Default)]
pub struct Indexing {
    pub(crate) busy: Mutex<HashSet<String>>,
    errors: Mutex<HashMap<String, String>>,
}

#[derive(Serialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Source {
    /// Path inside the knowledge base's folder: a bare name for Andai's own,
    /// repo-relative (`src/fare.ts`) for other projects.
    pub file: String,
    /// Display type: PDF, MD, TXT, CSV, CODE.
    pub kind: String,
    pub bytes: u64,
    /// ~4 chars per token; `None` for binary formats (PDF).
    pub approx_tokens: Option<u64>,
    /// When the file was last written (seconds): Andai's copy, or the file in the repo.
    pub added_at: u64,
    /// pending | indexed | failed
    pub status: String,
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

/// Most sources listed for one knowledge base; a large repo's rest is counted, not listed.
const MAX_LISTED_SOURCES: usize = 5000;

#[derive(Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct KbInfo {
    /// The ug project name, which is also the id the webview uses.
    pub slug: String,
    /// What the UI shows: the project name without `andai-`.
    pub name: String,
    /// Andai keeps this knowledge base's files (`kb/<slug>/docs`), so files
    /// can be added and removed. Other projects index the user's own folder.
    pub managed: bool,
    /// The folder ug indexes.
    pub root: String,
    pub created_at: u64,
    pub last_indexed_at: Option<u64>,
    pub last_error: Option<String>,
    pub sources: Vec<Source>,
    /// All of them, including those past `MAX_LISTED_SOURCES`.
    pub source_count: usize,
    /// offline (ug missing) | empty | pending | indexing | ready | failed
    pub status: String,
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
    /// Whether Andai can install ug itself here (ug_install.rs); else the website.
    pub can_install: bool,
    /// The terminal command UltraGraph's site gives, for installing by hand.
    pub install_command: &'static str,
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

/// UltraGraph's site: install script and downloads. It lives here, not in the
/// webview, so `src/` still names no remote URL (security.test.ts).
pub(crate) const UG_WEBSITE: &str = "https://ultra-graph.web.app";

/// Kept here with the site's URL rather than in `src/`, which names no remote URL.
const UG_INSTALL_COMMAND: &str = "curl -fsSL https://ultra-graph.web.app/install.sh | sh";

pub(crate) const UG_MISSING: &str =
    "The `ug` (UltraGraph) CLI was not found. Install it from Knowledge or Settings (Install UltraGraph), or from https://ultra-graph.web.app.";

fn ug() -> Result<Command, String> {
    let path = ug_path().ok_or(UG_MISSING)?;
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
pub(crate) fn create_private_dir(dir: &Path) -> std::io::Result<()> {
    let mut builder = fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    std::os::unix::fs::DirBuilderExt::mode(&mut builder, 0o700);
    builder.create(dir)
}

/// Opens `path` for writing, creating it owner-only (0600) if it is new.
pub(crate) fn private_file(path: &Path) -> std::io::Result<fs::File> {
    let mut opts = fs::OpenOptions::new();
    opts.write(true).create(true).truncate(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut opts, 0o600);
    opts.open(path)
}

pub(crate) fn write_private(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    private_file(path)?.write_all(bytes)
}

/// A private folder under app data. `ANDAI_DATA_DIR` relocates all of it
/// (knowledge bases, Laya checkpoints): the e2e runner points it at a temp dir
/// so tests never touch a user's real data.
pub(crate) fn data_dir(app: &AppHandle, parts: &[&str]) -> Result<PathBuf, String> {
    let mut dir = match std::env::var_os("ANDAI_DATA_DIR") {
        Some(dir) => PathBuf::from(dir),
        None => app.path().app_data_dir().map_err(|e| e.to_string())?,
    };
    dir.extend(parts);
    create_private_dir(&dir).map_err(|e| e.to_string())?;
    Ok(dir)
}

pub(crate) fn kb_root(app: &AppHandle) -> Result<PathBuf, String> {
    data_dir(app, &["kb"])
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

pub(crate) fn slugify(name: &str) -> String {
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

/// A ug project name as the webview may send it. It only ever names a project
/// `ug list` reports or one of Andai's folders, and reaches ug as the value of
/// `-n`, so it must not look like a flag.
pub(crate) fn valid_project(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 128
        && !name.starts_with(['-', '.'])
        && name.chars().all(|c| c.is_ascii_alphanumeric() || "._-".contains(c))
}

/// A repo-relative path with only normal components: no `..`, no root, no
/// backslash, no control characters, nothing that starts like a flag.
pub(crate) fn safe_rel_path(p: &str) -> bool {
    !p.is_empty()
        && p.len() <= 1024
        && !p.starts_with('-')
        && !p.contains('\\')
        && !p.chars().any(char::is_control)
        && Path::new(p).components().all(|c| matches!(c, std::path::Component::Normal(_)))
}

/// One project from `ug list --json` (ug 0.1.22), with what Andai uses.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct UgProject {
    pub name: String,
    pub repo_root: PathBuf,
    nodes: u64,
    edges: u64,
    size_bytes: u64,
    created_at: u64,
    updated_at: u64,
    /// Indexed files changed or deleted since (`isStale`), or never ingested (`hasDb` false).
    stale: bool,
    repo_missing: bool,
    kb_kind: Option<String>,
}

/// The projects in `ug list --json`, skipping any whose name or folders
/// aren't what a real project has.
fn parse_projects(list: &Value) -> Vec<UgProject> {
    let num = |p: &Value, k: &str| p.get(k).and_then(Value::as_u64).unwrap_or(0);
    let path = |p: &Value, k: &str| p.get(k).and_then(Value::as_str).map(PathBuf::from).filter(|p| p.is_absolute());
    let Some(projects) = list.get("projects").and_then(Value::as_array) else { return vec![] };
    projects
        .iter()
        .filter_map(|p| {
            let name = p.get("name")?.as_str()?.to_string();
            if !valid_project(&name) {
                return None;
            }
            Some(UgProject {
                repo_root: path(p, "repoRoot")?,
                nodes: num(p, "nodes"),
                edges: num(p, "edges"),
                size_bytes: num(p, "sizeBytes"),
                created_at: num(p, "createdAt"),
                updated_at: num(p, "updatedAt"),
                stale: p.get("isStale").and_then(Value::as_bool).unwrap_or(false)
                    || p.get("hasDb").and_then(Value::as_bool) == Some(false),
                repo_missing: p.get("repoMissing").and_then(Value::as_bool).unwrap_or(false),
                kb_kind: p.get("kbKind").and_then(Value::as_str).map(String::from),
                name,
            })
        })
        .collect()
}

/// Every ug project, or `None` without ug. With no projects at all, ug 0.1.22
/// exits 1 and prints a sentence instead of JSON, which reads as none here.
pub(crate) fn ug_projects() -> Option<Vec<UgProject>> {
    let mut cmd = ug().ok()?;
    let out = cmd.args(["list", "--json"]).output().ok()?;
    Some(serde_json::from_slice::<Value>(&out.stdout).map(|v| parse_projects(&v)).unwrap_or_default())
}

/// One file from `ug files --json`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct UgFile {
    pub path: String,
    language: String,
    bytes: u64,
    modified: u64,
    /// fresh | changed | missing, against the index.
    status: String,
}

/// What `ug files --json` reports for a project: its indexed files (the first
/// `MAX_LISTED_SOURCES`), how many there are, and how many drifted since.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct UgFiles {
    pub files: Vec<UgFile>,
    total: usize,
    drifted: usize,
}

/// Reads `ug files --json`, keeping only paths that stay inside the project.
fn parse_files(v: &Value) -> UgFiles {
    let num = |v: &Value, k: &str| v.get(k).and_then(Value::as_u64).unwrap_or(0);
    let files: Vec<UgFile> = v
        .get("files")
        .and_then(Value::as_array)
        .map(|fs| {
            fs.iter()
                .filter_map(|f| {
                    let path = f.get("path")?.as_str()?.to_string();
                    safe_rel_path(&path).then(|| UgFile {
                        path,
                        language: f.get("language").and_then(Value::as_str).unwrap_or("").to_string(),
                        bytes: num(f, "bytes"),
                        modified: num(f, "modified"),
                        status: f.get("status").and_then(Value::as_str).unwrap_or("").to_string(),
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    let counts = v.get("counts").cloned().unwrap_or(Value::Null);
    UgFiles {
        total: v.get("total").and_then(Value::as_u64).map(|t| t as usize).unwrap_or(files.len()),
        drifted: (num(&counts, "changed") + num(&counts, "missing")) as usize,
        files,
    }
}

/// A project's indexed files, through `ug files` (ug's interface; Andai never
/// reads ug's data folder). An older ug without the command says so.
pub(crate) fn ug_files(project: &str) -> Result<UgFiles, String> {
    let mut cmd = ug()?;
    cmd.args(["files", "-n", project, "--json", "-k", &MAX_LISTED_SOURCES.to_string(), "--no-banner"]);
    let out = cmd.output().map_err(|e| format!("failed to run ug: {e}"))?;
    match serde_json::from_slice::<Value>(&out.stdout) {
        Ok(v) if out.status.success() => Ok(parse_files(&v)),
        _ => Err(files_error(&String::from_utf8_lossy(&out.stderr))),
    }
}

/// What to tell the user when `ug files` fails. A ug from before the command
/// (0.1.22 and older) prints its whole help and ends stderr with
/// `error: unknown command: files`.
fn files_error(stderr: &str) -> String {
    let err = strip_ansi(stderr);
    let last = err.trim().lines().last().unwrap_or("").trim();
    if last.to_ascii_lowercase().contains("unknown command: files") {
        UG_TOO_OLD.into()
    } else if last.is_empty() {
        "ug files failed".into()
    } else {
        last.trim_start_matches("error:").trim().to_string()
    }
}

pub(crate) const UG_TOO_OLD: &str =
    "This version of ug can't list a project's files (`ug files`). Update it with `ug upgrade`, then refresh.";

/// Display type from a file. Andai stores TXT and CSV as Markdown it writes
/// (`ingest_file`), headed by the original name, which tells them apart.
fn kind_of(path: &Path) -> &'static str {
    const CODE: [&str; 9] = ["ts", "tsx", "js", "jsx", "mjs", "cjs", "py", "java", "rs"];
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    match ext.as_str() {
        "pdf" => "PDF",
        "txt" | "text" | "log" | "rst" => "TXT",
        "csv" | "tsv" => "CSV",
        e if CODE.contains(&e) => "CODE",
        "md" | "markdown" | "mdx" => {
            let mut head = [0u8; 256];
            let n = fs::File::open(path).and_then(|mut f| f.read(&mut head)).unwrap_or(0);
            let first = String::from_utf8_lossy(&head[..n]).lines().next().unwrap_or("").to_ascii_lowercase();
            let converted = |exts: &[&str]| first.starts_with("# ") && exts.iter().any(|e| first.ends_with(&format!(".{e}")));
            if converted(&["txt", "text", "log", "rst"]) {
                "TXT"
            } else if converted(&["csv", "tsv"]) {
                "CSV"
            } else {
                "MD"
            }
        }
        _ => "TXT",
    }
}

fn mtime(meta: &fs::Metadata) -> u64 {
    meta.modified().ok().and_then(|t| t.duration_since(UNIX_EPOCH).ok()).map(|d| d.as_secs()).unwrap_or(0)
}

/// One of Andai's own copies in `docs/` (Andai's data, read directly).
fn copy_at(docs: &Path, file: &str, status: &str) -> Source {
    let path = docs.join(file);
    let meta = fs::metadata(&path).ok();
    let bytes = meta.as_ref().map(|m| m.len()).unwrap_or(0);
    let kind = kind_of(&path);
    Source {
        file: file.into(),
        kind: kind.into(),
        bytes,
        approx_tokens: if kind == "PDF" { None } else { Some(bytes / 4) },
        added_at: meta.as_ref().map(mtime).unwrap_or(0),
        status: status.into(),
    }
}

/// A file of another ug project, as `ug files` reports it.
fn project_source(f: &UgFile) -> Source {
    let kind = match f.language.as_str() {
        "markdown" => "MD",
        "pdf" => "PDF",
        _ => "CODE",
    };
    Source {
        file: f.path.clone(),
        kind: kind.into(),
        bytes: f.bytes,
        approx_tokens: if kind == "PDF" { None } else { Some(f.bytes / 4) },
        added_at: f.modified,
        status: if f.status == "fresh" { "indexed" } else { "pending" }.into(),
    }
}

/// Andai's copies in `docs/`: indexed when ug lists the file as fresh, else
/// pending (failed after a failed run).
fn managed_sources(docs: &Path, ug: &HashMap<&str, &str>, failed: bool) -> Vec<Source> {
    let mut names: Vec<String> = fs::read_dir(docs)
        .map(|d| {
            d.flatten()
                .filter(|e| e.file_type().map(|t| t.is_file()).unwrap_or(false))
                .filter_map(|e| e.file_name().to_str().map(String::from))
                .filter(|n| !n.starts_with('.'))
                .collect()
        })
        .unwrap_or_default();
    names.sort();
    names
        .iter()
        .map(|n| {
            let status = match ug.get(n.as_str()) {
                Some(&"fresh") => "indexed",
                _ if failed => "failed",
                _ => "pending",
            };
            copy_at(docs, n, status)
        })
        .collect()
}

fn kb_status(engine: bool, indexing: bool, sources: usize, failed: bool, pending: bool) -> &'static str {
    if !engine {
        "offline"
    } else if indexing {
        "indexing"
    } else if sources == 0 {
        "empty"
    } else if failed {
        "failed"
    } else if pending {
        "pending"
    } else {
        "ready"
    }
}

fn kind_from_ug(kind: Option<&str>) -> Option<KbKind> {
    match kind? {
        "docs" | "document" | "documents" => Some(KbKind::Document),
        "code" => Some(KbKind::Code),
        "mixed" => Some(KbKind::Mixed),
        _ => None,
    }
}

/// `andai-<slug>` → the folder Andai keeps its files in, if it has one.
fn managed_docs(app: &AppHandle, project: &str) -> Result<Option<PathBuf>, String> {
    match project.strip_prefix(PROJECT_PREFIX).filter(|s| valid_slug(s)) {
        Some(slug) => Ok(Some(kb_dir(app, slug)?.join("docs")).filter(|d| d.is_dir())),
        None => Ok(None),
    }
}

/// Everything the UI shows about one knowledge base. `docs` is Andai's folder
/// for it, `p` ug's project and `files` what `ug files` said about it (none
/// before the first index or without ug).
fn build_info(
    project: &str,
    docs: Option<&Path>,
    p: Option<&UgProject>,
    files: Option<&Result<UgFiles, String>>,
    engine: bool,
    state: &Indexing,
) -> KbInfo {
    let indexing = state.busy.lock().unwrap().contains(project);
    let mut last_error = state.errors.lock().unwrap().get(project).cloned();
    let listed = match files {
        Some(Ok(f)) => Some(f),
        Some(Err(e)) => {
            last_error.get_or_insert_with(|| e.clone());
            None
        }
        None => None,
    };
    let (sources, source_count, root) = match docs {
        Some(docs) => {
            let ug: HashMap<&str, &str> =
                listed.map(|f| f.files.iter().map(|x| (x.path.as_str(), x.status.as_str())).collect()).unwrap_or_default();
            let sources = managed_sources(docs, &ug, last_error.is_some());
            let n = sources.len();
            (sources, n, docs.to_path_buf())
        }
        None => {
            let root = p.map(|p| p.repo_root.clone()).unwrap_or_default();
            let sources: Vec<Source> = listed.map(|f| f.files.iter().map(project_source).collect()).unwrap_or_default();
            let n = listed.map(|f| f.total).unwrap_or(sources.len());
            (sources, n, root)
        }
    };
    let pending = sources.iter().any(|s| s.status != "indexed")
        || (docs.is_none() && (p.is_some_and(|p| p.stale) || listed.is_some_and(|f| f.drifted > 0)));
    let last_error = last_error.or_else(|| {
        p.filter(|p| p.repo_missing).map(|p| format!("The indexed folder {} is gone.", p.repo_root.display()))
    });
    let status = kb_status(engine, indexing, source_count, last_error.is_some(), pending);
    let kind = match docs {
        Some(_) => derive_kind(&sources),
        None => kind_from_ug(p.and_then(|p| p.kb_kind.as_deref())).unwrap_or_else(|| derive_kind(&sources)),
    };
    let folder_created = docs.and_then(|d| fs::metadata(d).ok()).map(|m| {
        m.created().ok().and_then(|t| t.duration_since(UNIX_EPOCH).ok()).map(|d| d.as_secs()).unwrap_or_else(|| mtime(&m))
    });
    KbInfo {
        slug: project.into(),
        name: project.strip_prefix(PROJECT_PREFIX).unwrap_or(project).into(),
        managed: docs.is_some(),
        root: root.to_string_lossy().into(),
        created_at: p.map(|p| p.created_at).filter(|t| *t > 0).or(folder_created).unwrap_or(0),
        last_indexed_at: p.filter(|p| p.nodes > 0).map(|p| p.updated_at),
        last_error,
        sources,
        source_count,
        status: status.into(),
        kind,
        nodes: p.map(|p| p.nodes).unwrap_or(0),
        edges: p.map(|p| p.edges).unwrap_or(0),
        size_bytes: p.map(|p| p.size_bytes).unwrap_or(0),
    }
}

/// A knowledge base the webview named: one of Andai's folders, or a project
/// ug lists. Its root comes from app data or ug's registry, never the webview.
pub(crate) struct Kb {
    pub project: String,
    pub root: PathBuf,
    pub managed: bool,
}

pub(crate) fn resolve(app: &AppHandle, project: &str) -> Result<Kb, String> {
    if !valid_project(project) {
        return Err(format!("invalid knowledge base id: {project}"));
    }
    if let Some(docs) = managed_docs(app, project)? {
        return Ok(Kb { project: project.into(), root: docs, managed: true });
    }
    let p = ug_projects()
        .ok_or(UG_MISSING)?
        .into_iter()
        .find(|p| p.name == project)
        .ok_or_else(|| format!("There's no knowledge base named “{project}”."))?;
    Ok(Kb { project: p.name, root: p.repo_root, managed: false })
}

pub(crate) fn load_info(app: &AppHandle, project: &str, state: &Indexing) -> Result<KbInfo, String> {
    let kb = resolve(app, project)?;
    let projects = ug_projects();
    let p = projects.as_ref().and_then(|ps| ps.iter().find(|p| p.name == kb.project));
    let files = p.map(|p| ug_files(&p.name));
    Ok(build_info(&kb.project, kb.managed.then_some(kb.root.as_path()), p, files.as_ref(), projects.is_some(), state))
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
    UgStatus {
        found: path.is_some(),
        path: path.map(|p| p.to_string_lossy().into()),
        version,
        can_install: crate::ug_install::asset_name().is_some(),
        install_command: UG_INSTALL_COMMAND,
    }
}

/// Opens UltraGraph's site in the default browser. It takes no argument, so
/// the webview can't use it to open anything else, and the app itself makes
/// no request (the browser does).
#[tauri::command]
pub fn open_ug_website() -> Result<(), String> {
    open_in_system(UG_WEBSITE.as_ref()).map_err(|e| format!("could not open the browser: {e}"))
}

/// Hands `target` (a URL or a folder Rust chose, never one from the webview)
/// to the system's default handler: Finder or the browser.
pub(crate) fn open_in_system(target: &std::ffi::OsStr) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    let mut cmd = Command::new("/usr/bin/open");
    #[cfg(windows)]
    let mut cmd = {
        let root = std::env::var_os("SystemRoot").map(PathBuf::from).filter(|p| p.is_absolute());
        let mut c = Command::new(root.ok_or("SystemRoot is not set")?.join("System32").join("rundll32.exe"));
        c.arg("url.dll,FileProtocolHandler");
        c
    };
    #[cfg(all(unix, not(target_os = "macos")))]
    let mut cmd = Command::new("xdg-open");
    cmd.arg(target).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null());
    match cmd.status() {
        Ok(s) if s.success() => Ok(()),
        Ok(s) => Err(s.to_string()),
        Err(e) => Err(e.to_string()),
    }
}

/// Every knowledge base: each ug project, then Andai's folders ug has no
/// project for (never indexed, or the graph was removed). Without ug, only
/// Andai's folders, all `offline`.
#[tauri::command]
pub async fn kb_list(app: AppHandle, indexing: State<'_, Indexing>) -> Result<Vec<KbInfo>, String> {
    let root = kb_root(&app)?;
    // One `ug list`, then one `ug files` per project, side by side: each is a
    // short process, and in sequence a long list of projects would add up.
    let (projects, files) = tauri::async_runtime::spawn_blocking(|| {
        let projects = ug_projects();
        let files: Vec<Result<UgFiles, String>> = std::thread::scope(|s| {
            let handles: Vec<_> = projects.iter().flatten().map(|p| s.spawn(|| ug_files(&p.name))).collect();
            handles.into_iter().map(|h| h.join().unwrap_or_else(|_| Err("ug files failed".into()))).collect()
        });
        (projects, files)
    })
    .await
    .map_err(|e| e.to_string())?;
    let engine = projects.is_some();
    let projects = projects.unwrap_or_default();
    let mut out: Vec<KbInfo> = projects
        .iter()
        .zip(&files)
        .map(|(p, f)| Ok(build_info(&p.name, managed_docs(&app, &p.name)?.as_deref(), Some(p), Some(f), engine, &indexing)))
        .collect::<Result<_, String>>()?;
    for entry in fs::read_dir(&root).map_err(|e| e.to_string())?.flatten() {
        let Some(slug) = entry.file_name().to_str().map(String::from).filter(|s| valid_slug(s)) else { continue };
        let project = format!("{PROJECT_PREFIX}{slug}");
        let docs = entry.path().join("docs");
        if docs.is_dir() && !projects.iter().any(|p| p.name == project) {
            out.push(build_info(&project, Some(&docs), None, None, engine, &indexing));
        }
    }
    out.sort_by(|a, b| a.created_at.cmp(&b.created_at).then_with(|| a.slug.cmp(&b.slug)));
    Ok(out)
}

#[tauri::command]
pub fn kb_create(app: AppHandle, name: String, indexing: State<'_, Indexing>) -> Result<KbInfo, String> {
    let name = name.trim();
    if name.is_empty() {
        return Err("Give the knowledge base a name.".into());
    }
    let base = slugify(name);
    let taken: HashSet<String> = ug_projects().unwrap_or_default().into_iter().map(|p| p.name).collect();
    let mut slug = base.clone();
    let mut n = 2;
    while kb_dir(&app, &slug)?.exists() || taken.contains(&format!("{PROJECT_PREFIX}{slug}")) {
        slug = format!("{base}-{n}");
        n += 1;
    }
    create_kb(&app, &slug, &indexing)
}

/// A new, empty knowledge base `andai-<slug>`. With ug, it's registered as a
/// project right away (an empty `ug gen` takes a moment), so ug lists it too.
pub(crate) fn create_kb(app: &AppHandle, slug: &str, indexing: &Indexing) -> Result<KbInfo, String> {
    let docs = kb_dir(app, slug)?.join("docs");
    create_private_dir(&docs).map_err(|e| e.to_string())?;
    let project = format!("{PROJECT_PREFIX}{slug}");
    if let Ok(mut cmd) = ug() {
        let _ = cmd.arg("gen").arg(&docs).args(["-n", &project]).current_dir(&docs).stdout(Stdio::null()).output();
    }
    load_info(app, &project, indexing)
}

/// Ingests each path the user granted (see grants.rs) into `docs/`; returns
/// one actionable message per file that was skipped.
fn add_sources(docs: &Path, paths: &[String], grants: &FileGrants) -> Vec<String> {
    let mut errors = vec![];
    for p in paths {
        if Path::new(p).is_dir() {
            errors.push(format!("{p}: folders aren't supported yet — drop the files inside it"));
            continue;
        }
        if let Err(e) = grants.take(Path::new(p)).and_then(|src| ingest_file(docs, &src)) {
            errors.push(e);
        }
    }
    errors
}

/// Only Andai's own knowledge bases take files: another project indexes the
/// user's folder, which Andai never writes to.
fn managed(app: &AppHandle, project: &str) -> Result<Kb, String> {
    let kb = resolve(app, project)?;
    if !kb.managed {
        return Err(format!(
            "“{project}” indexes {} with ug; Andai doesn't change its files. Create a knowledge base in Andai to add files.",
            kb.root.display()
        ));
    }
    Ok(kb)
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
    let kb = managed(&app, &slug)?;
    let errors = add_sources(&kb.root, &paths, &grants);
    Ok((load_info(&app, &slug, &indexing)?, errors))
}

#[tauri::command]
pub fn kb_remove_source(app: AppHandle, slug: String, file: String, indexing: State<'_, Indexing>) -> Result<KbInfo, String> {
    let kb = managed(&app, &slug)?;
    if !valid_source_name(&file) {
        return Err("invalid file name".into());
    }
    let _ = fs::remove_file(kb.root.join(&file));
    load_info(&app, &slug, &indexing)
}

/// Removes ug's graph and, for Andai's own, its copies of the files. Another
/// project's folder is never touched: only ug's data for it goes.
#[tauri::command]
pub async fn kb_delete(app: AppHandle, slug: String, indexing: State<'_, Indexing>) -> Result<(), String> {
    let kb = resolve(&app, &slug)?;
    if indexing.busy.lock().unwrap().contains(&kb.project) {
        return Err("Wait for indexing to finish before deleting this knowledge base.".into());
    }
    indexing.errors.lock().unwrap().remove(&kb.project);
    tauri::async_runtime::spawn_blocking(move || {
        match ug() {
            Ok(mut cmd) => {
                let out = cmd.args(["remove", &kb.project, "-y"]).output().map_err(|e| e.to_string())?;
                if !out.status.success() && !kb.managed {
                    let err = strip_ansi(&String::from_utf8_lossy(&out.stderr));
                    return Err(err.trim().lines().last().unwrap_or("ug remove failed").to_string());
                }
            }
            Err(e) if !kb.managed => return Err(e),
            Err(_) => {}
        }
        if kb.managed {
            // `root` is `kb/<slug>/docs`; the folder to remove is its parent.
            let dir = kb.root.parent().ok_or("invalid knowledge base folder")?;
            fs::remove_dir_all(dir).map_err(|e| e.to_string())?;
        }
        Ok(())
    })
    .await
    .map_err(|e| e.to_string())?
}

/// Runs `ug gen --with-embed` over the knowledge base's folder, streaming its
/// progress lines as `kb-progress` events. Resolves with the updated KB once
/// the run finishes.
#[tauri::command]
pub async fn kb_index(app: AppHandle, slug: String, indexing: State<'_, Indexing>) -> Result<KbInfo, String> {
    let kb = resolve(&app, &slug)?;
    if !kb.root.is_dir() {
        return Err(format!("The folder {} is gone, so there's nothing to index.", kb.root.display()));
    }
    if !indexing.busy.lock().unwrap().insert(kb.project.clone()) {
        return Err("This knowledge base is already indexing.".into());
    }
    let _ = app.emit("kb-progress", Progress { slug: &slug, line: "Starting ug gen…".into() });

    let (app2, slug2) = (app.clone(), slug.clone());
    let (project, root, own) = (kb.project.clone(), kb.root.clone(), kb.managed);
    let result = tauri::async_runtime::spawn_blocking(move || -> Result<(), String> {
        let empty = fs::read_dir(&root).map(|mut d| d.next().is_none()).unwrap_or(true);
        if own && empty {
            // Nothing left to index: drop the graph so search can't return stale hits.
            let _ = ug()?.args(["remove", &project, "-y"]).output();
            return Ok(());
        }
        let mut child = ug()?
            .arg("gen")
            .arg(&root)
            .args(["-n", &project, "--with-embed"])
            .current_dir(&root)
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

    indexing.busy.lock().unwrap().remove(&kb.project);
    match &result {
        Ok(()) => indexing.errors.lock().unwrap().remove(&kb.project),
        Err(e) => indexing.errors.lock().unwrap().insert(kb.project.clone(), e.clone()),
    };
    let _ = app.emit(
        "kb-progress",
        Progress { slug: &slug, line: result.clone().map(|_| "Index ready.".into()).unwrap_or_else(|e| e) },
    );
    load_info(&app, &slug, &indexing)
}

/// GraphRAG search over one knowledge base: `ug search … --snippets --json`.
#[tauri::command]
pub async fn kb_search(app: AppHandle, slug: String, query: String, k: u32, max_chars: u32) -> Result<Value, String> {
    let kb = resolve(&app, &slug)?;
    let query = search_query(&query)?;
    let (k, max_chars) = search_limits(k, max_chars);
    tauri::async_runtime::spawn_blocking(move || {
        let mut cmd = ug()?;
        cmd.arg("search")
            .arg(&query)
            .args(["-n", &kb.project])
            .args(["-k", &k.to_string(), "--max-chars", &max_chars.to_string()])
            .arg("--snippets")
            .arg("--repo-root")
            .arg(&kb.root)
            .arg("--json")
            .current_dir(&kb.root);
        run_json(cmd)
    })
    .await
    .map_err(|e| e.to_string())?
}

/// Most of a source's text the source dialog is sent; a larger file is cut here.
const MAX_VIEW_BYTES: usize = 512 * 1024;
/// Budget for ug's report on one file: its outline comes first, relations after.
const STRUCTURE_CHARS: u32 = 30_000;

/// One source as the source dialog shows it: its metadata, its text, and what ug
/// indexed from it.
#[derive(Serialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct SourceView {
    pub source: Source,
    /// Where the file is on disk.
    pub path: String,
    /// The file as text, up to `MAX_VIEW_BYTES`. `None` for a PDF: its text
    /// exists only in ug's index (the outline's page entries).
    pub text: Option<String>,
    pub text_truncated: bool,
    /// ug's `file_context` report: facts, outline and related files.
    pub structure: Option<Value>,
    /// Why there's no structure (not indexed yet, ug missing, ug failed).
    pub structure_error: Option<String>,
}

/// The source the knowledge base lists under exactly this name. The webview
/// names a file; only one the knowledge base lists is ever read (AGENTS.md §9).
fn listed_source<'a>(sources: &'a [Source], file: &str) -> Result<&'a Source, String> {
    if !safe_rel_path(file) {
        return Err("invalid file name".into());
    }
    sources.iter().find(|s| s.file == file).ok_or_else(|| format!("“{file}” isn't in this knowledge base."))
}

/// Up to `cap` bytes of a regular file inside `root` as text. A symlink is
/// refused, not followed, and so is anything that resolves outside `root`.
fn read_text_capped(root: &Path, file: &str, cap: usize) -> Result<(String, bool), String> {
    let path = root.join(file);
    let meta = fs::symlink_metadata(&path).map_err(|e| format!("Can't read the file: {e}"))?;
    if !meta.is_file() {
        return Err("The file isn't a regular file.".into());
    }
    let (Ok(real), Ok(base)) = (path.canonicalize(), root.canonicalize()) else {
        return Err("Can't resolve the file.".into());
    };
    if !real.starts_with(&base) {
        return Err("The file is outside the knowledge base.".into());
    }
    let mut bytes = Vec::new();
    fs::File::open(&real)
        .and_then(|f| f.take(cap as u64 + 1).read_to_end(&mut bytes))
        .map_err(|e| format!("Can't read the file: {e}"))?;
    let truncated = bytes.len() > cap;
    bytes.truncate(cap);
    let mut text = String::from_utf8_lossy(&bytes).into_owned();
    if truncated {
        // The cut can split a character; drop the replacement it leaves.
        while text.ends_with('\u{FFFD}') {
            text.pop();
        }
    }
    Ok((text, truncated))
}

/// ug's arguments for one file's report. The file goes by its node id
/// (`file:<name>`), so a name that starts with `-` can't parse as a flag.
fn structure_args(project: &str, file: &str) -> Vec<String> {
    vec![
        "file_context".into(),
        format!("file:{file}"),
        "--max-chars".into(),
        STRUCTURE_CHARS.to_string(),
        "-k".into(),
        "1".into(),
        "-n".into(),
        project.into(),
        "--json".into(),
    ]
}

/// Everything the source dialog shows about one source: what the knowledge
/// base lists for it, its text, and ug's outline and relations for it.
#[tauri::command]
pub async fn kb_source(app: AppHandle, slug: String, file: String, indexing: State<'_, Indexing>) -> Result<SourceView, String> {
    let info = load_info(&app, &slug, &indexing)?;
    let kb = resolve(&app, &slug)?;
    let source = listed_source(&info.sources, &file)?.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let (text, text_truncated) = if source.kind == "PDF" {
            (None, false)
        } else {
            let (t, cut) = read_text_capped(&kb.root, &source.file, MAX_VIEW_BYTES)?;
            (Some(t), cut)
        };
        let report = if source.status != "indexed" {
            Err("Not indexed yet: its structure appears once ug has indexed it.".to_string())
        } else {
            ug_path().ok_or_else(|| UG_MISSING.to_string()).and_then(|bin| {
                let args = structure_args(&kb.project, &source.file);
                crate::tools::run(&bin, &args, &kb.root, std::time::Duration::from_secs(20), 1024 * 1024)
            })
        };
        let (structure, structure_error) = match report {
            Ok(out) if out.output.is_object() => (Some(out.output), None),
            Ok(_) => (None, Some("ug's report on this file was too large to read.".into())),
            Err(e) => (None, Some(e)),
        };
        let path = kb.root.join(&source.file).to_string_lossy().into_owned();
        Ok(SourceView { source, path, text, text_truncated, structure, structure_error })
    })
    .await
    .map_err(|e| e.to_string())?
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn source(status: &str) -> Source {
        source_of("MD", status)
    }

    fn source_of(kind: &str, status: &str) -> Source {
        Source {
            file: "a.md".into(),
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
    fn the_source_dialog_reads_only_listed_sources() {
        let nested = Source { file: "src/fare.ts".into(), ..source("indexed") };
        let listed = [source("indexed"), nested];
        assert_eq!(listed_source(&listed, "a.md").unwrap().file, "a.md");
        assert_eq!(listed_source(&listed, "src/fare.ts").unwrap().file, "src/fare.ts");
        for bad in ["kb.json", "../a.md", "b.md", "/a.md", "src/../a.md", ""] {
            assert!(listed_source(&listed, bad).is_err(), "{bad:?} should be refused");
        }
    }

    #[test]
    fn source_text_is_capped_and_never_follows_a_symlink() {
        let d = tempfile::tempdir().unwrap();
        fs::write(d.path().join("a.md"), "héllo").unwrap();
        assert_eq!(read_text_capped(d.path(), "a.md", 64).unwrap(), ("héllo".into(), false));
        // A cut inside "é" (2 bytes) drops the half character instead of showing U+FFFD.
        assert_eq!(read_text_capped(d.path(), "a.md", 2).unwrap(), ("h".into(), true));
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(d.path().join("a.md"), d.path().join("link.md")).unwrap();
            assert!(read_text_capped(d.path(), "link.md", 64).is_err());
            // A folder linked from outside the root doesn't let a file escape it.
            let outside = tempfile::tempdir().unwrap();
            fs::write(outside.path().join("secret.md"), "key").unwrap();
            std::os::unix::fs::symlink(outside.path(), d.path().join("out")).unwrap();
            assert!(read_text_capped(d.path(), "out/secret.md", 64).is_err());
        }
        fs::create_dir(d.path().join("dir")).unwrap();
        assert!(read_text_capped(d.path(), "dir", 64).is_err(), "a directory isn't a source");
    }

    #[test]
    fn structure_args_name_the_file_by_id_so_it_cannot_be_a_flag() {
        let a = structure_args("andai-kb", "--base-url.md");
        assert_eq!(a[..2], ["file_context", "file:--base-url.md"]);
        assert!(a.ends_with(&["-n".into(), "andai-kb".into(), "--json".into()]));
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

        let paths = [mine, secret].map(|p| p.to_string_lossy().to_string());
        let errors = add_sources(docs.path(), &paths, &grants);
        let added: Vec<_> = fs::read_dir(docs.path()).unwrap().flatten().map(|e| e.file_name()).collect();
        assert_eq!(added, ["mine.md"]);
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
    fn status_says_what_blocks_the_knowledge_base_first() {
        assert_eq!(kb_status(false, false, 3, false, false), "offline", "no ug: nothing works");
        assert_eq!(kb_status(true, true, 3, true, true), "indexing", "indexing wins");
        assert_eq!(kb_status(true, false, 0, false, false), "empty");
        assert_eq!(kb_status(true, false, 3, true, true), "failed");
        assert_eq!(kb_status(true, false, 3, false, true), "pending");
        assert_eq!(kb_status(true, false, 3, false, false), "ready");
    }

    #[test]
    fn kind_is_derived_from_sources_or_taken_from_ug() {
        assert_eq!(derive_kind(&[]), KbKind::Document);
        assert_eq!(derive_kind(&[source("indexed"), source_of("PDF", "indexed")]), KbKind::Document);
        assert_eq!(derive_kind(&[source_of("CODE", "indexed")]), KbKind::Code);
        assert_eq!(derive_kind(&[source("indexed"), source_of("CODE", "indexed")]), KbKind::Mixed);
        assert_eq!(kind_from_ug(Some("docs")), Some(KbKind::Document));
        assert_eq!(kind_from_ug(Some("code")), Some(KbKind::Code));
        assert_eq!(kind_from_ug(Some("mixed")), Some(KbKind::Mixed));
        assert_eq!(kind_from_ug(Some("weird")), None);
    }

    #[test]
    fn project_names_and_paths_from_the_webview_are_held_to_safe_shapes() {
        for ok in ["andai-docs", "tidewater-code", "my.repo_2"] {
            assert!(valid_project(ok), "{ok:?}");
        }
        for bad in ["", "-n", "--base-url", ".hidden", "a/b", "a b", "..", "a\0b", &"x".repeat(129)] {
            assert!(!valid_project(bad), "{bad:?} should be rejected");
        }
        for ok in ["a.md", "src/fare.ts", "docs/guide/intro.md"] {
            assert!(safe_rel_path(ok), "{ok:?}");
        }
        for bad in ["", "/etc/passwd", "../a", "src/../../a", "-flag", "a\\b", "./a", "a\nb"] {
            assert!(!safe_rel_path(bad), "{bad:?} should be rejected");
        }
    }

    fn project(name: &str, root: &Path) -> Value {
        json!({
            "name": name, "repoRoot": root, "nodes": 9, "edges": 8, "sizeBytes": 1234,
            "createdAt": 100, "updatedAt": 4_000_000_000u64, "isStale": false, "hasDb": true,
            "repoMissing": false, "kbKind": "code",
        })
    }

    #[test]
    fn reads_ug_list_and_skips_projects_that_dont_look_real() {
        let list = json!({ "projects": [
            project("andai-docs", Path::new("/r")),
            project("--evil", Path::new("/r")),
            { "name": "relative", "repoRoot": "r" },
            { "name": "no-root" },
        ]});
        let ps = parse_projects(&list);
        assert_eq!(ps.len(), 1);
        assert_eq!((ps[0].name.as_str(), ps[0].nodes, ps[0].updated_at), ("andai-docs", 9, 4_000_000_000));
        assert_eq!(ps[0].kb_kind.as_deref(), Some("code"));
        // ug 0.1.22 prints a sentence, not JSON, when there are no projects.
        assert!(parse_projects(&json!({})).is_empty());
        let mut never_ingested = project("p", Path::new("/r"));
        never_ingested["hasDb"] = json!(false);
        assert!(parse_projects(&json!({ "projects": [never_ingested] }))[0].stale);
    }

    fn ug_project(name: &str, root: &Path) -> UgProject {
        parse_projects(&json!({ "projects": [project(name, root)] })).remove(0)
    }

    /// What `ug files --json` prints: `(path, language, bytes, status)` per file.
    fn listed(files: &[(&str, &str, u64, &str)]) -> Result<UgFiles, String> {
        let changed = files.iter().filter(|f| f.3 != "fresh").count();
        Ok(parse_files(&json!({
            "total": files.len(),
            "counts": { "fresh": files.len() - changed, "changed": changed, "missing": 0 },
            "files": files.iter().map(|(path, language, bytes, status)| json!({
                "path": path, "ext": "", "language": language, "kind": "", "bytes": bytes, "modified": 7, "status": status,
            })).collect::<Vec<_>>(),
        })))
    }

    #[test]
    fn reads_ug_files_and_drops_paths_that_leave_the_project() {
        let f = listed(&[("src/fare.ts", "typescript", 19, "fresh"), ("../escape.md", "markdown", 1, "fresh"), ("/abs.md", "markdown", 1, "fresh")]).unwrap();
        assert_eq!(f.files.iter().map(|x| x.path.as_str()).collect::<Vec<_>>(), ["src/fare.ts"]);
        assert_eq!((f.total, f.drifted), (3, 0));
        let drift = parse_files(&json!({ "total": 40, "counts": { "changed": 2, "missing": 1 }, "files": [] }));
        assert_eq!((drift.total, drift.drifted), (40, 3), "counts cover every match, not just the listed window");
    }

    #[test]
    fn andai_knowledge_bases_list_their_copies_against_what_ug_indexed() {
        let docs = tempfile::tempdir().unwrap();
        fs::write(docs.path().join("a.md"), "# A").unwrap();
        fs::write(docs.path().join("b.md"), "# B").unwrap();
        fs::write(docs.path().join("c.md"), "# C").unwrap();
        fs::write(docs.path().join("log.md"), "# log.txt\n\nline").unwrap();
        fs::write(docs.path().join(".DS_Store"), "").unwrap();
        let p = ug_project("andai-docs", docs.path());
        let files = listed(&[("a.md", "markdown", 3, "fresh"), ("c.md", "markdown", 3, "changed"), ("log.md", "markdown", 14, "fresh")]);
        let state = Indexing::default();

        let kb = build_info("andai-docs", Some(docs.path()), Some(&p), Some(&files), true, &state);
        assert_eq!((kb.name.as_str(), kb.managed, kb.nodes, kb.size_bytes), ("docs", true, 9, 1234));
        let got: Vec<_> = kb.sources.iter().map(|s| (s.file.as_str(), s.kind.as_str(), s.status.as_str())).collect();
        assert_eq!(got, [("a.md", "MD", "indexed"), ("b.md", "MD", "pending"), ("c.md", "MD", "pending"), ("log.md", "TXT", "indexed")]);
        assert_eq!((kb.status.as_str(), kb.kind), ("pending", KbKind::Document), "new and edited files wait for the next index");

        state.errors.lock().unwrap().insert("andai-docs".into(), "ug gen failed: boom".into());
        let failed = build_info("andai-docs", Some(docs.path()), Some(&p), Some(&files), true, &state);
        assert_eq!((failed.status.as_str(), failed.sources[1].status.as_str()), ("failed", "failed"));
        assert_eq!(failed.last_error.as_deref(), Some("ug gen failed: boom"));

        state.busy.lock().unwrap().insert("andai-docs".into());
        assert_eq!(build_info("andai-docs", Some(docs.path()), Some(&p), Some(&files), true, &state).status, "indexing");

        // ug gone: the folder is still listed, every knowledge base offline.
        let offline = build_info("andai-docs", Some(docs.path()), None, None, false, &Indexing::default());
        assert_eq!((offline.status.as_str(), offline.sources.len(), offline.nodes), ("offline", 4, 0));
    }

    #[test]
    fn other_ug_projects_list_the_files_ug_reports() {
        let repo = tempfile::tempdir().unwrap();
        let mut p = ug_project("tidewater", repo.path());
        let files = listed(&[("README.md", "markdown", 3, "fresh"), ("docs/guide.pdf", "pdf", 900, "fresh"), ("src/fare.ts", "typescript", 19, "fresh")]);
        let kb = build_info("tidewater", None, Some(&p), Some(&files), true, &Indexing::default());
        assert_eq!((kb.name.as_str(), kb.managed, kb.status.as_str()), ("tidewater", false, "ready"));
        assert_eq!(kb.root, repo.path().to_string_lossy());
        let got: Vec<_> = kb.sources.iter().map(|s| (s.file.as_str(), s.kind.as_str(), s.bytes, s.added_at)).collect();
        assert_eq!(got, [("README.md", "MD", 3, 7), ("docs/guide.pdf", "PDF", 900, 7), ("src/fare.ts", "CODE", 19, 7)]);
        assert_eq!(kb.sources[1].approx_tokens, None);
        assert_eq!(kb.kind, KbKind::Code, "ug's own kind");

        let edited = listed(&[("src/fare.ts", "typescript", 19, "changed")]);
        let pending = build_info("tidewater", None, Some(&p), Some(&edited), true, &Indexing::default());
        assert_eq!((pending.status.as_str(), pending.sources[0].status.as_str()), ("pending", "pending"));
        p.stale = true;
        assert_eq!(build_info("tidewater", None, Some(&p), Some(&files), true, &Indexing::default()).status, "pending");
        p.repo_missing = true;
        let gone = build_info("tidewater", None, Some(&p), Some(&files), true, &Indexing::default());
        assert_eq!(gone.status, "failed");
        assert!(gone.last_error.unwrap().contains("is gone"));
    }

    #[test]
    fn an_old_ug_is_told_apart_from_other_failures() {
        // ug 0.1.22's stderr, colours and all.
        assert_eq!(files_error("\u{1b}[31merror:\u{1b}[0m unknown command: files\n"), UG_TOO_OLD);
        assert_eq!(files_error("error: No project named \"x\"\n"), "No project named \"x\"");
        assert_eq!(files_error(""), "ug files failed");
    }

    #[test]
    fn a_ug_that_cannot_list_files_says_how_to_fix_it() {
        let repo = tempfile::tempdir().unwrap();
        let p = ug_project("tidewater", repo.path());
        let old: Result<UgFiles, String> = Err(UG_TOO_OLD.into());
        let kb = build_info("tidewater", None, Some(&p), Some(&old), true, &Indexing::default());
        assert_eq!((kb.status.as_str(), kb.sources.len()), ("empty", 0));
        assert_eq!(kb.last_error.as_deref(), Some(UG_TOO_OLD));
    }

    #[test]
    fn kb_info_serializes_camel_case_for_the_frontend() {
        let docs = tempfile::tempdir().unwrap();
        fs::write(docs.path().join("a.md"), "# A").unwrap();
        let kb = build_info("andai-a", Some(docs.path()), None, None, true, &Indexing::default());
        let v = serde_json::to_value(&kb).unwrap();
        for key in ["slug", "name", "managed", "root", "createdAt", "sources", "sourceCount", "lastIndexedAt", "lastError", "status", "kind", "nodes", "sizeBytes"] {
            assert!(v.get(key).is_some(), "missing {key} in {v}");
        }
        assert!(v["sources"][0].get("approxTokens").is_some());
        assert!(v["sources"][0].get("addedAt").is_some());
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
        let listed = ug_projects().unwrap_or_default().into_iter().find(|p| p.name == project);
        let files = ug_files(&project);
        let structure = crate::tools::run(
            &ug_path().unwrap(),
            &structure_args(&project, "notes.md"),
            docs.path(),
            std::time::Duration::from_secs(20),
            1024 * 1024,
        );
        cleanup();

        // The source dialog's outline: the file's headings, with their lines.
        let report = structure.expect("file_context JSON").output;
        let outline = serde_json::to_string(&report["files"][0]["items"]).unwrap();
        assert!(outline.contains("\"Isolation\"") && outline.contains("\"outline\""), "outline should list the headings: {outline}");

        let items = result.expect("search JSON")["items"].as_array().cloned().unwrap_or_default();
        let listed = listed.expect("project should appear in `ug list --json`");
        assert_eq!(listed.repo_root.canonicalize().unwrap(), docs.path().canonicalize().unwrap());
        let files = files.expect("ug files --json");
        assert_eq!(files.files.iter().map(|f| (f.path.as_str(), f.status.as_str())).collect::<Vec<_>>(), [("notes.md", "fresh")]);
        assert_eq!(files.total, 1);
        assert!(!items.is_empty(), "search returned no items");
        let text = serde_json::to_string(&items).unwrap();
        assert!(text.contains("Cross-Origin-Embedder-Policy"), "snippet should carry the passage: {text}");
    }
}
