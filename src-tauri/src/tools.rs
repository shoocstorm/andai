//! Agent tools: the read-only ug operations the model may ask for.
//!
//! The webview proposes a call; this module is the trust boundary (AGENTS.md
//! §9). A call is a closed enum, so an unknown tool or field fails to
//! deserialize. Every argument is validated here and passed to ug as its own
//! `arg()`. The run is scoped to one knowledge base's ug project with its
//! `docs/` dir as cwd, and it is time-boxed and output-capped. The child gets
//! a scrubbed environment, so no inherited setting can point ug at a remote
//! embedder, and no embedder flag is ever passed.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::io::Read;
use std::path::{Component, Path};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};
use tauri::AppHandle;

use crate::ug::{resolve, search_query, strip_ansi, ug_path};

/// Wall-clock limit for one tool run; ug answers these from its graph in well under a second.
const TOOL_TIMEOUT: Duration = Duration::from_secs(20);
/// Most stdout kept from one run. Anything past it is read and dropped, so the child never blocks on a full pipe.
const MAX_OUTPUT_BYTES: usize = 256 * 1024;
/// Symbol names, file paths and prefixes are short; anything longer is not one.
const MAX_NAME_BYTES: usize = 256;
const MAX_SYMBOLS: usize = 5;
const MAX_CODE_LINES: u32 = 400;
/// Rows one `ug analyze` returns at most (its default is 50).
const MAX_ANALYZE_ROWS: u32 = 50;

/// The `ug analyze` presets Andai runs (ug 0.1.22), and whether each takes a
/// `target` file. A closed list: no raw GQL (`--gql`), no other graph
/// (`--db`) and no preset argument but `target` ever comes from the webview.
pub const ANALYZE_PRESETS: &[(&str, bool)] = &[
    ("language_breakdown", false),
    ("file_kinds", false),
    ("biggest_files", false),
    ("size_histogram", false),
    ("where_to_start", false),
    ("dependency_fanin", false),
    ("risky_symbols", false),
    ("untested_symbols", false),
    ("undocumented_hotspots", false),
    ("long_functions", false),
    ("coupling_matrix", false),
    ("dead_code", false),
    ("test_ratio", false),
    ("impact", true),
    ("impact_summary", true),
    ("retest_scope", true),
    ("boundary_impact", true),
];

/// ug node types a lookup may be restricted to (`ug graph_schema`, ug 0.1.21).
pub const NODE_TYPES: &[&str] =
    &["Function", "Method", "Class", "Interface", "Struct", "Enum", "Trait", "Type", "Constant", "Variable", "Module", "File", "Concept"];

/// One tool call as the webview sends it: `{ "tool": "kb_search", "query": … }`.
#[derive(Deserialize, Debug, Clone, PartialEq)]
#[serde(tag = "tool", deny_unknown_fields)]
pub enum ToolCall {
    #[serde(rename = "kb_search")]
    Search { query: String, k: u32, expand: bool, max_chars: u32 },
    #[serde(rename = "kb_find_symbols")]
    FindSymbols { names: Vec<String>, node_type: Option<String>, file_prefix: Option<String> },
    #[serde(rename = "kb_symbol_context")]
    SymbolContext { symbol: String, max_chars: u32 },
    #[serde(rename = "kb_get_code")]
    GetCode { symbol: Option<String>, file: Option<String>, start: Option<u32>, end: Option<u32> },
    #[serde(rename = "kb_find_usages")]
    FindUsages { symbol: String },
    #[serde(rename = "kb_file_context")]
    FileContext { file: String, max_chars: u32 },
    #[serde(rename = "kb_overview")]
    Overview {},
    /// Whole-repo statistics (`ug analyze`); `target` is a file, for the presets that take one.
    #[serde(rename = "kb_analyze")]
    Analyze { preset: String, target: Option<String>, limit: Option<u32> },
}

#[derive(Serialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct ToolOutput {
    /// Parsed JSON, or the raw (clipped) text when it didn't parse or was truncated.
    pub output: Value,
    pub truncated: bool,
    pub bytes: usize,
    pub ms: u64,
    /// The ug arguments as run, for the Execution Trace.
    pub argv: Vec<String>,
}

// ── validation ───────────────────────────────────────────────────────────

/// A symbol name, node id or pattern. ug has no `--` separator, so a leading
/// `-` would parse as a flag (AGENTS.md §2); no real symbol starts with one.
fn symbol(s: &str) -> Result<String, String> {
    let s = s.trim();
    if s.is_empty() || s.len() > MAX_NAME_BYTES {
        return Err(format!("Symbol must be 1–{MAX_NAME_BYTES} bytes."));
    }
    if s.starts_with('-') || s.chars().any(char::is_control) {
        return Err(format!("Not a symbol name: {s:?}"));
    }
    Ok(s.to_string())
}

/// A repo-relative path or glob inside the knowledge base's `docs/`. It must
/// not be absolute, climb out with `..`, or look like a flag. If it names a
/// file that exists, the resolved file must still be inside `docs/`.
fn rel_path(s: &str, docs: &Path) -> Result<String, String> {
    let s = s.trim();
    if s.is_empty() || s.len() > MAX_NAME_BYTES || s.starts_with('-') || s.contains('\\') {
        return Err(format!("Not a knowledge-base path: {s:?}"));
    }
    if s.chars().any(char::is_control) {
        return Err("Path contains a control character.".into());
    }
    let p = Path::new(s);
    if !p.components().all(|c| matches!(c, Component::Normal(_) | Component::CurDir)) {
        return Err(format!("Path must stay inside the knowledge base: {s:?}"));
    }
    let joined = docs.join(p);
    if joined.exists() {
        let (Ok(real), Ok(root)) = (joined.canonicalize(), docs.canonicalize()) else {
            return Err(format!("Can't resolve {s:?}"));
        };
        if !real.starts_with(&root) {
            return Err(format!("Path must stay inside the knowledge base: {s:?}"));
        }
    }
    Ok(s.to_string())
}

/// Validates a call and returns ug's arguments (after the binary), in order.
/// Pure, so the whole policy is unit-tested without running ug.
pub fn plan(call: &ToolCall, project: &str, docs: &Path) -> Result<Vec<String>, String> {
    let mut a: Vec<String> = vec![];
    match call {
        ToolCall::Search { query, k, expand, max_chars } => {
            if query.trim().is_empty() {
                return Err("Search query is empty.".into());
            }
            a.extend(["search".into(), search_query(query)?]);
            a.extend(["-k".into(), (*k).clamp(1, 20).to_string()]);
            a.extend(["--max-chars".into(), (*max_chars).clamp(100, 8000).to_string()]);
            if !expand {
                a.push("--no-expand".into());
            }
            a.extend(["--snippets".into(), "--repo-root".into(), docs.to_string_lossy().into()]);
        }
        ToolCall::FindSymbols { names, node_type, file_prefix } => {
            if names.is_empty() || names.len() > MAX_SYMBOLS {
                return Err(format!("Give 1–{MAX_SYMBOLS} symbol names."));
            }
            a.push("find_symbols".into());
            for n in names {
                a.push(symbol(n)?);
            }
            if let Some(t) = node_type {
                if !NODE_TYPES.contains(&t.as_str()) {
                    return Err(format!("Unknown node type {t:?}."));
                }
                a.extend(["--node-type".into(), t.clone()]);
            }
            if let Some(p) = file_prefix {
                a.extend(["--file-prefix".into(), rel_path(p, docs)?]);
            }
            a.extend(["-k".into(), "20".into(), "--include-docs".into()]);
        }
        ToolCall::SymbolContext { symbol: s, max_chars } => {
            a.extend(["context".into(), symbol(s)?]);
            a.extend(["--max-chars".into(), (*max_chars).clamp(500, 8000).to_string()]);
        }
        ToolCall::GetCode { symbol: s, file, start, end } => {
            a.push("get_code".into());
            match (s, file) {
                (Some(s), None) => a.push(symbol(s)?),
                (None, Some(f)) => {
                    let start = start.unwrap_or(1).max(1);
                    let end = end.unwrap_or(start + MAX_CODE_LINES - 1).clamp(start, start + MAX_CODE_LINES - 1);
                    a.extend(["-f".into(), rel_path(f, docs)?]);
                    a.extend(["-s".into(), start.to_string(), "-e".into(), end.to_string()]);
                }
                _ => return Err("get_code takes either a symbol or a file, not both.".into()),
            }
            a.extend(["--max-chars".into(), "8000".into()]);
        }
        ToolCall::FindUsages { symbol: s } => a.extend(["find_usages".into(), symbol(s)?]),
        ToolCall::FileContext { file, max_chars } => {
            a.extend(["file_context".into(), rel_path(file, docs)?]);
            a.extend(["--max-chars".into(), (*max_chars).clamp(500, 8000).to_string()]);
            a.extend(["-k".into(), "5".into()]);
        }
        ToolCall::Overview {} => a.push("project_overview".into()),
        ToolCall::Analyze { preset, target, limit } => {
            let Some(&(name, takes_target)) = ANALYZE_PRESETS.iter().find(|(p, _)| p == preset) else {
                return Err(format!("Unknown analysis {:?}.", preset.chars().take(64).collect::<String>()));
            };
            a.extend(["analyze".into(), name.into()]);
            match (takes_target, target) {
                // One argv `target=<path>`: ug splits it at the first `=`, and it can't parse as a flag.
                (true, Some(t)) => a.extend(["--arg".into(), format!("target={}", rel_path(t, docs)?)]),
                (true, None) => return Err(format!("{name} needs a file.")),
                (false, Some(_)) => return Err(format!("{name} takes no file.")),
                (false, None) => {}
            }
            a.extend(["-k".into(), limit.unwrap_or(MAX_ANALYZE_ROWS).clamp(1, MAX_ANALYZE_ROWS).to_string()]);
        }
    }
    a.extend(["-n".into(), project.into(), "--json".into()]);
    Ok(a)
}

// ── execution ────────────────────────────────────────────────────────────

/// What ug needs to find its data and model cache, and on Windows what any
/// process needs to start (`SystemRoot`) and find the user's profile.
#[cfg(not(windows))]
const KEPT_ENV: &[&str] = &["HOME", "TMPDIR", "LANG", "UG_HOME", "UG_MODEL_CACHE", "XDG_CACHE_HOME"];
#[cfg(windows)]
const KEPT_ENV: &[&str] = &[
    "SystemRoot", "SystemDrive", "windir", "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "HOME", "APPDATA",
    "LOCALAPPDATA", "TEMP", "TMP", "UG_HOME", "UG_MODEL_CACHE",
];

/// Only `KEPT_ENV` survives. Nothing inherited can point ug at a remote
/// embedder (`UG_*`, `OPENAI_*` and friends are dropped).
fn scrubbed(cmd: &mut Command) {
    cmd.env_clear().env("NO_COLOR", "1").env("CLICOLOR", "0");
    #[cfg(not(windows))]
    cmd.env("PATH", "/usr/bin:/bin");
    #[cfg(windows)]
    if let Some(root) = std::env::var_os("SystemRoot") {
        let root = PathBuf::from(root);
        if let Ok(path) = std::env::join_paths([root.join("System32"), root]) {
            cmd.env("PATH", path);
        }
    }
    for key in KEPT_ENV {
        if let Some(v) = std::env::var_os(key) {
            cmd.env(key, v);
        }
    }
}

/// Reads up to `cap` bytes and drains the rest.
fn read_capped(mut r: impl Read, cap: usize) -> (Vec<u8>, usize) {
    let mut kept = Vec::new();
    let mut total = 0;
    let mut buf = [0u8; 16 * 1024];
    while let Ok(n) = r.read(&mut buf) {
        if n == 0 {
            break;
        }
        total += n;
        let room = cap.saturating_sub(kept.len());
        kept.extend_from_slice(&buf[..n.min(room)]);
    }
    (kept, total)
}

/// The first `"error": "…"` string anywhere in ug's output.
fn first_error(v: &Value) -> Option<String> {
    match v {
        Value::Object(m) => {
            m.get("error").and_then(Value::as_str).map(String::from).or_else(|| m.values().find_map(first_error))
        }
        Value::Array(a) => a.iter().find_map(first_error),
        _ => None,
    }
}

/// Runs `bin args…` in `cwd` with the timeout and output cap.
pub fn run(bin: &Path, args: &[String], cwd: &Path, timeout: Duration, cap: usize) -> Result<ToolOutput, String> {
    let started = Instant::now();
    let mut cmd = Command::new(bin);
    scrubbed(&mut cmd);
    cmd.args(args).current_dir(cwd).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = cmd.spawn().map_err(|e| format!("failed to run ug: {e}"))?;
    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();
    let out_thread = std::thread::spawn(move || read_capped(stdout, cap));
    let err_thread = std::thread::spawn(move || read_capped(stderr, 64 * 1024).0);

    let status = loop {
        if let Some(status) = child.try_wait().map_err(|e| e.to_string())? {
            break status;
        }
        if started.elapsed() > timeout {
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!("Tool timed out after {} s and was stopped.", timeout.as_secs()));
        }
        std::thread::sleep(Duration::from_millis(15));
    };
    let (stdout, bytes) = out_thread.join().unwrap_or_default();
    let stderr = err_thread.join().unwrap_or_default();
    if !status.success() {
        // ug's lookups fail with the actionable message ("No symbol named …, try find_symbols")
        // as an `error` field in stdout JSON and a bare "error:" on stderr.
        if let Some(msg) = serde_json::from_slice::<Value>(&stdout).ok().as_ref().and_then(first_error) {
            return Err(msg);
        }
        let err = strip_ansi(&String::from_utf8_lossy(&stderr));
        let last = err.trim().lines().rev().find(|l| !l.trim().is_empty() && l.trim() != "error:");
        return Err(last.unwrap_or("ug failed").trim().to_string());
    }
    let truncated = bytes > stdout.len();
    let output = match (truncated, serde_json::from_slice::<Value>(&stdout)) {
        (false, Ok(v)) => v,
        _ => Value::String(strip_ansi(&String::from_utf8_lossy(&stdout))),
    };
    Ok(ToolOutput { output, truncated, bytes, ms: started.elapsed().as_millis() as u64, argv: args.to_vec() })
}

/// Runs one validated, read-only ug tool against a knowledge base: its ug
/// project, with the folder it indexes (Andai's copies, or the repo ug
/// recorded) as the root every path argument must stay inside.
#[tauri::command]
pub async fn kb_tool(app: AppHandle, slug: String, call: ToolCall) -> Result<ToolOutput, String> {
    let kb = resolve(&app, &slug)?;
    if !kb.root.is_dir() {
        return Err("This knowledge base's folder is missing.".into());
    }
    let docs = kb.root;
    let args = plan(&call, &kb.project, &docs)?;
    let bin = ug_path().ok_or(crate::ug::UG_MISSING)?;
    tauri::async_runtime::spawn_blocking(move || run(&bin, &args, &docs, TOOL_TIMEOUT, MAX_OUTPUT_BYTES))
        .await
        .map_err(|e| e.to_string())?
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn docs() -> tempfile::TempDir {
        let d = tempfile::tempdir().unwrap();
        std::fs::write(d.path().join("a.ts"), "export const a = 1;").unwrap();
        d
    }

    fn parse(v: Value) -> Result<ToolCall, String> {
        serde_json::from_value(v).map_err(|e| e.to_string())
    }

    #[test]
    fn unknown_tools_and_fields_are_rejected() {
        assert!(parse(json!({ "tool": "rm_rf", "path": "/" })).is_err());
        assert!(parse(json!({ "tool": "kb_overview", "base_url": "http://evil" })).is_err());
        assert!(parse(json!({ "tool": "kb_find_usages", "symbol": "x", "extra": 1 })).is_err());
        assert_eq!(parse(json!({ "tool": "kb_overview" })).unwrap(), ToolCall::Overview {});
    }

    #[test]
    fn analyze_runs_only_listed_presets_with_a_file_inside_the_kb() {
        let d = docs();
        let call = |preset: &str, target: Option<&str>, limit: Option<u32>| ToolCall::Analyze {
            preset: preset.into(),
            target: target.map(Into::into),
            limit,
        };
        let a = plan(&call("biggest_files", None, None), "andai-kb", d.path()).unwrap();
        assert_eq!(a, ["analyze", "biggest_files", "-k", "50", "-n", "andai-kb", "--json"]);
        let a = plan(&call("impact", Some("a.ts"), Some(999)), "andai-kb", d.path()).unwrap();
        assert_eq!(a[..6], ["analyze", "impact", "--arg", "target=a.ts", "-k", "50"]);
        assert!(plan(&call("dependency_fanin", None, Some(0)), "p", d.path()).unwrap().windows(2).any(|w| w == ["-k", "1"]));
        // Not a listed preset: raw GQL, a flag, or a preset Andai doesn't run.
        for bad in ["--gql", "nope", "layering_violations", "", "biggest_files --db /x"] {
            assert!(plan(&call(bad, None, None), "p", d.path()).is_err(), "{bad:?}");
        }
        // A target only where the preset takes one, and always inside the KB.
        assert!(plan(&call("impact", None, None), "p", d.path()).is_err());
        assert!(plan(&call("biggest_files", Some("a.ts"), None), "p", d.path()).is_err());
        for bad in ["-x", "../a.ts", "/etc/passwd", "a\\b.ts"] {
            assert!(plan(&call("impact", Some(bad), None), "p", d.path()).is_err(), "{bad:?}");
        }
        assert!(parse(json!({ "tool": "kb_analyze", "preset": "impact", "gql": "MATCH (n) RETURN n" })).is_err());
        assert_eq!(
            parse(json!({ "tool": "kb_analyze", "preset": "biggest_files" })).unwrap(),
            ToolCall::Analyze { preset: "biggest_files".into(), target: None, limit: None }
        );
    }

    #[test]
    fn search_is_scoped_clamped_and_never_a_flag() {
        let d = docs();
        let call = ToolCall::Search { query: "--base-url http://evil".into(), k: 999, expand: false, max_chars: 1 };
        let a = plan(&call, "andai-kb", d.path()).unwrap();
        assert_eq!(a[..2], ["search", " --base-url http://evil"]);
        assert!(a.windows(2).any(|w| w == ["-k", "20"]));
        assert!(a.windows(2).any(|w| w == ["--max-chars", "100"]));
        assert!(a.contains(&"--no-expand".to_string()));
        assert!(a.ends_with(&["-n".into(), "andai-kb".into(), "--json".into()]));
        let empty = ToolCall::Search { query: "  ".into(), k: 8, expand: true, max_chars: 4000 };
        assert!(plan(&empty, "andai-kb", d.path()).is_err());
    }

    #[test]
    fn no_argument_can_add_an_embedder_flag() {
        let d = docs();
        let calls = [
            ToolCall::FindSymbols { names: vec!["-n".into()], node_type: None, file_prefix: None },
            ToolCall::SymbolContext { symbol: "--base-url".into(), max_chars: 1000 },
            ToolCall::FindUsages { symbol: "-k".into() },
            ToolCall::FileContext { file: "--api-key".into(), max_chars: 1000 },
            ToolCall::GetCode { symbol: None, file: Some("-f".into()), start: None, end: None },
        ];
        for c in calls {
            assert!(plan(&c, "andai-kb", d.path()).is_err(), "{c:?} should be rejected");
        }
    }

    #[test]
    fn paths_cannot_leave_the_knowledge_base() {
        let d = docs();
        for bad in ["../kb.json", "/etc/passwd", "a/../../x", "..", "a\\b", "x\0y", "a\nb"] {
            let c = ToolCall::FileContext { file: bad.into(), max_chars: 1000 };
            assert!(plan(&c, "andai-kb", d.path()).is_err(), "{bad:?} should be rejected");
        }
        for ok in ["a.ts", "./a.ts", "src/**/*.ts", "missing.md"] {
            let c = ToolCall::FileContext { file: ok.into(), max_chars: 1000 };
            assert!(plan(&c, "andai-kb", d.path()).is_ok(), "{ok:?} should be allowed");
        }
    }

    #[cfg(unix)]
    #[test]
    fn symlinks_out_of_docs_are_rejected() {
        let d = docs();
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("secret"), "key").unwrap();
        std::os::unix::fs::symlink(outside.path().join("secret"), d.path().join("link")).unwrap();
        let c = ToolCall::GetCode { symbol: None, file: Some("link".into()), start: None, end: None };
        assert!(plan(&c, "andai-kb", d.path()).unwrap_err().contains("inside the knowledge base"));
    }

    #[test]
    fn symbol_lookups_are_bounded() {
        let d = docs();
        let many = ToolCall::FindSymbols { names: vec!["a".into(); 6], node_type: None, file_prefix: None };
        assert!(plan(&many, "andai-kb", d.path()).is_err());
        let bad_type = ToolCall::FindSymbols { names: vec!["a".into()], node_type: Some("Evil".into()), file_prefix: None };
        assert!(plan(&bad_type, "andai-kb", d.path()).is_err());
        let ok = ToolCall::FindSymbols {
            names: vec!["build*".into()],
            node_type: Some("Function".into()),
            file_prefix: Some("src/".into()),
        };
        let a = plan(&ok, "andai-kb", d.path()).unwrap();
        assert_eq!(a[..2], ["find_symbols", "build*"]);
        assert!(plan(&ToolCall::FindUsages { symbol: "x".repeat(MAX_NAME_BYTES + 1) }, "andai-kb", d.path()).is_err());
    }

    #[test]
    fn code_ranges_are_clamped() {
        let d = docs();
        let c = ToolCall::GetCode { symbol: None, file: Some("a.ts".into()), start: Some(0), end: Some(100_000) };
        let a = plan(&c, "andai-kb", d.path()).unwrap();
        assert!(a.windows(2).any(|w| w == ["-s", "1"]));
        assert!(a.windows(2).any(|w| w == ["-e", &MAX_CODE_LINES.to_string()]));
        let both = ToolCall::GetCode { symbol: Some("a".into()), file: Some("a.ts".into()), start: None, end: None };
        assert!(plan(&both, "andai-kb", d.path()).is_err());
        let neither = ToolCall::GetCode { symbol: None, file: None, start: None, end: None };
        assert!(plan(&neither, "andai-kb", d.path()).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn runs_are_time_boxed() {
        let d = tempfile::tempdir().unwrap();
        let t = Instant::now();
        let err = run(Path::new("/bin/sleep"), &["5".into()], d.path(), Duration::from_millis(200), 1024).unwrap_err();
        assert!(err.contains("timed out"), "{err}");
        assert!(t.elapsed() < Duration::from_secs(2), "the child was killed, not waited for");
    }

    #[cfg(unix)]
    #[test]
    fn output_is_capped_and_json_parsed() {
        let d = tempfile::tempdir().unwrap();
        let big = run(Path::new("/bin/sh"), &["-c".into(), "yes x | head -c 5000".into()], d.path(), TOOL_TIMEOUT, 100)
            .unwrap();
        assert!(big.truncated);
        assert_eq!(big.bytes, 5000);
        assert!(big.output.as_str().unwrap().len() <= 100);
        let json = run(Path::new("/bin/echo"), &[r#"{"items":[1]}"#.into()], d.path(), TOOL_TIMEOUT, 1024).unwrap();
        assert_eq!(json.output, json!({ "items": [1] }));
        assert!(!json.truncated);
    }

    #[cfg(unix)]
    #[test]
    fn the_child_environment_is_scrubbed() {
        let d = tempfile::tempdir().unwrap();
        std::env::set_var("UG_EMBED_BASE_URL", "http://evil");
        std::env::set_var("OPENAI_API_KEY", "sk-test");
        let out = run(Path::new("/usr/bin/env"), &[], d.path(), TOOL_TIMEOUT, 64 * 1024).unwrap();
        let text = out.output.as_str().unwrap().to_string();
        assert!(!text.contains("UG_EMBED_BASE_URL") && !text.contains("OPENAI_API_KEY"), "{text}");
        assert!(text.contains("NO_COLOR=1"));
    }

    #[cfg(unix)]
    #[test]
    fn failures_surface_the_last_stderr_line() {
        let d = tempfile::tempdir().unwrap();
        let err = run(Path::new("/bin/sh"), &["-c".into(), "echo one >&2; echo 'no such symbol' >&2; exit 1".into()], d.path(), TOOL_TIMEOUT, 1024)
            .unwrap_err();
        assert_eq!(err, "no such symbol");
        let json = r#"echo '{"nodes":[{"query":"x","error":"No symbol named x"}]}'; echo 'error:' >&2; exit 1"#;
        let err = run(Path::new("/bin/sh"), &["-c".into(), json.into()], d.path(), TOOL_TIMEOUT, 1024).unwrap_err();
        assert_eq!(err, "No symbol named x", "ug's JSON error beats the bare stderr line");
    }

    /// Every tool against a real mixed (Markdown + TypeScript) ug project.
    #[test]
    #[ignore = "requires the ug CLI"]
    fn every_tool_runs_against_real_ug() {
        let bin = ug_path().expect("ug not found");
        let docs = tempfile::tempdir().unwrap();
        std::fs::write(docs.path().join("notes.md"), "# Deploy\n\n## Isolation\n\nwllama needs COOP and COEP headers.\n").unwrap();
        std::fs::write(
            docs.path().join("math.ts"),
            "/** Adds two numbers. */\nexport function add(a: number, b: number) { return a + b; }\nexport function twice(x: number) { return add(x, x); }\n",
        )
        .unwrap();
        let slug = format!("tooltest-{}", std::process::id());
        let project = format!("andai-{slug}");
        let gen = Command::new(&bin).arg("gen").arg(docs.path()).args(["-n", &project, "--with-embed"]).output().unwrap();
        let cleanup = || {
            let _ = Command::new(&bin).args(["remove", &project, "-y"]).output();
        };
        if !gen.status.success() {
            cleanup();
            panic!("ug gen failed: {}", String::from_utf8_lossy(&gen.stderr));
        }
        let calls = [
            (ToolCall::Search { query: "which headers does wllama need".into(), k: 4, expand: true, max_chars: 2000 }, "items"),
            (ToolCall::FindSymbols { names: vec!["add".into()], node_type: Some("Function".into()), file_prefix: None }, "queries"),
            (ToolCall::SymbolContext { symbol: "add".into(), max_chars: 2000 }, "items"),
            (ToolCall::GetCode { symbol: Some("add".into()), file: None, start: None, end: None }, "slices"),
            (ToolCall::GetCode { symbol: None, file: Some("notes.md".into()), start: Some(1), end: Some(3) }, "slices"),
            (ToolCall::FindUsages { symbol: "add".into() }, "nodes"),
            (ToolCall::FileContext { file: "math.ts".into(), max_chars: 2000 }, "files"),
            (ToolCall::Overview {}, "node_count"),
            (ToolCall::Analyze { preset: "biggest_files".into(), target: None, limit: Some(5) }, "rows"),
            (ToolCall::Analyze { preset: "impact".into(), target: Some("math.ts".into()), limit: None }, "rows"),
        ];
        let mut failures = vec![];
        for (call, key) in calls {
            let args = plan(&call, &project, docs.path()).unwrap();
            match run(&bin, &args, docs.path(), TOOL_TIMEOUT, MAX_OUTPUT_BYTES) {
                Ok(out) if out.output.get(key).is_some() => {}
                Ok(out) => failures.push(format!("{call:?}: no {key:?} in {}", out.output)),
                Err(e) => failures.push(format!("{call:?}: {e}")),
            }
        }
        cleanup();
        assert!(failures.is_empty(), "{failures:#?}");
    }
}
