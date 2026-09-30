//! The activity log (Settings → Activity log; off by default, a product
//! decision, 2026-09-30): one JSON line per agent event (a turn, each step,
//! each argument writer call, each tool call, the context sent, the answer
//! and its checks), appended to `<app data>/logs/agent-YYYY-MM-DD.jsonl`
//! (UTC day). It holds the user's questions and passages from their documents,
//! so the folder is private (0700 / 0600), a day stops at `DAY_CAP`, and
//! files older than `KEEP_DAYS` are deleted.
//!
//! The webview writes only while the user has it on, but the webview is
//! untrusted (AGENTS.md §9), so Rust bounds what it accepts either way: a
//! closed set of kinds, sizes, and the daily cap. The file name and the time
//! of day come from Rust's clock, never from the webview. Deleting and
//! opening act on this folder only, and take no path; reading (the Logs
//! screen) takes a file name, which must be exactly one of ours.
//!
//! Besides the agent's events, the app logs what it did to models and
//! knowledge bases (turn id `app`), and every line may carry a one-line
//! `summary` and a `level`, so the files read without the app.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};
use tauri::{AppHandle, State};

use crate::ug::{data_dir, open_in_system};

/// What an event can be; anything else is refused.
pub const KINDS: &[&str] = &["turn", "step", "args", "tool", "retrieve", "relevance", "context", "answer", "claims", "error", "done", "model", "kb"];
/// How bad an event is; anything else is refused.
pub const LEVELS: &[&str] = &["info", "warn", "error"];
/// A summary's bytes.
const MAX_SUMMARY: usize = 1024;
/// Events one read returns: the newest ones of the day.
pub const READ_LIMIT: usize = 5000;
/// Events per call, and bytes per event once serialized.
const MAX_EVENTS: usize = 64;
const MAX_EVENT_BYTES: usize = 256 * 1024;
/// A day's file stops growing here; later events that day are dropped.
pub const DAY_CAP: u64 = 20 * 1024 * 1024;
/// Days of files kept, today included.
pub const KEEP_DAYS: i64 = 7;

#[derive(Deserialize)]
pub struct Event {
    kind: String,
    /// The assistant message the event belongs to.
    turn: String,
    /// One line a person can read ("Searched “ferries” · 6 passages · 40 ms").
    #[serde(default)]
    summary: Option<String>,
    #[serde(default)]
    level: Option<String>,
    data: Value,
}

#[derive(Serialize)]
struct Line<'a> {
    /// Milliseconds since the epoch, from Rust's clock.
    at: u64,
    kind: &'a str,
    turn: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    level: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    summary: Option<&'a str>,
    data: &'a Value,
}

/// One day's events as the Logs screen reads them.
#[derive(Serialize)]
pub struct LogDay {
    /// Parsed lines, oldest first; the newest `READ_LIMIT` when the day has more.
    events: Vec<Value>,
    /// Lines in the file, and lines that weren't JSON (a cut line, an edit by hand).
    total: usize,
    bad: usize,
}

/// Serializes writes, so two calls can't interleave inside a line or race the cap.
#[derive(Default)]
pub struct ActivityLock(Mutex<()>);

#[derive(Serialize)]
pub struct LogFile {
    name: String,
    bytes: u64,
}

#[derive(Serialize)]
pub struct LogInfo {
    dir: String,
    files: Vec<LogFile>,
}

fn logs_dir(app: &AppHandle) -> Result<PathBuf, String> {
    data_dir(app, &["logs"])
}

fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

/// Days since the epoch → (year, month, day), proleptic Gregorian (Howard Hinnant's `civil_from_days`).
fn civil(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (yoe + era * 400 + i64::from(m <= 2), m, d)
}

fn file_name(days: i64) -> String {
    let (y, m, d) = civil(days);
    format!("agent-{y:04}-{m:02}-{d:02}.jsonl")
}

/// The day a log file is for, when `name` is exactly one of ours.
fn day_of(name: &str) -> Option<i64> {
    let date = name.strip_prefix("agent-")?.strip_suffix(".jsonl")?;
    let b = date.as_bytes();
    if b.len() != 10 || b[4] != b'-' || b[7] != b'-' || !date.chars().enumerate().all(|(i, c)| i == 4 || i == 7 || c.is_ascii_digit()) {
        return None;
    }
    let (y, m, d): (i64, u32, u32) = (date[..4].parse().ok()?, date[5..7].parse().ok()?, date[8..].parse().ok()?);
    // The inverse of `civil`, checked by round trip so "2026-02-31" isn't a day.
    let y2 = if m <= 2 { y - 1 } else { y };
    let era = y2.div_euclid(400);
    let yoe = y2.rem_euclid(400);
    let mp = if m > 2 { m - 3 } else { m + 9 } as i64;
    let doy = (153 * mp + 2) / 5 + i64::from(d) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    (civil(days) == (y, m, d)).then_some(days)
}

/// Our log files in `dir`, with their day; nothing else in the folder is touched.
fn ours(dir: &Path) -> Vec<(PathBuf, i64)> {
    let Ok(entries) = fs::read_dir(dir) else { return vec![] };
    let mut out: Vec<(PathBuf, i64)> = entries
        .flatten()
        .filter(|e| e.file_type().map(|t| t.is_file()).unwrap_or(false))
        .filter_map(|e| day_of(&e.file_name().to_string_lossy()).map(|d| (e.path(), d)))
        .collect();
    out.sort_by_key(|(_, d)| *d);
    out
}

/// Deletes our files older than `KEEP_DAYS` before `today`.
fn prune(dir: &Path, today: i64) {
    for (path, day) in ours(dir) {
        if day <= today - KEEP_DAYS {
            let _ = fs::remove_file(path);
        }
    }
}

/// Checks one event: a known kind, a short turn id, a bounded payload. Returns its line.
fn line(e: &Event, at: u64) -> Result<String, String> {
    if !KINDS.contains(&e.kind.as_str()) {
        return Err(format!("Unknown activity kind “{}”.", e.kind.chars().take(32).collect::<String>()));
    }
    if e.turn.is_empty() || e.turn.len() > 64 || !e.turn.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
        return Err("An activity event's turn id must be 1–64 letters, digits, - or _.".into());
    }
    if let Some(l) = &e.level {
        if !LEVELS.contains(&l.as_str()) {
            return Err("An activity event's level must be info, warn or error.".into());
        }
    }
    if e.summary.as_ref().is_some_and(|s| s.len() > MAX_SUMMARY || s.contains('\n')) {
        return Err(format!("An activity summary is one line of at most {MAX_SUMMARY} bytes."));
    }
    let line = Line { at, kind: &e.kind, turn: &e.turn, level: e.level.as_deref(), summary: e.summary.as_deref(), data: &e.data };
    let text = serde_json::to_string(&line).map_err(|e| e.to_string())?;
    if text.len() > MAX_EVENT_BYTES {
        return Err(format!("An activity event is {} KB; the most is {} KB.", text.len() / 1024, MAX_EVENT_BYTES / 1024));
    }
    Ok(text)
}

/// Appends the events to `dir`'s file for the day of `at`, within the day's cap. Returns how many were written.
fn append(dir: &Path, events: &[Event], at: u64) -> Result<usize, String> {
    if events.len() > MAX_EVENTS {
        return Err(format!("At most {MAX_EVENTS} activity events per call."));
    }
    let lines = events.iter().map(|e| line(e, at)).collect::<Result<Vec<_>, _>>()?;
    let today = (at / 86_400_000) as i64;
    prune(dir, today);
    let path = dir.join(file_name(today));
    let mut size = fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
    let mut opts = fs::OpenOptions::new();
    opts.append(true).create(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut opts, 0o600);
    let mut file = opts.open(&path).map_err(|e| format!("Couldn't open the activity log: {e}"))?;
    let mut written = 0;
    for l in lines {
        let n = l.len() as u64 + 1;
        if size + n > DAY_CAP {
            break;
        }
        file.write_all(format!("{l}\n").as_bytes()).map_err(|e| format!("Couldn't write the activity log: {e}"))?;
        size += n;
        written += 1;
    }
    Ok(written)
}

#[tauri::command]
pub async fn activity_write(app: AppHandle, lock: State<'_, ActivityLock>, events: Vec<Event>) -> Result<usize, String> {
    let dir = logs_dir(&app)?;
    let _held = lock.0.lock().map_err(|_| "The activity log is unavailable.".to_string())?;
    // Small appends; blocking the command's own thread is fine, and the lock keeps lines whole.
    append(&dir, &events, now_ms())
}

#[tauri::command]
pub async fn activity_info(app: AppHandle) -> Result<LogInfo, String> {
    let dir = logs_dir(&app)?;
    prune(&dir, (now_ms() / 86_400_000) as i64);
    let files = ours(&dir)
        .into_iter()
        .rev()
        .map(|(p, _)| LogFile { name: p.file_name().unwrap_or_default().to_string_lossy().into(), bytes: fs::metadata(&p).map(|m| m.len()).unwrap_or(0) })
        .collect();
    Ok(LogInfo { dir: dir.to_string_lossy().into(), files })
}

/// Reads one day's file from `dir`: `name` must be exactly one of ours, a regular file (not a link).
fn read_day(dir: &Path, name: &str) -> Result<LogDay, String> {
    if day_of(name).is_none() {
        return Err("Not an activity log file.".into());
    }
    let path = dir.join(name);
    match fs::symlink_metadata(&path) {
        Ok(m) if m.is_file() => {}
        Ok(_) => return Err("Not an activity log file.".into()),
        Err(_) => return Ok(LogDay { events: vec![], total: 0, bad: 0 }),
    }
    // A day stops at DAY_CAP, so this is bounded; lossy, since a cut line may split a character.
    let bytes = fs::read(&path).map_err(|e| format!("Couldn't read the activity log: {e}"))?;
    let text = String::from_utf8_lossy(&bytes);
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    let total = lines.len();
    let mut bad = 0;
    let mut events: Vec<Value> = lines[total.saturating_sub(READ_LIMIT)..]
        .iter()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok().filter(Value::is_object).or_else(|| {
            bad += 1;
            None
        }))
        .collect();
    events.shrink_to_fit();
    Ok(LogDay { events, total, bad })
}

/// One day's events for the Logs screen.
#[tauri::command]
pub async fn activity_read(app: AppHandle, lock: State<'_, ActivityLock>, name: String) -> Result<LogDay, String> {
    let dir = logs_dir(&app)?;
    let _held = lock.0.lock().map_err(|_| "The activity log is unavailable.".to_string())?;
    read_day(&dir, &name)
}

/// Deletes every log file (only ours; the UI confirms first). Returns how many.
#[tauri::command]
pub async fn activity_clear(app: AppHandle, lock: State<'_, ActivityLock>) -> Result<usize, String> {
    let dir = logs_dir(&app)?;
    let _held = lock.0.lock().map_err(|_| "The activity log is unavailable.".to_string())?;
    let mut n = 0;
    for (path, _) in ours(&dir) {
        fs::remove_file(&path).map_err(|e| format!("Couldn't delete {}: {e}", path.display()))?;
        n += 1;
    }
    Ok(n)
}

/// Shows the logs folder in Finder (Explorer on Windows). No argument: it opens only that folder.
#[tauri::command]
pub fn activity_open(app: AppHandle) -> Result<(), String> {
    let dir = logs_dir(&app)?;
    open_in_system(dir.as_os_str()).map_err(|e| format!("Couldn't open the logs folder: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const DAY: u64 = 86_400_000;
    // 2026-09-30T12:00:00Z
    const AT: u64 = 1_790_769_600_000;

    fn ev(kind: &str, data: Value) -> Event {
        Event { kind: kind.into(), turn: "m1".into(), summary: None, level: None, data }
    }

    #[test]
    fn names_files_by_utc_day_and_reads_only_its_own_names_back() {
        let today = (AT / DAY) as i64;
        assert_eq!(file_name(today), "agent-2026-09-30.jsonl");
        assert_eq!(file_name(0), "agent-1970-01-01.jsonl");
        assert_eq!(day_of("agent-2026-09-30.jsonl"), Some(today));
        assert_eq!(day_of("agent-2024-02-29.jsonl").map(file_name).as_deref(), Some("agent-2024-02-29.jsonl"));
        for bad in ["agent-2026-02-31.jsonl", "agent-2026-9-30.jsonl", "agent-2026-09-30.jsonl.bak", "notes.jsonl", "agent-2026-09-3x.jsonl", "../agent-2026-09-30.jsonl"] {
            assert_eq!(day_of(bad), None, "{bad}");
        }
    }

    #[test]
    fn appends_one_json_line_per_event_with_rusts_clock() {
        let dir = tempfile::tempdir().unwrap();
        let n = append(dir.path(), &[ev("turn", json!({"question": "q"})), ev("args", json!({"tool": "kb_search"}))], AT).unwrap();
        assert_eq!(n, 2);
        append(dir.path(), &[ev("tool", json!({}))], AT + 1).unwrap();
        let text = fs::read_to_string(dir.path().join("agent-2026-09-30.jsonl")).unwrap();
        let lines: Vec<Value> = text.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0], json!({"at": AT, "kind": "turn", "turn": "m1", "data": {"question": "q"}}));
        assert_eq!(lines[2]["at"], json!(AT + 1));
    }

    #[cfg(unix)]
    #[test]
    fn the_file_is_owner_only() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        append(dir.path(), &[ev("turn", json!(1))], AT).unwrap();
        let mode = fs::metadata(dir.path().join("agent-2026-09-30.jsonl")).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600);
    }

    #[test]
    fn refuses_unknown_kinds_bad_turn_ids_oversized_events_and_batches() {
        let dir = tempfile::tempdir().unwrap();
        assert!(append(dir.path(), &[ev("shell", json!({}))], AT).unwrap_err().contains("Unknown activity kind"));
        let bad_turn = Event { kind: "turn".into(), turn: "../x".into(), summary: None, level: None, data: json!({}) };
        assert!(append(dir.path(), &[bad_turn], AT).is_err());
        let big = ev("context", json!({"system": "x".repeat(MAX_EVENT_BYTES)}));
        assert!(append(dir.path(), &[big], AT).unwrap_err().contains("KB"));
        let many: Vec<Event> = (0..=MAX_EVENTS).map(|_| ev("step", json!({}))).collect();
        assert!(append(dir.path(), &many, AT).is_err());
        // nothing was written by a refused call, not even the valid events before a bad one
        assert!(append(dir.path(), &[ev("turn", json!({})), ev("nope", json!({}))], AT).is_err());
        assert!(!dir.path().join("agent-2026-09-30.jsonl").exists());
    }

    #[test]
    fn stops_a_day_at_its_cap() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("agent-2026-09-30.jsonl");
        fs::write(&path, vec![b'x'; (DAY_CAP - 100) as usize]).unwrap();
        let n = append(dir.path(), &[ev("step", json!({})), ev("step", json!({"pad": "y".repeat(200)}))], AT).unwrap();
        assert_eq!(n, 1);
        assert!(fs::metadata(&path).unwrap().len() <= DAY_CAP);
        // the next day starts fresh
        assert_eq!(append(dir.path(), &[ev("step", json!({}))], AT + DAY).unwrap(), 1);
    }

    #[test]
    fn keeps_seven_days_and_never_touches_other_files() {
        let dir = tempfile::tempdir().unwrap();
        let today = (AT / DAY) as i64;
        for back in [0, 6, 7, 30] {
            fs::write(dir.path().join(file_name(today - back)), "{}\n").unwrap();
        }
        fs::write(dir.path().join("notes.txt"), "mine").unwrap();
        fs::write(dir.path().join("agent-2000-01-01.jsonl.keep"), "mine").unwrap();
        prune(dir.path(), today);
        let mut left: Vec<String> = fs::read_dir(dir.path()).unwrap().flatten().map(|e| e.file_name().to_string_lossy().into()).collect();
        left.sort();
        assert_eq!(left, vec!["agent-2000-01-01.jsonl.keep", "agent-2026-09-24.jsonl", "agent-2026-09-30.jsonl", "notes.txt"]);
    }

    #[test]
    fn writes_a_summary_and_level_when_given_and_refuses_bad_ones() {
        let dir = tempfile::tempdir().unwrap();
        let e = Event { kind: "kb".into(), turn: "app".into(), summary: Some("Indexed “Ferries”".into()), level: Some("warn".into()), data: json!({}) };
        append(dir.path(), &[e], AT).unwrap();
        let text = fs::read_to_string(dir.path().join("agent-2026-09-30.jsonl")).unwrap();
        assert_eq!(text, format!("{{\"at\":{AT},\"kind\":\"kb\",\"turn\":\"app\",\"level\":\"warn\",\"summary\":\"Indexed “Ferries”\",\"data\":{{}}}}\n"));
        let level = Event { kind: "kb".into(), turn: "app".into(), summary: None, level: Some("fatal".into()), data: json!({}) };
        assert!(append(dir.path(), &[level], AT).is_err());
        let long = Event { kind: "kb".into(), turn: "app".into(), summary: Some("x".repeat(MAX_SUMMARY + 1)), level: None, data: json!({}) };
        assert!(append(dir.path(), &[long], AT).is_err());
        let lines = Event { kind: "kb".into(), turn: "app".into(), summary: Some("a\nb".into()), level: None, data: json!({}) };
        assert!(append(dir.path(), &[lines], AT).is_err());
    }

    #[test]
    fn reads_back_a_day_newest_last_skipping_bad_lines() {
        let dir = tempfile::tempdir().unwrap();
        append(dir.path(), &[ev("turn", json!({"question": "q"})), ev("done", json!({}))], AT).unwrap();
        let path = dir.path().join("agent-2026-09-30.jsonl");
        let mut f = fs::OpenOptions::new().append(true).open(&path).unwrap();
        f.write_all(b"not json\n[1]\n\n").unwrap();
        let day = read_day(dir.path(), "agent-2026-09-30.jsonl").unwrap();
        assert_eq!((day.total, day.bad, day.events.len()), (4, 2, 2));
        assert_eq!(day.events[0]["kind"], json!("turn"));
        // a day without a file is empty, not an error
        let none = read_day(dir.path(), "agent-2026-09-29.jsonl").unwrap();
        assert_eq!((none.total, none.events.len()), (0, 0));
    }

    #[test]
    fn reads_only_the_newest_events_of_a_long_day() {
        let dir = tempfile::tempdir().unwrap();
        let text: String = (0..READ_LIMIT + 3).map(|i| format!("{{\"i\":{i}}}\n")).collect();
        fs::write(dir.path().join("agent-2026-09-30.jsonl"), text).unwrap();
        let day = read_day(dir.path(), "agent-2026-09-30.jsonl").unwrap();
        assert_eq!((day.total, day.events.len()), (READ_LIMIT + 3, READ_LIMIT));
        assert_eq!(day.events[0]["i"], json!(3));
    }

    #[test]
    fn reads_only_its_own_files_never_a_path_or_a_link() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("notes.txt"), "{}").unwrap();
        for bad in ["notes.txt", "../agent-2026-09-30.jsonl", "/etc/passwd", "agent-2026-09-30.jsonl/..", ""] {
            assert!(read_day(dir.path(), bad).is_err(), "{bad}");
        }
        #[cfg(unix)]
        {
            let outside = tempfile::tempdir().unwrap();
            fs::write(outside.path().join("secret"), "{\"secret\":1}\n").unwrap();
            std::os::unix::fs::symlink(outside.path().join("secret"), dir.path().join("agent-2026-09-28.jsonl")).unwrap();
            assert!(read_day(dir.path(), "agent-2026-09-28.jsonl").is_err());
        }
    }
}
