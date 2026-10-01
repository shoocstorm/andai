//! Installs the `ug` CLI from inside Andai, the same way UltraGraph's
//! `install.sh` does (https://ultra-graph.web.app/install.sh), without piping
//! a remote script into a shell:
//!
//!   1. ask GitHub for the latest release of `shoocstorm/ug`,
//!   2. download this platform's archive,
//!   3. check its size and sha256 against the release's own digest,
//!   4. unpack it into `~/.local/share/ultragraph/.ug` and link
//!      `~/.local/bin/ug` to it (where `ug_path()` already looks).
//!
//! The requests go through the system `curl` (Rust has no HTTP client,
//! AGENTS.md §9), started with `-q` so no `.curlrc` changes them, HTTPS only,
//! and a scrubbed environment. The webview only says "install": every URL,
//! path and version is decided here (a product decision recorded in §1.4).

use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter, State};

use crate::ug::{ug_path, ug_status, UgStatus};

const REPO: &str = "shoocstorm/ug";
const RELEASE_API: &str = "https://api.github.com/repos/shoocstorm/ug/releases/latest";
/// Every asset URL must start with this; anything else in the release JSON is refused.
const DOWNLOAD_PREFIX: &str = "https://github.com/shoocstorm/ug/releases/download/";
/// The release JSON is a few KB; the archive about 25 MB (v0.1.22).
const MAX_RELEASE_JSON: usize = 2 * 1024 * 1024;
const MAX_ARCHIVE_BYTES: u64 = 512 * 1024 * 1024;

/// One install at a time.
#[derive(Default)]
pub struct Installing(AtomicBool);

/// This platform's release asset, or `None` where no automatic install
/// exists (Windows ships a zip the user extracts, as `install.sh` says).
pub(crate) fn asset_name() -> Option<&'static str> {
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        Some("ultragraph-macos-arm64.tar.gz")
    } else if cfg!(all(target_os = "macos", target_arch = "x86_64")) {
        Some("ultragraph-macos-x64.tar.gz")
    } else if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        Some("ultragraph-linux-x64.tar.gz")
    } else {
        None
    }
}

#[derive(Serialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct InstallProgress {
    /// lookup | download | verify | install | check
    pub stage: &'static str,
    pub version: Option<String>,
    pub done: u64,
    pub total: u64,
}

#[derive(Debug, PartialEq)]
struct Asset {
    version: String,
    url: String,
    size: u64,
    /// Lowercase hex, from the asset's `digest`, else from its `.sha256` sibling.
    sha256: Option<String>,
    sha256_url: Option<String>,
}

fn is_hex64(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Picks `name` out of GitHub's release JSON, holding every field to what a
/// real release looks like: an exact name, a URL under this repo's downloads,
/// a sane size, a well-formed digest.
fn pick_asset(release: &Value, name: &str) -> Result<Asset, String> {
    let version = release["tag_name"].as_str().unwrap_or("").to_string();
    if version.is_empty() || version.len() > 64 || !version.bytes().all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b)) {
        return Err("The UltraGraph release has no valid version tag.".into());
    }
    let assets = release["assets"].as_array().ok_or("The UltraGraph release lists no downloads.")?;
    let find = |n: &str| assets.iter().find(|a| a["name"].as_str() == Some(n));
    let url_of = |a: &Value| -> Result<String, String> {
        let url = a["browser_download_url"].as_str().unwrap_or("");
        let expected = format!("{DOWNLOAD_PREFIX}{version}/{}", a["name"].as_str().unwrap_or(""));
        if url != expected {
            return Err(format!("Unexpected download address for {name}."));
        }
        Ok(url.to_string())
    };
    let a = find(name).ok_or_else(|| format!("The latest UltraGraph release ({version}) has no {name} yet."))?;
    let size = a["size"].as_u64().filter(|s| (1..=MAX_ARCHIVE_BYTES).contains(s)).ok_or("The UltraGraph download has an invalid size.")?;
    let sha256 = a["digest"].as_str().and_then(|d| d.strip_prefix("sha256:")).filter(|h| is_hex64(h)).map(str::to_ascii_lowercase);
    let sha256_url = match &sha256 {
        Some(_) => None,
        None => Some(url_of(find(&format!("{name}.sha256")).ok_or("The UltraGraph release has no checksum for this download.")?)?),
    };
    Ok(Asset { url: url_of(a)?, version, size, sha256, sha256_url })
}

/// `<hex>  <file>` (shasum's format), for the file we downloaded.
fn parse_sha256_file(text: &str, name: &str) -> Result<String, String> {
    let mut parts = text.split_whitespace();
    match (parts.next(), parts.next()) {
        (Some(h), Some(f)) if is_hex64(h) && f.trim_start_matches('*') == name => Ok(h.to_ascii_lowercase()),
        (Some(h), None) if is_hex64(h) => Ok(h.to_ascii_lowercase()),
        _ => Err("The UltraGraph checksum file is malformed.".into()),
    }
}

fn sha256_of(path: &Path) -> std::io::Result<String> {
    let mut f = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 16];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// The system curl, HTTPS only (redirects too), no config files, no inherited
/// environment (no proxy or CA overrides from wherever Andai was launched).
fn curl() -> Command {
    let mut cmd = Command::new("/usr/bin/curl");
    cmd.env_clear()
        .env("PATH", "/usr/bin:/bin")
        .args(["-q", "--proto", "=https", "--proto-redir", "=https", "--fail", "--silent", "--show-error", "--location"])
        .args(["--connect-timeout", "20", "--speed-limit", "1024", "--speed-time", "30"])
        .stdin(Stdio::null());
    cmd
}

fn curl_error(stderr: &[u8]) -> String {
    let msg = String::from_utf8_lossy(stderr);
    let msg = msg.trim().trim_start_matches("curl: ").lines().last().unwrap_or("").to_string();
    if msg.contains("resolve host") || msg.contains("connect to") || msg.contains("timed out") {
        format!("Couldn't reach GitHub. Check your internet connection and try again. ({msg})")
    } else if msg.is_empty() {
        "The download failed.".into()
    } else {
        format!("The download failed: {msg}")
    }
}

fn get_small(url: &str, limit: usize) -> Result<Vec<u8>, String> {
    let out = curl()
        .args(["--max-time", "30", "-H", "Accept: application/vnd.github+json", "-H", "User-Agent: Andai"])
        .arg(url)
        .output()
        .map_err(|e| format!("Couldn't start curl: {e}"))?;
    if !out.status.success() {
        return Err(curl_error(&out.stderr));
    }
    if out.stdout.len() > limit {
        return Err("GitHub sent an unexpectedly large reply.".into());
    }
    Ok(out.stdout)
}

/// Downloads `url` to `dest`, reporting the bytes written as they land.
fn download(url: &str, dest: &Path, size: u64, mut progress: impl FnMut(u64)) -> Result<(), String> {
    let _ = fs::remove_file(dest);
    let mut child = curl()
        .args(["--max-time", "900", "--max-filesize", &MAX_ARCHIVE_BYTES.to_string(), "-H", "User-Agent: Andai", "-o"])
        .arg(dest)
        .arg(url)
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Couldn't start curl: {e}"))?;
    let status = loop {
        if let Some(s) = child.try_wait().map_err(|e| e.to_string())? {
            break s;
        }
        progress(fs::metadata(dest).map(|m| m.len().min(size)).unwrap_or(0));
        std::thread::sleep(Duration::from_millis(120));
    };
    let mut err = Vec::new();
    if let Some(mut e) = child.stderr.take() {
        let _ = e.read_to_end(&mut err);
    }
    if !status.success() {
        let _ = fs::remove_file(dest);
        return Err(curl_error(&err));
    }
    progress(size);
    Ok(())
}

/// Unpacks a verified archive into `<root>/.ug` and links `<bin_dir>/ug` to
/// it, as `install.sh` does. It unpacks next to the old folder first, so a
/// bad archive leaves whatever was there untouched.
fn install_archive(archive: &Path, root: &Path, bin_dir: &Path) -> Result<PathBuf, String> {
    let io = |what: &str, e: std::io::Error| format!("Couldn't {what}: {e}");
    fs::create_dir_all(root).map_err(|e| io("create the UltraGraph folder", e))?;
    let staging = root.join(".ug.partial");
    let _ = fs::remove_dir_all(&staging);
    fs::create_dir_all(&staging).map_err(|e| io("create the UltraGraph folder", e))?;
    // bsdtar and GNU tar both refuse absolute paths and `..` entries by default.
    let out = Command::new("/usr/bin/tar")
        .env_clear()
        .arg("-xzf")
        .arg(archive)
        .arg("-C")
        .arg(&staging)
        .stdin(Stdio::null())
        .output()
        .map_err(|e| io("run tar", e))?;
    let bin = staging.join("ug");
    let valid = out.status.success() && fs::symlink_metadata(&bin).map(|m| m.is_file()).unwrap_or(false);
    if !valid {
        let _ = fs::remove_dir_all(&staging);
        let err = String::from_utf8_lossy(&out.stderr).trim().to_string();
        return Err(format!("The UltraGraph archive didn't contain the ug program. {err}").trim().to_string());
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&bin, fs::Permissions::from_mode(0o755)).map_err(|e| io("mark ug as executable", e))?;
    }
    let dir = root.join(".ug");
    if dir.exists() {
        fs::remove_dir_all(&dir).map_err(|e| io("replace the previous UltraGraph folder", e))?;
    }
    fs::rename(&staging, &dir).map_err(|e| io("move UltraGraph into place", e))?;

    fs::create_dir_all(bin_dir).map_err(|e| io(&format!("create {}", bin_dir.display()), e))?;
    let link = bin_dir.join("ug");
    match fs::symlink_metadata(&link) {
        Ok(m) if m.file_type().is_symlink() => fs::remove_file(&link).map_err(|e| io("replace the old ug link", e))?,
        Ok(_) => return Err(format!("{} already exists and isn't a link; remove it and try again.", link.display())),
        Err(_) => {}
    }
    #[cfg(unix)]
    std::os::unix::fs::symlink(dir.join("ug"), &link).map_err(|e| io(&format!("link {}", link.display()), e))?;
    Ok(link)
}

fn home() -> Result<PathBuf, String> {
    std::env::var_os("HOME").map(PathBuf::from).filter(|h| h.is_absolute()).ok_or_else(|| "HOME is not set.".to_string())
}

/// The whole install, reporting each stage. `home` is a parameter so tests
/// can install into a temporary one.
fn run(home: &Path, emit: &mut dyn FnMut(InstallProgress)) -> Result<(), String> {
    let name = asset_name().ok_or("Automatic install isn't available on this platform; download ug from its website.")?;
    let mut report = |stage, version: &Option<String>, done, total| emit(InstallProgress { stage, version: version.clone(), done, total });
    report("lookup", &None, 0, 0);
    let release: Value = serde_json::from_slice(&get_small(RELEASE_API, MAX_RELEASE_JSON)?)
        .map_err(|_| format!("GitHub's reply about {REPO} wasn't readable."))?;
    let asset = pick_asset(&release, name)?;
    let version = Some(asset.version.clone());
    let expected = match (&asset.sha256, &asset.sha256_url) {
        (Some(h), _) => h.clone(),
        (None, Some(url)) => parse_sha256_file(&String::from_utf8_lossy(&get_small(url, 4096)?), name)?,
        (None, None) => unreachable!("pick_asset returns one of them"),
    };

    let root = home.join(".local/share/ultragraph");
    fs::create_dir_all(&root).map_err(|e| format!("Couldn't create {}: {e}", root.display()))?;
    let archive = root.join(format!(".{name}.download"));
    let result = (|| {
        report("download", &version, 0, asset.size);
        let mut last = Instant::now() - Duration::from_secs(1);
        download(&asset.url, &archive, asset.size, |done| {
            if last.elapsed() >= Duration::from_millis(100) || done == asset.size {
                last = Instant::now();
                report("download", &version, done, asset.size);
            }
        })?;
        report("verify", &version, 0, 0);
        let got = fs::metadata(&archive).map(|m| m.len()).unwrap_or(0);
        if got != asset.size {
            return Err(format!("The download is incomplete ({got} of {} bytes). Try again.", asset.size));
        }
        let hash = sha256_of(&archive).map_err(|e| format!("Couldn't read the download: {e}"))?;
        if hash != expected {
            return Err("The download doesn't match UltraGraph's published checksum, so it was discarded. Try again.".into());
        }
        report("install", &version, 0, 0);
        install_archive(&archive, &root, &home.join(".local/bin"))
    })();
    let _ = fs::remove_file(&archive);
    result.map(|_| ())
}

/// Installs ug for this user (no admin rights) and returns the new status.
/// Refuses when ug is already installed or an install is running.
#[tauri::command]
pub async fn ug_install(app: AppHandle, installing: State<'_, Installing>) -> Result<UgStatus, String> {
    if ug_path().is_some() {
        return Ok(ug_status());
    }
    if installing.0.swap(true, Ordering::SeqCst) {
        return Err("UltraGraph is already being installed.".into());
    }
    let handle = app.clone();
    let result = tauri::async_runtime::spawn_blocking(move || {
        run(&home()?, &mut |p| {
            let _ = handle.emit("ug-install", p);
        })
    })
    .await
    .map_err(|e| e.to_string())
    .and_then(|r| r);
    installing.0.store(false, Ordering::SeqCst);
    result?;
    let _ = app.emit("ug-install", InstallProgress { stage: "check", version: None, done: 0, total: 0 });
    let status = tauri::async_runtime::spawn_blocking(ug_status).await.map_err(|e| e.to_string())?;
    if !status.found {
        return Err("ug was installed but Andai can't find it. Restart Andai and try again.".into());
    }
    Ok(status)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const HASH: &str = "331ea28b93558b97905d4dd7ff5c4a3235a1b3c656d9f6909a35363c774433ae";

    fn release(assets: Value) -> Value {
        json!({ "tag_name": "v0.1.22", "assets": assets })
    }
    fn asset(name: &str, digest: Option<&str>) -> Value {
        json!({
            "name": name,
            "size": 25292946,
            "digest": digest,
            "browser_download_url": format!("{DOWNLOAD_PREFIX}v0.1.22/{name}"),
        })
    }

    #[test]
    fn picks_the_exact_asset_with_its_digest() {
        let name = "ultragraph-macos-arm64.tar.gz";
        let r = release(json!([asset(&format!("{name}.sha256"), None), asset(name, Some(&format!("sha256:{HASH}")))]));
        let a = pick_asset(&r, name).unwrap();
        assert_eq!(a.url, format!("{DOWNLOAD_PREFIX}v0.1.22/{name}"));
        assert_eq!((a.size, a.sha256.as_deref(), a.sha256_url), (25292946, Some(HASH), None));
    }

    #[test]
    fn falls_back_to_the_checksum_file_without_a_digest() {
        let name = "ultragraph-macos-arm64.tar.gz";
        let r = release(json!([asset(name, None), asset(&format!("{name}.sha256"), None)]));
        let a = pick_asset(&r, name).unwrap();
        assert_eq!(a.sha256, None);
        assert_eq!(a.sha256_url.unwrap(), format!("{DOWNLOAD_PREFIX}v0.1.22/{name}.sha256"));
        // No digest and no checksum file: refused, never installed unverified.
        assert!(pick_asset(&release(json!([asset(name, None)])), name).is_err());
    }

    #[test]
    fn refuses_a_release_that_points_elsewhere() {
        let name = "ultragraph-macos-arm64.tar.gz";
        let digest = format!("sha256:{HASH}");
        let mut evil = asset(name, Some(&digest));
        evil["browser_download_url"] = json!("https://evil.example/ug.tar.gz");
        assert!(pick_asset(&release(json!([evil])), name).is_err());
        let mut other_tag = asset(name, Some(&digest));
        other_tag["browser_download_url"] = json!(format!("{DOWNLOAD_PREFIX}v9/{name}"));
        assert!(pick_asset(&release(json!([other_tag])), name).is_err());
        let mut huge = asset(name, Some(&digest));
        huge["size"] = json!(MAX_ARCHIVE_BYTES + 1);
        assert!(pick_asset(&release(json!([huge])), name).is_err());
        let bad_tag = json!({ "tag_name": "../x", "assets": [asset(name, Some(&digest))] });
        assert!(pick_asset(&bad_tag, name).is_err());
        assert!(pick_asset(&release(json!([asset("ultragraph-linux-x64.tar.gz", Some(&digest))])), name).is_err());
    }

    #[test]
    fn reads_shasum_output() {
        let name = "ultragraph-macos-arm64.tar.gz";
        assert_eq!(parse_sha256_file(&format!("{HASH}  {name}\n"), name).unwrap(), HASH);
        assert_eq!(parse_sha256_file(&format!("{}\n", HASH.to_uppercase()), name).unwrap(), HASH);
        assert!(parse_sha256_file(&format!("{HASH}  other.tar.gz"), name).is_err());
        assert!(parse_sha256_file("nope", name).is_err());
    }

    fn tarball(dir: &Path, files: &[(&str, &str)]) -> PathBuf {
        let src = dir.join("src");
        fs::create_dir_all(&src).unwrap();
        for (name, body) in files {
            fs::write(src.join(name), body).unwrap();
        }
        let out = dir.join("ug.tar.gz");
        let ok = Command::new("/usr/bin/tar").arg("-czf").arg(&out).arg("-C").arg(&src).arg(".").status().unwrap();
        assert!(ok.success());
        out
    }

    #[cfg(unix)]
    #[test]
    fn installs_like_install_sh_and_replaces_a_dangling_link() {
        let home = tempfile::tempdir().unwrap();
        let archive = tarball(home.path(), &[("ug", "#!/bin/sh\necho 'ug version 0.0.1'\n"), ("ug-app", "x")]);
        let root = home.path().join(".local/share/ultragraph");
        let bin = home.path().join(".local/bin");
        fs::create_dir_all(&bin).unwrap();
        std::os::unix::fs::symlink(home.path().join("gone"), bin.join("ug")).unwrap();
        fs::create_dir_all(root.join(".ug")).unwrap();
        fs::write(root.join(".ug/stale"), "old").unwrap();

        let link = install_archive(&archive, &root, &bin).unwrap();
        assert_eq!(fs::read_link(&link).unwrap(), root.join(".ug/ug"));
        let out = Command::new(&link).output().unwrap();
        assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "ug version 0.0.1");
        assert!(root.join(".ug/ug-app").is_file());
        assert!(!root.join(".ug/stale").exists(), "the old folder is replaced, as install.sh does");
        assert!(!root.join(".ug.partial").exists());
    }

    #[cfg(unix)]
    #[test]
    fn an_archive_without_ug_changes_nothing() {
        let home = tempfile::tempdir().unwrap();
        let archive = tarball(home.path(), &[("readme", "hi")]);
        let root = home.path().join(".local/share/ultragraph");
        fs::create_dir_all(root.join(".ug")).unwrap();
        fs::write(root.join(".ug/ug"), "previous").unwrap();
        assert!(install_archive(&archive, &root, &home.path().join(".local/bin")).is_err());
        assert_eq!(fs::read_to_string(root.join(".ug/ug")).unwrap(), "previous");
        assert!(!home.path().join(".local/bin/ug").exists());
        // Not a tarball at all.
        fs::write(&archive, "garbage").unwrap();
        assert!(install_archive(&archive, &root, &home.path().join(".local/bin")).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn never_replaces_a_real_file_at_the_link() {
        let home = tempfile::tempdir().unwrap();
        let archive = tarball(home.path(), &[("ug", "x")]);
        let bin = home.path().join(".local/bin");
        fs::create_dir_all(&bin).unwrap();
        fs::write(bin.join("ug"), "mine").unwrap();
        assert!(install_archive(&archive, &home.path().join("root"), &bin).is_err());
        assert_eq!(fs::read_to_string(bin.join("ug")).unwrap(), "mine");
    }

    #[test]
    fn hashes_files() {
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("abc");
        fs::write(&f, "abc").unwrap();
        assert_eq!(sha256_of(&f).unwrap(), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    }

    /// The real thing, into a temporary home: GitHub lookup, download,
    /// checksum, unpack, and the installed ug answers `-v`. Needs the network.
    #[test]
    #[ignore]
    fn installs_the_latest_release_for_real() {
        if asset_name().is_none() {
            return;
        }
        let home = tempfile::tempdir().unwrap();
        let mut stages = Vec::new();
        run(home.path(), &mut |p| {
            if stages.last() != Some(&p.stage) {
                stages.push(p.stage);
            }
        })
        .unwrap();
        assert_eq!(stages, ["lookup", "download", "verify", "install"]);
        let out = Command::new(home.path().join(".local/bin/ug")).arg("-v").output().unwrap();
        assert!(String::from_utf8_lossy(&out.stdout).contains("ug"), "{out:?}");
    }
}
