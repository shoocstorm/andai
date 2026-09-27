//! Laya checkpoint files on disk, under `<data>/models/laya/<id>/`.
//!
//! The webview downloads (the Rust side has no HTTP client, AGENTS.md §9) and
//! streams each file here in chunks, into `<file>.part`. Only `finish` turns
//! parts into the files the model loads, and only when every file has the
//! catalog's exact size and sha256; otherwise every part is removed. So the
//! webview can't place a file that `load` would read: that takes the hash.

use super::catalog::Checkpoint;
use crate::ug::{create_private_dir, private_file};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// Written last by `finish`: the commit whose files passed verification.
const MARKER: &str = ".verified";

pub fn dir(root: &Path, c: &Checkpoint) -> PathBuf {
    root.join(&*c.id)
}

fn part_path(root: &Path, c: &Checkpoint, file: &str) -> PathBuf {
    dir(root, c).join(format!("{file}.part"))
}

/// Appends `bytes` at `offset` to the part file of `file`; offset 0 starts it
/// over. Chunks must arrive in order. Returns the part's new length.
pub fn write_chunk(root: &Path, c: &Checkpoint, file: &str, offset: u64, bytes: &[u8]) -> Result<u64, String> {
    let spec = c.file(file)?;
    let part = part_path(root, c, file);
    let end = offset + bytes.len() as u64;
    if end > spec.bytes {
        let _ = fs::remove_file(&part);
        return Err(format!("{file} is larger than expected ({} bytes)", spec.bytes));
    }
    create_private_dir(part.parent().expect("part has a parent")).map_err(|e| e.to_string())?;
    let mut out = if offset == 0 {
        private_file(&part).map_err(|e| e.to_string())?
    } else {
        let have = fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
        if have != offset {
            return Err(format!("{file}: chunk at byte {offset}, but {have} bytes were written"));
        }
        fs::OpenOptions::new().append(true).open(&part).map_err(|e| e.to_string())?
    };
    out.write_all(bytes).map_err(|e| e.to_string())?;
    Ok(end)
}

fn sha256_file(path: &Path) -> std::io::Result<String> {
    let mut hasher = Sha256::new();
    let mut file = fs::File::open(path)?;
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

fn remove_parts(root: &Path, c: &Checkpoint) {
    for f in c.files.iter() {
        let _ = fs::remove_file(part_path(root, c, &f.path));
    }
}

/// Checks every part against the catalog (size, then sha256) and, only if all
/// match, moves them into place and marks the checkpoint verified. Any
/// mismatch removes every part: a bad download is never kept.
pub fn finish(root: &Path, c: &Checkpoint) -> Result<(), String> {
    for f in c.files.iter() {
        let part = part_path(root, c, &f.path);
        let fail = |why: String| {
            remove_parts(root, c);
            Err(format!("{}: {why}. The download was discarded; try again.", f.path))
        };
        let size = match fs::metadata(&part) {
            Ok(m) => m.len(),
            Err(_) => return fail("missing".into()),
        };
        if size != f.bytes {
            return fail(format!("{size} bytes, expected {}", f.bytes));
        }
        match sha256_file(&part) {
            Ok(h) if h == f.sha256 => {}
            Ok(_) => return fail("checksum mismatch".into()),
            Err(e) => return fail(e.to_string()),
        }
    }
    let d = dir(root, c);
    let _ = fs::remove_file(d.join(MARKER));
    for f in c.files.iter() {
        fs::rename(part_path(root, c, &f.path), d.join(&*f.path)).map_err(|e| e.to_string())?;
    }
    private_file(&d.join(MARKER))
        .and_then(|mut m| m.write_all(c.commit.as_bytes()))
        .map_err(|e| e.to_string())
}

/// Verified at download (marker for this commit) and still complete on disk.
pub fn is_downloaded(root: &Path, c: &Checkpoint) -> bool {
    let d = dir(root, c);
    fs::read_to_string(d.join(MARKER)).is_ok_and(|m| m == *c.commit)
        && c.files.iter().all(|f| fs::metadata(d.join(&*f.path)).is_ok_and(|m| m.len() == f.bytes))
}

/// The folder to load from, or why it can't be loaded.
pub fn resolve(root: &Path, c: &Checkpoint) -> Result<PathBuf, String> {
    if is_downloaded(root, c) {
        Ok(dir(root, c))
    } else {
        Err(format!("{} isn't downloaded yet; download it in Settings → Models.", c.id))
    }
}

pub fn remove(root: &Path, c: &Checkpoint) -> Result<(), String> {
    match fs::remove_dir_all(dir(root, c)) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e.to_string()),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::super::catalog::f;
    use super::*;

    // sha256("hello") and sha256("abc")
    const HELLO: &str = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824";
    const ABC: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    static TEST_FILES: [super::super::catalog::CheckpointFile; 2] = [f("model.safetensors", 5, HELLO), f("tokenizer/tokenizer.json", 3, ABC)];
    static TEST: Checkpoint = super::super::catalog::pinned("test-ckpt", "x/y", "0123456789abcdef0123456789abcdef01234567", &TEST_FILES);

    fn download(root: &Path) {
        write_chunk(root, &TEST, "model.safetensors", 0, b"hel").unwrap();
        assert_eq!(write_chunk(root, &TEST, "model.safetensors", 3, b"lo").unwrap(), 5);
        write_chunk(root, &TEST, "tokenizer/tokenizer.json", 0, b"abc").unwrap();
    }

    #[test]
    fn a_verified_download_is_moved_into_place() {
        let tmp = tempfile::tempdir().unwrap();
        download(tmp.path());
        assert!(!is_downloaded(tmp.path(), &TEST), "parts alone are not a download");
        assert!(resolve(tmp.path(), &TEST).is_err());
        finish(tmp.path(), &TEST).unwrap();
        assert!(is_downloaded(tmp.path(), &TEST));
        let d = resolve(tmp.path(), &TEST).unwrap();
        assert_eq!(fs::read(d.join("model.safetensors")).unwrap(), b"hello");
        assert!(!d.join("model.safetensors.part").exists());
    }

    #[test]
    fn a_checksum_mismatch_discards_every_part() {
        let tmp = tempfile::tempdir().unwrap();
        write_chunk(tmp.path(), &TEST, "model.safetensors", 0, b"HELLO").unwrap();
        write_chunk(tmp.path(), &TEST, "tokenizer/tokenizer.json", 0, b"abc").unwrap();
        let err = finish(tmp.path(), &TEST).unwrap_err();
        assert!(err.contains("model.safetensors: checksum mismatch"), "{err}");
        assert!(!part_path(tmp.path(), &TEST, "tokenizer/tokenizer.json").exists());
        assert!(!is_downloaded(tmp.path(), &TEST));
    }

    #[test]
    fn a_missing_or_short_file_fails_finish() {
        let tmp = tempfile::tempdir().unwrap();
        write_chunk(tmp.path(), &TEST, "model.safetensors", 0, b"hel").unwrap();
        assert!(finish(tmp.path(), &TEST).unwrap_err().contains("3 bytes, expected 5"));
        write_chunk(tmp.path(), &TEST, "model.safetensors", 0, b"hello").unwrap();
        assert!(finish(tmp.path(), &TEST).unwrap_err().contains("tokenizer/tokenizer.json: missing"));
    }

    #[test]
    fn chunks_must_arrive_in_order_and_fit_the_catalog_size() {
        let tmp = tempfile::tempdir().unwrap();
        write_chunk(tmp.path(), &TEST, "model.safetensors", 0, b"he").unwrap();
        assert!(write_chunk(tmp.path(), &TEST, "model.safetensors", 3, b"lo").unwrap_err().contains("2 bytes were written"));
        assert!(write_chunk(tmp.path(), &TEST, "model.safetensors", 2, b"llo!!").unwrap_err().contains("larger than expected"));
        assert!(!part_path(tmp.path(), &TEST, "model.safetensors").exists(), "an oversized file is dropped");
        assert!(write_chunk(tmp.path(), &TEST, "../escape", 0, b"x").is_err(), "only catalog files");
    }

    #[test]
    fn tampering_after_download_makes_it_not_downloaded() {
        let tmp = tempfile::tempdir().unwrap();
        download(tmp.path());
        finish(tmp.path(), &TEST).unwrap();
        fs::write(dir(tmp.path(), &TEST).join("model.safetensors"), b"hi").unwrap();
        assert!(!is_downloaded(tmp.path(), &TEST));
        fs::write(dir(tmp.path(), &TEST).join(MARKER), b"another-commit").unwrap();
        assert!(!is_downloaded(tmp.path(), &TEST));
    }

    #[test]
    fn remove_deletes_the_folder_and_tolerates_absence() {
        let tmp = tempfile::tempdir().unwrap();
        download(tmp.path());
        finish(tmp.path(), &TEST).unwrap();
        remove(tmp.path(), &TEST).unwrap();
        assert!(!dir(tmp.path(), &TEST).exists());
        remove(tmp.path(), &TEST).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn files_are_private() {
        use std::os::unix::fs::PermissionsExt;
        let tmp = tempfile::tempdir().unwrap();
        download(tmp.path());
        finish(tmp.path(), &TEST).unwrap();
        let d = dir(tmp.path(), &TEST);
        assert_eq!(fs::metadata(d.join("model.safetensors")).unwrap().permissions().mode() & 0o777, 0o600);
        assert_eq!(fs::metadata(d.join("tokenizer")).unwrap().permissions().mode() & 0o777, 0o700);
    }
}
