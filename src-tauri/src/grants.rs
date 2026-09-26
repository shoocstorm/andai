//! Which files the webview may ingest.
//!
//! The webview is untrusted (AGENTS.md §9): model output and retrieved
//! documents are rendered there, so it must not be able to name an arbitrary
//! path (`~/.ssh/id_rsa`) and have Rust copy it into a knowledge base. A path
//! is only ingestible after the *user* chose it through a channel the webview
//! can't forge:
//!
//! - a drag-and-drop onto the window (`WindowEvent::DragDrop`, seen by Rust),
//! - the file dialog, opened by Rust in `kb_pick_files`,
//! - `ANDAI_E2E_FILES`, read once at startup (the e2e harness).
//!
//! Each grant is for one canonical file and is consumed when it is ingested.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use tauri::AppHandle;
use tauri_plugin_dialog::DialogExt;

/// Extensions `ug::ingest_file` accepts; the dialog filters on the same list.
pub const SUPPORTED_EXTENSIONS: &[&str] = &[
    "pdf", "md", "markdown", "mdx", "txt", "text", "log", "rst", "csv", "tsv", "ts", "tsx", "js", "jsx", "mjs",
    "cjs", "py", "java", "rs",
];

#[derive(Default)]
pub struct FileGrants(Mutex<HashSet<PathBuf>>);

impl FileGrants {
    /// Grants each path the user chose. Paths are canonicalized so a later
    /// request through a symlink or `..` resolves to the same entry, or to an
    /// ungranted one. Paths that don't exist can't be granted.
    pub fn grant<P: AsRef<Path>>(&self, paths: impl IntoIterator<Item = P>) {
        let mut set = self.0.lock().unwrap();
        set.extend(paths.into_iter().filter_map(|p| p.as_ref().canonicalize().ok()));
    }

    /// Consumes the grant for `path`, returning its canonical form, or an
    /// error a user can act on. Whatever a symlink points at is what must have
    /// been granted.
    pub fn take(&self, path: &Path) -> Result<PathBuf, String> {
        let denied = || format!("{}: not added by you — drop the file or use Upload", path.display());
        let canonical = path.canonicalize().map_err(|_| denied())?;
        if self.0.lock().unwrap().remove(&canonical) {
            Ok(canonical)
        } else {
            Err(denied())
        }
    }
}

/// Grants the fixtures the e2e harness passes in `ANDAI_E2E_FILES`. The
/// environment is set by whoever launched the process, never by the webview.
pub fn grant_from_env(grants: &FileGrants) {
    if let Some(list) = std::env::var_os("ANDAI_E2E_FILES") {
        let list = list.to_string_lossy();
        grants.grant(list.split(',').filter(|p| !p.is_empty()));
    }
}

/// Opens the file dialog from Rust, grants what the user picked and returns
/// the paths. The JS dialog API is not granted (capabilities/default.json), so
/// this is the only way a picked path becomes ingestible.
#[tauri::command]
pub async fn kb_pick_files(
    app: AppHandle,
    title: String,
    grants: tauri::State<'_, FileGrants>,
) -> Result<Vec<String>, String> {
    let picked = tauri::async_runtime::spawn_blocking(move || {
        app.dialog()
            .file()
            .set_title(title)
            .add_filter("Documents", SUPPORTED_EXTENSIONS)
            .blocking_pick_files()
    })
    .await
    .map_err(|e| e.to_string())?;
    let paths: Vec<PathBuf> = picked.unwrap_or_default().into_iter().filter_map(|p| p.into_path().ok()).collect();
    grants.grant(&paths);
    Ok(paths.into_iter().map(|p| p.to_string_lossy().into()).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn ungranted_paths_are_refused() {
        let dir = tempfile::tempdir().unwrap();
        let secret = dir.path().join("id_rsa.md");
        fs::write(&secret, "key").unwrap();
        let err = FileGrants::default().take(&secret).unwrap_err();
        assert!(err.contains("not added by you"), "{err}");
    }

    #[test]
    fn a_grant_is_consumed_once() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("notes.md");
        fs::write(&file, "x").unwrap();
        let grants = FileGrants::default();
        grants.grant([&file]);
        assert_eq!(grants.take(&file).unwrap(), file.canonicalize().unwrap());
        assert!(grants.take(&file).is_err(), "a second ingest needs a new grant");
    }

    #[test]
    fn dot_dot_resolves_to_the_same_grant() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir(dir.path().join("sub")).unwrap();
        let file = dir.path().join("notes.md");
        fs::write(&file, "x").unwrap();
        let grants = FileGrants::default();
        grants.grant([&file]);
        assert!(grants.take(&dir.path().join("sub/../notes.md")).is_ok());
    }

    #[cfg(unix)]
    #[test]
    fn a_symlink_cannot_launder_an_ungranted_target() {
        let dir = tempfile::tempdir().unwrap();
        let granted = dir.path().join("granted.md");
        let secret = dir.path().join("secret.md");
        fs::write(&granted, "ok").unwrap();
        fs::write(&secret, "key").unwrap();
        let grants = FileGrants::default();
        grants.grant([&granted]);

        // Swap the granted name for a link to the secret after the grant.
        let link = dir.path().join("link.md");
        std::os::unix::fs::symlink(&secret, &link).unwrap();
        assert!(grants.take(&link).is_err());
        assert!(grants.take(&granted).is_ok());
    }

    #[test]
    fn missing_files_are_never_granted() {
        let grants = FileGrants::default();
        grants.grant(["/definitely/not/here.md"]);
        assert!(grants.0.lock().unwrap().is_empty());
    }

    #[test]
    fn dialog_filter_matches_what_ingest_accepts() {
        let src = tempfile::tempdir().unwrap();
        let docs = tempfile::tempdir().unwrap();
        for ext in SUPPORTED_EXTENSIONS {
            let p = src.path().join(format!("f.{ext}"));
            fs::write(&p, "x").unwrap();
            assert!(crate::ug::ingest_file(docs.path(), &p).is_ok(), "{ext} is in the filter but ingest rejects it");
        }
    }
}
