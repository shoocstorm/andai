//! Sample knowledge bases shipped with the app, so the agent can be tried
//! before the user has documents of their own: Tidewater Ferries, a fictional
//! ferry operator (the agent eval's fixtures, tests/fixtures/eval/), as
//! documents, as code, and as both.
//!
//! The source files are bundled (tauri.conf.json `bundle.resources`), not a
//! prebuilt graph: a ug graph records its absolute root and ug version, so a
//! copied one breaks on another machine. Adding a sample copies its files in
//! like a user's (`ug::ingest_file`); the webview then indexes it with the
//! usual `kb_index`, which takes a few seconds for these files.
//!
//! The webview names a sample by id from this closed list (AGENTS.md §9);
//! paths only come from here and the app's resource folder.

use crate::ug::{self, Indexing, KbInfo};
use std::fs;
use std::path::PathBuf;
use tauri::{AppHandle, Manager, State};

pub struct Sample {
    pub id: &'static str,
    /// The knowledge base it becomes: `andai-<slug>`, with its files in `kb/<slug>/docs`.
    pub slug: &'static str,
    pub name: &'static str,
    /// Folders under the bundled `samples/`, in order.
    pub folders: &'static [&'static str],
}

pub const SAMPLES: &[Sample] = &[
    Sample { id: "tidewater-docs", slug: "tidewater-ferries-documents", name: "Tidewater Ferries · Documents", folders: &["docs", "large"] },
    Sample { id: "tidewater-code", slug: "tidewater-ferries-code", name: "Tidewater Ferries · Code", folders: &["code"] },
    Sample { id: "tidewater-mixed", slug: "tidewater-ferries-docs-code", name: "Tidewater Ferries · Docs + code", folders: &["docs", "code"] },
];

fn sample(id: &str) -> Result<&'static Sample, String> {
    SAMPLES.iter().find(|s| s.id == id).ok_or_else(|| format!("unknown sample: {id}"))
}

/// The bundled `samples/` folder. Tauri copies bundle resources next to the
/// binary in dev builds too; a debug build falls back to the fixtures.
fn samples_root(app: &AppHandle) -> Result<PathBuf, String> {
    if let Some(dir) = app.path().resource_dir().ok().map(|d| d.join("samples")).filter(|d| d.is_dir()) {
        return Ok(dir);
    }
    #[cfg(debug_assertions)]
    {
        let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures/eval");
        if fixtures.is_dir() {
            return Ok(fixtures);
        }
    }
    Err("The sample files are missing from this build of Andai.".into())
}

/// The sample's files, in a stable order: each folder's files, sorted by name.
fn files(root: &std::path::Path, s: &Sample) -> Result<Vec<PathBuf>, String> {
    let mut out = Vec::new();
    for folder in s.folders {
        let mut here: Vec<PathBuf> = fs::read_dir(root.join(folder))
            .map_err(|e| format!("sample folder {folder}: {e}"))?
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_file())
            .collect();
        here.sort();
        out.extend(here);
    }
    if out.is_empty() {
        return Err(format!("{} has no files in this build.", s.name));
    }
    Ok(out)
}

/// Adds sample `sample` as knowledge base `andai-<slug>`, or returns it if
/// it's already there. Its files are copied in and pending; call `kb_index` next.
#[tauri::command]
pub fn kb_add_sample(app: AppHandle, sample: String, indexing: State<'_, Indexing>) -> Result<KbInfo, String> {
    let s = self::sample(&sample)?;
    let project = format!("{}{}", ug::PROJECT_PREFIX, s.slug);
    let docs = ug::kb_dir(&app, s.slug)?.join("docs");
    let has_files = fs::read_dir(&docs).map(|mut d| d.next().is_some()).unwrap_or(false);
    if !has_files {
        let paths = files(&samples_root(&app)?, s)?;
        ug::create_private_dir(&docs).map_err(|e| e.to_string())?;
        for p in &paths {
            ug::ingest_file(&docs, p)?;
        }
    }
    ug::load_info(&app, &project, &indexing)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixtures() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures/eval")
    }

    #[test]
    fn samples_are_a_closed_list_of_distinct_ids_and_names() {
        assert!(sample("tidewater-docs").is_ok());
        for bad in ["", "../x", "tidewater", "Tidewater-docs"] {
            assert!(sample(bad).is_err(), "{bad:?}");
        }
        let ids: std::collections::HashSet<_> = SAMPLES.iter().map(|s| s.id).collect();
        let names: std::collections::HashSet<_> = SAMPLES.iter().map(|s| s.name).collect();
        let slugs: std::collections::HashSet<_> = SAMPLES.iter().map(|s| s.slug).collect();
        assert_eq!((ids.len(), names.len(), slugs.len()), (SAMPLES.len(), SAMPLES.len(), SAMPLES.len()));
        // A sample's slug is what slugify makes of its name, as for a knowledge base the user names.
        for s in SAMPLES {
            assert_eq!(ug::slugify(s.name), s.slug);
        }
    }

    #[test]
    fn each_sample_has_the_files_of_its_kind() {
        let kinds = |id: &str| -> Vec<String> {
            let s = sample(id).unwrap();
            let tmp = tempfile::tempdir().unwrap();
            files(&fixtures(), s).unwrap().iter().map(|p| ug::ingest_file(tmp.path(), p).unwrap().kind).collect()
        };
        let docs = kinds("tidewater-docs");
        let code = kinds("tidewater-code");
        let mixed = kinds("tidewater-mixed");
        assert!(!docs.is_empty() && docs.iter().all(|k| k == "MD"), "{docs:?}");
        assert!(!code.is_empty() && code.iter().all(|k| k == "CODE"), "{code:?}");
        assert!(mixed.iter().any(|k| k == "MD") && mixed.iter().any(|k| k == "CODE"), "{mixed:?}");
        // The documents sample holds the long documents too (`large/`), not only the short ones the mixed one shares.
        assert_eq!(docs.len(), 7, "{docs:?}");
    }

    #[test]
    fn the_bundle_ships_every_sample_folder() {
        let conf: serde_json::Value = serde_json::from_str(include_str!("../tauri.conf.json")).unwrap();
        let resources = conf["bundle"]["resources"].as_object().expect("bundle.resources is a map");
        for s in SAMPLES {
            for folder in s.folders {
                assert!(resources.values().any(|v| v.as_str() == Some(&format!("samples/{folder}/"))), "samples/{folder}/ is bundled");
            }
        }
    }
}
