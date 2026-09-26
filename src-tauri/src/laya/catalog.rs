//! The Laya checkpoints Andai can download and load: a closed list, pinned to
//! Hugging Face commits, with every file's size and sha256. The webview names
//! a checkpoint and a file by id; paths and hashes only ever come from here.

use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct CheckpointFile {
    /// Path inside the checkpoint, as on Hugging Face and on disk.
    pub path: &'static str,
    pub bytes: u64,
    /// sha256 of the file: the LFS oid for large files, and for the small JSON
    /// files the hash of the file served at the pinned commit.
    pub sha256: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Checkpoint {
    pub id: &'static str,
    pub repo: &'static str,
    /// Immutable commit; `resolve/main` could be moved under us (AGENTS.md §2).
    pub commit: &'static str,
    pub files: &'static [CheckpointFile],
}

const fn f(path: &'static str, bytes: u64, sha256: &'static str) -> CheckpointFile {
    CheckpointFile { path, bytes, sha256 }
}

/// Measured sizes and hashes from `POST /api/models/<repo>/paths-info/<commit>`
/// and the downloaded files (2026-09-26).
pub const CHECKPOINTS: &[Checkpoint] = &[
    Checkpoint {
        id: "laya-multilingual",
        repo: "aac6fef/laya-multilingual-mlx",
        commit: "f2b4faf51023039425946074e2cf1361d2db11d5",
        files: &[
            f("model.safetensors", 643_835_426, "7fc5834af4d8fdfb268d272a9d1a66e5819a0daac98241651c4c888cc43adff1"),
            f("rl_agent_config.json", 473, "9a669a70961064c3c6cc76d2afb8bc5fb10dcd8349bb66e5f7b9b1afb74440d5"),
            f("encoder/config.json", 1_938, "83f6916d13ef0f556ac461f28308dc2bffa7ebeadee8ec9e2db5812020ea5bb4"),
            f("tokenizer/tokenizer.json", 34_363_188, "609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f"),
            f("tokenizer/tokenizer_config.json", 524, "6c6b2d8e3c84ce0e671c129cd6b374b235d6f9863042a5836358d00a89bbb5a1"),
        ],
    },
    Checkpoint {
        id: "laya-en",
        repo: "aac6fef/laya-mlx",
        commit: "20aed815fc6acde75733882e7ec0e3f28aeb9717",
        files: &[
            f("model.safetensors", 842_609_225, "b9c07bf14be2fa5c78a9193a3e6d840ac80e89e62fc40f425834c3d8a6eaa3de"),
            f("rl_agent_config.json", 746, "d96dc2cb39d6375e030ff48c9957088f3c52668f45c30c56504f4e801ed3ee62"),
            f("encoder/config.json", 2_083, "bf3ab80598fdccf414855a2ce80f22859e4492d06ca8a62ddd1cfb63972f8979"),
            f("tokenizer/tokenizer.json", 3_583_228, "6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30"),
            f("tokenizer/tokenizer_config.json", 308, "50044de60daaa73df97d262e15a40d4faf0160e7d742df64b377877a1320dd12"),
        ],
    },
];

pub fn checkpoint(id: &str) -> Result<&'static Checkpoint, String> {
    CHECKPOINTS.iter().find(|c| c.id == id).ok_or_else(|| format!("unknown Laya checkpoint: {id}"))
}

impl Checkpoint {
    pub fn file(&self, path: &str) -> Result<&'static CheckpointFile, String> {
        self.files.iter().find(|f| f.path == path).ok_or_else(|| format!("{} has no file {path}", self.id))
    }

    pub fn bytes(&self) -> u64 {
        self.files.iter().map(|f| f.bytes).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_checkpoint_is_pinned_and_hashed() {
        for c in CHECKPOINTS {
            assert_eq!(c.commit.len(), 40, "{}: commit must be a full sha", c.id);
            assert!(c.commit.chars().all(|ch| ch.is_ascii_hexdigit()));
            for f in c.files {
                assert_eq!(f.sha256.len(), 64, "{} {}", c.id, f.path);
                assert!(f.sha256.chars().all(|ch| ch.is_ascii_hexdigit() && !ch.is_ascii_uppercase()));
                assert!(f.bytes > 0);
                // relative, no traversal: joined under the checkpoint folder
                assert!(!f.path.starts_with('/') && !f.path.split('/').any(|p| p == ".." || p.is_empty()));
            }
            for needed in ["model.safetensors", "rl_agent_config.json", "encoder/config.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"] {
                c.file(needed).unwrap();
            }
        }
    }

    #[test]
    fn unknown_ids_and_files_are_refused() {
        assert!(checkpoint("laya-multilingual").is_ok());
        assert!(checkpoint("../laya").is_err());
        assert!(checkpoint("laya-multilingual").unwrap().file("../../etc/passwd").is_err());
    }
}
