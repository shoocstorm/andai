//! The native chat models Andai can download and load on MLX: a closed list,
//! pinned to Hugging Face commits, with every file's size and sha256 (the
//! same format and store as Laya's, laya/catalog.rs). The webview names a
//! checkpoint and a file by id; paths and hashes only ever come from here.

use crate::laya::catalog::{Checkpoint, CheckpointFile};

const fn f(path: &'static str, bytes: u64, sha256: &'static str) -> CheckpointFile {
    CheckpointFile { path, bytes, sha256 }
}

/// The Qwen3 tokenizer is the same file in both repos.
const TOKENIZER: CheckpointFile = f("tokenizer.json", 11_422_654, "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4");

/// Measured sizes and hashes from the Hugging Face tree API at the pinned
/// commits (the LFS oid for the weights and tokenizer) and the downloaded
/// files (2026-09-27).
pub const CHECKPOINTS: &[Checkpoint] = &[
    Checkpoint {
        id: "qwen3-1.7b-mlx",
        repo: "mlx-community/Qwen3-1.7B-4bit",
        commit: "3b1b1768f8f8cf8351c712464f906e86c2b8269e",
        files: &[
            f("model.safetensors", 968_080_210, "0e86d9677e519323849eac1bc272caae88567a481ff188c431f70be543d9995f"),
            f("config.json", 937, "507a6701220524eb8b283425bf0856a9ae4f21f4052e563896ddd668994b1dc7"),
            TOKENIZER,
            f("tokenizer_config.json", 9_706, "253153d0738ceb4c668d2eff957714dd2bea0b56de772a9fdccd96cbf517e6a0"),
        ],
    },
    Checkpoint {
        id: "qwen3-0.6b-mlx",
        repo: "mlx-community/Qwen3-0.6B-8bit",
        commit: "11de96878523501bcaa86104e3c186de07ff9068",
        files: &[
            f("model.safetensors", 633_442_994, "3ad5c96e0a476d4eee48e9f525d0ee0a3f1830b3e5b8e17b441de0661e921397"),
            f("config.json", 937, "9f387dcb1bf045cadb34a36e67569abd73c00e5322603df2b19d0c0f8cc9d240"),
            TOKENIZER,
            f("tokenizer_config.json", 10_219, "8e7dd4250db58b2a29792562d727b2715dc0fe79a94b446caa4c07db39eceba6"),
        ],
    },
];

pub fn checkpoint(id: &str) -> Result<&'static Checkpoint, String> {
    CHECKPOINTS.iter().find(|c| c.id == id).ok_or_else(|| format!("unknown model: {id}"))
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
                assert!(!f.path.starts_with('/') && !f.path.split('/').any(|p| p == ".." || p.is_empty()));
            }
            for needed in ["model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json"] {
                c.file(needed).unwrap();
            }
        }
    }

    #[test]
    fn ids_do_not_collide_with_laya() {
        for c in CHECKPOINTS {
            assert!(crate::laya::catalog::checkpoint(c.id).is_err(), "{} is also a Laya id", c.id);
        }
        assert!(checkpoint("laya-en").is_err());
        assert!(checkpoint("../qwen3-1.7b-mlx").is_err());
    }
}
