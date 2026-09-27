//! The native chat models Andai can download and load on MLX: a closed list,
//! pinned to Hugging Face commits, with every file's size and sha256 (the
//! same format and store as Laya's, laya/catalog.rs). The webview names a
//! checkpoint and a file by id; paths and hashes only ever come from here.

use crate::laya::catalog::{f, pinned, Checkpoint, CheckpointFile};

/// The Qwen3 tokenizer is the same file in both repos.
const TOKENIZER: CheckpointFile = f("tokenizer.json", 11_422_654, "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4");

static QWEN3_1_7B_MLX_FILES: [CheckpointFile; 4] = [
    f("model.safetensors", 968_080_210, "0e86d9677e519323849eac1bc272caae88567a481ff188c431f70be543d9995f"),
    f("config.json", 937, "507a6701220524eb8b283425bf0856a9ae4f21f4052e563896ddd668994b1dc7"),
    TOKENIZER,
    f("tokenizer_config.json", 9_706, "253153d0738ceb4c668d2eff957714dd2bea0b56de772a9fdccd96cbf517e6a0"),
];

static QWEN3_0_6B_MLX_FILES: [CheckpointFile; 4] = [
    f("model.safetensors", 633_442_994, "3ad5c96e0a476d4eee48e9f525d0ee0a3f1830b3e5b8e17b441de0661e921397"),
    f("config.json", 937, "9f387dcb1bf045cadb34a36e67569abd73c00e5322603df2b19d0c0f8cc9d240"),
    TOKENIZER,
    f("tokenizer_config.json", 10_219, "8e7dd4250db58b2a29792562d727b2715dc0fe79a94b446caa4c07db39eceba6"),
];

/// Qwen's own MLX builds, 4-bit in groups of 128 (Hugging Face tree API at
/// the pinned commits, and the downloaded files, 2026-09-27). 8B and up have
/// their own LM head; 14B and 32B are sharded, so their index is pinned too.
static QWEN3_4B_MLX_FILES: [CheckpointFile; 4] = [
    f("model.safetensors", 2_137_326_367, "e56a94b846a2dce7d1cc154ce58b9cfc4b8d99e8ffb74bfe256a265f4cbc7fb9"),
    f("config.json", 988, "76eae114d67a444c7296be7a6bc0409ecc15c340e113f1bac7572cb7d0928b6f"),
    TOKENIZER,
    f("tokenizer_config.json", 9_732, "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101"),
];

static QWEN3_8B_MLX_FILES: [CheckpointFile; 4] = [
    f("model.safetensors", 4_351_884_216, "fcc83d7537bee76cc143be3e25f8954b816798f93528103db1cddb02977c5617"),
    f("config.json", 941, "eb21fe151c8d2b761794129a0f92a8b719a76d97705dda758ed6746c829a7720"),
    TOKENIZER,
    f("tokenizer_config.json", 9_690, "f45a828b27bfe1ae2080246d6de51c29370a0978f15b3299123830d5ba9e08a1"),
];

static QWEN3_14B_MLX_FILES: [CheckpointFile; 6] = [
    f("model-00001-of-00002.safetensors", 5_360_593_224, "f2f502ca7604ad6789b124bf9cd7a192fda5a60e6f52b1fe521c3aa4dad598ce"),
    f("model-00002-of-00002.safetensors", 2_485_808_912, "d3758e05e08bcfb5fcd1b76d74758366355c80464398702051ce0388045e898e"),
    f("model.safetensors.index.json", 86_266, "4825c397a7eb6ddde3827310ced22cdc3f6d1c2d5ca393e111bdcd9ad5601b05"),
    f("config.json", 941, "b2b3caeb0beef3a231730ec846241585b64c7c88b99dba427450f6a9f2a04214"),
    TOKENIZER,
    f("tokenizer_config.json", 9_690, "f45a828b27bfe1ae2080246d6de51c29370a0978f15b3299123830d5ba9e08a1"),
];

static QWEN3_32B_MLX_FILES: [CheckpointFile; 8] = [
    f("model-00001-of-00004.safetensors", 5_363_162_698, "c09c797044b3a0481c8718c1e20342580d83e248238b4dfe93dd600c403f599d"),
    f("model-00002-of-00004.safetensors", 5_342_645_855, "fcc830611e03d175a034d04bfdafa433ca218e2574b2c034596a2ea348f5b608"),
    f("model-00003-of-00004.safetensors", 5_368_472_294, "efd53790497af9614d484437dbb6d8303e11ba961c765e859a5ac38a683582eb"),
    f("model-00004-of-00004.safetensors", 1_331_774_591, "61df74f545bc1ebf254dbce58e36b35f03c9bc1bbe722886250ddeedc3b167cf"),
    f("model.safetensors.index.json", 137_843, "ac9806a9f6768f35b404ae9179fc65231203a53476e606235d1413b4b2586374"),
    f("config.json", 941, "0059296f60b039b09e00bd5cbad130abd7277d2a2b6bf520839a3d633de4ae15"),
    TOKENIZER,
    f("tokenizer_config.json", 9_690, "f45a828b27bfe1ae2080246d6de51c29370a0978f15b3299123830d5ba9e08a1"),
];

/// Measured sizes and hashes from the Hugging Face tree API at the pinned
/// commits (the LFS oid for the weights and tokenizer) and the downloaded
/// files (2026-09-27).
pub static CHECKPOINTS: &[Checkpoint] = &[
    pinned("qwen3-1.7b-mlx", "mlx-community/Qwen3-1.7B-4bit", "3b1b1768f8f8cf8351c712464f906e86c2b8269e", &QWEN3_1_7B_MLX_FILES),
    pinned("qwen3-4b-mlx", "Qwen/Qwen3-4B-MLX-4bit", "52a5ab34fa604bc8af6d3ce0cac0cab10b7eb495", &QWEN3_4B_MLX_FILES),
    pinned("qwen3-8b-mlx", "Qwen/Qwen3-8B-MLX-4bit", "383413e909f3bc5303ce195ebbdf0339c5a1a2a3", &QWEN3_8B_MLX_FILES),
    pinned("qwen3-14b-mlx", "Qwen/Qwen3-14B-MLX-4bit", "ba63a5141812f9870287df53341123a71ba41433", &QWEN3_14B_MLX_FILES),
    pinned("qwen3-32b-mlx", "Qwen/Qwen3-32B-MLX-4bit", "ceb4c4aad0164033af0c8dd37a0641693d0d6321", &QWEN3_32B_MLX_FILES),
    pinned("qwen3-0.6b-mlx", "mlx-community/Qwen3-0.6B-8bit", "11de96878523501bcaa86104e3c186de07ff9068", &QWEN3_0_6B_MLX_FILES),
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
            for f in c.files.iter() {
                assert_eq!(f.sha256.len(), 64, "{} {}", c.id, f.path);
                assert!(f.sha256.chars().all(|ch| ch.is_ascii_hexdigit() && !ch.is_ascii_uppercase()));
                assert!(f.bytes > 0);
                assert!(!f.path.starts_with('/') && !f.path.split('/').any(|p| p == ".." || p.is_empty()));
            }
            for needed in ["config.json", "tokenizer.json", "tokenizer_config.json"] {
                c.file(needed).unwrap();
            }
            // Weights: one file, or an index and the shards it names (checked against the file in test:llm).
            let shards = c.files.iter().filter(|f| crate::llm::custom::is_shard_name(&f.path)).count();
            assert!(c.file("model.safetensors").is_ok() != (shards > 0 && c.file("model.safetensors.index.json").is_ok()), "{}", c.id);
        }
    }

    #[test]
    fn ids_do_not_collide_with_laya() {
        for c in CHECKPOINTS {
            assert!(crate::laya::catalog::checkpoint(&c.id).is_err(), "{} is also a Laya id", c.id);
        }
        assert!(checkpoint("laya-en").is_err());
        assert!(checkpoint("../qwen3-1.7b-mlx").is_err());
    }
}
