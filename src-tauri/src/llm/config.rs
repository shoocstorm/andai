//! What the native engine can run, read from a checkpoint's config.json and
//! tokenizer_config.json. Platform-independent, so a model added from Hugging
//! Face is checked before its weights download (llm/custom.rs), and the same
//! check runs again when it loads (model.rs).

use serde_json::Value;

pub type R<T> = Result<T, String>;

pub fn e<E: std::fmt::Display>(err: E) -> String {
    err.to_string()
}

#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub vocab: i32,
    pub hidden: i32,
    pub intermediate: i32,
    pub layers: usize,
    pub heads: i32,
    pub kv_heads: i32,
    pub head_dim: i32,
    pub eps: f32,
    pub theta: f32,
    pub group_size: i32,
    pub bits: i32,
    /// The LM head is the embedding (Qwen3 up to 4B); else its own weight.
    pub tied: bool,
}

impl Config {
    /// Reads config.json, refusing what this port doesn't implement.
    pub fn from_json(v: &Value) -> R<Self> {
        let int = |k: &str| v[k].as_i64().filter(|n| *n > 0 && *n <= i32::MAX as i64).map(|n| n as i32).ok_or_else(|| format!("model config: missing {k}"));
        if v["model_type"].as_str() != Some("qwen3") {
            return Err(format!("it's a {} model; the native engine runs Qwen3 (model_type qwen3)", v["model_type"].as_str().unwrap_or("unknown")));
        }
        if !v["rope_scaling"].is_null() {
            return Err("it uses scaled RoPE, which the native engine doesn't support".into());
        }
        if v["attention_bias"].as_bool() == Some(true) {
            return Err("it uses attention bias, which the native engine doesn't support".into());
        }
        let q = &v["quantization"];
        let (Some(group_size), Some(bits)) = (q["group_size"].as_i64(), q["bits"].as_i64()) else {
            return Err("its weights aren't quantized; the native engine runs MLX-quantized checkpoints (4-, 5-, 6- or 8-bit)".into());
        };
        if q["mode"].as_str().is_some_and(|m| m != "affine") {
            return Err(format!("it uses {} quantization; the native engine supports affine", q["mode"]));
        }
        // Per-layer settings: `false` (left unquantized) only on norms and RoPE, which this port
        // never quantizes; a layer with its own bits or group size isn't supported.
        if q.as_object().is_some_and(|o| o.iter().any(|(k, val)| !matches!(k.as_str(), "group_size" | "bits" | "mode") && !val.is_boolean())) {
            return Err("it mixes quantization settings per layer, which the native engine doesn't support".into());
        }
        if !matches!(bits, 2..=8) || !matches!(group_size, 32 | 64 | 128) {
            return Err(format!("{bits}-bit weights in groups of {group_size} aren't supported"));
        }
        let (hidden, heads, kv_heads) = (int("hidden_size")?, int("num_attention_heads")?, int("num_key_value_heads")?);
        if heads % kv_heads != 0 {
            return Err("model config: heads must be a multiple of kv heads".into());
        }
        let layers = int("num_hidden_layers")? as usize;
        if layers > 256 {
            return Err("model config: too many layers".into());
        }
        Ok(Self {
            vocab: int("vocab_size")?,
            hidden,
            intermediate: int("intermediate_size")?,
            layers,
            heads,
            kv_heads,
            head_dim: v["head_dim"].as_i64().map(|n| n as i32).unwrap_or(hidden / heads),
            eps: v["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
            theta: v["rope_theta"].as_f64().unwrap_or(1_000_000.0) as f32,
            group_size: group_size as i32,
            bits: bits as i32,
            tied: v["tie_word_embeddings"].as_bool().unwrap_or(false),
        })
    }
}

/// The chat template's text: tokenizer_config.json's `chat_template` (a
/// string, or named templates with a `default`), else chat_template.jinja.
pub fn chat_template(tokenizer_config: &Value, jinja: Option<&str>) -> Option<String> {
    let t = &tokenizer_config["chat_template"];
    t.as_str()
        .map(str::to_string)
        .or_else(|| t.as_array()?.iter().find(|x| x["name"] == "default")?["template"].as_str().map(str::to_string))
        .or_else(|| jinja.map(str::to_string))
}

/// Whether the engine can prompt a model with this template: it renders
/// Qwen's ChatML (template.rs). Returns whether the template has Qwen3's
/// thinking switch (`enable_thinking`); without it, no think block is added.
pub fn check_template(template: Option<&str>) -> R<bool> {
    let t = template.ok_or("it has no chat template")?;
    if !t.contains("<|im_start|>") {
        return Err("its chat template isn't ChatML, the format the native engine writes".into());
    }
    Ok(t.contains("enable_thinking"))
}

/// The end-of-turn token named in tokenizer_config.json.
pub fn eos_token(tokenizer_config: &Value) -> Option<&str> {
    let v = &tokenizer_config["eos_token"];
    v.as_str().or_else(|| v["content"].as_str())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn cfg_json() -> Value {
        json!({
            "model_type": "qwen3", "vocab_size": 151936, "hidden_size": 2048, "intermediate_size": 6144,
            "num_hidden_layers": 28, "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128,
            "rms_norm_eps": 1e-6, "rope_theta": 1000000, "rope_scaling": null, "tie_word_embeddings": true,
            "quantization": {"group_size": 64, "bits": 4}
        })
    }

    #[test]
    fn reads_the_qwen3_config() {
        let c = Config::from_json(&cfg_json()).unwrap();
        assert_eq!((c.hidden, c.layers, c.heads, c.kv_heads, c.head_dim, c.bits, c.group_size, c.tied), (2048, 28, 16, 8, 128, 4, 64, true));
        assert_eq!(c.theta, 1_000_000.0);
        let mut untied = cfg_json();
        untied["tie_word_embeddings"] = json!(false);
        assert!(!Config::from_json(&untied).unwrap().tied, "a separate LM head (Qwen3 8B) is supported");
        let mut norms_off = cfg_json();
        norms_off["quantization"]["model.layers.0.input_layernorm"] = json!(false);
        assert!(Config::from_json(&norms_off).is_ok(), "unquantized norms are what this port does anyway");
    }

    #[test]
    fn refuses_what_the_port_does_not_implement_and_says_why() {
        for (k, v, why) in [
            ("model_type", json!("llama"), "llama model"),
            ("rope_scaling", json!({"type": "yarn"}), "RoPE"),
            ("attention_bias", json!(true), "bias"),
            ("quantization", Value::Null, "quantized"),
            ("quantization", json!({"group_size": 64, "bits": 4, "mode": "mxfp4"}), "quantization"),
            ("quantization", json!({"group_size": 64, "bits": 4, "model.layers.0.mlp.down_proj": {"bits": 8, "group_size": 64}}), "per layer"),
            ("quantization", json!({"group_size": 64, "bits": 16}), "16-bit"),
            ("num_key_value_heads", json!(5), "kv heads"),
            ("num_hidden_layers", json!(-1), "layers"),
        ] {
            let mut j = cfg_json();
            j[k] = v;
            let err = Config::from_json(&j).unwrap_err();
            assert!(err.contains(why), "{k}: {err}");
        }
    }

    #[test]
    fn reads_and_checks_the_chat_template() {
        let chatml = "{{ '<|im_start|>' }}{% if enable_thinking is false %}...";
        assert_eq!(chat_template(&json!({"chat_template": chatml}), None).as_deref(), Some(chatml));
        assert_eq!(chat_template(&json!({"chat_template": [{"name": "tool_use", "template": "x"}, {"name": "default", "template": "d"}]}), None).as_deref(), Some("d"));
        assert_eq!(chat_template(&json!({}), Some("j")).as_deref(), Some("j"));
        assert!(check_template(Some(chatml)).unwrap());
        assert!(!check_template(Some("<|im_start|>{{ m }}")).unwrap(), "ChatML without the thinking switch");
        assert!(check_template(Some("[INST]")).unwrap_err().contains("ChatML"));
        assert!(check_template(None).is_err());
        assert_eq!(eos_token(&json!({"eos_token": {"content": "<|im_end|>"}})), Some("<|im_end|>"));
    }
}
