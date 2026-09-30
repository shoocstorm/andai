//! What the native engine can run, read from a checkpoint's config.json and
//! tokenizer_config.json. Platform-independent, so a model added from Hugging
//! Face is checked before its weights download (llm/custom.rs), and the same
//! check runs again when it loads (model.rs).

use serde_json::Value;

pub type R<T> = Result<T, String>;

pub fn e<E: std::fmt::Display>(err: E) -> String {
    err.to_string()
}

/// Which network the checkpoint is: Qwen3, or Qwen3.5's hybrid of linear
/// attention (Gated DeltaNet) and gated full attention.
#[derive(Debug, Clone, PartialEq)]
pub enum Arch {
    Qwen3,
    Qwen35(Hybrid),
}

/// Qwen3.5's extra shape (mlx-lm `models/qwen3_5.py`): every
/// `full_every`-th layer is full attention, the others Gated DeltaNet.
#[derive(Debug, Clone, PartialEq)]
pub struct Hybrid {
    pub full_every: usize,
    pub k_heads: i32,
    pub v_heads: i32,
    pub k_dim: i32,
    pub v_dim: i32,
    pub conv_kernel: i32,
}

impl Hybrid {
    pub fn is_linear(&self, layer: usize) -> bool {
        !(layer + 1).is_multiple_of(self.full_every)
    }
}

/// How one weight is quantized (MLX affine: packed values, a scale and a bias per group).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Quant {
    pub group_size: i32,
    pub bits: i32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub arch: Arch,
    pub vocab: i32,
    pub hidden: i32,
    pub intermediate: i32,
    pub layers: usize,
    pub heads: i32,
    pub kv_heads: i32,
    pub head_dim: i32,
    /// Leading dimensions of each head that RoPE rotates (Qwen3.5: a quarter).
    pub rotary_dims: i32,
    pub eps: f32,
    pub theta: f32,
    pub group_size: i32,
    pub bits: i32,
    /// Per-weight quantization that differs from the default (mixed-precision
    /// checkpoints), by weight name without `.weight`.
    pub overrides: std::collections::HashMap<String, Quant>,
    /// The LM head is the embedding (Qwen3 up to 4B); else its own weight.
    pub tied: bool,
}

/// MLX's quantized kernels take 2, 3, 4, 5, 6 or 8 bits (MLX 0.32; `mx.quantize`
/// refuses 1 and 7) in groups of 32, 64 or 128.
fn check_quant(bits: i64, group_size: i64) -> R<Quant> {
    if bits == 1 {
        return Err("it has 1-bit weights; MLX runs 2- to 8-bit weights (1-bit needs a patched MLX)".into());
    }
    if !matches!(bits, 2..=6 | 8) || !matches!(group_size, 32 | 64 | 128) {
        return Err(format!("{bits}-bit weights in groups of {group_size} aren't supported"));
    }
    Ok(Quant { group_size: group_size as i32, bits: bits as i32 })
}

impl Config {
    /// Reads config.json, refusing what this port doesn't implement.
    pub fn from_json(v: &Value) -> R<Self> {
        let model_type = v["model_type"].as_str();
        let qwen35 = match model_type {
            Some("qwen3") => false,
            Some("qwen3_5") => true,
            other => return Err(format!("it's a {} model; the native engine runs Qwen3 and Qwen3.5 (model_type qwen3, qwen3_5)", other.unwrap_or("unknown"))),
        };
        // Qwen3.5 nests the language model's settings (it's a vision-language
        // checkpoint; the engine runs its text part).
        let t = if qwen35 && v["text_config"].is_object() { &v["text_config"] } else { v };
        let int = |k: &str| t[k].as_i64().filter(|n| *n > 0 && *n <= i32::MAX as i64).map(|n| n as i32).ok_or_else(|| format!("model config: missing {k}"));
        if !t["rope_scaling"].is_null() && !qwen35 {
            return Err("it uses scaled RoPE, which the native engine doesn't support".into());
        }
        if t["attention_bias"].as_bool() == Some(true) {
            return Err("it uses attention bias, which the native engine doesn't support".into());
        }
        let q = if v["quantization"].is_object() { &v["quantization"] } else { &t["quantization"] };
        let (Some(group_size), Some(bits)) = (q["group_size"].as_i64(), q["bits"].as_i64()) else {
            return Err("its weights aren't quantized; the native engine runs MLX-quantized checkpoints (2- to 8-bit)".into());
        };
        if q["mode"].as_str().is_some_and(|m| m != "affine") {
            return Err(format!("it uses {} quantization; the native engine supports affine", q["mode"]));
        }
        let Quant { group_size, bits } = check_quant(bits, group_size)?;
        // Per-weight settings: `false` (left unquantized) only on norms and RoPE, which
        // this port never quantizes; `{bits, group_size}` for a mixed-precision weight.
        let mut overrides = std::collections::HashMap::new();
        for (k, val) in q.as_object().into_iter().flatten() {
            if matches!(k.as_str(), "group_size" | "bits" | "mode") || val.is_boolean() {
                continue;
            }
            if val["mode"].as_str().is_some_and(|m| m != "affine") {
                return Err(format!("it uses {} quantization; the native engine supports affine", val["mode"]));
            }
            let (Some(b), Some(g)) = (val["bits"].as_i64(), val["group_size"].as_i64()) else {
                return Err(format!("its quantization setting for {k} isn't one the native engine reads"));
            };
            overrides.insert(k.clone(), check_quant(b, g)?);
        }
        let (hidden, heads, kv_heads) = (int("hidden_size")?, int("num_attention_heads")?, int("num_key_value_heads")?);
        if heads % kv_heads != 0 {
            return Err("model config: heads must be a multiple of kv heads".into());
        }
        let layers = int("num_hidden_layers")? as usize;
        if layers > 256 {
            return Err("model config: too many layers".into());
        }
        let head_dim = t["head_dim"].as_i64().map(|n| n as i32).unwrap_or(hidden / heads);
        let (arch, rotary_dims, theta) = if qwen35 { Self::hybrid(t, layers, head_dim)? } else { (Arch::Qwen3, head_dim, t["rope_theta"].as_f64().unwrap_or(1_000_000.0) as f32) };
        Ok(Self {
            arch,
            vocab: int("vocab_size")?,
            hidden,
            intermediate: int("intermediate_size")?,
            layers,
            heads,
            kv_heads,
            head_dim,
            rotary_dims,
            eps: t["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
            theta,
            group_size,
            bits,
            overrides,
            tied: t["tie_word_embeddings"].as_bool().or(v["tie_word_embeddings"].as_bool()).unwrap_or(false),
        })
    }

    /// Qwen3.5's text settings, as mlx-lm reads them: dense layers only (no
    /// experts), a gated attention output, and plain RoPE on a part of each
    /// head. Its multimodal RoPE (`mrope_section`) gives every section the same
    /// position for text, so it reduces to that.
    fn hybrid(t: &Value, layers: usize, head_dim: i32) -> R<(Arch, i32, f32)> {
        let int = |k: &str| t[k].as_i64().filter(|n| *n > 0 && *n <= 4096).map(|n| n as i32).ok_or_else(|| format!("model config: missing {k}"));
        if t["num_experts"].as_i64().is_some_and(|n| n > 0) {
            return Err("it's a mixture-of-experts model, which the native engine doesn't run".into());
        }
        if t["attn_output_gate"].as_bool() == Some(false) {
            return Err("its attention has no output gate, which Qwen3.5's engine expects".into());
        }
        let rope = &t["rope_parameters"];
        let kind = rope["rope_type"].as_str().or(rope["type"].as_str()).unwrap_or("default");
        if kind != "default" {
            return Err(format!("it uses {kind} RoPE scaling, which the native engine doesn't support"));
        }
        let partial = rope["partial_rotary_factor"].as_f64().or(t["partial_rotary_factor"].as_f64()).unwrap_or(0.25);
        let rotary_dims = (head_dim as f64 * partial) as i32;
        if rotary_dims <= 0 || rotary_dims > head_dim || rotary_dims % 2 != 0 {
            return Err("model config: bad partial_rotary_factor".into());
        }
        let theta = rope["rope_theta"].as_f64().or(t["rope_theta"].as_f64()).unwrap_or(100_000.0) as f32;
        let h = Hybrid {
            full_every: t["full_attention_interval"].as_u64().unwrap_or(4) as usize,
            k_heads: int("linear_num_key_heads")?,
            v_heads: int("linear_num_value_heads")?,
            k_dim: int("linear_key_head_dim")?,
            v_dim: int("linear_value_head_dim")?,
            conv_kernel: int("linear_conv_kernel_dim")?,
        };
        if h.full_every == 0 || h.v_heads % h.k_heads != 0 {
            return Err("model config: bad linear attention shape".into());
        }
        // The recurrence kernel splits a key head over one SIMD group of 32 threads,
        // and a value head over threadgroups of 4 (llm/delta.rs).
        if h.k_dim % 32 != 0 || h.v_dim % 4 != 0 || h.conv_kernel < 2 {
            return Err("model config: linear attention head sizes the native engine can't run".into());
        }
        // mlx-lm places layers by the interval; a config that lists them otherwise isn't Qwen3.5.
        if let Some(types) = t["layer_types"].as_array() {
            let expected = (0..layers).map(|i| if h.is_linear(i) { "linear_attention" } else { "full_attention" });
            if types.len() != layers || !types.iter().map(|x| x.as_str().unwrap_or("")).eq(expected) {
                return Err("its layer_types don't follow full_attention_interval".into());
            }
        }
        Ok((Arch::Qwen35(h), rotary_dims, theta))
    }

    /// The quantization of the weight named `name` (without `.weight`).
    pub fn quant(&self, name: &str) -> Quant {
        self.overrides.get(name).copied().unwrap_or(Quant { group_size: self.group_size, bits: self.bits })
    }

    /// Where the language model's weights live: Qwen3.5 checkpoints nest it.
    pub fn prefix(&self) -> &'static str {
        match self.arch {
            Arch::Qwen3 => "",
            Arch::Qwen35(_) => "language_model.",
        }
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
            ("model_type", json!("qwen3_5_moe"), "qwen3_5_moe model"),
            ("rope_scaling", json!({"type": "yarn"}), "RoPE"),
            ("attention_bias", json!(true), "bias"),
            ("quantization", Value::Null, "quantized"),
            ("quantization", json!({"group_size": 64, "bits": 4, "mode": "mxfp4"}), "quantization"),
            ("quantization", json!({"group_size": 64, "bits": 4, "model.layers.0.mlp.down_proj": {"bits": 16, "group_size": 64}}), "16-bit"),
            ("quantization", json!({"group_size": 64, "bits": 4, "model.layers.0.mlp.down_proj": {"mode": "mxfp4"}}), "quantization"),
            ("quantization", json!({"group_size": 64, "bits": 16}), "16-bit"),
            ("quantization", json!({"group_size": 128, "bits": 1}), "1-bit"),
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
    fn reads_mixed_precision() {
        let mut j = cfg_json();
        j["quantization"]["model.layers.0.mlp.down_proj"] = json!({"bits": 8, "group_size": 64});
        let c = Config::from_json(&j).unwrap();
        assert_eq!(c.quant("model.layers.0.mlp.down_proj"), Quant { group_size: 64, bits: 8 });
        assert_eq!(c.quant("model.layers.0.mlp.up_proj"), Quant { group_size: 64, bits: 4 });
    }

    /// mlx-community/Qwen3.5-0.8B-OptiQ-4bit's config.json, trimmed.
    fn qwen35_json() -> Value {
        let types: Vec<&str> = (0..24).map(|i| if (i + 1) % 4 == 0 { "full_attention" } else { "linear_attention" }).collect();
        json!({
            "model_type": "qwen3_5", "tie_word_embeddings": true,
            "quantization": {"group_size": 64, "bits": 4, "mode": "affine",
                "language_model.model.embed_tokens": {"bits": 8, "group_size": 64}},
            "text_config": {
                "model_type": "qwen3_5_text", "attention_bias": false, "attn_output_gate": true,
                "full_attention_interval": 4, "head_dim": 256, "hidden_size": 1024, "intermediate_size": 3584,
                "layer_types": types, "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128,
                "linear_num_key_heads": 16, "linear_num_value_heads": 16, "linear_value_head_dim": 128,
                "num_attention_heads": 8, "num_hidden_layers": 24, "num_key_value_heads": 2, "rms_norm_eps": 1e-6,
                "tie_word_embeddings": true, "vocab_size": 248320,
                "rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_theta": 10000000,
                    "partial_rotary_factor": 0.25, "type": "default"}
            }
        })
    }

    #[test]
    fn reads_the_qwen35_config() {
        let c = Config::from_json(&qwen35_json()).unwrap();
        let Arch::Qwen35(h) = &c.arch else { panic!("not hybrid") };
        assert_eq!((h.full_every, h.k_heads, h.v_heads, h.k_dim, h.v_dim, h.conv_kernel), (4, 16, 16, 128, 128, 4));
        assert!(h.is_linear(0) && h.is_linear(2) && !h.is_linear(3) && !h.is_linear(23));
        assert_eq!((c.hidden, c.layers, c.heads, c.kv_heads, c.head_dim, c.rotary_dims, c.vocab), (1024, 24, 8, 2, 256, 64, 248320));
        assert_eq!((c.theta, c.bits, c.group_size, c.tied), (10_000_000.0, 4, 64, true));
        assert_eq!(c.quant("language_model.model.embed_tokens").bits, 8);
        assert_eq!(c.prefix(), "language_model.");
        // Bonsai 27B (1-bit): quantization inside text_config as well, own LM head.
        let mut bonsai = qwen35_json();
        bonsai["quantization"] = json!({"group_size": 128, "bits": 1});
        assert!(Config::from_json(&bonsai).unwrap_err().contains("1-bit"));
    }

    #[test]
    fn refuses_qwen35_variants_the_port_does_not_implement() {
        for (k, v, why) in [
            ("num_experts", json!(256), "mixture-of-experts"),
            ("attn_output_gate", json!(false), "gate"),
            ("rope_parameters", json!({"rope_type": "yarn", "factor": 4.0}), "yarn"),
            ("full_attention_interval", json!(3), "layer_types"),
            ("linear_key_head_dim", json!(100), "head sizes"),
            ("linear_num_value_heads", json!(24), "linear attention shape"),
        ] {
            let mut j = qwen35_json();
            j["text_config"][k] = v;
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
