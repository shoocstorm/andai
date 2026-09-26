//! A loaded Laya checkpoint: config, tokenizer, calibration and network, and
//! the one thing Andai asks of it, a choice between options.

use super::model::{EncoderConfig, Model, R};
use super::prompt::{self, Encode, Special};
use serde::Serialize;
use serde_json::Value;
use std::path::Path;
use std::time::Instant;

struct Tok(tokenizers::Tokenizer);

impl Encode for Tok {
    fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        Ok(self.0.encode(text, false).map_err(|e| e.to_string())?.get_ids().to_vec())
    }
}

pub struct Engine {
    model: Model,
    tok: Tok,
    special: Special,
    max_len: usize,
    head_max_len: usize,
    /// Choice temperature, and per option-count bucket overrides (clamped).
    temperature: f64,
    by_options: Vec<(String, f64)>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Choice {
    /// Calibrated probability per option, in the order given.
    pub probabilities: Vec<f64>,
    pub input_tokens: usize,
    /// An option, the question or the state was cut to fit the model's input.
    pub truncated: bool,
    pub ms: f64,
}

fn read_json(path: &Path) -> R<Value> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
}

/// A special token named in tokenizer_config.json (a string or `{content}`).
fn special_token(tok: &tokenizers::Tokenizer, cfg: &Value, name: &str) -> R<(String, u32)> {
    let v = &cfg[name];
    let text = v.as_str().or_else(|| v["content"].as_str()).ok_or_else(|| format!("tokenizer config has no {name}"))?;
    let id = tok.token_to_id(text).ok_or_else(|| format!("tokenizer has no id for {name} {text:?}"))?;
    Ok((text.to_string(), id))
}

impl Engine {
    pub fn load(dir: &Path) -> R<Self> {
        let agent = read_json(&dir.join("rl_agent_config.json"))?;
        let enc = EncoderConfig::from_json(&read_json(&dir.join("encoder/config.json"))?)?;
        let max_len = agent["max_len"].as_u64().unwrap_or(512) as usize;
        let head_max_len = agent["head_max_len"].as_u64().unwrap_or(192) as usize;
        if !(4 < head_max_len && head_max_len < max_len) {
            return Err("Laya config: expected 4 < head_max_len < max_len".into());
        }
        let temps = agent["temperature"].as_array().cloned().unwrap_or_default();
        let temperature = prompt::clamp_temperature(temps.first().and_then(Value::as_f64).unwrap_or(1.0));
        let by_options = agent["temperature_by_options"]
            .as_object()
            .map(|m| m.iter().filter_map(|(k, v)| v.as_f64().map(|t| (k.clone(), prompt::clamp_temperature(t)))).collect())
            .unwrap_or_default();
        let head_layers = agent["head_layers"].as_u64().ok_or("Laya config: missing head_layers")? as usize;
        let act_outputs = agent["act_costs"].as_object().map_or(0, |m| m.len()) as i32 + 1;

        let mut tk = tokenizers::Tokenizer::from_file(dir.join("tokenizer/tokenizer.json")).map_err(|e| e.to_string())?;
        tk.with_padding(None);
        tk.with_truncation(None).map_err(|e| e.to_string())?;
        let tcfg = read_json(&dir.join("tokenizer/tokenizer_config.json"))?;
        let (_, cls) = special_token(&tk, &tcfg, "cls_token")?;
        let (_, sep) = special_token(&tk, &tcfg, "sep_token")?;
        let (mask_text, mask) = special_token(&tk, &tcfg, "mask_token")?;

        let model = Model::load(dir, enc, head_layers, act_outputs)?;
        Ok(Self { model, tok: Tok(tk), special: Special { cls, sep, mask, mask_text }, max_len, head_max_len, temperature, by_options })
    }

    pub fn choose(&self, state: &str, question: &str, options: &[(String, String)]) -> R<Choice> {
        let started = Instant::now();
        let rendered: Vec<String> = options.iter().map(|(id, text)| prompt::render_option(id, text)).collect();
        let seq = prompt::build(&self.tok, &self.special, state, question, &rendered, self.max_len, self.head_max_len)?;
        let logits = self.model.logits(&seq.ids, &seq.markers, 0)?;
        if logits.iter().any(|l| !l.is_finite()) {
            return Err("the decision model returned non-finite scores".into());
        }
        let bucket = prompt::choice_bucket(options.len());
        let t = self.by_options.iter().find(|(k, _)| k == bucket).map_or(self.temperature, |(_, t)| *t);
        Ok(Choice {
            probabilities: prompt::probabilities(&logits, t),
            input_tokens: seq.ids.len(),
            truncated: seq.truncated,
            ms: started.elapsed().as_secs_f64() * 1e3,
        })
    }
}

/// Parity against laya-mlx, on real checkpoints. Needs the Hugging Face
/// cache (`hf download aac6fef/laya-multilingual-mlx` / `aac6fef/laya-mlx`
/// at the catalog commits) or `LAYA_DIR_<ID>`; run with
/// `cargo test --release -- --ignored laya`.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::laya::catalog;
    use std::path::PathBuf;

    fn checkpoint_dir(id: &str) -> PathBuf {
        let c = catalog::checkpoint(id).unwrap();
        if let Some(d) = std::env::var_os(format!("LAYA_DIR_{}", id.replace('-', "_").to_uppercase())) {
            return d.into();
        }
        let home = std::env::var_os("HOME").expect("HOME");
        PathBuf::from(home)
            .join(".cache/huggingface/hub")
            .join(format!("models--{}", c.repo.replace('/', "--")))
            .join("snapshots")
            .join(c.commit)
    }

    fn fixture() -> Value {
        serde_json::from_str(include_str!("../../tests/fixtures/laya/fixture.json")).unwrap()
    }

    fn run(id: &str) {
        let golden: Value = serde_json::from_str(&std::fs::read_to_string(format!("{}/tests/fixtures/laya/golden-{id}.json", env!("CARGO_MANIFEST_DIR"))).unwrap()).unwrap();
        let fx = fixture();
        let engine = Engine::load(&checkpoint_dir(id)).unwrap();
        let options: Vec<(String, String)> = fx["options"].as_array().unwrap().iter().map(|o| (o[0].as_str().unwrap().into(), o[1].as_str().unwrap().into())).collect();
        let (state, question) = (fx["state"].as_str().unwrap(), fx["question"].as_str().unwrap());

        let rendered: Vec<String> = options.iter().map(|(i, t)| prompt::render_option(i, t)).collect();
        let seq = prompt::build(&engine.tok, &engine.special, state, question, &rendered, engine.max_len, engine.head_max_len).unwrap();
        let want_ids: Vec<u32> = serde_json::from_value(golden["ids"].clone()).unwrap();
        let want_markers: Vec<usize> = serde_json::from_value(golden["markers"].clone()).unwrap();
        assert_eq!(seq.ids, want_ids, "token ids match laya-mlx");
        assert_eq!(seq.markers, want_markers);

        let got = engine.choose(state, question, &options).unwrap();
        let want: Vec<f64> = serde_json::from_value(golden["probabilities_fp32"].clone()).unwrap();
        let worst = got.probabilities.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        let argmax = |v: &[f64]| v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
        assert_eq!(argmax(&got.probabilities), argmax(&want), "same choice as laya-mlx");
        // FP16 vs laya-mlx FP32: measured ≤ 0.0015 on these fixtures; laya-mlx's own FP16 run differs by 0.005.
        assert!(worst < 0.006, "max |Δp| {worst} vs laya-mlx FP32");

        // Latency: this port measured 9 ms (multilingual) and 20 ms (English) P50 on an M5 Max.
        let mut t: Vec<f64> = (0..25).map(|_| engine.choose(state, question, &options).unwrap().ms).collect();
        t.sort_by(f64::total_cmp);
        println!("{id}: P50 {:.1} ms, {} tokens, max |Δp| {worst:.4}", t[12], got.input_tokens);
        assert!(t[12] < 100.0, "a decision must take < 100 ms (P50 {:.1} ms)", t[12]);
    }

    #[test]
    #[ignore = "needs the multilingual checkpoint"]
    fn laya_multilingual_matches_laya_mlx() {
        run("laya-multilingual");
    }

    #[test]
    #[ignore = "needs the English checkpoint"]
    fn laya_en_matches_laya_mlx() {
        run("laya-en");
    }
}
