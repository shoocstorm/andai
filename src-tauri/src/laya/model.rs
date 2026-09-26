//! Laya's network on MLX (mlx-rs), inference only, one question at a time:
//! ModernBERT → + question-type embedding → 2 decision-head layers → scorer
//! at each option marker. Ported line by line from laya-mlx
//! `laya_mlx/model.py`; the action head isn't needed for a choice, so it's
//! loaded (strict names) but never run.
//!
//! With a single unpadded sequence every token is valid, so global attention
//! and the decision head need no mask, and local layers use a band mask.

use mlx_rs::{fast, nn, ops, Array, Dtype};
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

pub type R<T> = Result<T, String>;

fn e<E: std::fmt::Display>(err: E) -> String {
    err.to_string()
}

#[derive(Debug, Clone, PartialEq)]
pub struct EncoderConfig {
    pub vocab: i32,
    pub hidden: i32,
    pub intermediate: i32,
    pub heads: i32,
    /// Per layer: full (global) attention, else a sliding window.
    pub global: Vec<bool>,
    pub window: i32,
    pub global_theta: f32,
    pub local_theta: f32,
    pub eps: f32,
}

impl EncoderConfig {
    /// Mirrors laya-mlx `EncoderConfig.from_dict`, including its refusals.
    pub fn from_json(v: &Value) -> R<Self> {
        let int = |k: &str| v[k].as_i64().map(|n| n as i32).ok_or_else(|| format!("encoder config: missing {k}"));
        let num = |k: &str, d: f64| v[k].as_f64().unwrap_or(d);
        if v["model_type"].as_str().unwrap_or("modernbert") != "modernbert" {
            return Err("unsupported encoder: expected modernbert".into());
        }
        if v["hidden_activation"].as_str().unwrap_or("gelu") != "gelu" {
            return Err("unsupported encoder activation".into());
        }
        let (hidden, heads, layers) = (int("hidden_size")?, int("num_attention_heads")?, int("num_hidden_layers")? as usize);
        if heads <= 0 || hidden % heads != 0 || (hidden / heads) % 2 != 0 {
            return Err("ModernBERT needs an even, integral attention head size".into());
        }
        let every = num("global_attn_every_n_layers", 3.0) as usize;
        let global: Vec<bool> = match v["layer_types"].as_array() {
            Some(types) => types
                .iter()
                .map(|t| match t.as_str() {
                    Some("full_attention") => Ok(true),
                    Some("sliding_attention") => Ok(false),
                    _ => Err("invalid ModernBERT layer_types".to_string()),
                })
                .collect::<R<_>>()?,
            None => (0..layers).map(|i| i % every.max(1) == 0).collect(),
        };
        if global.len() != layers {
            return Err("invalid ModernBERT layer_types".into());
        }
        // rope_parameters, when present, wins over the flat thetas: mmBERT sets
        // 160000 for sliding layers too (the Phase 0 spike got this wrong at
        // first, and every probability with it).
        let rope = |kind: &str, flat: &str, d: f64| -> R<f32> {
            let p = &v["rope_parameters"][kind];
            if p["rope_type"].as_str().unwrap_or("default") != "default" {
                return Err("only default (unscaled) RoPE is supported".into());
            }
            Ok(p["rope_theta"].as_f64().unwrap_or(num(flat, d)) as f32)
        };
        Ok(Self {
            vocab: int("vocab_size")?,
            hidden,
            intermediate: int("intermediate_size")?,
            heads,
            global,
            window: num("local_attention", 128.0) as i32,
            global_theta: rope("full_attention", "global_rope_theta", 160_000.0)?,
            local_theta: rope("sliding_attention", "local_rope_theta", 10_000.0)?,
            eps: num("norm_eps", 1e-5) as f32,
        })
    }
}

struct Linear {
    /// Stored transposed, so a forward is `x @ wt (+ b)`.
    wt: Array,
    b: Option<Array>,
}

impl Linear {
    fn call(&self, x: &Array) -> R<Array> {
        let y = ops::matmul(x, &self.wt).map_err(e)?;
        match &self.b {
            Some(b) => ops::add(&y, b).map_err(e),
            None => Ok(y),
        }
    }
}

struct Norm {
    w: Array,
    b: Option<Array>,
    eps: f32,
}

impl Norm {
    fn call(&self, x: &Array) -> R<Array> {
        fast::layer_norm(x, Some(&self.w), self.b.as_ref(), self.eps).map_err(e)
    }
}

struct EncoderLayer {
    attn_norm: Option<Norm>,
    wqkv: Linear,
    wo: Linear,
    mlp_norm: Norm,
    wi: Linear,
    mlp_wo: Linear,
    global: bool,
}

struct HeadLayer {
    norm1: Norm,
    in_proj: Linear,
    out_proj: Linear,
    norm2: Norm,
    linear1: Linear,
    linear2: Linear,
}

pub struct Model {
    pub cfg: EncoderConfig,
    tok_embeddings: Array,
    emb_norm: Norm,
    layers: Vec<EncoderLayer>,
    final_norm: Norm,
    type_emb: Array,
    head: Vec<HeadLayer>,
    head_heads: i32,
    scorer_norm: Norm,
    scorer1: Linear,
    scorer2: Linear,
}

/// Takes named tensors out of the checkpoint, checking shapes; whatever is
/// left over at the end is an error (like laya-mlx's `strict=True`).
struct Weights(HashMap<String, Array>);

impl Weights {
    fn take(&mut self, name: &str, shape: &[i32]) -> R<Array> {
        let a = self.0.remove(name).ok_or_else(|| format!("checkpoint is missing {name}"))?;
        if a.shape() != shape {
            return Err(format!("{name} has shape {:?}, expected {shape:?}", a.shape()));
        }
        Ok(a)
    }
    fn linear(&mut self, p: &str, out: i32, inp: i32, bias: bool) -> R<Linear> {
        let w = self.take(&format!("{p}.weight"), &[out, inp])?;
        let b = if bias { Some(self.take(&format!("{p}.bias"), &[out])?) } else { None };
        Ok(Linear { wt: w.t(), b })
    }
    fn norm(&mut self, p: &str, dims: i32, bias: bool, eps: f32) -> R<Norm> {
        let w = self.take(&format!("{p}.weight"), &[dims])?;
        let b = if bias { Some(self.take(&format!("{p}.bias"), &[dims])?) } else { None };
        Ok(Norm { w, b, eps })
    }
}

impl Model {
    pub fn load(dir: &Path, cfg: EncoderConfig, head_layers: usize, act_outputs: i32) -> R<Self> {
        let raw = Array::load_safetensors(dir.join("model.safetensors")).map_err(e)?;
        let mut w = Weights(raw);
        let (d, eps) = (cfg.hidden, cfg.eps);
        let tok_embeddings = w.take("encoder.embeddings.tok_embeddings.weight", &[cfg.vocab, d])?;
        let emb_norm = w.norm("encoder.embeddings.norm", d, false, eps)?;
        let mut layers = Vec::with_capacity(cfg.global.len());
        for (i, &global) in cfg.global.iter().enumerate() {
            let p = format!("encoder.layers.{i}");
            layers.push(EncoderLayer {
                // ModernBERT's first layer has no attention norm (the embedding norm precedes it)
                attn_norm: if i == 0 { None } else { Some(w.norm(&format!("{p}.attn_norm"), d, false, eps)?) },
                wqkv: w.linear(&format!("{p}.attn.Wqkv"), 3 * d, d, false)?,
                wo: w.linear(&format!("{p}.attn.Wo"), d, d, false)?,
                mlp_norm: w.norm(&format!("{p}.mlp_norm"), d, false, eps)?,
                wi: w.linear(&format!("{p}.mlp.Wi"), 2 * cfg.intermediate, d, false)?,
                mlp_wo: w.linear(&format!("{p}.mlp.Wo"), d, cfg.intermediate, false)?,
                global,
            });
        }
        let final_norm = w.norm("encoder.final_norm", d, false, eps)?;
        let type_emb = w.take("type_emb.weight", &[3, d])?;
        let mut head = Vec::with_capacity(head_layers);
        for i in 0..head_layers {
            let p = format!("head.layers.{i}");
            head.push(HeadLayer {
                norm1: w.norm(&format!("{p}.norm1"), d, true, 1e-5)?,
                in_proj: w.linear(&format!("{p}.self_attn.in_proj"), 3 * d, d, true)?,
                out_proj: w.linear(&format!("{p}.self_attn.out_proj"), d, d, true)?,
                norm2: w.norm(&format!("{p}.norm2"), d, true, 1e-5)?,
                linear1: w.linear(&format!("{p}.linear1"), 4 * d, d, true)?,
                linear2: w.linear(&format!("{p}.linear2"), d, 4 * d, true)?,
            });
        }
        let scorer_norm = w.norm("scorer.layers.0", d, true, 1e-5)?;
        let scorer1 = w.linear("scorer.layers.1", d, d, true)?;
        let scorer2 = w.linear("scorer.layers.3", 1, d, true)?;
        // Loaded for a strict name check, not used for a choice.
        w.linear("act_head.layers.0", 256, d + 4, true)?;
        w.linear("act_head.layers.2", act_outputs, 256, true)?;
        w.take("temperature", &[3])?;
        if let Some(extra) = w.0.keys().next() {
            return Err(format!("unexpected tensor in checkpoint: {extra}"));
        }
        let model = Self {
            head_heads: (d / 64).max(1),
            cfg,
            tok_embeddings,
            emb_norm,
            layers,
            final_norm,
            type_emb,
            head,
            scorer_norm,
            scorer1,
            scorer2,
        };
        model.materialize()?;
        Ok(model)
    }

    /// Reads every weight into memory now, not on the first decision.
    fn materialize(&self) -> R<()> {
        fn linear<'a>(all: &mut Vec<&'a Array>, l: &'a Linear) {
            all.push(&l.wt);
            all.extend(l.b.as_ref());
        }
        fn norm<'a>(all: &mut Vec<&'a Array>, n: &'a Norm) {
            all.push(&n.w);
            all.extend(n.b.as_ref());
        }
        let mut all = vec![&self.tok_embeddings, &self.type_emb];
        for n in [&self.emb_norm, &self.final_norm, &self.scorer_norm] {
            norm(&mut all, n);
        }
        for l in &self.layers {
            for x in [&l.wqkv, &l.wo, &l.wi, &l.mlp_wo] {
                linear(&mut all, x);
            }
            norm(&mut all, &l.mlp_norm);
            if let Some(n) = &l.attn_norm {
                norm(&mut all, n);
            }
        }
        for h in &self.head {
            for x in [&h.in_proj, &h.out_proj, &h.linear1, &h.linear2] {
                linear(&mut all, x);
            }
            norm(&mut all, &h.norm1);
            norm(&mut all, &h.norm2);
        }
        linear(&mut all, &self.scorer1);
        linear(&mut all, &self.scorer2);
        mlx_rs::transforms::eval(all).map_err(e)
    }

    /// `qkv`: [B, L, 3·D] → attention output [B, L, D].
    fn attention(&self, qkv: &Array, heads: i32, rope: Option<f32>, mask: Option<&Array>) -> R<Array> {
        let d = self.cfg.hidden;
        let hd = d / heads;
        let (b, l) = (qkv.shape()[0], qkv.shape()[1]);
        let parts = qkv
            .reshape(&[b, l, 3, heads, hd])
            .and_then(|a| a.transpose_axes(&[2, 0, 3, 1, 4]))
            .and_then(|a| a.split_equal(3, 0))
            .map_err(e)?;
        let pick = |i: usize| parts[i].squeeze_axes(&[0]).map_err(e);
        let (mut q, mut k, v) = (pick(0)?, pick(1)?, pick(2)?);
        if let Some(base) = rope {
            q = fast::rope(&q, hd, false, base, 1.0, 0, None).map_err(e)?;
            k = fast::rope(&k, hd, false, base, 1.0, 0, None).map_err(e)?;
        }
        let scale = (hd as f32).powf(-0.5);
        let out = match mask {
            Some(m) => fast::scaled_dot_product_attention(&q, &k, &v, scale, m, None),
            None => fast::scaled_dot_product_attention(&q, &k, &v, scale, None, None),
        }
        .map_err(e)?;
        out.transpose_axes(&[0, 2, 1, 3]).and_then(|a| a.reshape(&[b, l, d])).map_err(e)
    }

    /// Local layers attend within `window / 2` tokens either side (inclusive).
    fn band_mask(&self, l: i32) -> R<Array> {
        let pos = ops::arange::<_, i32>(0, l, 1).map_err(e)?;
        let dist = ops::abs(ops::subtract(&pos.reshape(&[l, 1]).map_err(e)?, &pos.reshape(&[1, l]).map_err(e)?).map_err(e)?).map_err(e)?;
        dist.le(Array::from_int(self.cfg.window / 2)).and_then(|m| m.reshape(&[1, 1, l, l])).map_err(e)
    }

    /// One logit per marker for each row (its token ids, marker positions and
    /// question type), all rows in one forward pass. Rows are padded to the
    /// longest, as laya-mlx `collate_items` does: padding is never a key, and
    /// padded queries see the valid keys (so no softmax row is all masked);
    /// they're never read. A single row needs no masks beyond the band.
    pub fn logits(&self, rows: &[(&[u32], &[usize], i32)], pad: u32) -> R<Vec<Vec<f32>>> {
        let b = rows.len() as i32;
        let l = rows.iter().map(|r| r.0.len()).max().unwrap_or(0) as i32;
        let k = rows.iter().map(|r| r.1.len()).max().unwrap_or(0).max(1) as i32;
        let mut ids = vec![pad as i32; (b * l) as usize];
        let mut valid = vec![false; (b * l) as usize];
        let mut at = vec![0i32; (b * k) as usize];
        for (i, (row, markers, _)) in rows.iter().enumerate() {
            for (j, &t) in row.iter().enumerate() {
                ids[i * l as usize + j] = t as i32;
                valid[i * l as usize + j] = true;
            }
            for (j, &m) in markers.iter().enumerate() {
                at[i * k as usize + j] = i as i32 * l + m as i32;
            }
        }
        let band = self.band_mask(l)?;
        let padded = valid.iter().any(|v| !v);
        let (full, local) = if padded {
            let valid = Array::from_slice(&valid, &[b, l]);
            let full = valid.reshape(&[b, 1, 1, l]).map_err(e)?;
            let pad_query = valid.logical_not().and_then(|v| v.reshape(&[b, 1, l, 1])).map_err(e)?;
            let local = ops::logical_or(&band, &pad_query).and_then(|m| ops::logical_and(&m, &full)).map_err(e)?;
            (Some(full), local)
        } else {
            (None, band)
        };

        let mut x = self.tok_embeddings.take_axis(Array::from_slice(&ids, &[b, l]), 0).map_err(e)?;
        x = self.emb_norm.call(&x)?;
        for layer in &self.layers {
            let h = match &layer.attn_norm {
                Some(n) => n.call(&x)?,
                None => x.clone(),
            };
            let theta = if layer.global { self.cfg.global_theta } else { self.cfg.local_theta };
            let mask = if layer.global { full.as_ref() } else { Some(&local) };
            let a = self.attention(&layer.wqkv.call(&h)?, self.cfg.heads, Some(theta), mask)?;
            x = ops::add(&x, &layer.wo.call(&a)?).map_err(e)?;
            let gate = layer.wi.call(&layer.mlp_norm.call(&x)?)?.split_equal(2, -1).map_err(e)?;
            let m = ops::multiply(&nn::gelu(&gate[0]).map_err(e)?, &gate[1]).map_err(e)?;
            x = ops::add(&x, &layer.mlp_wo.call(&m)?).map_err(e)?;
        }
        x = self.final_norm.call(&x)?;
        let qtypes: Vec<i32> = rows.iter().map(|r| r.2).collect();
        let te = self.type_emb.take_axis(Array::from_slice(&qtypes, &[b]), 0).and_then(|t| t.reshape(&[b, 1, self.cfg.hidden])).map_err(e)?;
        x = ops::add(&x, &te).map_err(e)?;
        for h in &self.head {
            let a = self.attention(&h.in_proj.call(&h.norm1.call(&x)?)?, self.head_heads, None, full.as_ref())?;
            x = ops::add(&x, &h.out_proj.call(&a)?).map_err(e)?;
            // PyTorch's TransformerEncoderLayer defaults to ReLU here, unlike the GELU elsewhere.
            let f = nn::relu(&h.linear1.call(&h.norm2.call(&x)?)?).map_err(e)?;
            x = ops::add(&x, &h.linear2.call(&f)?).map_err(e)?;
        }
        // Gather every row's markers from the flattened [B·L, D] states.
        let flat = x.reshape(&[b * l, self.cfg.hidden]).map_err(e)?;
        let picked = flat.take_axis(Array::from_slice(&at, &[b * k]), 0).map_err(e)?;
        let s = nn::gelu(&self.scorer1.call(&self.scorer_norm.call(&picked)?)?).map_err(e)?;
        let out = self.scorer2.call(&s)?.reshape(&[b * k]).and_then(|a| a.as_dtype(Dtype::Float32)).map_err(e)?;
        out.eval().map_err(e)?;
        let all = out.as_slice::<f32>();
        Ok(rows.iter().enumerate().map(|(i, r)| all[i * k as usize..i * k as usize + r.1.len()].to_vec()).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn reads_rope_parameters_before_the_flat_thetas() {
        let c = EncoderConfig::from_json(&json!({
            "vocab_size": 10, "hidden_size": 8, "intermediate_size": 4, "num_attention_heads": 2, "num_hidden_layers": 3,
            "local_rope_theta": 10000.0,
            "rope_parameters": {"full_attention": {"rope_theta": 160000}, "sliding_attention": {"rope_theta": 160000}}
        }))
        .unwrap();
        assert_eq!(c.local_theta, 160_000.0);
        assert_eq!(c.global, vec![true, false, false]);
        let flat = EncoderConfig::from_json(&json!({
            "vocab_size": 10, "hidden_size": 8, "intermediate_size": 4, "num_attention_heads": 2, "num_hidden_layers": 1
        }))
        .unwrap();
        assert_eq!((flat.global_theta, flat.local_theta, flat.window), (160_000.0, 10_000.0, 128));
    }

    #[test]
    fn refuses_what_the_port_does_not_implement() {
        let base = json!({"vocab_size": 10, "hidden_size": 8, "intermediate_size": 4, "num_attention_heads": 2, "num_hidden_layers": 1});
        let with = |k: &str, v: Value| {
            let mut c = base.clone();
            c[k] = v;
            EncoderConfig::from_json(&c)
        };
        assert!(with("model_type", json!("bert")).is_err());
        assert!(with("hidden_activation", json!("silu")).is_err());
        assert!(with("num_attention_heads", json!(3)).is_err());
        assert!(with("layer_types", json!(["full_attention", "sliding_attention"])).is_err());
        assert!(with("rope_parameters", json!({"full_attention": {"rope_type": "yarn"}})).is_err());
    }
}
