//! Qwen3 on MLX (mlx-rs), inference only: ported from mlx-lm
//! `models/qwen3.py` for the quantized mlx-community checkpoints. Embedding →
//! per layer (RMSNorm → attention with q/k norms, RoPE and GQA → RMSNorm →
//! SwiGLU MLP) → RMSNorm → the tied embedding as the LM head.

use mlx_rs::fast::{self, ScaledDotProductAttentionMask};
use mlx_rs::ops::indexing::{IndexOp, TryIndexMutOp};
use mlx_rs::{ops, Array};
use std::collections::HashMap;
use std::path::Path;

pub use super::config::{e, Config, R};

/// A quantized `[out, in]` weight: packed values, per-group scales and biases.
struct QLinear {
    w: Array,
    scales: Array,
    biases: Array,
    group_size: i32,
    bits: i32,
}

impl QLinear {
    fn call(&self, x: &Array) -> R<Array> {
        ops::quantized_matmul(x, &self.w, &self.scales, &self.biases, true, self.group_size, self.bits).map_err(e)
    }
}

struct Layer {
    input_norm: Array,
    q: QLinear,
    k: QLinear,
    v: QLinear,
    o: QLinear,
    q_norm: Array,
    k_norm: Array,
    post_norm: Array,
    gate: QLinear,
    up: QLinear,
    down: QLinear,
}

/// One layer's keys and values, grown in steps (mlx-lm `KVCache`). `offset`
/// is how many positions are valid; the arrays may be longer.
#[derive(Default)]
pub struct Kv {
    k: Option<Array>,
    v: Option<Array>,
    pub offset: i32,
}

const KV_STEP: i32 = 256;

impl Kv {
    fn update(&mut self, k: &Array, v: &Array) -> R<(Array, Array)> {
        let prev = self.offset;
        let n = k.shape()[2];
        let cap = self.k.as_ref().map_or(0, |a| a.shape()[2]);
        if prev + n > cap {
            let s = k.shape();
            let steps = (KV_STEP + n - 1) / KV_STEP;
            let grow = |dt| ops::zeros_dtype(&[s[0], s[1], steps * KV_STEP, s[3]], dt).map_err(e);
            let (nk, nv) = (grow(k.dtype())?, grow(v.dtype())?);
            match (self.k.take(), self.v.take()) {
                (Some(ok), Some(ov)) => {
                    // Positions past `prev` are stale (a trimmed prefix): drop them before growing.
                    let (ok, ov) = (ok.index((.., .., ..prev, ..)), ov.index((.., .., ..prev, ..)));
                    self.k = Some(ops::concatenate(&[ok, nk], 2).map_err(e)?);
                    self.v = Some(ops::concatenate(&[ov, nv], 2).map_err(e)?);
                }
                _ => {
                    self.k = Some(nk);
                    self.v = Some(nv);
                }
            }
        }
        self.offset = prev + n;
        let (ck, cv) = (self.k.as_mut().unwrap(), self.v.as_mut().unwrap());
        ck.try_index_mut((.., .., prev..self.offset, ..), k).map_err(e)?;
        cv.try_index_mut((.., .., prev..self.offset, ..), v).map_err(e)?;
        Ok((ck.index((.., .., ..self.offset, ..)), cv.index((.., .., ..self.offset, ..))))
    }

    /// Forgets every position from `n` on (prompt-prefix reuse).
    pub fn trim_to(&mut self, n: i32) {
        self.offset = self.offset.min(n);
    }

    pub fn arrays(&self) -> impl Iterator<Item = &Array> {
        self.k.iter().chain(self.v.iter())
    }
}

/// `silu(gate) * up` as one fused kernel, like mlx-lm's compiled `swiglu`
/// (two kernels per layer otherwise).
fn swiglu(gate: &Array, up: &Array) -> R<Array> {
    let f = |(g, u): (&Array, &Array)| ops::multiply(ops::multiply(g, ops::sigmoid(g)?)?, u);
    let mut compiled = mlx_rs::transforms::compile::compile(f, true);
    compiled((gate, up)).map_err(e)
}

pub struct Model {
    pub cfg: Config,
    embed: QLinear,
    layers: Vec<Layer>,
    norm: Array,
    /// Untied checkpoints only; tied ones use the embedding.
    lm_head: Option<QLinear>,
}

/// The weight files: model.safetensors, or else the shards its index names
/// (`model-00001-of-00002.safetensors`, …), merged. Some repos ship an index
/// that just names model.safetensors; the single file wins.
fn load_weights(dir: &Path) -> R<HashMap<String, Array>> {
    let single = dir.join("model.safetensors");
    let index = dir.join("model.safetensors.index.json");
    if single.is_file() || !index.is_file() {
        return Array::load_safetensors(single).map_err(e);
    }
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&index).map_err(e)?).map_err(e)?;
    let mut shards: Vec<&str> = v["weight_map"].as_object().ok_or("the weight index has no weight_map")?.values().filter_map(|f| f.as_str()).collect();
    shards.sort_unstable();
    shards.dedup();
    let mut all = HashMap::new();
    for shard in shards {
        if !super::custom::is_shard_name(shard) {
            return Err(format!("unexpected weight file {shard:?}"));
        }
        all.extend(Array::load_safetensors(dir.join(shard)).map_err(e)?);
    }
    Ok(all)
}

/// Takes named tensors out of the checkpoint, checking shapes; whatever is
/// left over at the end is an error (like mlx-lm's `strict=True`).
struct Weights(HashMap<String, Array>);

impl Weights {
    fn take(&mut self, name: &str, shape: &[i32]) -> R<Array> {
        let a = self.0.remove(name).ok_or_else(|| format!("checkpoint is missing {name}"))?;
        if a.shape() != shape {
            return Err(format!("{name} has shape {:?}, expected {shape:?}", a.shape()));
        }
        Ok(a)
    }

    fn qlinear(&mut self, p: &str, out: i32, inp: i32, cfg: &Config) -> R<QLinear> {
        // `bits`-bit values packed into u32 words; one scale and bias per group.
        let packed = inp * cfg.bits / 32;
        let groups = inp / cfg.group_size;
        Ok(QLinear {
            w: self.take(&format!("{p}.weight"), &[out, packed])?,
            scales: self.take(&format!("{p}.scales"), &[out, groups])?,
            biases: self.take(&format!("{p}.biases"), &[out, groups])?,
            group_size: cfg.group_size,
            bits: cfg.bits,
        })
    }
}

impl Model {
    pub fn load(dir: &Path, cfg: Config) -> R<Self> {
        let mut w = Weights(load_weights(dir)?);
        let (d, hd) = (cfg.hidden, cfg.head_dim);
        let embed = w.qlinear("model.embed_tokens", cfg.vocab, d, &cfg)?;
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let p = format!("model.layers.{i}");
            layers.push(Layer {
                input_norm: w.take(&format!("{p}.input_layernorm.weight"), &[d])?,
                q: w.qlinear(&format!("{p}.self_attn.q_proj"), cfg.heads * hd, d, &cfg)?,
                k: w.qlinear(&format!("{p}.self_attn.k_proj"), cfg.kv_heads * hd, d, &cfg)?,
                v: w.qlinear(&format!("{p}.self_attn.v_proj"), cfg.kv_heads * hd, d, &cfg)?,
                o: w.qlinear(&format!("{p}.self_attn.o_proj"), d, cfg.heads * hd, &cfg)?,
                q_norm: w.take(&format!("{p}.self_attn.q_norm.weight"), &[hd])?,
                k_norm: w.take(&format!("{p}.self_attn.k_norm.weight"), &[hd])?,
                post_norm: w.take(&format!("{p}.post_attention_layernorm.weight"), &[d])?,
                gate: w.qlinear(&format!("{p}.mlp.gate_proj"), cfg.intermediate, d, &cfg)?,
                up: w.qlinear(&format!("{p}.mlp.up_proj"), cfg.intermediate, d, &cfg)?,
                down: w.qlinear(&format!("{p}.mlp.down_proj"), d, cfg.intermediate, &cfg)?,
            });
        }
        let norm = w.take("model.norm.weight", &[d])?;
        let lm_head = if cfg.tied {
            // A tied checkpoint may still carry an lm_head copy; mlx-lm drops it too.
            w.0.retain(|k, _| !k.starts_with("lm_head."));
            None
        } else {
            Some(w.qlinear("lm_head", cfg.vocab, d, &cfg)?)
        };
        if let Some(extra) = w.0.keys().next() {
            return Err(format!("unexpected tensor in checkpoint: {extra}"));
        }
        let model = Self { cfg, embed, layers, norm, lm_head };
        model.materialize()?;
        Ok(model)
    }

    /// Reads every weight into memory now, not on the first token.
    fn materialize(&self) -> R<()> {
        let q = |l: &QLinear| [l.w.clone(), l.scales.clone(), l.biases.clone()];
        let mut all: Vec<Array> = q(&self.embed).to_vec();
        if let Some(h) = &self.lm_head {
            all.extend(q(h));
        }
        all.push(self.norm.clone());
        for l in &self.layers {
            for x in [&l.q, &l.k, &l.v, &l.o, &l.gate, &l.up, &l.down] {
                all.extend(q(x));
            }
            all.extend([l.input_norm.clone(), l.q_norm.clone(), l.k_norm.clone(), l.post_norm.clone()]);
        }
        mlx_rs::transforms::eval(&all).map_err(e)
    }

    pub fn new_cache(&self) -> Vec<Kv> {
        (0..self.layers.len()).map(|_| Kv::default()).collect()
    }

    /// Token ids `[1, L]` → hidden states `[1, L, D]`, appending to `cache`.
    pub fn forward(&self, ids: &Array, cache: &mut [Kv]) -> R<Array> {
        let c = &self.cfg;
        let e_ = &self.embed;
        let rows = |a: &Array| a.take_axis(ids, 0).map_err(e);
        let mut h = ops::dequantize(rows(&e_.w)?, rows(&e_.scales)?, &rows(&e_.biases)?, e_.group_size, e_.bits).map_err(e)?;
        let (b, l) = (ids.shape()[0], ids.shape()[1]);
        let scale = (c.head_dim as f32).powf(-0.5);
        for (layer, kv) in self.layers.iter().zip(cache.iter_mut()) {
            let x = fast::rms_norm(&h, Some(&layer.input_norm), c.eps).map_err(e)?;
            let heads = |a: Array, n: i32, norm: Option<&Array>| -> R<Array> {
                let a = a.reshape(&[b, l, n, c.head_dim]).map_err(e)?;
                let a = match norm {
                    Some(w) => fast::rms_norm(&a, Some(w), c.eps).map_err(e)?,
                    None => a,
                };
                a.transpose_axes(&[0, 2, 1, 3]).map_err(e)
            };
            let q = heads(layer.q.call(&x)?, c.heads, Some(&layer.q_norm))?;
            let k = heads(layer.k.call(&x)?, c.kv_heads, Some(&layer.k_norm))?;
            let v = heads(layer.v.call(&x)?, c.kv_heads, None)?;
            let q = fast::rope(&q, c.head_dim, false, c.theta, 1.0, kv.offset, None).map_err(e)?;
            let k = fast::rope(&k, c.head_dim, false, c.theta, 1.0, kv.offset, None).map_err(e)?;
            let (k, v) = kv.update(&k, &v)?;
            // One new token attends to everything cached; a chunk also needs causality
            // within itself, aligned to the end of the keys.
            let attn = if l > 1 {
                fast::scaled_dot_product_attention(&q, &k, &v, scale, ScaledDotProductAttentionMask::Causal, None)
            } else {
                fast::scaled_dot_product_attention(&q, &k, &v, scale, None, None)
            }
            .map_err(e)?;
            let attn = attn.transpose_axes(&[0, 2, 1, 3]).and_then(|a| a.reshape(&[b, l, c.heads * c.head_dim])).map_err(e)?;
            h = ops::add(&h, layer.o.call(&attn)?).map_err(e)?;
            let x = fast::rms_norm(&h, Some(&layer.post_norm), c.eps).map_err(e)?;
            let act = swiglu(&layer.gate.call(&x)?, &layer.up.call(&x)?)?;
            h = ops::add(&h, layer.down.call(&act)?).map_err(e)?;
        }
        fast::rms_norm(&h, Some(&self.norm), c.eps).map_err(e)
    }

    /// Hidden states `[.., D]` → logits over the vocabulary.
    pub fn logits(&self, h: &Array) -> R<Array> {
        self.lm_head.as_ref().unwrap_or(&self.embed).call(h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_kv_cache_grows_in_steps_and_trims() {
        let mut kv = Kv::default();
        let a = |n: i32, x: f32| ops::full::<f32>(&[1, 2, n, 4], Array::from_f32(x)).unwrap();
        let (k, _) = kv.update(&a(3, 1.0), &a(3, 1.0)).unwrap();
        assert_eq!((k.shape()[2], kv.offset, kv.k.as_ref().unwrap().shape()[2]), (3, 3, KV_STEP));
        kv.trim_to(2);
        let (k, _) = kv.update(&a(KV_STEP, 2.0), &a(KV_STEP, 2.0)).unwrap();
        // 2 kept + 256 new: grown by one more step, stale position 2 overwritten.
        assert_eq!((k.shape()[2], kv.offset), (KV_STEP + 2, KV_STEP + 2));
        let col: Vec<f32> = k.index((0, 0, .., 0)).as_slice::<f32>().to_vec();
        assert_eq!(&col[..3], &[1.0, 1.0, 2.0]);
    }
}
