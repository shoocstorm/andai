//! Qwen3 and Qwen3.5 on MLX (mlx-rs), inference only: ported from mlx-lm
//! `models/qwen3.py` and `models/qwen3_5.py` for MLX-quantized checkpoints.
//!
//! - **Qwen3:** embedding → per layer (RMSNorm → attention with q/k norms,
//!   RoPE and GQA → RMSNorm → SwiGLU MLP) → RMSNorm → LM head (tied or not).
//! - **Qwen3.5** (dense; `qwen3_5`): the same frame, but three of every four
//!   layers mix tokens with Gated DeltaNet (a causal depthwise convolution,
//!   then a gated linear recurrence, llm/delta.rs) instead of attention, whose
//!   state is fixed-size; the fourth is attention with a sigmoid output gate
//!   and RoPE on a quarter of each head. The vision tower is skipped.

use super::config::{Arch, Hybrid};
use mlx_rs::fast::{self, ScaledDotProductAttentionMask};
use mlx_rs::ops::indexing::{IndexOp, TryIndexMutOp};
use mlx_rs::{ops, Array, Dtype};
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

    fn arrays(&self) -> [Array; 3] {
        [self.w.clone(), self.scales.clone(), self.biases.clone()]
    }
}

struct Attention {
    q: QLinear,
    k: QLinear,
    v: QLinear,
    o: QLinear,
    q_norm: Array,
    k_norm: Array,
    /// Qwen3.5: `q_proj` also yields a gate, and the output is `o(attn · σ(gate))`.
    gated: bool,
}

/// Qwen3.5's linear-attention mixer (mlx-lm `GatedDeltaNet`).
struct DeltaNet {
    qkv: QLinear,
    z: QLinear,
    b: QLinear,
    a: QLinear,
    out: QLinear,
    /// Depthwise causal convolution, `[conv_dim, kernel, 1]`.
    conv: Array,
    dt_bias: Array,
    a_log: Array,
    norm: Array,
}

enum Mixer {
    Attention(Attention),
    Delta(DeltaNet),
}

struct Layer {
    input_norm: Array,
    mixer: Mixer,
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


/// A DeltaNet layer's state: the convolution's last `kernel - 1` inputs and
/// the recurrence's `[1, Hv, Dv, Dk]` float32 matrix. Unlike keys and values it
/// can't be cut back to an earlier position, so the engine keeps snapshots
/// (`Cache::snapshot`); both arrays are replaced on every step, never written
/// in place, so a snapshot is just the handles.
#[derive(Default, Clone)]
pub struct DeltaState {
    conv: Option<Array>,
    state: Option<Array>,
}

pub enum LayerCache {
    Kv(Kv),
    Delta(DeltaState),
}

/// Every layer's cache, and the DeltaNet states at earlier positions.
pub struct Cache {
    pub layers: Vec<LayerCache>,
    /// `(position, one state per DeltaNet layer)`, oldest first.
    snapshots: Vec<(usize, Vec<DeltaState>)>,
    /// How many snapshots to keep (from their size; at least 2).
    max_snapshots: usize,
}

/// Memory the snapshots of one model may hold.
const SNAPSHOT_BUDGET: usize = 256 << 20;

impl Cache {
    /// Whether any layer's state can't be trimmed (a Qwen3.5 model).
    pub fn recurrent(&self) -> bool {
        self.layers.iter().any(|l| matches!(l, LayerCache::Delta(_)))
    }

    fn deltas(&self) -> Vec<DeltaState> {
        self.layers.iter().filter_map(|l| if let LayerCache::Delta(d) = l { Some(d.clone()) } else { None }).collect()
    }

    /// Remembers the DeltaNet states as they are at `position` (the cache holds exactly
    /// that many tokens). The first snapshot (usually the end of the system prompt) is
    /// kept longest; past the limit the next oldest goes.
    pub fn snapshot(&mut self, position: usize) {
        if !self.recurrent() {
            return;
        }
        self.snapshots.retain(|(p, _)| *p < position);
        self.snapshots.push((position, self.deltas()));
        if self.snapshots.len() > self.max_snapshots {
            self.snapshots.remove(1);
        }
    }

    /// Goes back to at most `want` tokens of the `live` ones held; returns how many are
    /// kept. Keys and values are trimmed; DeltaNet states come back from the latest
    /// snapshot at or before `want` (or start over).
    pub fn rewind(&mut self, want: usize, live: usize) -> usize {
        let keep = if !self.recurrent() || want >= live {
            want.min(live)
        } else {
            match self.snapshots.iter().rev().find(|(p, _)| *p <= want) {
                Some((p, states)) => {
                    let (p, mut states) = (*p, states.clone().into_iter());
                    for l in &mut self.layers {
                        if let LayerCache::Delta(d) = l {
                            *d = states.next().unwrap_or_default();
                        }
                    }
                    p
                }
                None => {
                    for l in &mut self.layers {
                        if let LayerCache::Delta(d) = l {
                            *d = DeltaState::default();
                        }
                    }
                    0
                }
            }
        };
        for l in &mut self.layers {
            if let LayerCache::Kv(kv) = l {
                kv.trim_to(keep as i32);
            }
        }
        self.snapshots.retain(|(p, _)| *p <= keep);
        keep
    }

    pub fn arrays(&self) -> Vec<&Array> {
        self.layers
            .iter()
            .flat_map(|l| -> Vec<&Array> {
                match l {
                    LayerCache::Kv(kv) => kv.arrays().collect(),
                    LayerCache::Delta(d) => d.conv.iter().chain(d.state.iter()).collect(),
                }
            })
            .collect()
    }
}

/// `silu(gate) * up` as one fused kernel, like mlx-lm's compiled `swiglu`
/// (two kernels per layer otherwise).
fn swiglu(gate: &Array, up: &Array) -> R<Array> {
    let f = |(g, u): (&Array, &Array)| ops::multiply(ops::multiply(g, ops::sigmoid(g)?)?, u);
    let mut compiled = mlx_rs::transforms::compile::compile(f, true);
    compiled((gate, up)).map_err(e)
}

/// `x · σ(x)`, compiled like MLX's `nn.silu` (its rounding differs from separate ops).
fn silu(x: &Array) -> R<Array> {
    let f = |x: &Array| ops::multiply(x, ops::sigmoid(x)?);
    let mut compiled = mlx_rs::transforms::compile::compile(f, true);
    compiled(x).map_err(e)
}

/// DeltaNet's decay `exp(-exp(A_log) · softplus(a + dt_bias))`, compiled like
/// mlx-lm's `compute_g`: a fused kernel rounds differently from separate ops,
/// enough to move a bf16 logit by a step.
fn decay(a_log: &Array, a: &Array, dt_bias: &Array) -> R<Array> {
    let f = |(a_log, a, dt_bias): (&Array, &Array, &Array)| {
        let x = ops::add(a, dt_bias)?;
        let softplus = ops::logaddexp(&x, Array::from_f32(0.0).as_dtype(x.dtype())?)?;
        ops::exp(ops::negative(ops::multiply(ops::exp(a_log.as_dtype(Dtype::Float32)?)?, &softplus)?)?)
    };
    let mut compiled = mlx_rs::transforms::compile::compile(f, true);
    compiled((a_log, a, dt_bias)).map_err(e)
}

/// `silu(gate) · x` in float32, back in `x`'s dtype (mlx-lm `_precise_swiglu`, compiled).
fn precise_swiglu(gate: &Array, x: &Array) -> R<Array> {
    let f = |(gate, x): (&Array, &Array)| {
        let g = gate.as_dtype(Dtype::Float32)?;
        let g = ops::multiply(&g, ops::sigmoid(&g)?)?;
        ops::multiply(&g, x.as_dtype(Dtype::Float32)?)?.as_dtype(x.dtype())
    };
    let mut compiled = mlx_rs::transforms::compile::compile(f, true);
    compiled((gate, x)).map_err(e)
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

/// Qwen3.5 checkpoints as mlx-lm's `sanitize` leaves them: the vision tower
/// dropped (never read from disk), `model.language_model` renamed, and every
/// name under `language_model.`. Multi-token-prediction weights are dropped;
/// a checkpoint that still had them wasn't converted by mlx-lm, whose norms
/// then get the +1 that mlx-lm adds (Qwen3.5 stores them zero-centered).
fn sanitize_qwen35(weights: HashMap<String, Array>) -> R<HashMap<String, Array>> {
    let mut out = HashMap::new();
    let mut had_mtp = false;
    for (k, v) in weights {
        if k.starts_with("vision_tower") || k.starts_with("model.visual") {
            continue;
        }
        let k = if let Some(rest) = k.strip_prefix("model.language_model") {
            format!("language_model.model{rest}")
        } else if k.starts_with("language_model.") {
            k
        } else {
            format!("language_model.{k}")
        };
        if k.contains("mtp.") {
            had_mtp = true;
            continue;
        }
        out.insert(k, v);
    }
    let unconverted = out.iter().any(|(k, v)| k.ends_with("conv1d.weight") && v.shape().last() != Some(&1));
    if unconverted {
        return Err("its weights aren't converted for MLX (convert them with mlx-lm first)".into());
    }
    if had_mtp {
        let norms = [".input_layernorm.weight", ".post_attention_layernorm.weight", "model.norm.weight", ".q_norm.weight", ".k_norm.weight"];
        for (k, v) in out.iter_mut() {
            if v.ndim() == 1 && norms.iter().any(|s| k.ends_with(s)) {
                *v = ops::add(&*v, Array::from_f32(1.0)).and_then(|x| x.as_dtype(v.dtype())).map_err(e)?;
            }
        }
    }
    Ok(out)
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
        let q = cfg.quant(p);
        if inp % q.group_size != 0 || (inp * q.bits) % 32 != 0 {
            return Err(format!("{p}: {inp} inputs don't split into {}-bit groups of {}", q.bits, q.group_size));
        }
        let packed = inp * q.bits / 32;
        let groups = inp / q.group_size;
        Ok(QLinear {
            w: self.take(&format!("{p}.weight"), &[out, packed])?,
            scales: self.take(&format!("{p}.scales"), &[out, groups])?,
            biases: self.take(&format!("{p}.biases"), &[out, groups])?,
            group_size: q.group_size,
            bits: q.bits,
        })
    }
}

impl Model {
    pub fn load(dir: &Path, cfg: Config) -> R<Self> {
        let raw = load_weights(dir)?;
        let mut w = Weights(if matches!(cfg.arch, Arch::Qwen35(_)) { sanitize_qwen35(raw)? } else { raw });
        let pre = cfg.prefix();
        let (d, hd) = (cfg.hidden, cfg.head_dim);
        let embed = w.qlinear(&format!("{pre}model.embed_tokens"), cfg.vocab, d, &cfg)?;
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let p = format!("{pre}model.layers.{i}");
            let mixer = match &cfg.arch {
                Arch::Qwen35(h) if h.is_linear(i) => Mixer::Delta(Self::delta(&mut w, &format!("{p}.linear_attn"), h, &cfg)?),
                arch => {
                    let gated = matches!(arch, Arch::Qwen35(_));
                    let a = format!("{p}.self_attn");
                    Mixer::Attention(Attention {
                        q: w.qlinear(&format!("{a}.q_proj"), cfg.heads * hd * if gated { 2 } else { 1 }, d, &cfg)?,
                        k: w.qlinear(&format!("{a}.k_proj"), cfg.kv_heads * hd, d, &cfg)?,
                        v: w.qlinear(&format!("{a}.v_proj"), cfg.kv_heads * hd, d, &cfg)?,
                        o: w.qlinear(&format!("{a}.o_proj"), d, cfg.heads * hd, &cfg)?,
                        q_norm: w.take(&format!("{a}.q_norm.weight"), &[hd])?,
                        k_norm: w.take(&format!("{a}.k_norm.weight"), &[hd])?,
                        gated,
                    })
                }
            };
            layers.push(Layer {
                input_norm: w.take(&format!("{p}.input_layernorm.weight"), &[d])?,
                mixer,
                post_norm: w.take(&format!("{p}.post_attention_layernorm.weight"), &[d])?,
                gate: w.qlinear(&format!("{p}.mlp.gate_proj"), cfg.intermediate, d, &cfg)?,
                up: w.qlinear(&format!("{p}.mlp.up_proj"), cfg.intermediate, d, &cfg)?,
                down: w.qlinear(&format!("{p}.mlp.down_proj"), d, cfg.intermediate, &cfg)?,
            });
        }
        let norm = w.take(&format!("{pre}model.norm.weight"), &[d])?;
        let head = format!("{pre}lm_head");
        let lm_head = if cfg.tied {
            // A tied checkpoint may still carry an lm_head copy; mlx-lm drops it too.
            w.0.retain(|k, _| !k.starts_with(&format!("{head}.")));
            None
        } else {
            Some(w.qlinear(&head, cfg.vocab, d, &cfg)?)
        };
        if let Some(extra) = w.0.keys().next() {
            return Err(format!("unexpected tensor in checkpoint: {extra}"));
        }
        let model = Self { cfg, embed, layers, norm, lm_head };
        model.materialize()?;
        Ok(model)
    }

    fn delta(w: &mut Weights, p: &str, h: &Hybrid, cfg: &Config) -> R<DeltaNet> {
        let (key_dim, value_dim) = (h.k_heads * h.k_dim, h.v_heads * h.v_dim);
        let conv_dim = key_dim * 2 + value_dim;
        let d = cfg.hidden;
        Ok(DeltaNet {
            qkv: w.qlinear(&format!("{p}.in_proj_qkv"), conv_dim, d, cfg)?,
            z: w.qlinear(&format!("{p}.in_proj_z"), value_dim, d, cfg)?,
            b: w.qlinear(&format!("{p}.in_proj_b"), h.v_heads, d, cfg)?,
            a: w.qlinear(&format!("{p}.in_proj_a"), h.v_heads, d, cfg)?,
            out: w.qlinear(&format!("{p}.out_proj"), d, value_dim, cfg)?,
            conv: w.take(&format!("{p}.conv1d.weight"), &[conv_dim, h.conv_kernel, 1])?,
            dt_bias: w.take(&format!("{p}.dt_bias"), &[h.v_heads])?,
            // mlx-lm keeps A_log in float32 (`cast_predicate`).
            a_log: w.take(&format!("{p}.A_log"), &[h.v_heads])?.as_dtype(Dtype::Float32).map_err(e)?,
            norm: w.take(&format!("{p}.norm.weight"), &[h.v_dim])?,
        })
    }

    /// Reads every weight into memory now, not on the first token.
    fn materialize(&self) -> R<()> {
        let mut all: Vec<Array> = self.embed.arrays().to_vec();
        if let Some(h) = &self.lm_head {
            all.extend(h.arrays());
        }
        all.push(self.norm.clone());
        for l in &self.layers {
            for x in [&l.gate, &l.up, &l.down] {
                all.extend(x.arrays());
            }
            all.extend([l.input_norm.clone(), l.post_norm.clone()]);
            match &l.mixer {
                Mixer::Attention(a) => {
                    for x in [&a.q, &a.k, &a.v, &a.o] {
                        all.extend(x.arrays());
                    }
                    all.extend([a.q_norm.clone(), a.k_norm.clone()]);
                }
                Mixer::Delta(n) => {
                    for x in [&n.qkv, &n.z, &n.b, &n.a, &n.out] {
                        all.extend(x.arrays());
                    }
                    all.extend([n.conv.clone(), n.dt_bias.clone(), n.a_log.clone(), n.norm.clone()]);
                }
            }
        }
        mlx_rs::transforms::eval(&all).map_err(e)
    }

    pub fn new_cache(&self) -> Cache {
        let layers: Vec<LayerCache> = self
            .layers
            .iter()
            .map(|l| match l.mixer {
                Mixer::Attention(_) => LayerCache::Kv(Kv::default()),
                Mixer::Delta(_) => LayerCache::Delta(DeltaState::default()),
            })
            .collect();
        let per_snapshot = match &self.cfg.arch {
            Arch::Qwen35(h) => {
                let conv_dim = (2 * h.k_heads * h.k_dim + h.v_heads * h.v_dim) as usize;
                let one = (h.v_heads * h.v_dim * h.k_dim) as usize * 4 + (h.conv_kernel as usize - 1) * conv_dim * 2;
                one * (0..self.cfg.layers).filter(|&i| h.is_linear(i)).count()
            }
            Arch::Qwen3 => 1,
        };
        Cache { layers, snapshots: Vec::new(), max_snapshots: (SNAPSHOT_BUDGET / per_snapshot.max(1)).clamp(2, 16) }
    }

    fn attention(&self, a: &Attention, x: &Array, kv: &mut Kv) -> R<Array> {
        let c = &self.cfg;
        let (b, l) = (x.shape()[0], x.shape()[1]);
        let heads = |t: Array, n: i32, norm: Option<&Array>| -> R<Array> {
            let t = t.reshape(&[b, l, n, c.head_dim]).map_err(e)?;
            let t = match norm {
                Some(w) => fast::rms_norm(&t, Some(w), c.eps).map_err(e)?,
                None => t,
            };
            t.transpose_axes(&[0, 2, 1, 3]).map_err(e)
        };
        let (q, gate) = if a.gated {
            // Per head: the query, then its gate.
            let both = a.q.call(x)?.reshape(&[b, l, c.heads, 2 * c.head_dim]).map_err(e)?;
            let parts = ops::split_at_indices(&both, &[c.head_dim], -1).map_err(e)?;
            (parts[0].reshape(&[b, l, c.heads * c.head_dim]).map_err(e)?, Some(parts[1].reshape(&[b, l, c.heads * c.head_dim]).map_err(e)?))
        } else {
            (a.q.call(x)?, None)
        };
        let q = heads(q, c.heads, Some(&a.q_norm))?;
        let k = heads(a.k.call(x)?, c.kv_heads, Some(&a.k_norm))?;
        let v = heads(a.v.call(x)?, c.kv_heads, None)?;
        let q = fast::rope(&q, c.rotary_dims, false, c.theta, 1.0, kv.offset, None).map_err(e)?;
        let k = fast::rope(&k, c.rotary_dims, false, c.theta, 1.0, kv.offset, None).map_err(e)?;
        let (k, v) = kv.update(&k, &v)?;
        let scale = (c.head_dim as f32).powf(-0.5);
        // One new token attends to everything cached; a chunk also needs causality
        // within itself, aligned to the end of the keys.
        let attn = if l > 1 {
            fast::scaled_dot_product_attention(&q, &k, &v, scale, ScaledDotProductAttentionMask::Causal, None)
        } else {
            fast::scaled_dot_product_attention(&q, &k, &v, scale, None, None)
        }
        .map_err(e)?;
        let mut attn = attn.transpose_axes(&[0, 2, 1, 3]).and_then(|t| t.reshape(&[b, l, c.heads * c.head_dim])).map_err(e)?;
        if let Some(g) = gate {
            attn = ops::multiply(&attn, ops::sigmoid(&g).map_err(e)?).map_err(e)?;
        }
        a.o.call(&attn)
    }

    /// mlx-lm `GatedDeltaNet.__call__` for one sequence, no mask.
    fn delta_net(&self, n: &DeltaNet, x: &Array, st: &mut DeltaState) -> R<Array> {
        let Arch::Qwen35(h) = &self.cfg.arch else { unreachable!("a DeltaNet layer outside Qwen3.5") };
        let (b, s) = (x.shape()[0], x.shape()[1]);
        let key_dim = h.k_heads * h.k_dim;
        let conv_dim = 2 * key_dim + h.v_heads * h.v_dim;
        let qkv = n.qkv.call(x)?;
        let z = n.z.call(x)?.reshape(&[b, s, h.v_heads, h.v_dim]).map_err(e)?;
        let beta = ops::sigmoid(n.b.call(x)?).map_err(e)?;
        let a = n.a.call(x)?;

        // Causal depthwise convolution over the kept inputs and the new ones.
        let keep = h.conv_kernel - 1;
        let prev = match st.conv.take() {
            Some(c) => c,
            None => ops::zeros_dtype(&[b, keep, conv_dim], x.dtype()).map_err(e)?,
        };
        let input = ops::concatenate(&[prev, qkv], 1).map_err(e)?;
        let len = input.shape()[1];
        st.conv = Some(input.index((.., (len - keep)..len, ..)));
        let conv = silu(&ops::conv1d(&input, &n.conv, 1, 0, 1, conv_dim).map_err(e)?)?;

        let parts = ops::split_at_indices(&conv, &[key_dim, 2 * key_dim], -1).map_err(e)?;
        let q = parts[0].reshape(&[b, s, h.k_heads, h.k_dim]).map_err(e)?;
        let k = parts[1].reshape(&[b, s, h.k_heads, h.k_dim]).map_err(e)?;
        let v = parts[2].reshape(&[b, s, h.v_heads, h.v_dim]).map_err(e)?;
        let inv = (h.k_dim as f32).powf(-0.5);
        // Scaled in the activations' dtype, as Python's scalar is.
        let scalar = |f: f32| Array::from_f32(f).as_dtype(x.dtype()).map_err(e);
        let q = ops::multiply(fast::rms_norm(&q, None, 1e-6).map_err(e)?, scalar(inv * inv)?).map_err(e)?;
        let k = ops::multiply(fast::rms_norm(&k, None, 1e-6).map_err(e)?, scalar(inv)?).map_err(e)?;

        let g = decay(&n.a_log, &a, &n.dt_bias)?;
        let state = match st.state.take() {
            Some(s) => s,
            None => ops::zeros_dtype(&[b, h.v_heads, h.v_dim, h.k_dim], Dtype::Float32).map_err(e)?,
        };
        let (y, state) = super::delta::update(&q, &k, &v, &g, &beta, &state)?;
        st.state = Some(state);

        // Gated RMSNorm: rms_norm(y) · silu(z), in float32.
        let normed = fast::rms_norm(&y, Some(&n.norm), self.cfg.eps).map_err(e)?;
        let out = precise_swiglu(&z, &normed)?;
        n.out.call(&out.reshape(&[b, s, -1]).map_err(e)?)
    }

    /// Token ids `[1, L]` → hidden states `[1, L, D]`, appending to `cache`.
    pub fn forward(&self, ids: &Array, cache: &mut Cache) -> R<Array> {
        let c = &self.cfg;
        let e_ = &self.embed;
        let rows = |a: &Array| a.take_axis(ids, 0).map_err(e);
        let mut h = ops::dequantize(rows(&e_.w)?, rows(&e_.scales)?, &rows(&e_.biases)?, e_.group_size, e_.bits).map_err(e)?;
        for (layer, lc) in self.layers.iter().zip(cache.layers.iter_mut()) {
            let x = fast::rms_norm(&h, Some(&layer.input_norm), c.eps).map_err(e)?;
            let r = match (&layer.mixer, lc) {
                (Mixer::Attention(a), LayerCache::Kv(kv)) => self.attention(a, &x, kv)?,
                (Mixer::Delta(n), LayerCache::Delta(st)) => self.delta_net(n, &x, st)?,
                _ => return Err("the cache doesn't match the model's layers".into()),
            };
            h = ops::add(&h, r).map_err(e)?;
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
    // The only unignored test that evaluates MLX arrays: where its metallib is
    // missing (a CI runner whose restored cargo cache never built `~/.mlx`),
    // the MLX error aborts the whole test process, not just this test
    // (AGENTS.md §2). Run it explicitly: `cargo test -- --ignored kv`.
    #[ignore = "evaluates MLX arrays; aborts the whole test process where mlx.metallib is missing"]
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

    fn hybrid_cache() -> Cache {
        let layers = (0..4).map(|i| if i == 3 { LayerCache::Kv(Kv::default()) } else { LayerCache::Delta(DeltaState::default()) }).collect();
        Cache { layers, snapshots: Vec::new(), max_snapshots: 3 }
    }

    #[test]
    fn a_recurrent_cache_rewinds_to_a_snapshot_or_starts_over() {
        let mut c = hybrid_cache();
        assert!(c.recurrent());
        for p in [10, 20, 30] {
            c.snapshot(p);
        }
        // Everything held is a prefix of the new prompt: keep it all, no snapshot needed.
        assert_eq!(c.rewind(40, 40), 40);
        // Shares 25 of 40 tokens: back to the snapshot at 20.
        assert_eq!(c.rewind(25, 40), 20);
        assert_eq!(c.snapshots.iter().map(|s| s.0).collect::<Vec<_>>(), vec![10, 20], "later snapshots are gone");
        // Shares less than the first snapshot: start over.
        assert_eq!(c.rewind(5, 20), 0);
        assert!(c.snapshots.is_empty());
    }

    #[test]
    fn snapshots_keep_the_first_and_the_latest() {
        let mut c = hybrid_cache();
        for p in [10, 20, 30, 40] {
            c.snapshot(p);
        }
        assert_eq!(c.snapshots.iter().map(|s| s.0).collect::<Vec<_>>(), vec![10, 30, 40]);
        // A snapshot at an earlier position replaces the ones after it.
        c.snapshot(15);
        assert_eq!(c.snapshots.iter().map(|s| s.0).collect::<Vec<_>>(), vec![10, 15]);
    }

    #[test]
    fn a_trimmable_cache_keeps_what_is_shared() {
        let mut c = Cache { layers: vec![LayerCache::Kv(Kv::default())], snapshots: Vec::new(), max_snapshots: 2 };
        c.snapshot(10);
        assert!(c.snapshots.is_empty(), "no snapshots without recurrent layers");
        assert_eq!(c.rewind(25, 40), 25);
    }

    #[test]
    #[ignore = "evaluates MLX arrays; aborts the whole test process where mlx.metallib is missing"]
    fn the_gated_delta_kernel_matches_the_step_by_step_ops() {
        use mlx_rs::random;
        let n = |shape: &[i32], i: u64| random::normal::<f32>(shape, None, None, &random::key(i).unwrap()).unwrap();
        let (b, t, hk, hv, dk, dv) = (1, 9, 2, 4, 64, 32);
        let bf = |a: Array| a.as_dtype(Dtype::Bfloat16).unwrap();
        let q = bf(ops::multiply(n(&[b, t, hk, dk], 0), Array::from_f32(0.1)).unwrap());
        let k = bf(ops::multiply(n(&[b, t, hk, dk], 1), Array::from_f32(0.1)).unwrap());
        let v = bf(n(&[b, t, hv, dv], 2));
        let g = ops::sigmoid(n(&[b, t, hv], 3)).unwrap();
        let beta = bf(ops::sigmoid(n(&[b, t, hv], 4)).unwrap());
        let s0 = ops::multiply(n(&[b, hv, dv, dk], 5), Array::from_f32(0.1)).unwrap();
        let (y1, s1) = super::super::delta::update(&q, &k, &v, &g, &beta, &s0).unwrap();
        let (y2, s2) = super::super::delta::reference(&q, &k, &v, &g, &beta, &s0).unwrap();
        let diff = |a: &Array, b: &Array| ops::abs(ops::subtract(a.as_dtype(Dtype::Float32).unwrap(), b.as_dtype(Dtype::Float32).unwrap()).unwrap()).unwrap().max(None).unwrap().item_exact::<f32>();
        assert_eq!(y1.shape(), &[b, t, hv, dv]);
        assert!(diff(&y1, &y2) < 0.05, "outputs {}", diff(&y1, &y2));
        assert!(diff(&s1, &s2) < 1e-3, "states {}", diff(&s1, &s2));
    }
}
