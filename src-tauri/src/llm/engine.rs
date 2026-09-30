//! A loaded native chat model: Qwen3 or Qwen3.5 on MLX plus its tokenizer,
//! and the generation loop.
//!
//! - **Prompt-prefix reuse.** The KV cache keeps the token ids it holds; a new
//!   prompt that starts with the same tokens (the same system prompt and
//!   history, a follow-up decision) only runs the rest. It replaces wllama's
//!   `cache_prompt`, and the result is the same as running the whole prompt
//!   (checked in the parity test). Qwen3.5's DeltaNet state can't be cut
//!   back, so its prompts are read in pieces that end where a message starts,
//!   the state is kept at each (`Cache::snapshot`), and a new prompt resumes
//!   from the last one it shares.
//! - **Pipelined decoding**, like mlx-lm's `generate_step`: the next step is
//!   queued on the GPU (`async_eval`) before the current token is read back,
//!   so the GPU never waits on Rust. The sampled token stays an MLX array
//!   between steps. Without this, generation was about half as fast.
//! - **Choices.** A `root ::= "A" | "B"` grammar (grammar.rs) restricts the
//!   reply to those strings, and the first token's log-probabilities are read
//!   from the raw distribution (llm/decide.ts reads the option letters there).
//! - **Grammars.** Any other grammar is followed token by token through its
//!   DFA (grammar.rs `Constraint`): each step picks among the tokens whose
//!   bytes the grammar allows. No pipelining there, since the next mask
//!   depends on the token picked; replies held to a grammar are short.

use super::grammar::{Constraint, TokenBytes};
use super::config::Arch;
use super::model::{e, Cache, Config, Model, R};
use super::template::{self, Flavor, Message};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::{ops, random, Array, Dtype};
use serde::Serialize;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

/// Prompt tokens per forward pass while reading a prompt (mlx-lm uses 2048;
/// 512 keeps the activation memory of a chunk small at no measured cost).
const PREFILL_CHUNK: usize = 512;

pub struct Params {
    pub max_tokens: usize,
    /// 0 picks the most likely token (greedy).
    pub temperature: f32,
    /// 0 means no top-k cut.
    pub top_k: usize,
    pub top_p: f32,
    pub seed: Option<u64>,
    /// How many of the first token's most likely alternatives to report.
    pub top_logprobs: usize,
    /// Restrict the reply to one of these strings (a choice grammar).
    pub choices: Option<Vec<String>>,
    /// Any other GBNF grammar the reply must follow (grammar.rs).
    pub grammar: Option<String>,
    /// Reuse the cached prefix of the last prompt (see the module docs).
    pub reuse_prefix: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct TopLogprob {
    pub token: String,
    pub bytes: Vec<u8>,
    pub logprob: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct FirstToken {
    pub token: String,
    pub bytes: Vec<u8>,
    pub logprob: f32,
    /// Most likely first tokens, from the distribution before any choice restriction.
    pub top_logprobs: Vec<TopLogprob>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Finish {
    Stop,
    Length,
    Cancelled,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Generated {
    pub text: String,
    /// All prompt tokens, including those reused from the cache.
    pub prompt_tokens: usize,
    /// Prompt tokens whose keys and values came from the cache.
    pub cached_tokens: usize,
    pub completion_tokens: usize,
    /// From the call to the first token: reading the prompt and the first step.
    pub prompt_ms: f64,
    /// From the first token to the last.
    pub gen_ms: f64,
    pub finish: Finish,
    pub first: Option<FirstToken>,
}

pub struct Engine {
    model: Model,
    tok: tokenizers::Tokenizer,
    stop: Vec<u32>,
    cache: Cache,
    /// Token ids whose keys and values are in `cache`, in order.
    cached: Vec<u32>,
    pub n_ctx: usize,
    /// The chat template has Qwen3's `enable_thinking` switch.
    pub think_switch: bool,
    /// Which chat template to write.
    pub flavor: Flavor,
    /// `<|im_start|>`: where a message starts, so where a Qwen3.5 prompt is split.
    msg_start: Option<u32>,
    /// Each token id's raw bytes (`None`: a special or added token), built on first use by a grammar.
    token_bytes: Option<Arc<TokenBytes>>,
}

fn read_json(path: &Path) -> R<Value> {
    let text = std::fs::read_to_string(path).map_err(|err| format!("{}: {err}", path.display()))?;
    serde_json::from_str(&text).map_err(|err| format!("{}: {err}", path.display()))
}

/// Streams text from token ids without splitting a character: decodes the
/// window since the last emitted text and holds back an incomplete UTF-8
/// sequence (the byte-level tokenizer decodes it as U+FFFD).
struct Detok {
    ids: Vec<u32>,
    prefix: usize,
    read: usize,
}

impl Detok {
    fn push(&mut self, tok: &tokenizers::Tokenizer, id: u32) -> R<String> {
        self.ids.push(id);
        let before = tok.decode(&self.ids[self.prefix..self.read], true).map_err(e)?;
        let now = tok.decode(&self.ids[self.prefix..], true).map_err(e)?;
        if now.ends_with('\u{FFFD}') {
            return Ok(String::new());
        }
        match now.get(before.len()..) {
            Some(delta) if now.starts_with(&before) => {
                let delta = delta.to_string();
                self.prefix = self.read;
                self.read = self.ids.len();
                Ok(delta)
            }
            _ => Ok(String::new()),
        }
    }
}

impl Engine {
    pub fn load(dir: &Path, n_ctx: usize) -> R<Self> {
        let cfg = Config::from_json(&read_json(&dir.join("config.json"))?)?;
        let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).map_err(e)?;
        let tcfg = read_json(&dir.join("tokenizer_config.json"))?;
        let jinja = std::fs::read_to_string(dir.join("chat_template.jinja")).ok();
        let think_switch = super::config::check_template(super::config::chat_template(&tcfg, jinja.as_deref()).as_deref())?;
        // The end-of-turn token, and end-of-text in case a model emits it instead.
        let mut stop = Vec::new();
        for name in ["eos_token", "pad_token"] {
            let v = &tcfg[name];
            if let Some(id) = v.as_str().or_else(|| v["content"].as_str()).and_then(|t| tok.token_to_id(t)) {
                stop.push(id);
            }
        }
        if stop.is_empty() {
            return Err("tokenizer config names no end-of-turn token".into());
        }
        let flavor = match cfg.arch {
            Arch::Qwen3 => Flavor::Qwen3,
            Arch::Qwen35(_) => Flavor::Qwen35,
        };
        let msg_start = tok.token_to_id("<|im_start|>");
        let model = Model::load(dir, cfg)?;
        let cache = model.new_cache();
        Ok(Self { model, tok, stop, cache, cached: Vec::new(), n_ctx, think_switch, flavor, msg_start, token_bytes: None })
    }

    /// The prompt for `messages` in this model's chat template. A template
    /// without a thinking switch is rendered as with thinking on, which adds
    /// no think block for Qwen3.
    pub fn render(&self, messages: &[Message], thinking: bool) -> String {
        template::render_as(self.flavor, messages, thinking || !self.think_switch)
    }

    /// Layer count and weight bits.
    pub fn shape(&self) -> (usize, i32) {
        (self.model.cfg.layers, self.model.cfg.bits)
    }

    pub fn encode(&self, text: &str) -> R<Vec<u32>> {
        Ok(self.tok.encode(text, false).map_err(e)?.get_ids().to_vec())
    }

    /// Every token's raw bytes. Qwen's tokenizer is byte-level BPE: a token's
    /// string spells its bytes in GPT-2's printable alphabet, inverted here.
    /// (`decode` can't be used: it turns a partial UTF-8 sequence into U+FFFD.)
    fn token_bytes(&mut self) -> Arc<TokenBytes> {
        if let Some(t) = &self.token_bytes {
            return t.clone();
        }
        let mut to_byte = HashMap::new();
        let mut shifted = 0;
        for b in 0..=255u32 {
            let printable = (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
            let c = if printable {
                b
            } else {
                shifted += 1;
                255 + shifted
            };
            to_byte.insert(char::from_u32(c).expect("valid scalar"), b as u8);
        }
        let added: HashSet<u32> = self.tok.get_added_tokens_decoder().keys().copied().collect();
        let table: TokenBytes = (0..self.model.cfg.vocab as u32)
            .map(|id| {
                if added.contains(&id) {
                    return None;
                }
                let piece = self.tok.id_to_token(id)?;
                piece.chars().map(|c| to_byte.get(&c).copied()).collect::<Option<Vec<u8>>>().map(Vec::into_boxed_slice)
            })
            .collect();
        let table = Arc::new(table);
        self.token_bytes = Some(table.clone());
        table
    }

    fn token_text(&self, id: u32) -> (String, Vec<u8>) {
        let text = self.tok.decode(&[id], false).unwrap_or_default();
        let bytes = text.as_bytes().to_vec();
        (text, bytes)
    }

    /// Runs `ids` through the model; returns the last position's logits as f32 `[V]`.
    fn step(&mut self, ids: &Array) -> R<Array> {
        let h = self.model.forward(ids, &mut self.cache)?;
        let last = h.index((.., -1, ..));
        let logits = self.model.logits(&last)?;
        logits.reshape(&[-1]).and_then(|l| l.as_dtype(Dtype::Float32)).map_err(e)
    }

    fn log_softmax(logits: &Array) -> R<Array> {
        ops::subtract(logits, ops::logsumexp(logits, true).map_err(e)?).map_err(e)
    }

    fn top_logprobs(&self, logprobs: &Array, n: usize) -> R<Vec<TopLogprob>> {
        if n == 0 {
            return Ok(vec![]);
        }
        let n = n.min(logprobs.shape()[0] as usize) as i32;
        let idx = ops::argsort(ops::negative(logprobs).map_err(e)?).map_err(e)?.index(..n);
        let vals = logprobs.take(&idx).map_err(e)?;
        mlx_rs::transforms::eval([&idx, &vals]).map_err(e)?;
        Ok(idx
            .as_slice::<u32>()
            .iter()
            .zip(vals.as_slice::<f32>())
            .map(|(&id, &logprob)| {
                let (token, bytes) = self.token_text(id);
                TopLogprob { token, bytes, logprob }
            })
            .collect())
    }

    /// Samples one token id `[1, 1]` from `logits` on the GPU, lazily.
    fn sample(logits: &Array, p: &Params, key: &mut Array) -> R<Array> {
        if p.temperature <= 0.0 {
            return ops::indexing::argmax_axis(logits, 0, false).and_then(|t| t.reshape(&[1, 1])).map_err(e);
        }
        let v = ops::divide(logits, Array::from_f32(p.temperature)).map_err(e)?;
        let vocab = v.shape()[0] as usize;
        let neg = ops::negative(&v).map_err(e)?;
        // Candidates, most likely first: the top k, or the whole vocabulary.
        let idx = if p.top_k > 0 && p.top_k < vocab {
            let k = p.top_k as i32;
            let top = ops::argpartition(&neg, k - 1).map_err(e)?.index(..k);
            let order = ops::argsort(ops::negative(v.take(&top).map_err(e)?).map_err(e)?).map_err(e)?;
            top.take(&order).map_err(e)?
        } else {
            ops::argsort(&neg).map_err(e)?
        };
        let mut vals = v.take(&idx).map_err(e)?;
        if p.top_p < 1.0 {
            // Keep the smallest set whose probability reaches top_p (always the first).
            let probs = ops::softmax(&vals, true).map_err(e)?;
            let before = ops::subtract(ops::cumsum(&probs, None, None, None).map_err(e)?, &probs).map_err(e)?;
            let keep = before.lt(Array::from_f32(p.top_p)).map_err(e)?;
            vals = ops::select(&keep, &vals, Array::from_f32(f32::NEG_INFINITY)).map_err(e)?;
        }
        let (next, sub) = random::split(&*key, 2).map_err(e)?;
        *key = next;
        let pick = random::categorical(&vals, 0, None, &sub).map_err(e)?;
        idx.take(&pick).and_then(|t| t.reshape(&[1, 1])).map_err(e)
    }

    /// Reads `ids[..len-1]` into the cache, reusing a cached prefix when allowed.
    fn prefill(&mut self, ids: &[u32], reuse_prefix: bool) -> R<usize> {
        let common = if reuse_prefix { self.cached.iter().zip(ids).take_while(|(a, b)| a == b).count() } else { 0 };
        // At least the last prompt token runs, to produce the first logits.
        let reuse = self.cache.rewind(common.min(ids.len() - 1), self.cached.len());
        self.cached.truncate(reuse);
        let end = ids.len() - 1;
        let recurrent = self.cache.recurrent();
        let mut at = reuse;
        while at < end {
            let mut n = PREFILL_CHUNK.min(end - at);
            // A recurrent model stops where the next message starts, to keep its state there.
            if let (true, Some(start)) = (recurrent, self.msg_start) {
                if let Some(i) = ids[at + 1..at + n].iter().position(|&t| t == start) {
                    n = i + 1;
                }
            }
            let chunk = Array::from_slice(&ids[at..at + n], &[1, n as i32]);
            self.model.forward(&chunk, &mut self.cache)?;
            mlx_rs::transforms::eval(self.cache.arrays()).map_err(e)?;
            self.cached.extend_from_slice(&ids[at..at + n]);
            at += n;
            self.cache.snapshot(at);
            mlx_rs::memory::clear_cache().map_err(e)?;
        }
        Ok(reuse)
    }

    /// Generates a reply to the rendered prompt `text`. `on_text` gets each
    /// new piece of text; `cancelled` is checked after every token.
    pub fn generate(&mut self, text: &str, p: &Params, on_text: &mut dyn FnMut(&str), cancelled: &dyn Fn() -> bool) -> R<Generated> {
        let ids = self.encode(text)?;
        if ids.is_empty() {
            return Err("the prompt is empty".into());
        }
        // Compile the grammar first: a bad one fails before any model work.
        let constraint = p.grammar.as_deref().map(Constraint::new).transpose()?;
        if ids.len() >= self.n_ctx {
            return Err(format!("The prompt is {} tokens; this model reads at most {}.", ids.len(), self.n_ctx - 1));
        }
        let max_new = p.max_tokens.min(self.n_ctx - ids.len());
        let started = Instant::now();
        let cached_tokens = self.prefill(&ids, p.reuse_prefix)?;
        let last = Array::from_slice(&ids[ids.len() - 1..], &[1, 1]);
        let logits = self.step(&last)?;
        self.cached.push(ids[ids.len() - 1]);
        let seed = p.seed.unwrap_or_else(|| std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0));
        let mut key = random::key(seed).map_err(e)?;

        let mut out = Generated {
            text: String::new(),
            prompt_tokens: ids.len(),
            cached_tokens,
            completion_tokens: 0,
            prompt_ms: 0.0,
            gen_ms: 0.0,
            finish: Finish::Length,
            first: None,
        };
        if p.top_logprobs > 0 || p.choices.is_some() {
            let lp = Self::log_softmax(&logits)?;
            let top = self.top_logprobs(&lp, p.top_logprobs)?;
            out.first = Some(FirstToken { token: String::new(), bytes: vec![], logprob: 0.0, top_logprobs: top });
            if let Some(choices) = &p.choices {
                return self.generate_choice(choices, lp, p, &mut key, started, out, on_text);
            }
        }
        if max_new == 0 {
            out.prompt_ms = started.elapsed().as_secs_f64() * 1e3;
            return Ok(out);
        }
        if let Some(c) = constraint {
            return self.generate_constrained(c, logits, p, max_new, &mut key, started, out, on_text, cancelled);
        }

        let mut detok = Detok { ids: Vec::new(), prefix: 0, read: 0 };
        let mut y = Self::sample(&logits, p, &mut key)?;
        mlx_rs::transforms::async_eval([&y]).map_err(e)?;
        let mut first_at: Option<Instant> = None;
        for n in 0..max_new {
            // Queue the next step before reading this token back.
            let next = if n + 1 < max_new {
                let logits = self.step(&y)?;
                let next = Self::sample(&logits, p, &mut key)?;
                mlx_rs::transforms::async_eval([&next]).map_err(e)?;
                Some(next)
            } else {
                None
            };
            let id = y.item_exact::<u32>();
            if next.is_some() {
                // `y` went into the cache with that step.
                self.cached.push(id);
            }
            let now = Instant::now();
            if first_at.is_none() {
                first_at = Some(now);
                out.prompt_ms = (now - started).as_secs_f64() * 1e3;
                if let Some(f) = &mut out.first {
                    (f.token, f.bytes) = self.token_text(id);
                    f.logprob = f.top_logprobs.iter().find(|t| t.token == f.token).map_or(f32::NEG_INFINITY, |t| t.logprob);
                }
            }
            if self.stop.contains(&id) {
                out.finish = Finish::Stop;
                break;
            }
            out.completion_tokens += 1;
            let piece = detok.push(&self.tok, id)?;
            if !piece.is_empty() {
                out.text.push_str(&piece);
                on_text(&piece);
            }
            if cancelled() {
                out.finish = Finish::Cancelled;
                break;
            }
            match next {
                Some(next) => y = next,
                None => break,
            }
        }
        out.gen_ms = first_at.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1e3);
        Ok(out)
    }

    /// The reply held to a grammar, token by token: each step samples among the
    /// tokens whose bytes it allows, plus the end-of-turn token once it may end.
    #[allow(clippy::too_many_arguments)]
    fn generate_constrained(
        &mut self,
        mut c: Constraint,
        mut logits: Array,
        p: &Params,
        max_new: usize,
        key: &mut Array,
        started: Instant,
        mut out: Generated,
        on_text: &mut dyn FnMut(&str),
        cancelled: &dyn Fn() -> bool,
    ) -> R<Generated> {
        let table = self.token_bytes();
        let eos = self.stop[0];
        let mut detok = Detok { ids: Vec::new(), prefix: 0, read: 0 };
        let mut first_at: Option<Instant> = None;
        for _ in 0..max_new {
            let allowed = c.allowed(&table)?;
            let can_end = c.can_end()?;
            let candidates: std::borrow::Cow<[u32]> = if can_end {
                let mut v = allowed.to_vec();
                v.push(eos);
                v.into()
            } else {
                allowed.as_slice().into()
            };
            if candidates.is_empty() {
                // Only reachable with a grammar that can neither continue nor end.
                out.finish = Finish::Stop;
                break;
            }
            let vals = logits.take(Array::from_slice(&candidates, &[candidates.len() as i32])).map_err(e)?;
            let pos = if p.temperature <= 0.0 || candidates.len() == 1 {
                ops::indexing::argmax_axis(&vals, 0, false).map_err(e)?.item_exact::<u32>()
            } else {
                let scaled = ops::divide(&vals, Array::from_f32(p.temperature)).map_err(e)?;
                let (next, sub) = random::split(&*key, 2).map_err(e)?;
                *key = next;
                random::categorical(&scaled, 0, None, &sub).map_err(e)?.item_exact::<u32>()
            };
            let id = candidates[pos as usize];
            if first_at.is_none() {
                first_at = Some(Instant::now());
                out.prompt_ms = started.elapsed().as_secs_f64() * 1e3;
                if let Some(f) = &mut out.first {
                    (f.token, f.bytes) = self.token_text(id);
                    let lp = Self::log_softmax(&logits)?.take(Array::from_slice(&[id], &[1])).map_err(e)?;
                    f.logprob = lp.item_exact::<f32>();
                }
            }
            if self.stop.contains(&id) {
                out.finish = Finish::Stop;
                break;
            }
            c.advance(table[id as usize].as_deref().ok_or("picked a token the grammar can't spell")?)?;
            out.completion_tokens += 1;
            let piece = detok.push(&self.tok, id)?;
            if !piece.is_empty() {
                out.text.push_str(&piece);
                on_text(&piece);
            }
            if cancelled() {
                out.finish = Finish::Cancelled;
                break;
            }
            // Complete, with nothing that could follow: done, without another model step.
            if c.can_end()? && c.allowed(&table)?.is_empty() {
                out.finish = Finish::Stop;
                break;
            }
            logits = self.step(&Array::from_slice(&[id], &[1, 1]))?;
            self.cached.push(id);
        }
        out.gen_ms = first_at.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1e3);
        Ok(out)
    }

    /// The reply restricted to one of `choices`, token by token: at each step,
    /// only a token that continues some choice can be picked.
    #[allow(clippy::too_many_arguments)]
    fn generate_choice(
        &mut self,
        choices: &[String],
        mut logprobs: Array,
        p: &Params,
        key: &mut Array,
        started: Instant,
        mut out: Generated,
        on_text: &mut dyn FnMut(&str),
    ) -> R<Generated> {
        let seqs: Vec<Vec<u32>> = choices.iter().map(|c| self.encode(c)).collect::<R<_>>()?;
        let mut chosen: Vec<u32> = Vec::new();
        loop {
            let mut allowed: Vec<u32> = seqs.iter().filter(|s| s.len() > chosen.len() && s.starts_with(&chosen)).map(|s| s[chosen.len()]).collect();
            allowed.sort_unstable();
            allowed.dedup();
            if allowed.is_empty() {
                out.finish = Finish::Stop;
                break;
            }
            if chosen.len() >= p.max_tokens {
                out.finish = if seqs.contains(&chosen) { Finish::Stop } else { Finish::Length };
                break;
            }
            let vals = logprobs.take(Array::from_slice(&allowed, &[allowed.len() as i32])).map_err(e)?;
            let pick = if p.temperature <= 0.0 || allowed.len() == 1 {
                vals.eval().map_err(e)?;
                let v = vals.as_slice::<f32>();
                (0..v.len()).max_by(|&a, &b| v[a].total_cmp(&v[b])).unwrap()
            } else {
                let scaled = ops::divide(&vals, Array::from_f32(p.temperature)).map_err(e)?;
                let (next, sub) = random::split(&*key, 2).map_err(e)?;
                *key = next;
                random::categorical(&scaled, 0, None, &sub).map_err(e)?.item_exact::<u32>() as usize
            };
            let id = allowed[pick];
            if chosen.is_empty() {
                out.prompt_ms = started.elapsed().as_secs_f64() * 1e3;
                if let Some(f) = &mut out.first {
                    (f.token, f.bytes) = self.token_text(id);
                    let v = logprobs.take(Array::from_slice(&[id], &[1])).map_err(e)?;
                    f.logprob = v.item_exact::<f32>();
                }
            }
            chosen.push(id);
            out.completion_tokens += 1;
            // A choice that is complete and can't be extended ends the reply.
            if seqs.contains(&chosen) && !seqs.iter().any(|s| s.len() > chosen.len() && s.starts_with(&chosen)) {
                out.finish = Finish::Stop;
                break;
            }
            let logits = self.step(&Array::from_slice(&[id], &[1, 1]))?;
            self.cached.push(id);
            logprobs = Self::log_softmax(&logits)?;
        }
        out.text = self.tok.decode(&chosen, true).map_err(e)?;
        on_text(&out.text);
        out.gen_ms = (started.elapsed().as_secs_f64() * 1e3 - out.prompt_ms).max(0.0);
        Ok(out)
    }

    /// Frees the cached keys and values (the next prompt starts cold).
    #[cfg(test)]
    pub fn clear_cache(&mut self) {
        self.cache = self.model.new_cache();
        self.cached.clear();
    }
}

/// Parity against transformers and mlx-lm on the real checkpoints. Needs the
/// Hugging Face cache (`hf download mlx-community/Qwen3-1.7B-4bit` /
/// `Qwen3-0.6B-8bit` at the catalog commits) or `LLM_DIR_<ID>`; run with
/// `bun run test:llm`.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::catalog;
    use crate::llm::template::{render, Message};
    use std::path::PathBuf;

    fn checkpoint_dir(id: &str) -> PathBuf {
        let c = catalog::checkpoint(id).unwrap();
        if let Some(d) = std::env::var_os(format!("LLM_DIR_{}", id.replace(['-', '.'], "_").to_uppercase())) {
            return d.into();
        }
        let home = std::env::var_os("HOME").expect("HOME");
        PathBuf::from(home).join(".cache/huggingface/hub").join(format!("models--{}", c.repo.replace('/', "--"))).join("snapshots").join(&*c.commit)
    }

    fn golden(id: &str) -> Value {
        let path = format!("{}/tests/fixtures/llm/golden-{}.json", env!("CARGO_MANIFEST_DIR"), id.trim_end_matches("-mlx"));
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    fn messages_system(g: &Value) -> String {
        g["cases"]["system_user_nothink"]["messages"][0]["content"].as_str().unwrap().to_string()
    }

    fn greedy(max_tokens: usize) -> Params {
        Params { max_tokens, temperature: 0.0, top_k: 0, top_p: 1.0, seed: Some(1), top_logprobs: 0, choices: None, grammar: None, reuse_prefix: true }
    }

    fn run(id: &str) {
        run_on(id, &checkpoint_dir(id), golden(id));
    }

    fn run_on(id: &str, dir: &Path, g: Value) {
        let mut engine = Engine::load(dir, 8192).unwrap();
        for (name, case) in g["cases"].as_object().unwrap() {
            let messages: Vec<Message> = serde_json::from_value(case["messages"].clone()).unwrap();
            let text = engine.render(&messages, case["enable_thinking"].as_bool().unwrap());
            assert_eq!(text, case["text"].as_str().unwrap(), "{id} {name}: template matches transformers");
            let want: Vec<u32> = serde_json::from_value(case["ids"].clone()).unwrap();
            assert_eq!(engine.encode(&text).unwrap(), want, "{id} {name}: token ids match transformers");
        }

        // Greedy continuation, token for token (4-bit and 8-bit weights, same kernels as mlx-lm).
        let case = &g["cases"]["system_user_nothink"];
        let text = case["text"].as_str().unwrap();
        let want: Vec<u32> = serde_json::from_value(case["greedy"].clone()).unwrap();
        let want_text = case["greedy_text"].as_str().unwrap().replace("<|im_end|>", "");
        engine.clear_cache();
        let out = engine.generate(text, &greedy(want.len()), &mut |_| {}, &|| false).unwrap();
        assert_eq!(out.text, want_text, "{id}: greedy text matches mlx-lm");
        assert_eq!(out.cached_tokens, 0);

        // Same prompt again: everything but the last token comes from the cache, same output.
        let again = engine.generate(text, &greedy(want.len()), &mut |_| {}, &|| false).unwrap();
        assert_eq!(again.cached_tokens, out.prompt_tokens - 1, "{id}: prefix reused");
        assert_eq!(again.text, out.text, "{id}: a reused prefix gives the same reply");

        // A partly shared prompt (same system prompt, another question) reuses only that
        // prefix and runs the rest as a chunk at an offset. Its first-token distribution
        // must match a cold run's within bf16 noise: splitting a prompt moves logits by up
        // to ~0.4 in mlx-lm too (enough to flip a greedy tie, so replies aren't compared),
        // while a wrong mask or RoPE offset moves them by whole nats.
        let other = engine.render(
            &[
                Message { role: crate::llm::template::Role::System, content: messages_system(&g) },
                Message { role: crate::llm::template::Role::User, content: "Which vessels carry cars, and how many?".into() },
            ],
            false,
        );
        let top = Params { top_logprobs: 20, ..greedy(1) };
        engine.clear_cache();
        let cold = engine.generate(&other, &top, &mut |_| {}, &|| false).unwrap().first.unwrap();
        engine.generate(text, &greedy(4), &mut |_| {}, &|| false).unwrap();
        let warm_out = engine.generate(&other, &top, &mut |_| {}, &|| false).unwrap();
        assert!(
            warm_out.cached_tokens > 10 && warm_out.cached_tokens < warm_out.prompt_tokens - 1,
            "{id}: partial reuse ({} cached)",
            warm_out.cached_tokens
        );
        let warm = warm_out.first.unwrap();
        let drift = cold.top_logprobs[..5]
            .iter()
            .map(|c| warm.top_logprobs.iter().find(|w| w.token == c.token).map_or(f32::INFINITY, |w| (w.logprob - c.logprob).abs()))
            .fold(0.0f32, f32::max);
        assert!(drift < 0.75, "{id}: a partly reused prefix keeps the first-token distribution (drift {drift})");

        // Decision readout: the letters' first-token log-probabilities, restricted to the
        // letters. From a clean cache, so the prompt runs the way the golden's did: the
        // 4-bit model's values move by nats with how a prompt is split (in mlx-lm too),
        // e.g. C from -1.5 to -3.0 behind a 3-token cached prefix.
        engine.clear_cache();
        let d = &g["cases"]["decision"];
        let choice = Params { choices: Some(vec!["A".into(), "B".into(), "C".into()]), top_logprobs: 20, max_tokens: 1, ..greedy(1) };
        let got = engine.generate(d["text"].as_str().unwrap(), &choice, &mut |_| {}, &|| false).unwrap();
        let first = got.first.unwrap();
        let mut worst = 0.0f32;
        for (letter, want) in d["first_logprobs"].as_object().unwrap() {
            let want = want.as_f64().unwrap() as f32;
            match first.top_logprobs.iter().find(|t| t.token == *letter) {
                Some(t) => worst = worst.max((t.logprob - want).abs()),
                None => assert!(want < first.top_logprobs.last().unwrap().logprob + 0.05, "{id}: {letter} ({want}) missing from the top 20"),
            }
        }
        let best = d["first_logprobs"].as_object().unwrap().iter().max_by(|a, b| a.1.as_f64().unwrap().total_cmp(&b.1.as_f64().unwrap())).unwrap().0;
        assert_eq!(&got.text, best, "{id}: same letter as mlx-lm");
        assert_eq!((got.finish, got.completion_tokens), (Finish::Stop, 1));
        assert!(worst < 0.05, "{id}: letter logprobs within 0.05 of mlx-lm float32 (worst {worst})");

        // The byte table spells every token exactly, multi-byte characters included.
        let table = engine.token_bytes();
        for sample in ["Ünïcödé ✓ 日本 — {\"query\": \"x\"}", "  indented\n\tcode();"] {
            let bytes: Vec<u8> = engine.encode(sample).unwrap().iter().flat_map(|&id| table[id as usize].clone().unwrap().into_vec()).collect();
            assert_eq!(bytes, sample.as_bytes(), "{id}: token bytes");
        }
        assert!(table[engine.stop[0] as usize].is_none(), "{id}: end-of-turn is never spelled by a grammar");

        // Argument filling held to its grammar: a JSON object in the schema's shape.
        let grammars: Value = serde_json::from_str(include_str!("../../tests/fixtures/llm/grammars.json")).unwrap();
        let fill = engine.render(
            &[Message {
                role: crate::llm::template::Role::User,
                content: "Write a knowledge base search for: which function implements the group discount?\nArguments schema:\n{\"query\": string, \"scope\": \"broad\" | \"focused\"}\nReply with only the JSON object.".into(),
            }],
            false,
        );
        let held = Params { grammar: Some(grammars["kb_search"].as_str().unwrap().into()), ..greedy(200) };
        let t = Instant::now();
        let args = engine.generate(&fill, &held, &mut |_| {}, &|| false).unwrap();
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let v: Value = serde_json::from_str(&args.text).unwrap_or_else(|e| panic!("{id}: {e}: {:?}", args.text));
        assert!(v["query"].is_string() && ["broad", "focused"].contains(&v["scope"].as_str().unwrap()), "{id}: {}", args.text);
        assert_eq!(args.finish, Finish::Stop, "{id}: ends with the object");
        println!("{id}: grammar-held arguments {} in {ms:.0} ms ({} tokens)", args.text, args.completion_tokens);

        // Speed on a grounded-size prompt (~550 tokens), like bench:engine.
        let filler = "The ferry leaves the north pier at nine and returns at five. ".repeat(40);
        let prompt = engine.render(&[Message { role: crate::llm::template::Role::User, content: format!("{filler}\nSummarize the schedule in detail.") }], false);
        engine.clear_cache();
        let run = engine.generate(&prompt, &Params { max_tokens: 256, ..greedy(256) }, &mut |_| {}, &|| false).unwrap();
        let tps = (run.completion_tokens.saturating_sub(1)) as f64 / (run.gen_ms / 1e3);
        let pps = run.prompt_tokens as f64 / (run.prompt_ms / 1e3);
        println!(
            "{id}: {} prompt tokens at {pps:.0} tok/s, {} generated at {tps:.0} tok/s; decision logprob Δ {worst:.3}; partial-reuse drift {drift:.3}",
            run.prompt_tokens, run.completion_tokens
        );
    }

    /// Qwen's own MLX builds in the catalog (4B–32B): each loads from its
    /// pinned files, matches mlx-lm's greedy text, and reports its speed on a
    /// grounded-size prompt. Set `LLM_BIG` to a subset (e.g. `4B,8B`); needs the
    /// checkpoints in the Hugging Face cache.
    #[test]
    #[ignore = "needs Qwen's MLX checkpoints (2–17 GB each)"]
    fn llm_qwen3_big_match_mlx_lm() {
        let golden: Value = serde_json::from_str(include_str!("../../tests/fixtures/llm/golden-qwen3-mlx-4bit.json")).unwrap();
        let only = std::env::var("LLM_BIG").unwrap_or_else(|_| "4B,8B,14B,32B".into());
        for size in only.split(',') {
            let g = &golden[size];
            let id = format!("qwen3-{}-mlx", size.to_lowercase());
            let c = catalog::checkpoint(&id).unwrap();
            assert_eq!((c.repo.as_ref(), c.commit.as_ref()), (g["repo"].as_str().unwrap(), g["commit"].as_str().unwrap()));
            let t = Instant::now();
            let mut engine = Engine::load(&checkpoint_dir(&id), 4096).unwrap();
            let load_ms = t.elapsed().as_secs_f64() * 1e3;
            let out = engine.generate(g["text"].as_str().unwrap(), &greedy(12), &mut |_| {}, &|| false).unwrap();
            assert_eq!(out.text, g["greedy_text"].as_str().unwrap(), "{id}: same greedy text as mlx-lm");
            let filler = "The ferry leaves the north pier at nine and returns at five. ".repeat(40);
            let prompt = render(&[Message { role: crate::llm::template::Role::User, content: format!("{filler}\nSummarize the schedule in detail.") }], false);
            engine.clear_cache();
            let run = engine.generate(&prompt, &Params { max_tokens: 128, ..greedy(128) }, &mut |_| {}, &|| false).unwrap();
            println!(
                "{id}: load {load_ms:.0} ms; {} prompt tokens at {:.0} tok/s (first token {:.0} ms), {} generated at {:.0} tok/s",
                run.prompt_tokens,
                run.prompt_tokens as f64 / (run.prompt_ms / 1e3),
                run.prompt_ms,
                run.completion_tokens,
                (run.completion_tokens.saturating_sub(1)) as f64 / (run.gen_ms / 1e3)
            );
        }
    }

    /// A model added from Hugging Face, end to end in Rust: the spec the webview
    /// builds (llm/hub.ts), `custom::add`, a chunked download through the store
    /// with its sha256 check, then load and generate. Qwen3 8B is untied (its own
    /// LM head), which the catalog models aren't. Needs
    /// `hf download mlx-community/Qwen3-8B-4bit --revision 545dc42…` (4.6 GB).
    #[test]
    #[ignore = "needs the Qwen3 8B MLX checkpoint"]
    fn llm_custom_qwen3_8b_untied_matches_mlx_lm() {
        use crate::laya::store;
        use crate::llm::custom::{self, InlineFile, Spec, SpecFile};
        use sha2::{Digest, Sha256};
        let repo = "mlx-community/Qwen3-8B-4bit";
        let commit = "545dc4251c05440727734bcd94334791f6ab0192";
        let snap = PathBuf::from(std::env::var_os("HOME").unwrap()).join(".cache/huggingface/hub/models--mlx-community--Qwen3-8B-4bit/snapshots").join(commit);
        let read = |p: &str| std::fs::read(snap.join(p)).unwrap();
        let hash = |b: &[u8]| Sha256::digest(b).iter().map(|x| format!("{x:02x}")).collect::<String>();
        let big = |p: &str| {
            let b = read(p);
            SpecFile { path: p.into(), bytes: b.len() as u64, sha256: hash(&b) }
        };
        let text = |p: &str| InlineFile { path: p.into(), content: String::from_utf8(read(p)).unwrap() };
        let spec = Spec { repo: repo.into(), commit: commit.into(), files: vec![big("model.safetensors"), big("tokenizer.json")], inline: vec![text("config.json"), text("tokenizer_config.json")] };

        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let m = custom::add(root, &spec, 1).unwrap();
        assert_eq!((m.layers, m.bits, m.thinking), (36, 4, true));
        let c = m.checkpoint();
        for f in c.files.iter() {
            let bytes = read(&f.path);
            let mut offset = 0u64;
            for chunk in bytes.chunks(16 << 20) {
                offset = store::write_chunk(root, &c, &f.path, offset, chunk).unwrap();
            }
        }
        assert!(!custom::is_complete(root, &m), "parts aren't a download");
        store::finish(root, &c).unwrap();
        assert!(custom::is_complete(root, &m));

        let golden: Value = serde_json::from_str(include_str!("../../tests/fixtures/llm/golden-qwen3-8b-untied.json")).unwrap();
        let mut engine = Engine::load(&store::dir(root, &c), 4096).unwrap();
        assert!(engine.think_switch);
        let t = Instant::now();
        let out = engine.generate(golden["text"].as_str().unwrap(), &greedy(16), &mut |_| {}, &|| false).unwrap();
        assert_eq!(out.text, golden["greedy_text"].as_str().unwrap(), "untied LM head: same greedy text as mlx-lm");
        println!("qwen3-8b (custom): {} tokens in {:.0} ms, {:.0} tok/s", out.completion_tokens, t.elapsed().as_secs_f64() * 1e3, (out.completion_tokens - 1) as f64 / (out.gen_ms / 1e3));

        // Tampering after the download: the model no longer counts as complete.
        std::fs::write(store::dir(root, &c).join("config.json"), "{}").unwrap();
        assert!(!custom::is_complete(root, &m));
    }

    /// Qwen3.5 (hybrid DeltaNet + gated attention, mixed 4/8-bit weights): the
    /// same checks as the Qwen3 catalog models against golden35.py's goldens.
    /// Needs `hf download mlx-community/Qwen3.5-0.8B-OptiQ-4bit --revision ef60586…`.
    #[test]
    #[ignore = "needs the Qwen3.5 0.8B OptiQ MLX checkpoint"]
    fn llm_qwen35_0_8b_optiq_matches_mlx_lm() {
        let home = std::env::var_os("HOME").expect("HOME");
        let dir = PathBuf::from(home).join(".cache/huggingface/hub/models--mlx-community--Qwen3.5-0.8B-OptiQ-4bit/snapshots/ef60586933bd2cc02b763f77eb8839a5114bbec1");
        let g: Value = serde_json::from_str(include_str!("../../tests/fixtures/llm/golden-qwen3.5-0.8b-optiq.json")).unwrap();
        run_on("qwen3.5-0.8b-optiq", &dir, g);
    }

    #[test]
    #[ignore = "needs the Qwen3 1.7B MLX checkpoint"]
    fn llm_qwen3_1_7b_matches_mlx_lm() {
        run("qwen3-1.7b-mlx");
    }

    #[test]
    #[ignore = "needs the Qwen3 0.6B MLX checkpoint"]
    fn llm_qwen3_0_6b_matches_mlx_lm() {
        run("qwen3-0.6b-mlx");
    }
}
