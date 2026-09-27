//! A loaded Laya checkpoint: config, tokenizer, calibration and network, and
//! what Andai asks of it: one or more typed questions about the same state
//! (a choice between options, or whether a statement holds), in one pass.

use super::model::{EncoderConfig, Model, R};
use super::prompt::{self, Encode, Kind, Special};
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
    pad: u32,
    max_len: usize,
    head_max_len: usize,
    /// Per question type (choice, score, noul), and per bucket overrides; clamped.
    temperature: [f64; 3],
    by_options: Vec<(String, f64)>,
}

/// One question about the state. A `Noul` has no options of its own.
#[derive(Debug, Clone)]
pub struct Question {
    pub kind: Kind,
    pub instructions: String,
    /// `(id, text)` per option, for a choice.
    pub options: Vec<(String, String)>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Answer {
    /// Calibrated probability per option in the order given; for a noul,
    /// `[P(false), P(true)]`.
    pub probabilities: Vec<f64>,
    pub input_tokens: usize,
    /// An option, the question or the state was cut to fit the model's input.
    pub truncated: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Asked {
    pub answers: Vec<Answer>,
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
        let t = |i: usize| prompt::clamp_temperature(temps.get(i).and_then(Value::as_f64).unwrap_or(1.0));
        let temperature = [t(0), t(1), t(2)];
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
        let (_, pad) = special_token(&tk, &tcfg, "pad_token")?;

        let model = Model::load(dir, enc, head_layers, act_outputs)?;
        Ok(Self { model, tok: Tok(tk), special: Special { cls, sep, mask, mask_text }, pad, max_len, head_max_len, temperature, by_options })
    }

    /// Scores every question against `state` in one batched forward pass.
    pub fn ask(&self, state: &str, questions: &[Question]) -> R<Asked> {
        let rows: Vec<(&str, &Question)> = questions.iter().map(|q| (state, q)).collect();
        self.ask_rows(&rows)
    }

    /// Each row is its own state and question (one passage per row for the
    /// relevance check), scored in padded batches of at most `BATCH` rows.
    pub fn ask_rows(&self, rows: &[(&str, &Question)]) -> R<Asked> {
        const BATCH: usize = 8;
        let started = Instant::now();
        let mut answers = Vec::with_capacity(rows.len());
        for chunk in rows.chunks(BATCH) {
            let seqs = chunk
                .iter()
                .map(|(state, q)| {
                    let options = match q.kind {
                        Kind::Choice => q.options.iter().map(|(id, text)| prompt::render_option(id, text)).collect(),
                        Kind::Noul => prompt::noul_options(),
                    };
                    prompt::build(&self.tok, &self.special, q.kind, state, &q.instructions, &options, self.max_len, self.head_max_len)
                })
                .collect::<R<Vec<_>>>()?;
            let batch: Vec<_> = seqs.iter().zip(chunk).map(|(s, (_, q))| (s.ids.as_slice(), s.markers.as_slice(), q.kind.qtype())).collect();
            let logits = self.model.logits(&batch, self.pad)?;
            for ((seq, (_, q)), l) in seqs.iter().zip(chunk).zip(logits) {
                if l.iter().any(|v| !v.is_finite()) {
                    return Err("the decision model returned non-finite scores".into());
                }
                let bucket = prompt::bucket(q.kind, l.len());
                let t = self.by_options.iter().find(|(k, _)| *k == bucket).map_or(self.temperature[q.kind.qtype() as usize], |(_, t)| *t);
                answers.push(Answer { probabilities: prompt::probabilities(&l, t), input_tokens: seq.ids.len(), truncated: seq.truncated });
            }
        }
        Ok(Asked { answers, ms: started.elapsed().as_secs_f64() * 1e3 })
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
            .join(&*c.commit)
    }

    fn fixture() -> Value {
        serde_json::from_str(include_str!("../../tests/fixtures/laya/fixture.json")).unwrap()
    }

    /// Largest |Δp| between two distributions, after checking they pick the same option.
    fn close(got: &[f64], want: &Value, what: &str) -> f64 {
        let want: Vec<f64> = serde_json::from_value(want.clone()).unwrap();
        let argmax = |v: &[f64]| v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
        assert_eq!(argmax(got), argmax(&want), "{what}: same answer as laya-mlx");
        let worst = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        // FP16 vs laya-mlx FP32: measured ≤ 0.0015 on these fixtures; laya-mlx's own FP16 run differs by 0.005.
        assert!(worst < 0.006, "{what}: max |Δp| {worst} vs laya-mlx FP32");
        worst
    }

    fn run(id: &str) {
        let golden: Value = serde_json::from_str(&std::fs::read_to_string(format!("{}/tests/fixtures/laya/golden-{id}.json", env!("CARGO_MANIFEST_DIR"))).unwrap()).unwrap();
        let fx = fixture();
        let engine = Engine::load(&checkpoint_dir(id)).unwrap();
        let options: Vec<(String, String)> = fx["options"].as_array().unwrap().iter().map(|o| (o[0].as_str().unwrap().into(), o[1].as_str().unwrap().into())).collect();
        let (state, question) = (fx["state"].as_str().unwrap(), fx["question"].as_str().unwrap());
        let choice = Question { kind: Kind::Choice, instructions: question.into(), options: options.clone() };
        let stop = Question { kind: Kind::Noul, instructions: fx["stop"].as_str().unwrap().into(), options: vec![] };

        let rendered: Vec<String> = options.iter().map(|(i, t)| prompt::render_option(i, t)).collect();
        let seq = prompt::build(&engine.tok, &engine.special, Kind::Choice, state, question, &rendered, engine.max_len, engine.head_max_len).unwrap();
        let want_ids: Vec<u32> = serde_json::from_value(golden["ids"].clone()).unwrap();
        let want_markers: Vec<usize> = serde_json::from_value(golden["markers"].clone()).unwrap();
        assert_eq!(seq.ids, want_ids, "token ids match laya-mlx");
        assert_eq!(seq.markers, want_markers);

        let one = engine.ask(state, std::slice::from_ref(&choice)).unwrap();
        let worst = close(&one.answers[0].probabilities, &golden["probabilities_fp32"], "choice");
        // A batch pads the shorter noul row: the choice must come out the same.
        let both = engine.ask(state, &[choice.clone(), stop.clone()]).unwrap();
        let worst_b = close(&both.answers[0].probabilities, &golden["batch"]["choice_fp32"], "batched choice")
            .max(close(&both.answers[1].probabilities, &golden["batch"]["stop_fp32"], "batched noul"));
        assert!(both.answers[1].input_tokens < both.answers[0].input_tokens, "the noul row is shorter, so it was padded");

        // Latency: this port measured 9 ms (multilingual) and 20 ms (English) P50 on an M5 Max.
        let p50 = |qs: &[Question]| {
            let mut t: Vec<f64> = (0..25).map(|_| engine.ask(state, qs).unwrap().ms).collect();
            t.sort_by(f64::total_cmp);
            t[12]
        };
        // Relevance: one noul row per passage, each with its own state (mod.rs `laya_relevance`).
        let rel = &fx["relevance"];
        let noul = Question { kind: Kind::Noul, instructions: crate::laya::RELEVANT.into(), options: vec![] };
        assert_eq!(rel["statement"].as_str().unwrap(), crate::laya::RELEVANT);
        let states: Vec<String> = rel["passages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| crate::laya::passage_state(rel["request"].as_str().unwrap(), &crate::laya::LayaPassage { source: p[0].as_str().unwrap().into(), text: p[1].as_str().unwrap().into() }))
            .collect();
        let rows: Vec<(&str, &Question)> = states.iter().map(|s| (s.as_str(), &noul)).collect();
        let got: Vec<f64> = engine.ask_rows(&rows).unwrap().answers.iter().map(|a| a.probabilities[1]).collect();
        let want: Vec<f64> = serde_json::from_value(golden["relevance_fp32"].clone()).unwrap();
        let worst_r = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(worst_r < 0.006 && got[0] > got[1], "relevance {got:?} vs laya-mlx {want:?}");

        let (t1, t2) = (p50(std::slice::from_ref(&choice)), p50(&[choice, stop]));
        println!("{id}: P50 {t1:.1} ms (choice), {t2:.1} ms (choice + noul), {} tokens, max |Δp| {worst:.4} / batched {worst_b:.4} / relevance {worst_r:.4}", one.answers[0].input_tokens);
        assert!(t2 < 100.0, "a decision must take < 100 ms (P50 {t2:.1} ms)");
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

    /// The intent question the probe measures; `agent/loop.ts` would ask the same (item 12).
    const INTENT: &str = "What is the user doing with this request?";
    const INTENTS: [(&str, &str); 4] = [
        ("small_talk", "Greeting, thanks, sign-off or other small talk that needs no information"),
        ("about_assistant", "Asking about the assistant itself: who it is or what it can do"),
        ("kb_content", "Asking about the content of the knowledge base: its documents, code or facts"),
        ("follow_up", "Following up on the earlier conversation's topic"),
    ];

    /// Item 12 probe (docs/agentic-rag-improvements.md): can Laya tell small
    /// talk and questions to the assistant from lookups, on the agent's own
    /// step-1 state? Prints every request's intent scores and a confusion
    /// matrix; the tracker records the result. Not an assertion: a measurement.
    fn intent_probe(id: &str) {
        let engine = Engine::load(&checkpoint_dir(id)).unwrap();
        let cases: Value = serde_json::from_str(include_str!("../../../tests/fixtures/eval/cases.json")).unwrap();
        let kbs = [("docs", "document", 3, 17), ("code", "code", 4, 24), ("mixed", "mixed", 7, 40)];
        // (request, earlier turns as (role, text), kb, expected: lookup or not)
        type Row<'a> = (String, Vec<(String, String)>, &'a str, bool);
        let mut rows: Vec<Row> = vec![];
        for c in cases["cases"].as_array().unwrap() {
            let history = c["history"]
                .as_array()
                .map(|h| h.iter().map(|m| (m["role"].as_str().unwrap().to_string(), m["content"].as_str().unwrap().to_string())).collect())
                .unwrap_or_default();
            let kb = kbs.iter().find(|k| k.0 == c["kb"].as_str().unwrap()).unwrap().0;
            let lookup = !c["first"].as_array().unwrap().iter().any(|f| f == "answer_now");
            rows.push((c["prompt"].as_str().unwrap().into(), history, kb, lookup));
        }
        for p in [
            "hi", "hello there", "thanks!", "thank you so much", "ok cool", "bye", "good morning", "great, that helps", "nice one", "cheers mate",
            "who are you?", "what can you do?", "are you an AI?", "what model are you?", "how do you work?", "what's your name?", "can you help me?",
            "what are you able to answer?", "are you running locally?", "who made you?",
        ] {
            rows.push((p.into(), vec![], "docs", false));
        }
        let q = Question {
            kind: Kind::Choice,
            instructions: INTENT.into(),
            options: INTENTS.iter().map(|(i, t)| (i.to_string(), t.to_string())).collect(),
        };
        let states: Vec<String> = rows
            .iter()
            .map(|(p, h, kb, _)| {
                let (_, kind, files, nodes) = kbs.iter().find(|k| k.0 == *kb).unwrap();
                let mut s = format!("User request:\n{p}");
                if !h.is_empty() {
                    s += &format!("\n\nRecent conversation:\n{}", h.iter().map(|(r, c)| format!("{r}: {c}")).collect::<Vec<_>>().join("\n"));
                }
                s + &format!("\n\nKnowledge base: “{kb}”, a {kind} knowledge base ({files} files, {nodes} graph nodes).\n\nTool results so far: none.\n\nTool calls used: 0 of 4.")
            })
            .collect();
        let batch: Vec<(&str, &Question)> = states.iter().map(|s| (s.as_str(), &q)).collect();
        let asked = engine.ask_rows(&batch).unwrap();
        let names: Vec<&str> = INTENTS.iter().map(|(i, _)| *i).collect();
        let no_lookup = |p: &[f64]| p[0] + p[1];
        let (mut false_skip, mut caught, mut chat) = (vec![], 0, 0);
        println!("{id} intent probe ({} requests, {:.0} ms):", rows.len(), asked.ms);
        for ((p, _, _, lookup), a) in rows.iter().zip(&asked.answers) {
            let best = (0..names.len()).max_by(|x, y| a.probabilities[*x].total_cmp(&a.probabilities[*y])).unwrap();
            let skip = no_lookup(&a.probabilities);
            println!(
                "  {:>6} {:<16} P(no lookup) {:.2} · {}  {p:?}",
                if *lookup { "lookup" } else { "chat" },
                names[best],
                skip,
                a.probabilities.iter().map(|v| format!("{v:.2}")).collect::<Vec<_>>().join(" ")
            );
            if *lookup && skip >= 0.5 {
                false_skip.push(p.clone());
            }
            if !*lookup {
                chat += 1;
                if skip >= 0.5 {
                    caught += 1;
                }
            }
        }
        println!("{id}: no-lookup requests caught {caught}/{chat}; lookups wrongly skipped {} {false_skip:?}", false_skip.len());
    }

    #[test]
    #[ignore = "needs the multilingual checkpoint; a probe that prints"]
    fn laya_multilingual_intent_probe() {
        intent_probe("laya-multilingual");
    }

    #[test]
    #[ignore = "needs the English checkpoint; a probe that prints"]
    fn laya_en_intent_probe() {
        intent_probe("laya-en");
    }

    /// Item 13 probe (docs/agentic-rag-improvements.md): does Laya tell a
    /// cited sentence's own passage from a passage cited for another question?
    /// Prints the AUC and, per threshold, how many own passages it would flag
    /// (false alarms) and how many others it would catch. A measurement.
    fn claims_probe(id: &str) {
        let engine = Engine::load(&checkpoint_dir(id)).unwrap();
        let fx: Value = serde_json::from_str(include_str!("../../tests/fixtures/laya/claims.json")).unwrap();
        let rows = fx["rows"].as_array().unwrap();
        let states: Vec<String> = rows
            .iter()
            .map(|r| crate::laya::claim_state(&serde_json::from_value(serde_json::json!({ "statement": r["sentence"], "source": r["source"], "text": r["text"] })).unwrap()))
            .collect();
        let q = Question { kind: Kind::Noul, instructions: crate::laya::SUPPORTS.into(), options: vec![] };
        let batch: Vec<(&str, &Question)> = states.iter().map(|s| (s.as_str(), &q)).collect();
        let asked = engine.ask_rows(&batch).unwrap();
        let scored: Vec<(f64, bool)> = asked.answers.iter().zip(rows).map(|(a, r)| (a.probabilities[1], r["own"].as_bool().unwrap())).collect();
        let (pos, neg): (Vec<f64>, Vec<f64>) = (
            scored.iter().filter(|x| x.1).map(|x| x.0).collect(),
            scored.iter().filter(|x| !x.1).map(|x| x.0).collect(),
        );
        let wins: f64 = pos.iter().map(|p| neg.iter().map(|n| if p > n { 1.0 } else if p == n { 0.5 } else { 0.0 }).sum::<f64>()).sum();
        let auc = wins / (pos.len() * neg.len()) as f64;
        println!("{id} claim probe: {} rows in {:.0} ms, AUC {auc:.3}", rows.len(), asked.ms);
        for (r, (p, own)) in rows.iter().zip(&scored) {
            if *own && *p < 0.1 {
                println!("  own passage below 0.10 ({p:.3}): {:?} ← {}", r["sentence"].as_str().unwrap(), r["source"].as_str().unwrap());
            }
        }
        for t in [0.05, 0.1, 0.2, 0.3, 0.5] {
            let flagged_own = pos.iter().filter(|p| **p < t).count();
            let caught = neg.iter().filter(|n| **n < t).count();
            println!("  flag below {t:.2}: own passages flagged {flagged_own}/{}, other passages caught {caught}/{}", pos.len(), neg.len());
        }
    }

    #[test]
    #[ignore = "needs the multilingual checkpoint; a probe that prints"]
    fn laya_multilingual_claims_probe() {
        claims_probe("laya-multilingual");
    }

    #[test]
    #[ignore = "needs the English checkpoint; a probe that prints"]
    fn laya_en_claims_probe() {
        claims_probe("laya-en");
    }
}
