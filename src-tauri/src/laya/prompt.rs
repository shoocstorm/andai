//! Laya's input layout and calibration, ported from laya-mlx
//! `laya_mlx/common.py` (upstream NandhaKishorM/laya @ 573e5b6, Apache-2.0):
//!
//! `[CLS] choice question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 … [SEP] <state> [SEP]`
//!
//! Each option is scored at its `[MASK]` marker. Options share a
//! `head_max_len` token budget; the state gets whatever room is left of
//! `max_len` and is cut from the end, as upstream does.

/// Just what the layout needs from a tokenizer (no special tokens added).
pub trait Encode {
    fn encode(&self, text: &str) -> Result<Vec<u32>, String>;
}

#[derive(Debug, Clone)]
pub struct Special {
    pub cls: u32,
    pub sep: u32,
    pub mask: u32,
    /// The mask token's text, removed from all input so text can't add markers.
    pub mask_text: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Sequence {
    pub ids: Vec<u32>,
    /// Position of each option's `[MASK]`, in option order.
    pub markers: Vec<usize>,
    /// Something was cut to fit: an option, the instructions or the state.
    pub truncated: bool,
}

/// Upstream caps each option at this many tokens after its marker.
const OPTION_TOKENS: usize = 48;

/// `id: text`, as upstream renders a choice criterion with a description.
pub fn render_option(id: &str, text: &str) -> String {
    if text.is_empty() {
        id.to_string()
    } else {
        format!("{id}: {text}")
    }
}

pub fn build(
    tok: &impl Encode,
    sp: &Special,
    state: &str,
    instructions: &str,
    options: &[String],
    max_len: usize,
    head_max_len: usize,
) -> Result<Sequence, String> {
    let clean = |s: &str| s.replace(&sp.mask_text, " ");
    let mut truncated = false;

    let head = tok.encode(&format!("choice question: {}", clean(instructions)))?;
    let mut opts = Vec::with_capacity(options.len());
    for o in options {
        let mut ids = tok.encode(&format!(" {}", clean(o)))?;
        if ids.len() > OPTION_TOKENS {
            ids.truncate(OPTION_TOKENS);
            truncated = true;
        }
        let mut v = vec![sp.mask];
        v.extend(ids);
        opts.push(v);
    }
    let used = |opts: &[Vec<u32>]| opts.iter().map(Vec::len).sum::<usize>() as i64;
    let mut budget = head_max_len as i64 - used(&opts);
    if budget < 16 {
        // Too many options for the budget: every option gets an equal share.
        let per = 4.max((head_max_len as i64 - 16) / opts.len().max(1) as i64) as usize;
        for o in &mut opts {
            if o.len() > per {
                o.truncate(per);
                truncated = true;
            }
        }
        budget = head_max_len as i64 - used(&opts);
    }
    let keep = budget.max(8) as usize;
    truncated |= head.len() > keep;

    let mut ids = vec![sp.cls];
    ids.extend(head.into_iter().take(keep));
    ids.push(sp.sep);
    let mut markers = Vec::with_capacity(opts.len());
    for o in opts {
        markers.push(ids.len());
        ids.extend(o);
    }
    ids.push(sp.sep);

    let room = max_len.saturating_sub(ids.len() + 1);
    let state_ids = tok.encode(&clean(state))?;
    truncated |= state_ids.len() > room;
    ids.extend(state_ids.into_iter().take(room));
    ids.push(sp.sep);
    if ids.len() > max_len {
        ids.truncate(max_len);
        truncated = true;
    }
    markers.retain(|&m| m < max_len);
    if markers.len() != options.len() {
        return Err(format!("{} options don't fit the decision model's {max_len}-token input.", options.len()));
    }
    Ok(Sequence { ids, markers, truncated })
}

/// A fitted temperature below 1 sharpens instead of softening (the English
/// checkpoint's `choice:11+` is 0.1006), so like upstream v0.3.5 it's clamped.
pub fn clamp_temperature(t: f64) -> f64 {
    if t.is_finite() {
        t.clamp(0.5, 5.0)
    } else {
        1.0
    }
}

/// The calibration bucket for a choice with `k` options.
pub fn choice_bucket(k: usize) -> &'static str {
    match k {
        0..=2 => "choice:2",
        3..=5 => "choice:3-5",
        6..=10 => "choice:6-10",
        _ => "choice:11+",
    }
}

/// Softmax of `logits / temperature`.
pub fn probabilities(logits: &[f32], temperature: f64) -> Vec<f64> {
    let z: Vec<f64> = logits.iter().map(|&l| l as f64 / temperature).collect();
    let max = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = z.iter().map(|v| (v - max).exp()).collect();
    let total: f64 = exps.iter().sum();
    exps.iter().map(|e| e / total).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One token per character: lengths are easy to reason about.
    struct Chars;
    impl Encode for Chars {
        fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
            Ok(text.chars().map(|c| c as u32).collect())
        }
    }
    const CLS: u32 = 1;
    const SEP: u32 = 2;
    const MASK: u32 = 4;
    fn sp() -> Special {
        Special { cls: CLS, sep: SEP, mask: MASK, mask_text: "<mask>".into() }
    }
    fn text(ids: &[u32]) -> String {
        ids.iter().map(|&i| char::from_u32(i).unwrap()).collect()
    }

    #[test]
    fn lays_out_question_options_and_state() {
        let opts = vec![render_option("a", "first"), render_option("b", "")];
        let s = build(&Chars, &sp(), "ST", "Q?", &opts, 512, 192).unwrap();
        let head = "choice question: Q?";
        assert_eq!(s.ids[0], CLS);
        assert_eq!(text(&s.ids[1..1 + head.len()]), head);
        assert_eq!(s.ids[1 + head.len()], SEP);
        let m0 = head.len() + 2;
        assert_eq!(s.markers, vec![m0, m0 + 1 + " a: first".len()]);
        assert_eq!(s.ids[m0], MASK);
        assert_eq!(text(&s.ids[m0 + 1..s.markers[1]]), " a: first");
        assert_eq!(text(&s.ids[s.markers[1] + 1..s.markers[1] + 3]), " b");
        assert_eq!(&s.ids[s.ids.len() - 4..], &[SEP, 'S' as u32, 'T' as u32, SEP]);
        assert!(!s.truncated);
    }

    #[test]
    fn text_cannot_add_markers() {
        let s = build(&Chars, &sp(), "x<mask>y", "a<mask>b", &["o<mask>".into(), "p".into()], 512, 192).unwrap();
        assert_eq!(s.ids.iter().filter(|&&i| i == MASK).count(), 2, "only the two option markers");
        assert!(text(&s.ids).contains("x y"));
    }

    #[test]
    fn long_options_are_cut_to_48_tokens_and_flagged() {
        let long = "z".repeat(100);
        let s = build(&Chars, &sp(), "", "q", &[long, "b".into()], 512, 192).unwrap();
        assert_eq!(s.markers[1] - s.markers[0], 1 + 48);
        assert!(s.truncated);
    }

    #[test]
    fn many_options_share_the_budget_equally() {
        // 10 options of 30 tokens + marker overflow a 192 budget: (192 - 16) / 10 = 17 each
        let opts: Vec<String> = (0..10).map(|i| format!("{i}{}", "y".repeat(29))).collect();
        let s = build(&Chars, &sp(), "", "q", &opts, 1024, 192).unwrap();
        assert!(s.markers.windows(2).all(|w| w[1] - w[0] == 17));
        assert!(s.truncated);
    }

    #[test]
    fn the_state_fills_the_room_left_and_is_cut_from_the_end() {
        let s = build(&Chars, &sp(), &"s".repeat(100), "q", &["a".into(), "b".into()], 60, 32).unwrap();
        assert_eq!(s.ids.len(), 60);
        assert_eq!(*s.ids.last().unwrap(), SEP);
        assert!(s.truncated);
    }

    #[test]
    fn options_that_dont_fit_the_input_are_an_error() {
        let opts: Vec<String> = (0..8).map(|i| format!("{i}")).collect();
        assert!(build(&Chars, &sp(), "", &"q".repeat(40), &opts, 12, 200).unwrap_err().contains("don't fit"));
    }

    #[test]
    fn calibration_clamps_and_buckets() {
        assert_eq!(clamp_temperature(0.1006), 0.5);
        assert_eq!(clamp_temperature(1.76), 1.76);
        assert_eq!(clamp_temperature(9.0), 5.0);
        assert_eq!(clamp_temperature(f64::NAN), 1.0);
        assert_eq!(choice_bucket(2), "choice:2");
        assert_eq!(choice_bucket(9), "choice:6-10");
        assert_eq!(choice_bucket(11), "choice:11+");
        let p = probabilities(&[2.0, 1.0, 0.0], 2.0);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        let q = probabilities(&[2.0, 1.0, 0.0], 1.0);
        assert!(p[0] < q[0], "a higher temperature softens");
    }
}
