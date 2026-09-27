//! Qwen3's chat template (tokenizer_config.json `chat_template`), ported for
//! the messages Andai sends: system, user and assistant, no tools. Checked
//! against transformers' `apply_chat_template` in the golden fixtures
//! (tests/fixtures/llm/golden.py). wllama renders the same Jinja template
//! from the GGUF, so both engines see the same prompt text.

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

/// An assistant message's `(reasoning, answer)`, split at the last `</think>`
/// the way the template does it.
fn split_think(content: &str) -> (String, String) {
    match content.rfind("</think>") {
        None => (String::new(), content.to_string()),
        Some(_) => {
            let answer = content.split("</think>").last().unwrap_or("").trim_start_matches('\n');
            let before = content.split("</think>").next().unwrap_or("").trim_end_matches('\n');
            let reasoning = before.split("<think>").last().unwrap_or("").trim_start_matches('\n');
            (reasoning.to_string(), answer.to_string())
        }
    }
}

/// The prompt text for `messages`, ending with the assistant turn to write.
/// With `thinking` off, that turn starts with an empty think block.
pub fn render(messages: &[Message], thinking: bool) -> String {
    let mut out = String::new();
    // The last user message: assistant turns after it keep their reasoning.
    let last_query = messages.iter().rposition(|m| m.role == Role::User).unwrap_or(messages.len().saturating_sub(1));
    for (i, m) in messages.iter().enumerate() {
        match m.role {
            Role::System if i == 0 => {
                out.push_str("<|im_start|>system\n");
                out.push_str(&m.content);
                out.push_str("<|im_end|>\n");
            }
            Role::User | Role::System => {
                out.push_str(if m.role == Role::User { "<|im_start|>user\n" } else { "<|im_start|>system\n" });
                out.push_str(&m.content);
                out.push_str("<|im_end|>\n");
            }
            Role::Assistant => {
                let (reasoning, answer) = split_think(&m.content);
                out.push_str("<|im_start|>assistant\n");
                let last = i + 1 == messages.len();
                if i > last_query && (last || !reasoning.is_empty()) {
                    out.push_str("<think>\n");
                    out.push_str(reasoning.trim_matches('\n'));
                    out.push_str("\n</think>\n\n");
                    out.push_str(answer.trim_start_matches('\n'));
                } else {
                    out.push_str(&answer);
                }
                out.push_str("<|im_end|>\n");
            }
        }
    }
    out.push_str("<|im_start|>assistant\n");
    if !thinking {
        out.push_str("<think>\n\n</think>\n\n");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn golden(id: &str) -> Value {
        let path = format!("{}/tests/fixtures/llm/golden-{id}.json", env!("CARGO_MANIFEST_DIR"));
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    #[test]
    fn matches_transformers_on_every_golden_case() {
        for id in ["qwen3-1.7b", "qwen3-0.6b"] {
            let g = golden(id);
            for (name, case) in g["cases"].as_object().unwrap() {
                let messages: Vec<Message> = serde_json::from_value(case["messages"].clone()).unwrap();
                let got = render(&messages, case["enable_thinking"].as_bool().unwrap());
                assert_eq!(got, case["text"].as_str().unwrap(), "{id} {name}");
            }
        }
    }

    #[test]
    fn earlier_reasoning_is_dropped_from_history() {
        let m = |role, content: &str| Message { role, content: content.into() };
        let text = render(&[m(Role::User, "q1"), m(Role::Assistant, "<think>\nhmm\n</think>\n\na1"), m(Role::User, "q2")], true);
        assert_eq!(text, "<|im_start|>user\nq1<|im_end|>\n<|im_start|>assistant\na1<|im_end|>\n<|im_start|>user\nq2<|im_end|>\n<|im_start|>assistant\n");
    }

    #[test]
    fn a_later_system_message_is_rendered_in_place() {
        let m = |role, content: &str| Message { role, content: content.into() };
        let text = render(&[m(Role::User, "q"), m(Role::System, "note")], false);
        assert!(text.starts_with("<|im_start|>user\nq<|im_end|>\n<|im_start|>system\nnote<|im_end|>\n"));
    }
}
