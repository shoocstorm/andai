# Goldens for the src-tauri native LLM parity tests (llm/): the Qwen3 chat
# template as transformers renders it, its token ids, and mlx-lm's greedy
# continuation and first-token log-probabilities.
# Run with a venv that has mlx-lm and transformers:
#   python golden.py <checkpoint dir> golden-<id>.json
import json, sys
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache
from transformers import AutoTokenizer

ckpt, out = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(ckpt)
model, _ = load(ckpt)

SYSTEM = "You are Andai, a local assistant. Answer from the knowledge base when it helps."
cases = {
    "system_user_nothink": ([{"role": "system", "content": SYSTEM}, {"role": "user", "content": "How long are refunds valid for a cancelled ferry?"}], False),
    "system_user_think": ([{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Hi!"}], True),
    "history": (
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "What is Andai?"},
            {"role": "assistant", "content": "<think>\nThe user asks about the app.\n</think>\n\nAndai is a local-first agent."},
            {"role": "user", "content": "And what does it add for a vehicle? Ünïcödé ✓ 日本"},
        ],
        False,
    ),
    "user_only": ([{"role": "user", "content": "  leading and trailing spaces  "}], False),
    "decision": (
        [
            {"role": "system", "content": "Make the requested decision from the supplied state. Follow the output format exactly."},
            {"role": "user", "content": "State:\nUser request: What is the refund window for weather cancellations?\nKnowledge base: documents.\n\nQuestion:\nWhat should the agent do next?\n\nAllowed options:\nA. Answer now\nB. Search the knowledge base\nC. Ask a clarifying question\n\nReply with exactly one option letter from: A, B, C."},
        ],
        False,
    ),
}

golden = {"source": f"transformers + mlx-lm ({ckpt.rstrip('/').split('/')[-3] if '/snapshots/' in ckpt else ckpt})", "cases": {}}
for name, (messages, think) in cases.items():
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=think)
    ids = tok.encode(text, add_special_tokens=False)
    entry = {"messages": messages, "enable_thinking": think, "text": text, "ids": ids}
    if name in ("system_user_nothink", "decision"):
        prompt = mx.array(ids)
        greedy, first = [], None
        for (t, logprobs), _ in zip(generate_step(prompt, model, max_tokens=24), range(24)):
            if first is None:
                first = logprobs
            greedy.append(int(t))
            if int(t) in (151645, 151643):
                break
        entry["greedy"] = greedy
        entry["greedy_text"] = tok.decode(greedy)
        letters = {l: tok.encode(l, add_special_tokens=False)[0] for l in "ABC"}
        entry["letter_ids"] = letters
        # First-token log-probabilities in float32, on the Rust engine's path:
        # every prompt token but the last in one pass, then the last. (mlx-lm's
        # own `first` is bf16, rounded to 0.25 at these magnitudes, and the
        # 4-bit model's values move with how the prompt is split.)
        cache = make_prompt_cache(model)
        model(prompt[None, :-1], cache=cache)
        logits = model(prompt[None, -1:], cache=cache)[0, -1].astype(mx.float32)
        lp = logits - mx.logsumexp(logits)
        entry["first_logprobs"] = {l: float(lp[i]) for l, i in letters.items()}
        top = mx.argsort(-lp)[:5].tolist()
        entry["first_top5"] = [[int(i), float(lp[i])] for i in top]
    golden["cases"][name] = entry

json.dump(golden, open(out, "w"), indent=1, ensure_ascii=False)
for n, e in golden["cases"].items():
    print(n, len(e["ids"]), repr(e.get("greedy_text", ""))[:80])
