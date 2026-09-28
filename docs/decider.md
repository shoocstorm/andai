# The decider: how a regular model picks the agent's next action

Andai's agent loop needs a decision at every step: which tool to run next,
whether to answer now, whether to ask a clarifying question. Small local models
can't be trusted to write that choice free-form (they name tools that don't
exist, hedge, or break the format), so the model never writes the decision at
all. It picks a **letter** from a list, and one forward pass is read out as a
probability per option.

The same interface has two backends: the **Laya** decision checkpoint (a small
encoder served by Rust on Apple Silicon, ~8–20 ms per decision) or **any
regular chat model** the app can run — Qwen3 0.6B on wllama in the webview, or
the same model natively on MLX. This document is about the regular-model path:
the technique, the exact prompt and readout, the measured facts it stands on,
and how to port it to another project. The technique is SemIf-style, after the
sibling `../SemIf` project (`webgpu-demo/worker.js`).

Everything here lives in `src/llm/decide.ts` (the mechanism), `src/llm/engine.ts`
(the model slot) and `src/agent/loop.ts` (the consumer). Function names below
refer to those files.

---

## The idea

- The model gets a **state** (the user's request, the knowledge base, the tool
  results so far), a **question**, and 2–16 **lettered options**.
- It replies with exactly one option letter (`A`–`P`), held to that by a GBNF
  grammar on the sampled token.
- The *unsampled* next-token distribution still scores **every** letter: the
  first token's `top_logprobs` are read, a softmax over just the letters'
  logprobs gives a probability per option, and the argmax is the choice.

That yields, in one forward pass: an always-valid choice, a probability for
every option (not only the winner), and a confidence signal the loop can gate
on and the user can inspect in the Execution Trace.

## The decision prompt

`decisionMessages()` builds two messages, nothing else:

```
system: Make the requested decision from the supplied state. Follow the output
        format exactly.

user:   State:
        <the situation: request, knowledge base, observations so far>

        Question:
        What should the assistant do next to fulfil the user's request?

        Allowed options:
        A. Answer without using the knowledge base: …
        B. Search the knowledge base for passages matching a query …
        C. …
        
        Reply with exactly one option letter from: A, B, C.
```

The wording of the options is **part of the mechanism, not cosmetic**.
Measured with Qwen3 0.6B (5 requests × 3 option orders): a broad *answer*
option mentioning "general knowledge" won 9/9 knowledge questions, and the
narrower wording now in `agent/loop.ts` (`PSEUDO`) scored 15/15 with *answer*
first, 12/15 as the last option, 11/15 shuffled. Rewording options for a new
project is a measured change, not an edit: run an eval before and after.

## Reading the answer out of one token

One non-streaming completion with these parameters (`decide()`):

| Parameter | Value | Why |
|---|---|---|
| `max_tokens` | `1` | Only the first token's distribution is needed. |
| `temperature` | `1` | Neutral sampling, so the logprobs are a plain readout of the logits. |
| `top_k` / `top_p` | `0` / `1` | Same: no truncated sampling distribution. |
| `logprobs` | `true` | Ask for the readout. |
| `top_logprobs` | `20` | Room for every letter even when a few other tokens rank among them. |
| `grammar` | `root ::= "A" | "B" | …` | GBNF: keeps the *sampled* token a valid letter. |
| `cache_prompt` | `true` on a dedicated decider slot | Consecutive decisions share most of their prompt, so the KV cache is reused. The chat model alternates decisions with answers and would evict it. |
| `chat_template_kwargs` | `{ enable_thinking: false }` | Reasoning must not consume the one token. |

Then `optionLogprobs()` reads the answer:

1. Take the first content token's `top_logprobs` list.
2. For each option letter, find its entry (by token text or single-byte value).
3. Softmax over the letters' logprobs only → a probability per option.
4. A letter that isn't listed **cannot be treated as zero**: it scored at most
   the lowest listed logprob. It gets that floor value and is reported as
   `bounded` (an upper bound) in the decision record.
5. If **no** letter is listed at all, that is no decision: it throws, and the
   agent loop falls back (below).

The probabilities are an **uncalibrated ranking signal**, not a confidence in
the statistical sense (SemIf `docs/CALIBRATION.md`). Andai treats them that
way: the trace shows them, and a threshold (`minConfidence`) triggers a safe
fallback rather than being read as "68% sure".

### When a decision fails

Any of these throws `DecisionError` (carrying the full request/response for
the trace): the call failed, no letter appeared in the readout, or the chosen
option wasn't offered. The loop's fallback keeps the turn useful:
**if nothing has been looked up yet and a search tool exists, do one plain
search (the fixed pipeline's behavior); otherwise answer from what's known**
(`fallbackAction()` in `agent/loop.ts`).

## Where it runs: the decider slot

A dedicated decision model is **optional** (`loadDecider()` in
`src/llm/engine.ts`):

- With one loaded, `slotFor('decider')` returns the decider instance: a second
  wllama instance (GGUF, context capped at 4096 — decisions are short prompts)
  or the native MLX engine's `decider` slot on Apple Silicon. It only ever sees
  decisions, so `cache_prompt` pays off.
- Without one, `slotFor` falls back to the **chat model**: decisions still
  work, they just evict the chat model's cache between answers.
- Models that may serve as deciders are marked `decider: true` in
  `src/llm/models.ts` (today: Qwen3 0.6B and 1.7B, GGUF and MLX). Measured in
  the agent eval: Qwen3 0.6B deciding while Qwen3 1.7B answered was the best
  wllama pairing; larger deciders chose worse.
- Routing to Laya happens first (`deciderLaya()`): when the loaded decider is
  a Laya checkpoint, `decide()` sends the state to Rust instead and reads
  calibrated scores — no letters, no logprobs. A yes/no *stop* question and
  enum-argument choices ride along in the same pass; regular deciders ignore
  both.

Cost of a decision (measured): ~0.7–0.9 s with Qwen3 0.6B on wllama,
23–38 ms on the native MLX engine, 8–20 ms on Laya.

## The flow in the agent loop

```
each step of runAgent (src/agent/loop.ts):
│
├─ searchFirst? ── the request plainly asks about content and nothing was
│                  looked up yet → run the search tool WITHOUT a decision
│                  (measured: the decision model picked search for 23 of 24
│                  such requests, so the decision cost ~0.7 s for nothing)
│
├─ build the option list (decisionOptions):
│    answer_now is always option A, ask_clarification is always last,
│    the tools between them are shuffled with a seeded shuffle
│    (seededShuffle, mulberry32; the seed goes into the trace so a run
│    replays — option ORDER moves small models' choices: SemIf measured
│    reversing it flipped 10 of 36)
│
├─ filter the tools first (offered()):
│    a tool that already returned results is not offered again — with
│    kb_search still offered after a search that found passages, Qwen3 0.6B
│    searched again 9/9 times; without it, it answered 9/9. A tool that
│    failed or came back empty gets one more try.
│
├─ decide(state, QUESTION, options)          ← the mechanism above
│
├─ trust gates, in order:
│    chose something not offered      → DecisionError → fallback
│    confidence < minConfidence       → fallback (search-once-then-answer)
│    decision failed                  → fallback
│
├─ tool-shaped guards (both measured, see AGENTS.md §2):
│    a symbol tool chosen before any symbol was seen → Find symbols/search first
│    Read lines chosen before any line range existed → search first
│
├─ fill arguments with the CHAT model (never the decider): JSON held to the
│  tool's schema by a grammar, values constrained to files/symbols/ranges
│  already seen
│
├─ policy gate (Auto / Ask) → run the tool → observe
│
└─ loop, until the decision is answer/clarify or the step budget runs out
```

Every decision — all options with their probabilities, the seed, the exact
messages sent and the raw logprob readout — is written to the message's
`agent` record and shown in the Execution Trace, so an odd decision can be
inspected and copied.

## Key code

| Piece | Where |
|---|---|
| Prompt, readout, softmax, seeded shuffle, `decide()` router, `DecisionError` | `src/llm/decide.ts` |
| Decider slot: `loadDecider`, `unloadDecider`, `slotFor`, `complete` (wllama or native MLX) | `src/llm/engine.ts` |
| Consumer: `decisionOptions`, `offered`, `PSEUDO` wording, `QUESTION`, fallbacks, guards | `src/agent/loop.ts` |
| Catalog flag for which models may decide | `src/llm/models.ts` (`decider: true`) |
| Tests: letter readout, bounded letters, Laya routing, fallback behavior | `src/llm/decide.test.ts`, `src/agent/turn.test.ts` |
| Origin of the technique (standalone demo) | `../SemIf/webgpu-demo/worker.js` |

## Measured facts this design stands on

Recorded in [AGENTS.md §2](../AGENTS.md#2-architecture) and in code comments;
each one was probed, not assumed. If you port the design, re-measure on your
stack:

- **wllama's chat logprobs are the raw next-token distribution** (wllama 3.6.1,
  measured in the app): every option letter shows up in `top_logprobs` when the
  prompt asks for one; a logit bias doesn't change the reported values; and
  `post_sampling_probs: true` makes the reply carry **no** `top_logprobs`.
- **This wllama build can't turn a JSON Schema into a grammar**
  (`response_format: json_schema` fails with "Failed to initialize samplers"),
  but a GBNF `grammar` string works.
- **Option order and wording move small models' choices** (SemIf: reversing
  flipped 10 of 36; the Qwen3 0.6B numbers above). Hence the seeded shuffle,
  the fixed A-for-answer slot, and eval runs before any rewording.
- **Removing options moves choices too.** Hiding tools until their inputs
  existed made the model answer without searching (12 of 16 code questions),
  so tools are filtered by *usage* instead, and guards redirect rather than
  hide.
- **Web Crypto/wllama aside, this needs no special runtime** — only a
  completion backend that exposes first-token logprobs and (ideally) supports a
  grammar constraint. See the portability notes below.

## Reusing it in another project

The mechanism is deliberately separable; only a thin ring of it is Andai-specific.

**Portable as-is** (pure TypeScript, zero Andai imports): `labelsFor`,
`decisionMessages`, `softmax`, `optionLogprobs`, `seededShuffle`, and the shape
of `decide()`. Copy them or lift them into a small library.

**The one seam to replace**: `decide()` calls `complete(target.slot, …)` and
`slotFor('decider')` from `src/llm/engine.ts`, and routes to Laya first. Swap
those ~10 lines for any OpenAI-compatible completion call and you have a
standalone `decide()`.

**Backend requirements**:

- first-token `top_logprobs` (OpenAI-style `logprobs`/`top_logprobs`) — required;
- a grammar/constrained-decoding knob (GBNF on the llama.cpp family, including
  wllama; Andai's native MLX engine implements the same for choice grammars in
  `src-tauri/src/llm/grammar.rs`) — optional but recommended: it only
  guarantees the *sampled* token, the readout itself works from `top_logprobs`
  even without it.

**Carry the know-how with the code** (this is the part that's easy to skip and
expensive to relearn): neutral sampling parameters, `enable_thinking: false`,
`top_logprobs` ≥ 20 headroom, the `bounded`-letter handling, the seeded option
shuffle, measured option wording, the min-confidence fallback instead of
trusting the probability as a calibrated confidence, and *never* letting the
decider write tool arguments — that stays with a grammar-held chat completion.

---

## Appendix: implementation prompt for an AI agent

Copy everything inside the fence below into an AI coding agent to implement
this mechanism from scratch in a new project. It is self-contained: fill in the
two bracketed placeholders first.

````markdown
Implement a choice-based decision mechanism ("decider") in this project,
after the SemIf technique, as described below. Do not improvise the
parameters or the prompt format: they are measured, and each deviation
listed under "Pitfalls" is a known, observed failure mode.

## Goal

`decide(state, question, options)` lets a small local LLM pick one of 2–16
options and returns a probability for EVERY option, in one forward pass.
The model never writes free-form decisions: it "chooses a letter" and the
answer is read out of the first token's logprob distribution.

## Prompt (exact)

Two messages:

- system: `Make the requested decision from the supplied state. Follow the output format exactly.`
- user:
  ```
  State:
  {state}

  Question:
  {question}

  Allowed options:
  A. {option text}
  B. {option text}
  …

  Reply with exactly one option letter from: A, B, …
  ```

Labels are `A`–`P` (uppercase). Building this prompt is a pure function.

## Completion parameters (exact)

One non-streaming chat completion:

- `max_tokens: 1`
- `temperature: 1`, `top_k: 0`, `top_p: 1` — neutral sampling so logprobs are
  the plain next-token distribution
- `logprobs: true`, `top_logprobs: 20` — headroom so every letter is listed
  even when other tokens rank among them
- if the backend supports grammars (GBNF, llama.cpp family):
  `grammar: root ::= "A" | "B" | …` — constrains only the SAMPLED token
- if the model has a thinking/reasoning mode: disable it (it must not spend
  the one token)
- if the backend supports prompt caching and this call goes to a slot that
  only ever sees decisions: enable it (consecutive decisions share most of
  their prompt)

## Readout (exact)

1. From the first content token, take `top_logprobs`
   (list of `{token, logprob, bytes?}`).
2. For each option letter, find its entry by token text or single-byte value
   (`bytes[0] === charCode`).
3. softmax over the found logprobs ONLY → probability per option.
4. A letter missing from the list is NOT zero: assign it the LOWEST logprob
   found in the list (an upper bound) and flag it as `bounded`.
5. If NO letter is found at all: throw a decision error carrying the exact
   request and the raw readout (for debugging/trace UI). The caller must
   have a fallback.

Return: options with `{probability, logprob}`, the argmax `chosen`, its
`confidence`, `bounded`, the model name, elapsed ms, prompt token count,
and the request/response pair for tracing.

## Shuffling

Option ORDER affects small models' choices (measured: reversing the order
flipped 10 of 36 decisions). Shuffle the middle options with a seeded
PRNG (e.g. mulberry32), keep a designated first option (e.g. "answer
without tools") pinned as A and a last option (e.g. "ask for
clarification") pinned last, and record the seed with the decision so runs
replay. Derive per-step seeds from one per-turn seed.

## Consumer pattern (how a tool-using agent should use it)

- Ask ONE question per step: "What should the assistant do next?", options =
  the available tools + answer/clarify.
- Filter the offered tools first: a tool that already returned results is
  removed from the options (measured: with the search tool still offered
  after a productive search, Qwen3 0.6B searched again 9/9 times; with it
  removed it answered 9/9). A tool that failed or returned nothing may be
  retried once.
- Treat `confidence` as an UNCALIBRATED ranking signal. If the chosen tool's
  confidence is below a configurable `minConfidence`, do not run it: fall
  back to a fixed, safe behavior (e.g. one plain search if nothing has been
  looked up yet, otherwise answer from what is known).
- If the decision fails or picks an option that wasn't offered, use the same
  fallback. Never loop on a failed decision.
- The DECIDER only picks actions. Tool arguments are written separately by a
  chat completion held to the tool's JSON schema (grammar-constrained when
  possible), and are validated in code before use.
- Keep the wording of options stable and MEASURED: small models are highly
  sensitive to it. When you change wording or option sets, re-run an eval
  before and after and compare.

## Pitfalls (measured failure modes — avoid each explicitly)

1. `post_sampling_probs: true` (wllama/llama.cpp) makes the response carry NO
   `top_logprobs`. Don't set it.
2. A logit bias on the letter tokens does not change the REPORTED logprobs in
   wllama. Don't rely on it.
3. `response_format: json_schema` fails on some wllama builds ("Failed to
   initialize samplers"). Use a GBNF `grammar` string instead.
4. Temperature/top-k/top-p other than neutral distort the readout (the
   reported distribution is pre-sampling, but keep the sampled token honest).
5. Thinking mode on: the first token is `<think>`, the readout is garbage.
6. Treating an absent letter as probability 0 overstates certainty; use the
   bounded floor and report it.
7. Trusting the probability as calibrated confidence; gate with a threshold
   and a safe fallback instead.

## Engine requirements

Any OpenAI-compatible completion backend with first-token `top_logprobs`.
A grammar/constrained-decoding knob is optional but recommended. Know your
backend's logprob semantics before trusting the readout: verify that all
letters appear in `top_logprobs` for a decision-shaped prompt, and record
the finding (model, backend version, parameters).

## Tests to ship

1. Prompt building: pure function, exact format, labels match option count.
2. Readout: given a fixture `top_logprobs` payload, probabilities match a
   hand-computed softmax; a missing letter gets the floor value and is
   flagged `bounded`; no letters at all throws.
3. Seeded shuffle: deterministic for a seed, first/last options pinned,
   all permutations keep every option.
4. End-to-end with the real (or a mocked) backend: a decision returns valid
   probabilities over all options; a failed call surfaces the fallback path.
````

---

Questions about this document or the mechanism: see
[AGENTS.md §2](../AGENTS.md#2-architecture) for the platform facts and
[development.md](development.md) for how a turn works end to end.
