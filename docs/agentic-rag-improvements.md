# Agentic RAG improvements: tracker

Planned work to make the agent loop (`src/agent/loop.ts`) more accurate and
faster. Items are done **one at a time, in order**, and only when a human
says to start the next one. **Item 1 goes first**: every later item is
judged by the numbers it produces.

Status: `todo` · `in progress` · `done` (with the date and the measured effect).
An item is done only when `bun run check` passes, the eval (item 1) shows
its effect, and the docs describe what ships (AGENTS.md §8). When an item
is done, move what it established into AGENTS.md (§2 platform facts, or the
agent design note), so this file stays a plan and not a second source of
truth.

Measured baseline (Qwen3 0.6B as the chat and decision model, an Apple Silicon
Mac, 2026-09-26): about 0.6 s per decision and 0.5–0.9 s per argument fill.
The e2e question takes `kb_search` (96–100%), then `answer_now` (81–92%).

| # | Item | Status |
|---|---|---|
| 1 | [Agent eval set](#1-agent-eval-set) | done 2026-09-26 · baseline below |
| 2 | [Known symbols for code tools](#2-known-symbols-for-code-tools) | done 2026-09-26 · no change on the eval; hiding tools measured worse |
| 3 | [Grounded line ranges for Read lines](#3-grounded-line-ranges-for-read-lines) | done 2026-09-26 · Read lines lands on a found passage; one call fewer |
| 4 | [Richer observations](#4-richer-observations) | measured 2026-09-26 · no gain, not shipped; re-measure after item 5 |
| 5 | [Merge evidence before answering](#5-merge-evidence-before-answering) | done 2026-09-26 · 0.6B facts 82.8% → 86.2%, no wasted calls |
| 6 | [Prompt-prefix caching](#6-prompt-prefix-caching) | done 2026-09-26 · later decisions ~20% faster on a decision model; same outcomes |
| 7 | [Skip the obvious first decision](#7-skip-the-obvious-first-decision) | done 2026-09-26 · first action 100%, 45% fewer decisions |
| 8 | [Measure a larger decision model](#8-measure-a-larger-decision-model) | done 2026-09-26 · 1.7B decides worse; recommend 1.7B answers + 0.6B decisions |
| 9 | [Two query phrasings per search](#9-two-query-phrasings-per-search) | measured 2026-09-26 · worse, not shipped |

**Where it ended (2026-09-26).** Shipped: 1, 2, 3, 5, 6, 7; measured and
not shipped: 4, 9; 8 changed the recommendation (a small decision model with
a larger chat model). With Qwen3 0.6B alone the agent now takes the right
first action on 32 of 32 questions (was 31) with 1.1 decisions per question
(was 2.0), and finds the expected facts in 86.2% of answers (82.8% on the
same set before items 3–7). Qwen3 1.7B answering with 0.6B deciding reaches
96.6% facts at about 7 s per question. The remaining misses are in the
answer text, not the retrieval.

**Deferred decision:** [a native llama.cpp engine](#deferred-a-native-llamacpp-engine)
(bundled `llama-server` on Metal), to decide once items 3–9 are done.

---

## 1. Agent eval set

**Why.** The decision-wording and offer-rule measurements in AGENTS.md §2
were one-off probes that were thrown away. Without a repeatable eval, a
change can only be argued about, not measured.

**What.**
- A fixture knowledge base (Markdown and TypeScript, with a known answer for
  each question) and about 20–30 questions. Each has the expected first
  action (a tool, or `answer_now` for small talk) and facts the answer must
  contain (a regex). Include document, code and mixed cases, follow-ups, and
  a question the knowledge base can't answer.
- A `VITE_SMOKE=eval` harness mode (like `runE2E` in `src/smoke.ts`) that runs
  each question through `runTurn` in the real app and writes a JSON report:
  per question, the actions with their confidences, fallbacks, calls,
  argument validity, whether the answer is grounded, and timings.
- `bun run eval:agent` (a script like `scripts/e2e.mjs`) prints the
  scorecard: first-action accuracy, answer-fact hit rate, wasted calls
  (errors, empty, skipped), mean decisions and seconds per question. It can
  also compare against a saved baseline, like `perf/baseline.json`.
- Machine-bound and model-bound like the perf baselines, and not part of
  `bun run check` (it needs a model and ug).

**Done when.** The eval runs from one command and gives a stable scorecard
(two runs agree), and the baseline is recorded here.

**Done (2026-09-26).** `bun run eval:agent`: 27 questions over three
knowledge bases (documents, code, both) built from `tests/fixtures/eval/`,
a fictional ferry operator, so the model can't answer from general
knowledge. The option shuffle is seeded (`runTurn(text, { seed })`), the
answer is greedy, and each run writes a full report to `eval/`. How to run
it and the metrics: [performance.md](performance.md#agent-eval-bun-run-evalagent-section-agent-eval).
Three runs gave the same outcome on every question. `bun run eval:view`
reads and compares the reports.

Other setups (2026-09-26, same questions and seed; each has its own baseline
section in `perf/baseline.json`, `agent-eval:<setup>`):

| Setup (answers + decisions) | First action | Answer facts | Grounded | ms / decision | s / q |
|---|---|---|---|---|---|
| Qwen3 0.6B + 0.6B (the default) | 96.3% | 83.3% | 56.5% | 668 | 3.84 |
| Qwen3 1.7B + 0.6B decision model | 96.3% | 95.8% | 91.3% | 726 | 6.47 |
| Qwen3 1.7B + 1.7B | 77.8% | 87.5% | 90.5% | 1,662 | 7.88 |

A bigger *answer* model fixed most of what items 4 and 5 aimed at. A bigger
*decision* model made decisions worse: Qwen3 1.7B chose *ask a clarifying
question* first for all 3 small-talk questions and 3 lookups, and after a
search it often asked instead of answering. The option wording in `loop.ts`
was tuned on 0.6B (AGENTS.md §2), so this is a first result for item 8, not
its conclusion.

Baseline (Qwen3 0.6B chat and decision model, Apple M5 Max):

| First action | Answer facts | Grounded | Wasted calls | Decisions / q | ms / decision | Prompt tokens / decision | s / q |
|---|---|---|---|---|---|---|---|
| 96.3% | 83.3% | 56.5% | 0 | 1.85 | 668 | 302 | 3.84 |

What it says about the later items:
- **Items 2 and 3** have little to show yet: every lookup question started
  with `kb_search`, no code tool or Read lines was ever chosen, and there
  were no "No symbol named" errors. Their effect will only show once the
  agent uses those tools (or with questions that need them).
- **Item 4 / 5:** the misses are in the answer, not the retrieval: the
  weather answer drops "30 days", the refund comparison doesn't say 50%, an
  unanswerable mixed question invents a database, and only 56% of answers
  cite their sources.
- **Item 7:** with the first decision `kb_search` 23/24 times and small talk
  `answer_now` 3/3, a fast path has the numbers to be judged against.
- The one first-action miss is a pronoun follow-up ("What does it add for a
  vehicle?") answered from history without a lookup.

## 2. Known symbols for code tools

**Why.** Symbol context, Read symbol source and Find usages take a
free-text `symbol`. It's the same failure mode the `file` enum fixed
(`schemaFor` in `agent/tools/argfill.ts`): a small model writes names that
aren't in the graph, and ug answers "No symbol named …".

**What.** Collect the symbol names seen so far in the turn (Find symbols
items, search hits that are code nodes, context items) and pass them, like
`files`, as an `enum` on `symbol` when there are 1–64 of them. When no
symbol has been seen yet, don't offer the three tools in `offered()`, so
the agent has to look one up or search first.

**Done when.** The eval's code questions show fewer "No symbol named" errors
and no drop in first-action accuracy.

**Done (2026-09-26).** The eval got 3 questions that call for the code tools
(`code-source-retry`, `code-usages-refund`, `mixed-context-createbooking`),
making 30. The "before" run on those 30 is the new `agent-eval` baseline
(`eval/item2-before-0.6b.json`: 96.7% first action, 81.5% facts, 50.0%
grounded).

- **Not offering the symbol tools until a symbol is seen made things much
  worse**, so it isn't shipped. With the shorter option list, Qwen3 0.6B
  answered without searching on 12 of the 16 code and mixed questions: first
  action 96.7% → 56.7%, facts 81.5% → 48.1% (`eval/item2-after-0.6b.json`
  from that attempt was overwritten; the numbers are from its scorecard).
- **What ships:** the option list is unchanged. `symbol` is held to the code
  symbols seen so far in the turn (`symbolsIn` in `loop.ts`: Find symbols
  items, search and context hits whose node type is a code symbol; ug calls
  document sections `Concept`), as an enum when there are 1–64, and they're
  listed in the argument prompt. If the model picks a symbol tool before any
  symbol has been seen, the loop runs Find symbols first and records it on
  the trace (`fallback: 'needs-symbol'`).
- **Measured:** the same outcome on all 30 questions as before (`--against`
  lists no change). There were no "No symbol named" errors before either:
  0.6B never picks a symbol tool before searching, so the gain can't show on
  this setup. The unit tests cover both paths. Timings in the after runs were
  taken on a loaded machine (the unchanged document questions were 29%
  slower too), so the baseline's timings were kept.
- **Found on the way (item 5):** on `code-usages-refund`, Symbol context
  and Find usages returned the call site of `refundFraction` in
  `cancelBooking`, but the loop dropped both passages because a search had
  already returned the same ug node ids with a shorter snippet. They counted
  as empty calls and the answer never saw the call site.

## 3. Grounded line ranges for Read lines

**Why.** In a probe the model asked for lines 100–200 of a file it hadn't
seen. Read lines is only useful around something already found.

**What.** Offer `kb_read_lines` only after a result with a file and line
range. Give the model those ranges (padded by about ±20 lines) as the
choices, e.g. a `range` enum of `"file:start-end"` strings mapped back to
`file/start/end`, instead of free integers.

**Done when.** Every Read lines call in the eval lands on a range that
overlaps an earlier hit.

**Done (2026-09-26).** Two questions joined the eval (`doc-quote-checkin`,
`code-lines-cancel`), making 32; the "before" run is the new `agent-eval`
baseline (`eval/item3-before-0.6b.json`).

- **What ships:** Read lines takes one `range` argument, `file:start-end`,
  held to the passages found so far padded by 20 lines (`rangesIn` in
  `loop.ts`, `schemaFor` in `argfill.ts`) and listed in the argument prompt.
  Rust still validates the file and clamps the lines. As in item 2, the
  option isn't hidden until a range exists (that moved 0.6B's choices);
  picking Read lines first searches instead (`fallback: 'needs-range'`).
- **Measured** (`eval/item3-after-0.6b.json`): the one Read lines call read
  `booking.ts:15-60`, the `cancelBooking` passage (35–40) the search had
  found, padded; before, it chose 35–40 itself and then also asked for the
  file outline, so the turn made one call fewer. Every other question had
  the same outcome; the scorecard passed on every metric.

## 4. Richer observations

**Why.** The next decision and argument fill only see a summary line of
file names and line ranges (`Evidence.summary`), which gives the model
nothing to judge relevance by.

**What.** Add the first ~100 characters of the top 2–3 passages to each
observation in `agentState` (fenced as untrusted, like everything else from
tools). Watch the prompt size: the eval should report prompt tokens per
decision.

**Done when.** The eval shows better answer-fact hits or fewer wasted calls,
at no more than +25% prompt tokens per decision.

**Measured (2026-09-26), not shipped.** The first 100 characters of the top
three passages went under each result, one line each (a quoted excerpt).
Prompt tokens per decision rose 12% (310 → 346).

| Setup | Answer facts | Grounded | Wasted calls |
|---|---|---|---|
| 0.6B, without → with excerpts | 82.8% → 79.3% | 46.4% → 46.4% | 2 → 0 |
| 1.7B answers + 0.6B decisions, without → with | 96.6% → 96.6% | 89.3% → 89.3% | 2 → 0 |

The "wasted" calls that went away are the two item 5 bug calls on
`code-usages-refund` (they found the call site and were thrown away as
duplicates); with excerpts the model skipped them and answered, with the
same miss. The 0.6B fact lost was `code-lines-cancel`: with excerpts in its
prompt, the argument writer picked `booking.ts:1-34` instead of the passage
around `cancelBooking`. Nothing got better, so the change was reverted
(reports: `eval/item4-*`). Re-measure after item 5, when those calls add
what they find.

## 5. Merge evidence before answering

**Why.** Passages from later calls are appended in arrival order, and
`loop.ts` drops a passage whose node id was already seen, even when the later
one has more in it (item 2 measured this: the call site Find usages found was
thrown away because a search had returned the same node). Also
`buildSystem` cuts at the context budget from the end, so a better passage
found later can be dropped. Overlapping ranges from different tools (search
hit vs. Read lines vs. symbol source) can repeat the same text.

**What.** Before `buildSystem`, merge all hits across calls: drop ranges
contained in another range of the same file (`dedupeHits` does this within
one search), then order them. Put results the agent read on purpose first
(Read lines, symbol source), then search hits in ug's order. Keep `[n]`
aligned with the UI source list.

**Done when.** The eval shows no answer-fact regressions, and prompts that
repeated text before now don't.

**Done (2026-09-26).** `agent/evidence.ts`, used by `loop.ts`:
- A passage whose ug node was already found keeps its place and gains the
  text the new one adds (Find usages' call-site lines, a context bundle's
  source); that call counts as having added something.
- A passage inside a range read on purpose (Read lines, Read symbol source)
  is dropped, since the read repeats its text; of two reads of one range, the
  first stays.
- Order for the answer and the UI source list (so `[n]` still matches):
  reads, then lookups (symbols, context, usages, outlines, overview), then
  search hits, each in arrival order. `buildSystem` cuts from the end, so
  broad search hits give way to budget first.

Measured against item 3's final run:

| Setup | Answer facts | Grounded | Wasted calls |
|---|---|---|---|
| 0.6B | 82.8% → 86.2% | 46.4% → 53.6% | 2 → 0 |
| 1.7B answers + 0.6B decisions | 96.6% → 96.6% | 89.3% → 85.7% | 2 → 0 |

The 0.6B gain is `code-usages-refund`, the item 2 bug: the answer now names
`cancelBooking`. The 1.7B grounding change is the same question: the answer
got more exact (the call is on line 37, from the merged call site; before
it said 35) but left out its `[1]` this time. No other question changed
(`eval/item5-after-*`).

## 6. Prompt-prefix caching

**Why.** Every decision and fill sends `cache_prompt: false`, so the whole
prompt is processed again each time. Consecutive decisions share most of
their prompt (system text, request, knowledge base line, earlier results).

**What.** Turn on `cache_prompt` for decisions on a **separate** decision
model: the chat model alternates between decisions, fills and answers, so
its cache would keep being thrown away. Measure the time per decision, and
check the scores don't move meaningfully (SemIf saw prefix reuse flip 5–6 of
777 decisions).

**Done when.** Decision time drops by a measured amount, and the eval's
first-action accuracy doesn't drop.

**Done (2026-09-26).** `decide` sends `cache_prompt: true` when it runs on
the decision model and `false` on the chat model. Measured with Qwen3 1.7B
answering and 0.6B deciding (`eval/item6-after-1.7b+0.6b.json` against
`item5-after-…`): the same outcome on all 32 questions. A turn's first
decision can't reuse anything; the later ones can:

| | First decision | Second | Third and later |
|---|---|---|---|
| Cache off | 711 ms | 840 ms | 856 ms |
| Cache on | 806 ms | 768 ms | 627 ms |

The machine was loaded (first decisions moved 13% with no change to them),
so read it relative to the first decision: later decisions went from 18%
slower than the first to 5% faster, about 150–200 ms each. Small, because a
turn makes about two decisions and the second prompt's new part (the tool
results, the reshuffled options) comes after a short shared prefix.

## 7. Skip the obvious first decision

**Why.** On step 1, with a knowledge base selected, the decision nearly
always picks search (96–100% in the e2e runs). That costs about 0.6 s per
question.

**What.** An optional fast path: on step 1, if the request isn't small talk
(a cheap check, or a two-option decision — search or answer — which is
faster to score), go straight to argument filling for `kb_search`. Record it
in the trace as a decision with a note, so it stays visible.

**Done when.** The eval shows the time saved and that small talk still gets
`answer_now`.

**Done (2026-09-26).** On step 1, if `kb_search` is offered and the request
isn't small talk (`isSmallTalk` in `prompt.ts`: short, no question mark,
starts like a greeting, thanks or sign-off), the loop searches without a
decision and says so on the trace. Small talk still goes to the model. The
switch is `searchFirst` in the tools settings (on by default, no UI yet).

| Setup | First action | Decisions / question | Seconds / question |
|---|---|---|---|
| 0.6B | 96.9% → 100% | 1.97 → 1.09 | 3.93 → 3.62 |
| 1.7B answers + 0.6B decisions | 96.9% → 100% | 2.00 → 1.13 | 7.79 → 7.05 |

The first-action gain is the follow-up "What does it add for a vehicle?",
which the model had answered from history without a lookup; it now
searches (its answer still misses the 18.5 surcharge). Small talk got
`answer_now` 3 of 3 times, as before. The load average was 6.8 during the
"after" run against 2.7 for "before", so the time saved is if anything
understated.

## 8. Measure a larger decision model

**Why.** SemIf measured choice accuracy rising sharply with model size
(0.44 → 0.69 → 0.81 balanced accuracy for 0.6B → 2B → 4B). Qwen3 1.7B is
already selectable as the decision model (`Settings → Decision model`).

**What.** Run the eval with Qwen3 0.6B chat + Qwen3 1.7B decider against
the 0.6B-only baseline. If it clearly helps, recommend it in the docs, and
consider adding a pinned 2–4B model under 2 GB (for example MiniCPM5 2B
Q4_K_M, 1.56 GB, SemIf's default) to the catalog.

**Done when.** The comparison is recorded here and in `docs/features.md`, with
the decision and the reason.

**Done (2026-09-26).** All four setups on the same 32 questions, with items
2–7 in place:

| Answers + decisions | First action | Answer facts | Grounded | ms / decision | s / question |
|---|---|---|---|---|---|
| 0.6B + 0.6B (default) | 100% | 86.2% | 55.2% | 751 | 3.62 |
| **1.7B + 0.6B** | **100%** | **96.6%** | **86.2%** | 888 | 7.05 |
| 0.6B + 1.7B | 90.6% | 79.3% | 51.7% | 1,645 | 6.34 |
| 1.7B + 1.7B | 90.6% | 93.1% | 79.3% | 2,060 | 11.38 |

(`eval/item7-after-*`, `eval/item8-*`; each setup has its own baseline
section.) **Decision:** a bigger decision model is worse here, not better.
Qwen3 1.7B deciding chose *ask a clarifying question* after the search on
most questions (22 of 29 lookups in both setups) and for all 3
small-talk questions, and each decision took 2–3× longer. The answer model
is what matters: 1.7B answering with 0.6B deciding is the best setup, and
the docs and the Settings text now recommend it. No larger model was added
to the catalog.

The option wording and the `offered` rule were tuned on 0.6B (AGENTS.md §2);
1.7B may do better with its own wording, but that's a new measurement, not
this item.

## 9. Two query phrasings per search

**Why.** One search phrase can miss: a keyword phrasing and a descriptive
phrasing find different passages. A second decision and search to recover
costs about 1.5 s.

**What.** Let `kb_search` take `queries: [1–2 strings]`. Rust runs ug once
per query (still validated, time-boxed and capped), then results are merged
with `dedupeHits`, with the arguments shown in the trace.

**Done when.** The eval shows higher answer-fact hits for document questions
without more decisions per question.

**Measured (2026-09-26), not shipped.** `kb_search` took `queries` (1–2
phrases); the loop ran the same validated Rust search once per phrase and
merged the items (`dedupeHits`), as one call on the trace. Both models used
two phrasings almost every time (0.6B 27 of 29 searches, 1.7B 29 of 29):

| Setup | Answer facts | Seconds / question |
|---|---|---|
| 0.6B, one → two phrasings | 86.2% → 82.8% (1 gained, 2 lost) | 3.62 → 3.93 |
| 1.7B answers + 0.6B decisions | 96.6% → 86.2% (3 lost) | 7.05 → 9.21 |

The second search roughly doubles the passages, so the relevant ones are
diluted and cut by the context budget, and the answer gets slower; no
document question improved. Reverted (`eval/item9-after-*`). The misses left
are in how the answer is written, not what the search finds, so a better
retrieval item would have to show a retrieval miss first. Finding this also
exposed a harness bug (records split across output chunks were dropped),
fixed separately.


---

## Deferred: a native llama.cpp engine

**Status:** to decide after the RAG items above are done (a human decision:
it adds a native-code dependency, AGENTS.md §1.10).

**Why.** WKWebView can only run wllama's compat (WebAssembly, Asyncify)
build. It does use the GPU (every layer on WebGPU), but measured on an Apple
M5 Max, Qwen3 1.7B reads prompts at about 185 tok/s and writes 30–65 tok/s,
so a grounded answer waits about 3 s before its first word. Flash attention
is unavailable on that path, and no wllama setting (threads, batch size)
moved the numbers (docs/performance.md, *Engine*). Native llama.cpp on Metal
would likely be several times faster at both; measure before deciding.

**What it would take.**
- Ship `llama-server` (or link llama.cpp through a Rust crate) for macOS and
  Windows, pinned and checksummed like the models; sign it with the app.
- Rust starts it bound to loopback on a random port with a per-launch token,
  or talks to it over a pipe; the webview never gets a raw socket (§9: the
  webview is untrusted, and a local server is a new attack surface).
- Reuse the same GGUF files and the pinned catalog; keep the choice-based
  decisions (`decide` needs raw next-token logprobs and GBNF, which
  `llama-server` has).
- Keep wllama as the fallback, and let `bun run eval:agent` and
  `bun run bench:engine` compare the two engines on the same questions.

**Open questions.** Binary size per platform (Metal and CPU builds), code
signing and notarization, how models move between the two engines, and
whether the speed-up survives the eval (answer quality should not change
with the same weights).

