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
| 10 | [Robust to the search scope](#10-robust-to-the-search-scope) | done 2026-09-27 with Laya · facts 75.9% → 89.7% (Multilingual), 82.8% → 89.7% (English); Qwen deciders unchanged |
| 11 | [Rerank kept passages by relevance](#11-rerank-kept-passages-by-relevance) | measured 2026-09-27 · one fact lost per checkpoint, none gained; not shipped |
| 12 | [Intent gate with Laya](#12-intent-gate-with-laya) | probed 2026-09-27 · fails the bar on both checkpoints; not shipped |
| 13 | [Answer claim check](#13-answer-claim-check) | done 2026-09-27 · AUC 0.67 / 0.82; flags 3 of 42 and 8 of 41 cited sentences on the eval, 14 / 27 ms |
| 14 | [Let the decision model choose the first step on code](#14-let-the-decision-model-choose-the-first-step-on-code) | measured 2026-09-27 · fewer facts on every setup; not shipped |
| 15 | [Decide on what was found, not on what fits](#15-decide-on-what-was-found-not-on-what-fits) | done 2026-09-28 · facts 75.0% → 85.0% (Laya English), 82.5% → 87.5% (Multilingual), 65.0% → 72.5% (Qwen deciding) |

**Where it ended (2026-09-26).** Shipped: 1, 2, 3, 5, 6, 7; measured and
not shipped: 4, 9; 8 changed the recommendation (a small decision model with
a larger chat model). With Qwen3 0.6B alone the agent now takes the right
first action on 32 of 32 questions (was 31) with 1.1 decisions per question
(was 2.0), and finds the expected facts in 86.2% of answers (82.8% on the
same set before items 3–7). Qwen3 1.7B answering with 0.6B deciding reaches
96.6% facts at about 7 s per question. The remaining misses are in the
answer text, not the retrieval.

**Engine (decided 2026-09-27):** a native engine on Apple Silicon, MLX in
Rust rather than llama.cpp ([below](#decided-a-native-engine-mlx)).

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
plainly asks about content, the loop searches without a decision and says
so on the trace; everything else goes to the model. The switch is
`searchFirst` in the tools settings (on by default, no UI yet).

**Fixed the same day:** the first version skipped the decision for anything
that wasn't a greeting or thanks, so "who are u?" was searched for and found
ten unrelated passages (reported from the app). `needsLookup` in
`prompt.ts` is conservative now: not small talk, at least one content word,
and not addressed to the assistant ("you", "your", "u"). A request it lets
through only costs a decision, which picks search when it should. Two eval
questions cover it (`doc-about-assistant`, `doc-assistant-abilities`, 34 in
all): both answered without a search on 0.6B and on 1.7B + 0.6B, and no
other question changed (`eval/item7fix-*`).

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

## 10. Robust to the search scope

**Why.** Measuring the native MLX engine (below) showed the agent's answers
hinge on one argument. For "Which function implements the group discount
from the release notes?", Qwen3 1.7B filled `kb_search` with `"scope":
"broad"` in MLX 4-bit, 8-bit and bf16, and `"focused"` only in GGUF Q4_K_M.
A broad search returned one-line fragments (`fares.ts:19-19`) instead of the
function, and the answer missed it. Three of the MLX 1.7B + 0.6B misses
(82.8% facts, against 96.6% on wllama) are this; the full-precision model
does it too, so it's the agent's fragility, not the engine's.

**What.** Options to measure: word the `scope` guidance in the argument
prompt differently; pick the scope by rule (a name-like query → focused);
or run a broad search's fragments through the evidence merge so they carry
their surrounding lines.

**Done when.** MLX 1.7B + 0.6B reaches wllama's fact rate, and wllama
doesn't lose any.

**Done with a Laya decider (2026-09-27).** An enum argument is a typed
choice, which is Laya's job: `kb_search` declares `scope` in
`ToolDef.choices`, Laya asks it as a `choice` row in the decision's batch
(or in its own ~8 ms pass on the search-first path), and `schemaFor` holds
the argument to the picked value, so the chat model writes only the query.
The baseline showed the failure plainly: the chat model wrote `broad` for
13 of 16 code and mixed searches, `computeFare` and `withRetry` included.
Qwen3 1.7B on MLX answering, 34 questions, seed 7 (`eval/laya10-*`):

| Decider | Facts | Grounded | ms / decision | s / question |
|---|---|---|---|---|
| Laya Multilingual, before → after | 75.9% → 89.7% | 75.8% → 84.8% | 49 → 21 | 1.56 → 0.83 |
| Laya English, before → after | 82.8% → 89.7% | 76.7% → 80.0% | 36 → 37 | 0.87 → 0.89 |

Gained: `code-surcharge`, `code-peak`, `code-lines-cancel` on both, and
`mixed-cancel` on Multilingual. Lost: `mixed-followup-surcharge` on
English, where Laya chose `focused` for the topical follow-up query
"vehicle features" and the answer named `VEHICLE_SURCHARGE` without 18.5
(it misses on Multilingual before and after). The Multilingual "before"
run's 49 ms per decision is about twice what Laya Multilingual measured
before (22 ms, AGENTS.md §2); the cause wasn't investigated, so only the
English row's timings are a fair comparison: the extra row costs nothing
measurable. **Still open:** a Qwen decider
(wllama or MLX 0.6B) has no argument choices, so MLX 1.7B + 0.6B keeps the
gap this item was opened for. Small-talk misses with Laya are item 12.

## 11. Rerank kept passages by relevance

**Why.** The relevance check scores every passage but only drops the ones
below 0.10; `buildSystem` cuts from the end, so a high-scoring passage late
in the list can lose to a low-scoring one.

**What.** Keep the top `KEEP_TOP` in place, order the other kept passages by
score (stable), and set `sources` after the check as now, so `[n]` stays
aligned. The trace shows each passage's rank.

**Done when.** Facts and grounding hold or improve on both Laya setups with
no question losing a fact; prompt tokens and first-token time are reported.

**Measured (2026-09-27), not shipped.** `keptOrder` put the top two in place
and the other kept passages by score, with `[n]` renumbered to match.
Against item 10's final runs (`eval/laya11-after-*` vs `laya10-after-*`):

| Decider | Facts | Grounded | s / question |
|---|---|---|---|
| Laya Multilingual | 89.7% → 86.2% | 84.8% → 81.8% | 0.83 → 0.84 |
| Laya English | 89.7% → 86.2% | 80.0% → 80.0% | 0.89 → 0.91 |

Lost `mixed-refund-match` (Multilingual) and `mixed-context-createbooking`
(English); nothing else changed and nothing was gained. A likely reason,
not checked: the eval's knowledge bases are small, so the context budget
may rarely cut a passage, and then the reorder only changes what the answer
leads with, not what it sees. Reverted. Worth a
re-measure on a knowledge base big enough for the budget to bite.

## 12. Intent gate with Laya

**Why.** With Laya deciding, small talk still searches ("hi", "who are
you?") or clarifies ("thanks"): 5 first-action misses with Multilingual.
A `noul` "needs the knowledge base" probe failed (AGENTS.md §2).

**What.** Probe first: a `choice` over `small_talk`, `about_assistant`,
`kb_content`, `follow_up` on every eval question plus ~20 small-talk and
about-the-assistant lines, both checkpoints, confusion matrix here. Ship
only if no lookup question lands on a no-lookup intent at ≥ 0.5; then ask
it in the first decision's batch when `needsLookup` is false and answer on
`small_talk` / `about_assistant`.

**Done when.** Small talk gets `answer_now` with Laya and lookup
first-action accuracy doesn't drop, or the probe is recorded as not shipped.

**Probed (2026-09-27), not shipped.** `laya_*_intent_probe` in
`src-tauri/src/laya/engine.rs` (run by `bun run test:laya`) asks the choice
on the loop's own step-1 state for the 34 eval requests plus 20 small-talk
and about-the-assistant lines (54 in all), and prints every score. "No
lookup" is P(small_talk) + P(about_assistant).

| Checkpoint | No-lookup requests caught (≥ 0.5) | Lookups wrongly skipped (≥ 0.5) |
|---|---|---|
| Laya Multilingual | 18 of 25 | **19 of 29** |
| Laya English | 1 of 25 | 3 of 29 (all three follow-ups) |

Multilingual gets greetings and thanks right (0.67–0.99) but scores plain
lookups the same way: "Is there a discount for large groups…" 0.85, "What
code is used to page the on-call duty manager?" 0.77 (as
`about_assistant`). English puts almost everything in `kb_content`, "hi"
included (0.58). The gate would only see requests `needsLookup` lets
through, but those include lookups addressed to the assistant ("Can you
tell me the refund window?"), and on this evidence Multilingual would skip
them. Nothing shipped; the probe stays so a new wording or checkpoint can
be measured the same way.

## 13. Answer claim check

**Why.** 76–85% of answers are grounded, and nothing checks whether a cited
sentence is supported by the passage it cites.

**What.** After the answer, with Laya, score each `[n]`-cited sentence
against its passage (a `noul` per row, a Rust command generalized from
`laya_relevance` with a closed set of statements), and show a quiet note
under the answer listing sentences that may not be supported. Never block
or change the answer. The threshold comes from an AUC probe on the eval's
answers (own passage vs. another), not a guess.

**Done when.** AUC and the flag rate are recorded, and e2e and release e2e
pass.

**Done (2026-09-27).** A new command, `laya_support` (claims of
`{statement, source, text}`; the statement "The passage supports this
statement." is a Rust constant), `agent/claims.ts` (`citedClaims` pure,
`checkClaims`), a `verify` step after `generate`, `Message.support`, and a
note under the answer only when something is flagged.

*Probe first* (`laya_*_claims_probe` in `laya/engine.rs`, fixture
`src-tauri/tests/fixtures/laya/claims.json`: 60 cited sentences from the
item 10 answers, each with its own passage and with one cited for another
question):

| Checkpoint | AUC | Own passages below 0.10 | Other passages below 0.10 |
|---|---|---|---|
| Laya Multilingual | 0.67 | 4 / 60 | 24 / 60 |
| Laya English | 0.82 | 8 / 60 | 36 / 60 |

"Own" isn't "supported": read one by one, most own passages below 0.10
were real gaps (a release note cited for a function name it never
mentions; `VEHICLE_SURCHARGE = 18.5` cited for "computeFare adds it"; a
sentence reduced to ", , , and are the relevant sections"). The clear false
alarms were 1 (Multilingual: the weather limits) and 2 (English:
`PEAK_DAYS`, `refundFraction`). Hence `FLAG_BELOW = 0.10` and "may not be
supported" in the UI, never "wrong".

*In the agent eval* (`eval/laya13-after-*`; no answer changed, as the check
only annotates): Multilingual checked 20 answers, 32 cited sentences, and
flagged 1 (the `computeFare` surcharge sentence, whose answer misses 18.5),
15 ms median; English checked 17 answers, 33 sentences, and flagged 4 (two
junk sentences, the surcharge one, and the `refundFraction` false alarm),
41 ms median. `bun run test:e2e` and `test:e2e:release` pass; their step
checks now accept the Laya-only relevance and claim steps as skipped, which
they are when there's nothing to check.

**Fixed (2026-09-27): citations after the full stop.** Reported from the
app ("the claim check step is skipped but I do see cites", Qwen3 8B
answering): an answer ending "… at the terminal. [6]" split into the
sentence and a bare "[6]", which had no words to check and was dropped, so
the step said "The answer cites no passages". `citedClaims` now gives a
citation-only fragment to the uncited sentences just before it (up to 3),
and the skip message says when citations exist but can't be checked.
Re-run (`eval/laya13b-*`, no answer changed): Multilingual checked 28
answers instead of 20, English 24 instead of 17, so about a third of cited
answers had been skipped. Of the new flags, two are real miscitations the
old split hid: `doc-checkin-cars` cites `[2]`, the knowledge-base overview,
for the check-in rule that is in `[1]`, and `doc-group` cites `[2]`,
`operations.md`, for the discount in the release notes (`[1]`). Four were
sentences about what the sources lack ("The information provided does not
mention the CEO [1][2][3]"), which no passage can support; `aboutMissing`
now leaves those out (checked on the eval's 68 sentences: it drops exactly
those two sentences). With that, Multilingual flags 3 of 42 cited sentences
(14 ms median) and English 8 of 41 (27 ms).

**Claim dialog (2026-09-27).** Every checked sentence opens a dialog like
*Why this step?*: the sentence, the passage it was compared with (from the
message's sources), the yes/no question, Laya's answer against the cut, the
exact input (`claimState`), input tokens and whether it was cut to fit
(`laya_support` now returns both per claim), and what the probe measured
for that checkpoint (`MEASURED`). Known limits, stated in the docs:
proxy labels with easy negatives (an upper bound), uncited sentences
unchecked, multi-cite sentences checked per source, the stored snippet
rather than everything the chat model read.

## Decided: a native engine (MLX)

**Decided 2026-09-27; what ships is described in AGENTS.md §2** (the native
chat model facts) and docs/performance.md (*Engine*). In short: on Apple
Silicon, Qwen3 runs natively on MLX in Rust (`src-tauri/src/llm/`), in the
MLX thread Laya already used; wllama stays everywhere else. Measured on an
M5 Max for Qwen3 1.7B: llama.cpp on Metal 285 tok/s, mlx-lm 347, Andai's
port 349 (wllama: 30–65), prompts read 50× faster. MLX 4-bit ships over
5-bit (302 tok/s, 89.7% facts) by a human decision; its quality gap is
item 10.

## 14. Let the decision model choose the first step on code

**Why.** Item 7 searches first, without a decision, whenever the request
plainly asks about content. On a code or mixed knowledge base a blind search
isn't obviously the best start: finding a named symbol, reading its source
or listing its callers can be (raised from the app, 2026-09-27; the trace's
note also claimed "a question almost always needs a search", which item 7
only measured with Qwen3 0.6B deciding).

**Measured (2026-09-27), not shipped.** Two variants against the setups'
last reports (`eval/laya13b-*`, `eval/item7fix-0.6b.json`; runs
`eval/item14-*` and `eval/item14b-*`):

| Setup | Facts, search first | Decision on every code/mixed question | Decision only when a symbol is named |
|---|---|---|---|
| MLX 1.7B + Laya Multilingual | 89.7% | 82.8% (wasted calls 4 → 12) | 82.8% (4 → 6) |
| MLX 1.7B + Laya English | 89.7% | 82.8% (3 → 18; 3.7 decisions/q) | 86.2% (3 → 6) |
| Qwen3 0.6B | 86.2% | 79.3% | 82.8% |

"Named" meant a camelCase, PascalCase or snake_case identifier, backticks,
`foo()` or a source file name (9 of the 16 code and mixed questions).

- **Without a name,** the argument writer guesses one: Find symbols for
  "vehicle" as a class, or for "eval-code" (the knowledge base's name) as a
  file. It finds nothing, the repeat is skipped, and only then comes the
  search.
- **With a name,** Find symbols finds it, but it returns names and line
  numbers, not code. The Laya deciders then judge the results sufficient and
  answer without the source (`code-lines-cancel`, `code-retry-default`,
  `mixed-context-createbooking`), or Laya Multilingual asks a clarifying
  question straight away ("How does computeFare work out the price?").
  A search returns the matching text itself, which is why it wins here.

What ships: search first everywhere, as before; the trace note now says why
("a question about the knowledge base's content starts with a search, which
reads the matching text") instead of "almost always needs a search".
**Next to try:** when Find symbols returns a symbol, read its source in the
same step (or offer Read symbol source as the only next code tool), so a
symbol-first start ends with code in the prompt; then re-measure this item.

## 15. Decide on what was found, not on what fits

**Why.** Laya reads 512 or 1,024 tokens and cuts the state from the end,
where the newest results are, while one search returns up to 6,000
characters (raised from the app, 2026-09-28: "it has to cap seriously on
the search result, barely putting 1/10 of it into the state"). In fact the
state never held the results' text: each call was one line of names and
locations, so the stop question ("the results already contain the
information needed") was answered from section titles. The eval's knowledge
bases were too small to show it (every document under 1 KB), so this item
adds one of long documents.

**What shipped.**
- **A long-document eval set:** `tests/fixtures/eval/large/` (four
  documents, 2.7–6.4 KB, facts late in long sections, similar-looking
  distractors) and 11 questions, including an 808-character request and a
  follow-up after a long answer; 45 questions in all. The scorecard counts
  Laya inputs cut to fit and splits by knowledge base.
- **Per-passage scores** (`relevance.ts` `PassageScorer`): as each tool
  returns, Laya's relevance question is asked of every new passage in its
  own row, a long one in overlapping pieces scored by the best; cached per
  turn, so the check before the answer rescores nothing. The result line
  names the most useful passage and its score. Rows ≤ 24 per call.
- **Stop gate:** a "results suffice" ≥ 0.5 is overruled while no passage
  scored ≥ 0.5 (`STOP_EVIDENCE`). Laya said 65% "enough" after a search
  whose passages all scored ≤ 8%.
- **Read a clipped passage whole** before answering from it (`read-whole`):
  ug clips each search passage to a share of `max_chars` (~750 characters),
  which cut the fact off in 3 of Laya Multilingual's 4 large-set misses
  before this item (the fourth section was never retrieved). Scored ≥ 0.3
  with Laya, the first clipped one without; once a turn; then answer.
- **Budgeted decision state** (`agentState` `budget`): fitted to Laya's
  input, request (≤ half, start and end kept) and step count first, oldest
  results left out first. The largest decision input fell from 495 to 375
  tokens on Laya English.
- **Follow-ups scored with the question before them** (`scoringRequest`).
- **The argument writer reads the two best passages** (≤ 500 characters
  each), fenced like `buildSystem`'s.
- **A bug it exposed:** a search passage and a read of the same lines were
  both kept (`mergeEvidence` spared the earlier one); now the read wins.

**Measured, not shipped:** the best passages' *text* in Laya's decision
state (as much as fitted: ~500 characters on English, ~1,900 on
Multilingual). Laya stopped sooner (1.80 against 2.29 decisions per question
on English) and found fewer facts: 80.0% against 85.0% (English), 85.0%
against 87.5% (Multilingual). The argument writer's passages were measured
the same way: without them 82.5% (English), 87.5% (Multilingual), 72.5%
(Qwen), so they add one English question and nothing else.

**Result** (Qwen3 1.7B MLX answering, seed 7, M5 Max; `eval/item15-before-*`
against `eval/item15-a1r-*`, a repeat that matched its first run on every
question):

| Decider | Facts | Large set | Original 29 | Grounded | Decisions / q | s / q |
|---|---|---|---|---|---|---|
| Laya English | 75.0% → 85.0% | 4 → 9 of 11 | 26 → 25 | 80.0% → 85.0% | 1.76 → 2.29 | 0.87 → 1.05 |
| Laya Multilingual | 82.5% → 87.5% | 7 → 8 | 26 → 27 | 84.1% → 81.8% | 2.00 → 1.96 | 0.89 → 0.92 |
| Qwen3 0.6B MLX | 65.0% → 72.5% | 3 → 6 | 23 → 23 | 77.5% → 80.0% | 1.02 → 1.02 | 0.63 → 0.65 |

Lost: `mixed-context-createbooking` on Laya English (it went to Symbol
context and answered without `withRetry`); Multilingual gained it. Still
missed on the large set: `wheelchair-harlow` (the argument writer reads a
range just past the fact), the 808-character request on English (answered
without a lookup) and Multilingual's `pets-kestrel` (the Pets section is
never retrieved). Timing varies between runs on this machine: the same
Multilingual code measured 0.92 and 1.35 s per question with identical
answers, and two earlier runs of 17 minutes were the machine, not the code
(a clean rerun took one).

**Next to try:** the argument writer's Read lines range is still free to
miss (hold it to the node of the best-scoring passage); the relevance
statement asks whether a passage *helps*, not whether it *answers*, so a
second fixed statement for the stop gate is worth a probe.

