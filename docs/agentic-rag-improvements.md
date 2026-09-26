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
| 3 | [Grounded line ranges for Read lines](#3-grounded-line-ranges-for-read-lines) | todo |
| 4 | [Richer observations](#4-richer-observations) | todo |
| 5 | [Merge evidence before answering](#5-merge-evidence-before-answering) | todo |
| 6 | [Prompt-prefix caching](#6-prompt-prefix-caching) | todo |
| 7 | [Skip the obvious first decision](#7-skip-the-obvious-first-decision) | todo |
| 8 | [Measure a larger decision model](#8-measure-a-larger-decision-model) | todo |
| 9 | [Two query phrasings per search](#9-two-query-phrasings-per-search) | todo |

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

## 9. Two query phrasings per search

**Why.** One search phrase can miss: a keyword phrasing and a descriptive
phrasing find different passages. A second decision and search to recover
costs about 1.5 s.

**What.** Let `kb_search` take `queries: [1–2 strings]`. Rust runs ug once
per query (still validated, time-boxed and capped), then results are merged
with `dedupeHits`, with the arguments shown in the trace.

**Done when.** The eval shows higher answer-fact hits for document questions
without more decisions per question.
