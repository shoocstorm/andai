# Performance baselines

Andai runs the model, the retrieval and the UI on your computer, so performance is
part of the product. A slower answer or a bigger bundle is a regression like
any other bug. This page covers what we measure, the current baseline, how
much drift is allowed, and how to update the baseline when a change makes
things slower on purpose.

The baselines are in [`perf/baseline.json`](../perf/baseline.json). Every
number below was measured, never estimated (AGENTS.md §8).

## Three suites

| Suite | Command | Where it runs | Machine-bound |
|---|---|---|---|
| **Bundle**: size of the built UI and runtime | `bun run perf bundle` | Locally and in CI (every push) | No, enforced everywhere |
| **Micro**: hot paths of a turn, in isolation | `bun run perf micro` | Locally | Yes |
| **End-to-end**: the real app, real model, real ug | `bun run test:e2e`, `bun run test:e2e:release` | Locally (needs ug and the model) | Yes |

`bun run perf` runs bundle and micro together. The e2e runs check their
perf numbers as one of their pass/fail checks, and only with the default
model `qwen3-0.6b`.

**Machine-bound** means timings are enforced only on the machine that
recorded the baseline (the `machine` field: CPU, cores, RAM). Anywhere else
the numbers are printed with a note and don't fail. Set `PERF_ENFORCE=1` to
enforce them anyway. File sizes don't depend on the machine, so the bundle
suite is always enforced.

## How a metric passes

Each metric stores its baseline `value`, a `tolerance`, and optionally a
`slack` in its own unit:

- lower is better (sizes, latencies): it passes if `measured ≤ value × tolerance + slack`
- higher is better (throughput): it passes if `measured ≥ value ÷ tolerance`

Slack keeps small timings, like a 100 ms search, from failing on scheduler
jitter. A metric that's in the baseline but no longer measured also fails,
because a metric that disappears quietly would hide its regressions.

## The baseline

### Bundle: `dist/` after `bun run build`, tolerance 1.10

Raw bytes, not gzip. The release UI loads from a loopback server, so the
cost is parsing and compiling, not transfer.

| Metric | Baseline | What it is |
|---|---|---|
| `app-js` | 938 KB | All app JavaScript (React, react-markdown, zustand, motion, wllama glue) |
| `app-css` | 104 KB | All app CSS |
| `fonts` | 1.41 MB | Bundled `@fontsource` files (`.woff` + `.woff2`) |
| `wllama-runtime` | 22.9 MB | wllama wasm (default + compat builds) and its JS |
| `dist-total` | 25.3 MB | Everything the app ships in its UI |

A deliberate bump (a new dependency, a wllama upgrade) moves these on
purpose: update the baseline in the same change and say why.

### Micro: `tests/perf/hot-paths.perf.tsx`, tolerance 1.5

Recorded on Apple M5 Max · 18 cores · 128 GB (Node 22 + jsdom, via Vitest).
Three consecutive runs stayed within ±5% of the baseline.

| Metric | Baseline | Why it matters |
|---|---|---|
| `sha256-throughput` | 323 MB/s | Verifies each downloaded model before it loads (`llm/integrity.ts`). At this rate the 1.1 GB model takes about 3.5 s in V8. WKWebView is slower (see e2e) |
| `stream-patch-400` | 1.06 ms | One streaming repaint with a 400-message chat. `turn.ts` patches every 33 ms, and the persist middleware rewrites the whole chat to localStorage on each patch. This is the metric to watch as chats grow |
| `markdown-render-8k` | 2.98 ms | Rendering a long answer (headings, list, code, table) through `<Markdown>`, which re-renders while streaming |
| `keywords-2k` | 11.6 µs | The "Analyze query" step on a 2,000-character prompt |
| `build-system-8` | 3.10 µs | System prompt from 8 retrieved passages (`agent/prompt.ts`) |
| `build-history-400` | 1.54 µs | History from a 400-message chat with think blocks |
| `dedupe-hits-64` | 2.94 µs | Deduplicating 64 ug search hits (`kb/api.ts`) |

The inputs are sized like a long real session, so code that grows worse
than linearly shows up here before users notice it. The µs-scale metrics
are far from mattering today. They're here to catch an accidental
quadratic, not to shave microseconds.

### End-to-end: the real app, tolerance 1.5

The e2e run ingests `tests/fixtures/`, loads Qwen3 0.6B and asks one
grounded question. Recorded on Apple M5 Max · 18 cores · 128 GB.

| Metric | Dev | Release | Slack | What it is |
|---|---|---|---|---|
| `ingest-ms` | 1,955 ms | 2,135 ms | 1,000 ms | ug ingests and indexes a Markdown file and a PDF |
| `search-ms` | 101 ms | 91 ms | 150 ms | One `ug search` through the Rust bridge |
| `warm-load-ms` | 1,309 ms | 1,338 ms | 1,000 ms | Loading an already downloaded and verified model into wllama |
| `first-token-ms` | 943 ms | 935 ms | 300 ms | Time from sending the question to the first streamed token: analyze, retrieve, build and prompt eval. This is the wait the user sees |
| `generation-tok-per-sec` | 17.8 tok/s | 22.0 tok/s | — | Streamed tokens ÷ time since generation started, prompt eval included |
| `prompt-tokens` | 361 | 367 | tolerance 1.25 | Size of the grounded prompt. Deterministic for the fixtures, so growth means the prompt got bigger |
| `verify-mb-per-sec` | not measured yet | not measured yet | — | SHA-256 in WKWebView, only on a run that downloads or re-verifies the model. AGENTS.md §2 has an earlier release-build measurement: about 93 MB/s |

A second run on each build stayed within 10% on every metric except
`generation-tok-per-sec`, which moved by up to 22% (17.2–22.0 tok/s in
release). Treat tok/s changes under about 25% as noise. The two builds
retrieve slightly different passages (361 vs. 367 prompt tokens), which is
why each has its own baseline.

Each run measures either warm load or verify throughput, never both,
because a load that hashed the model isn't a warm load. The metric a run
can't measure is left out of the comparison and kept in the baseline.

### Agent eval: `bun run eval:agent`, section `agent-eval`

Not a speed suite only: it scores what the agent *does* on 27 fixed
questions (`tests/fixtures/eval/cases.json`) over three knowledge bases built
from `tests/fixtures/eval/` (documents, code, both). Each question lists the
acceptable first actions and regexes the answer must match. The option
shuffle is seeded and the answer is greedy (temperature 0), so a run is
repeatable. It needs ug and the model, so it isn't part of `bun run check`.
Recorded on Apple M5 Max · 18 cores · 128 GB with Qwen3 0.6B.

| Metric | Baseline | Tolerance | What it is |
|---|---|---|---|
| `first-action-accuracy` | 96.3% | 1.1 (higher) | First action taken (after any fallback) is one the question allows |
| `fact-hit-rate` | 83.3% | 1.1 (higher) | Answers that match every fact regex, over questions with facts |
| `grounded-rate` | 56.5% | 1.1 (higher) | Answers with sources that cite at least one `[n]`, and only listed ones |
| `wasted-calls-per-question` | 0.00 | 1.5 + 0.2 | Tool calls that errored, came back empty, or were skipped as repeats |
| `decisions-per-question` | 1.85 | 1.25 + 0.1 | Decision readouts per turn |
| `seconds-per-question` | 4.04 s | 1.5 + 1 s | Whole turn, answer included |
| `ms-per-decision` | 710 ms | 1.5 + 100 ms | One decision readout |
| `prompt-tokens-per-decision` | 302 | 1.25 | Size of the decision prompt |

Two runs gave the same outcome on every question (`--against` listed no
change); only timings moved (3.45 and 4.04 s per question). Each run writes a
full report (every step, argument, source and answer) to `.eval/`.

```bash
bun run eval:agent                                # scorecard, compared with the baseline
bun run eval:agent --against .eval/<earlier>.json # which questions changed
bun run eval:agent --only doc-wind,code-peak      # a subset (not compared)
bun run eval:agent --update                       # re-record the baseline
```

## Workflow

**Before calling a change done** (AGENTS.md §6), if it touches a hot path
(prompt assembly, retrieval, the chat store, Markdown rendering, the engine,
dependencies):

```bash
bun run perf                 # bundle + micro
bun run test:e2e             # also checks e2e-dev perf (engine, ug or Tauri changes)
```

**When a regression is intended** (a feature that is worth the cost), re-record
and explain it in the commit message:

```bash
bun run perf --update                    # or: bun run perf bundle --update
bun run test:e2e --update-perf           # e2e-dev
bun run test:e2e:release --update-perf   # e2e-release
git diff perf/baseline.json              # review every number that moved
```

`--update` keeps each metric's tolerance and slack. Loosening a tolerance is
a deliberate edit of `perf/baseline.json`, with the reason in the commit.

**On a new machine**, the machine-bound suites only print numbers. Record your
own baseline with `--update` if you want them enforced locally, but don't
commit it over the shared one unless the team agrees to move the reference
machine.

## Adding a metric

- **Micro**: add an `it(...)` to `tests/perf/hot-paths.perf.tsx` (or a new
  `tests/perf/*.perf.ts` that calls `reportTo(import.meta.filename)`). Time it
  with `msPerOp` and `record` the result. Size the input like a real long
  session, then run `bun run perf micro --update`.
- **End-to-end**: add the raw number to `perf` in `RESULT` (`src/smoke.ts`)
  and a metric for it in `scripts/e2e.mjs`.
- **Show that it can fail** (AGENTS.md §6): plant a slowdown once and watch
  the suite go red. When this suite was added, an extra `JSON.stringify` in
  `dedupeHits` gave `dedupe-hits-64 … +1518%` and 200 KB of padding in the JS
  bundle gave `app-js … +20.8%`. Both runs exited 1.

## Not measured yet

These gaps are known. Add them once there's a reliable way to measure them:

- **Cold start**: from launching the app until the window is interactive
- **Memory**: webview and wasm memory with the model loaded
- **Large knowledge bases**: ingest and search time on hundreds of documents
  (the e2e fixtures are two small files)
- **Frame timing while streaming** in WKWebView (the micro suite measures
  jsdom, not WebKit layout and paint)
