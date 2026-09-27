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
| `app-js` | 1.04 MB | All app JavaScript (React, react-markdown, zustand, motion, wllama glue) |
| `app-css` | 113 KB | All app CSS |
| `fonts` | 1.41 MB | Bundled `@fontsource` files (`.woff` + `.woff2`) |
| `wllama-runtime` | 22.9 MB | wllama wasm (default + compat builds) and its JS |
| `dist-total` | 25.5 MB | Everything the app ships in its UI |

A deliberate bump (a new dependency, a wllama upgrade) moves these on
purpose: update the baseline in the same change and say why.

Recorded 2026-09-27: +~41 KB over 2026-09-26 for the web deployment's
Firebase Analytics chunks (`src/lib/firebase.ts`). They are lazy chunks a
browser session downloads; the desktop app never loads them, they only ride
along in its bundle on disk.

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
| `ingest-ms` | 2,122 ms | 2,095 ms | 1,000 ms | ug ingests and indexes a Markdown file and a PDF |
| `search-ms` | 110 ms | 95 ms | 150 ms | One `ug search` through the Rust bridge |
| `warm-load-ms` | 1,800 ms | 1,324 ms | 1,000 ms | Loading an already downloaded and verified model into wllama |
| `first-token-ms` | 967 ms | 944 ms | 300 ms | Time from sending the question to the first streamed token: analyze, retrieve, build and prompt eval. This is the wait the user sees |
| `generation-tok-per-sec` | 66.6 tok/s | 68.7 tok/s | — | Answer tokens per second from the first token on (llama.cpp's own timing). Until 2026-09-26 this counted the prompt eval too and read 17.8 / 22.0 |
| `prompt-tok-per-sec` | 520 tok/s | 532 tok/s | — | How fast the prompt is read before the first token (llama.cpp's timing) |
| `prompt-tokens` | 361 | 367 | tolerance 1.25 | Size of the grounded prompt. Deterministic for the fixtures, so growth means the prompt got bigger |
| `verify-mb-per-sec` | not measured yet | not measured yet | — | SHA-256 in WKWebView, only on a run that downloads or re-verifies the model. AGENTS.md §2 has an earlier release-build measurement: about 93 MB/s |

Generation speed depends on how busy the machine is: `bun run bench:engine`
measured the same Qwen3 1.7B setting at 22–64 tok/s within minutes, while
its prompt speed held at 180–186 tok/s. Treat generation changes under about
50% as noise unless the machine was idle; prompt speed is the steadier
number. The earlier baseline's second runs stayed within 10% on every other
metric. The two builds
retrieve slightly different passages (361 vs. 367 prompt tokens), which is
why each has its own baseline.

Each run measures either warm load or verify throughput, never both,
because a load that hashed the model isn't a warm load. The metric a run
can't measure is left out of the comparison and kept in the baseline.

### Agent eval: `bun run eval:agent`, section `agent-eval`

Not a speed suite only: it scores what the agent *does* on 30 fixed
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
| `seconds-per-question` | 3.84 s | 1.5 + 1 s | Whole turn, answer included |
| `ms-per-decision` | 668 ms | 1.5 + 100 ms | One decision readout |
| `prompt-tokens-per-decision` | 302 | 1.25 | Size of the decision prompt |

Other model setups have their own sections, `agent-eval:<answers>[+<decider>]`,
recorded the same day: Qwen3 1.7B answering with 0.6B deciding scored 96.3%
first action, 95.8% facts, 91.3% grounded at 6.47 s per question; 1.7B for
both scored 77.8%, 87.5% and 90.5% at 7.88 s (details in
[agentic-rag-improvements.md](agentic-rag-improvements.md#1-agent-eval-set)).

The native MLX models have their own sections too (2026-09-27, 34 questions,
`agent-eval:qwen3-1.7b-mlx+qwen3-0.6b-mlx` and the single-model ones): MLX
1.7B answering with MLX 0.6B deciding scored 100% first action, 82.8% facts,
86.2% grounded at **0.58 s per question** (decisions 23 ms); MLX 1.7B alone
85.3%, 79.3%, 93.1% at 0.58 s; MLX 0.6B alone 100%, 82.8%, 55.2% at 0.46 s.
Why they find fewer facts than wllama's 1.7B (96.6% at 7.65 s on the same
34 questions) is in AGENTS.md §2 and tracker item 10.

Decisions run on the chat model unless `EVAL_DECIDER` names a decision
model: the harness sets it explicitly, because the app otherwise restores the
decision model saved in Settings, which would leak into the run. Three runs
gave the same outcome on every question (`--against` listed no change); only
timings moved (3.45–4.04 s per question). Each run writes a full report
(the questions, every step, argument, source and answer) to `eval/`.
`bun run eval:view` opens a viewer with every report there listed, to read one or compare two
(scorecard deltas, which questions improved or regressed, and both traces
and answers side by side).

```bash
bun run eval:agent                                # scorecard, compared with the baseline
bun run eval:agent --against eval/<earlier>.json # which questions changed
bun run eval:agent --only doc-wind,code-peak      # a subset (not compared)
bun run eval:agent --update                       # re-record the baseline
EVAL_MODEL=qwen3-1.7b bun run eval:agent          # another answer model (not compared with the baseline)
EVAL_DECIDER=qwen3-1.7b bun run eval:agent        # a separate decision model
EVAL_MODEL=qwen3-1.7b-mlx EVAL_DECIDER=qwen3-0.6b-mlx bun run eval:agent  # native MLX (Apple Silicon)
ANDAI_HARNESS_PORT=1432 bun run eval:agent        # when something else holds the harness port (1431)
```

### Engine: `bun run bench:engine`

Not a baseline: a probe for engine questions ("is the GPU used?", "do more
threads help?"). It loads one model under several wllama settings in the
real webview (`src/bench.ts`) and prints llama.cpp's own load log (GPU
adapter, layers offloaded, threads) with prompt and generation speed for a
grounded-size prompt (about 550 tokens). Measured 2026-09-26, Qwen3 1.7B,
Apple M5 Max, dev build:

| Setting | Generation | Prompt |
|---|---|---|
| Default: WebGPU (29/29 layers), 4 threads | 29–64 tok/s | 183–186 tok/s |
| WebGPU, 2 threads | 34–64 tok/s | 177–186 tok/s |
| WebGPU, 8 threads | 13–16 tok/s | 180 tok/s |
| CPU only (`n_gpu_layers: 0`), 4 threads | 3.6 tok/s | 27 tok/s |

Also measured, and not worth changing:
- **Flash attention** is requested but llama.cpp turns it off on WebGPU
  ("Flash Attention not supported"); it only runs CPU-only, which is 10×
  slower overall. wllama 3.6.1 is the newest release.
- **Batch size** (`n_ubatch`): 1024 read a 548-token prompt at 212 tok/s
  against 178–183 at the default 512, but only because 548 tokens spill
  just past one 512 chunk; at 1,448 tokens 512, 1024 and 2048 all read
  196–219 tok/s. Each doubling doubles the WebGPU compute buffer (160 → 304
  → 608 MiB), and Qwen3 1.7B already takes about 1.66 GiB of the 2 GiB
  WebGPU budget llama.cpp reports.

Settings in one launch affect each other (the first is always fastest), so
compare settings in separate launches: `BENCH_CONFIGS='[{"name":"x","n_threads":2}]'`.

**Native MLX** (`BENCH_MODEL=qwen3-1.7b-mlx`, Apple Silicon): the same
prompt, natively in Rust (`src-tauri/src/llm/`). Measured 2026-09-27, Apple
M5 Max, dev build, two launches each:

| Model | Generation | Prompt | First token | Load |
|---|---|---|---|---|
| Qwen3 1.7B · MLX 4-bit | 348–352 tok/s | 9,850 tok/s | 56 ms | 0.49 s |
| Qwen3 0.6B · MLX 8-bit | 448–449 tok/s | 17,400–17,850 tok/s | 31–32 ms | 0.47 s |
| Qwen3 1.7B · MLX 5-bit (not shipped) | 298–303 tok/s | 9,350 tok/s | 58 ms | 0.51 s |
| Qwen3 4B · MLX 4-bit (Qwen's build) | 184 tok/s | 2,900 tok/s | 185 ms | 0.19 s |
| Qwen3 8B · MLX 4-bit | 113 tok/s | 2,800 tok/s | 192 ms | 0.22 s |
| Qwen3 14B · MLX 4-bit | 64 tok/s | 1,550 tok/s | 348 ms | 0.29 s |
| Qwen3 32B · MLX 4-bit | 28 tok/s | 640 tok/s | 848 ms | 0.48 s |
| Qwen3 1.7B · wllama Q4_K_M (above) | 29–64 tok/s | 185 tok/s | ~3 s | — |

The 4B–32B rows are from `bun run test:llm` (Rust, release build, same
prompt); mlx-lm measured 188, 114, 64 and 29 tok/s for them. For reference,
outside the app on the same Mac: mlx-lm 0.31 (Python) 347
tok/s generation and 12,200 prompt for the 4-bit 1.7B; llama.cpp b11205 on
Metal with flash attention, the GGUF Q4_K_M wllama loads, 285 and 8,770.
`bun run test:llm` prints the same numbers from Rust directly. A reply held
to a grammar (argument filling) isn't pipelined: 80–90 ms for 17 tokens.

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
