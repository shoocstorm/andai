# AGENTS.md — Andai engineering guide

Andai is a **local-first agentic RAG agent**, shipped as a desktop app for
macOS and Windows: it
answers from the user's knowledge bases by using tools (read-only ug commands,
`agent/tools/`), chosen at each step by the local model. It is a product, not a demo:
every change ships to users who trust it with their documents. This file is the
contract for anyone changing the code, human or AI agent. Read it before
editing. When this file and your instincts disagree, this file wins. When it
is wrong, fix it in the same change.

> **Pre-launch status (no live users yet).** Breaking changes and significant
> refactors are allowed. Don't add backward compatibility: no migrations for
> old storage or data formats, no deprecated aliases, no compatibility shims.
> Change the format and update every caller and test. The rules in §1 still
> apply, including tests, truthfulness and §1.5 for data on the developer's
> machine. Remove this note when Andai has its first real users.

---

## 1. Grounding rules (non-negotiable)

These exist because each one was either violated once or would silently break
the product.

1. **Verify, don't assume.** Before claiming something works, run it. "It
   compiles" is not "it works". Use the smallest check that proves the claim:
   a unit test, `bun run test:e2e`, or a screenshot you actually looked at.
2. **Report outcomes truthfully.** If a test fails, say so and quote the
   output. If you skipped a step (no network, no ug, no GPU), say which one and
   why. Never write "should work" about something you did not run.
3. **Real vs. simulated is explicit.** Features are either *real* or
   *simulated*; see §3. Never present a simulated capability as real in UI copy,
   docs, or commit messages. A simulated action shows a toast that says it is
   simulated.
4. **Nothing leaves the machine.** No telemetry, analytics, remote logging, or
   cloud APIs. The only network traffic is a user-initiated model download from
   Hugging Face. Adding any other outbound request needs an explicit product
   decision recorded in this file.
   **Recorded exception (product decision, 2026-09-27):** the *web
   deployment* (`firebase.json`, Firebase Hosting site `andai-agent`) loads
   Google Analytics (`src/lib/firebase.ts`, `firebase` npm package). It runs
   only outside the desktop app — `initWebAnalytics` returns immediately when
   `isTauri()`, the firebase packages live in lazy chunks the desktop never
   downloads, and the desktop CSP names no Google host. Both lines are held by
   `tests/unit/firebase-hosting.test.ts`; loosening them needs a new decision.
   **Recorded decision (2026-09-27):** *Add from Hugging Face* (Settings →
   Models) searches and reads model metadata from `huggingface.co/api` when
   the user types or picks a model, besides the download. Same host as the
   downloads (no new CSP source); no credentials, no user data in the
   request beyond the search text; `llm/hub.ts` is the only caller and
   `security.test.ts` holds its one guarded `fetch`.
   **Recorded decision (2026-09-30):** *Install UltraGraph* (shown when ug is
   missing: a launch dialog, Knowledge, Settings, the top-bar badge) installs
   ug from inside Andai. Rust (`ug_install.rs`, the only place that runs
   `curl`) asks `api.github.com` for `shoocstorm/ug`'s latest release and
   downloads the platform's archive from `github.com/shoocstorm/ug/releases/download/…`
   (redirected to GitHub's asset CDN) with the system `/usr/bin/curl`:
   `-q` (no `.curlrc`), HTTPS only incl. redirects, scrubbed env. The
   webview passes no argument and makes no request (no CSP change); the
   archive must match the release's size and sha256 or it is deleted; it
   installs where `install.sh` does (`~/.local/share/ultragraph/.ug`, link in
   `~/.local/bin`) and refuses when ug is already found. macOS and Linux
   x64 only; Windows keeps the website button. `security.test.ts` holds the
   URLs and flags.
5. **User data is sacred.** Never delete, overwrite, or migrate user data
   (`~/Library/Application Support/dev.andai.agent/` or
   `%APPDATA%\dev.andai.agent\` on Windows, including its `logs/`,
   `~/.ug/andai-*`, webview storage) without an explicit user action and a confirm step. Tests must
   never touch it: the e2e harness uses `ANDAI_DATA_DIR` and restores webview
   state (see §6).
6. **Don't guess model or library behavior. Measure it.** wllama, ug, and
   WebKit have surprised us (Memory64, custom-scheme isolation, ACL). When
   behavior matters, probe it and write down the result here or in a test.
7. **Answers must be grounded.** The agent cites retrieved passages as `[n]`,
   and `n` must match the n-th source in the UI (`agent/prompt.ts`). When
   retrieval finds nothing, the prompt says so. Never inject content the user
   did not provide.
8. **Keep the tree green.** `bun run check` passes before any change is
   considered done. Do not disable, skip, or loosen a test to make a change
   pass. Fix the code, or fix the test with a written reason if the test was
   wrong.
9. **Reviewable changes.** Keep each change focused enough to review.
   Refactoring code you pass through is welcome when it makes the code
   clearer, and larger refactors are fine (see the pre-launch note). Say what
   you refactored and why. Don't mass-reformat files for style alone.
10. **Ask when the decision is the user's.** Product scope, new dependencies
    with native code, data-format changes, and anything touching §1.4 or §1.5
    need a human decision. Everything else: make a sensible choice and state it.

---

## 2. Architecture

```
Andai/
├─ src/                      React 19 + Vite UI (runs in the Tauri webview)
│  ├─ llm/engine.ts          load / cache / stream on either engine: wllama (compat build on WKWebView) or native MLX
│  ├─ llm/native.ts          native MLX chat models: command wrappers, streaming channel, wllama-shaped completions
│  ├─ llm/hub.ts             Add from Hugging Face: search, inspect a repo at its commit, compatibility verdict
│  ├─ llm/custom.ts          models the user added: GGUF kept in webview storage, MLX ones read from Rust
│  ├─ llm/models.ts          model catalog: GGUF for wllama (< 2 GB, pinned commit + sha256), MLX checkpoints (pinned in Rust)
│  ├─ llm/integrity.ts       incremental SHA-256 + download verification
│  ├─ llm/decide.ts          choice-based decisions (SemIf): lettered options → one-pass logprob readout, or Laya in Rust
│  ├─ llm/laya.ts            Laya decision model: command wrappers + the checkpoint download (streamed to Rust; Laya and MLX models)
│  ├─ agent/prompt.ts        PURE prompt assembly: keywords, system prompt, history, budgets, agent state
│  ├─ agent/turn.ts          one turn: plan (agent loop) or analyze → retrieve (fixed) → build → generate
│  ├─ agent/loop.ts          agent loop: decide → fill args → policy gate → run tool → observe; writes Message.agent
│  ├─ agent/relevance.ts     relevance check (Laya): score retrieved passages, drop clear misses before the prompt
│  ├─ agent/claims.ts        claim check (Laya): score each cited sentence against its passage, note the unsupported ones
│  ├─ agent/evidence.ts      PURE: merges what the tools found into the answer's passages (same node, covered ranges, order)
│  ├─ agent/tools/           tool registry (code only), ug tools, argument filling, schema validation + GBNF
│  ├─ kb/api.ts              typed wrappers over the Rust ug bridge + hit dedupe
│  ├─ kb/source.ts          PURE: reads ug's file_context report (outline, facts, related files) for the source dialog
│  ├─ kb/samples.ts          the bundled sample knowledge bases (Tidewater Ferries) and their suggested questions
│  ├─ state/                 zustand stores: chat, kb, ugInstall, tools, persona, theme, layout, ui (persisted where noted)
│  ├─ screens/, shell/, components/
│  ├─ theme/tokens.css       ALL colors, both themes
│  ├─ lib/platform.ts        macOS vs Windows in the UI: shortcut labels (⌘ / Ctrl), traffic-light room
│  ├─ lib/activityText.ts    PURE: activity log summaries + levels per event, grouping a day by question (Logs screen)
│  ├─ mock/workflows.ts      data for the simulated Workflows screens
│  ├─ smoke.ts               in-webview test harness (VITE_SMOKE)
│  ├─ bench.ts               engine benchmark (VITE_SMOKE=bench; not in production builds)
│  └─ eval.ts                agent eval harness (VITE_SMOKE=eval; not in production builds)
├─ src-tauri/
│  ├─ src/lib.rs             app setup, navigation lock, drop → file grants
│  ├─ src/activity.rs        activity log (off by default): JSONL per UTC day in app data logs/, 7 days, 20 MB/day; read back by the Logs screen
│  ├─ src/ui_server.rs       loopback server for the release UI (http://localhost:14230)
│  ├─ src/grants.rs          which files the webview may ingest (drop / Rust dialog only)
│  ├─ src/ug.rs              knowledge bases = ug projects (`ug list`), Andai's own `andai-<slug>` over kb/<slug>/docs; `ug gen/search/remove`
│  ├─ src/ug_install.rs      Install UltraGraph: GitHub release → system curl → sha256 → ~/.local/bin/ug (§1.4)
│  ├─ src/tools.rs           agent tool calls: closed enum → validated argv → ug (scrubbed env, 20 s, 256 KB)
│  ├─ src/samples.rs         sample knowledge bases: closed list → bundled files (tests/fixtures/eval/) copied in, then indexed
│  ├─ src/mlx.rs             the one MLX thread (Apple Silicon): owns every model on MLX, runs jobs from commands
│  ├─ src/laya/              Laya decision model on MLX: catalog, verified store (shared with llm/), prompt, model
│  ├─ src/llm/               native chat models on MLX: catalog, Qwen3 port, KV cache + prefix reuse, generation, template;
│  │                         config.rs (what runs), custom.rs (models the user adds: validated manifests)
│  ├─ tauri.laya.conf.json   Apple Silicon build overlay: bundles mlx.metallib (bun run build:mac-arm64)
│  ├─ build.rs               app command manifest (ACL)
│  └─ capabilities/default.json
├─ docs/                     user-facing docs (features.md, …) — index in docs/README.md
│  └─ andai-website/         static product site: index.html, agent-loop.html (how the agent works) + img/
├─ tests/                    setup, guard tests, e2e fixtures, perf/ micro-benchmarks
├─ perf/baseline.json        performance baselines (docs/performance.md)
└─ scripts/                  copy-wllama (postinstall), e2e runner, perf runner, agent eval runner, mlx-metallib, checkpoint-cache
```

**Data flow of a turn (agent mode, a KB selected):** `runTurn` → `runAgent`:
[`decide` (llm/decide.ts: Laya via Rust `laya_decide`, else the wllama decision slot or chat model) → `fillArgs` (chat
model, GBNF from the tool schema) → `validate` → policy gate (Auto / Ask →
approval card) → `kbTool` (Rust `kb_tool` → `ug <cmd> --json`) →
`tool.observe` → `addEvidence` → with Laya, `PassageScorer` scores the new
passages (one `laya_relevance` row per passage, a long one in pieces)] ×
up to `maxSteps`; before answering from a passage the search clipped, it is
read whole, and on code a symbol the request names is fetched (source, or
usages for "who calls") → `mergeEvidence` →
relevance check (with Laya: `checkRelevance` reuses the loop's scores and drops passages scored < 0.10
past the top 2) → `buildSystem` (tool results fenced as passages) + `buildHistory` → `engine.chat` (native
MLX `llm_generate`, streamed over a channel, or wllama) → answer → claim check (with Laya: `checkClaims`
scores each `[n]`-cited sentence against its passage and notes those < 0.10 under the answer). Every decision and call
is written to `Message.agent`, which drives the tool chips, approval cards and
the Execution Trace. **Agent mode off, or no KB:** `runTurn` → `kbSearch` (one
`ug search` with the question) → `buildSystem` → `engine.chat`.

**Agent design (why it is shaped this way).** Small local models can't be
trusted to write tool calls free-form, so the model never names a tool: it
picks a letter from a list (`decide`), which is always valid and yields a
probability per option for the trace. Arguments are the only free text, and
they are held to a grammar, validated in TS, then validated again in Rust,
which is the trust boundary. When a decision can't be trusted (it failed,
chose something not offered, or is below `minConfidence`), the loop falls
back to what the fixed pipeline does: one plain search, then answer. Tools
are offered by KB kind (`KbKind`: code tools only for code/mixed). A `file`,
`symbol` or Read lines `range` argument is held to the files, code symbols
and passage line ranges known so far (`schemaFor`), because free-text names
and line numbers were the commonest failed or aimless calls. A tool picked
before anything has shown such a value looks it up or searches first; its
option is never hidden, since removing options moves a small model's other
choices (§2). The first step of a turn is a search without a decision
when the request plainly asks about content (`searchFirst`, `needsLookup`):
the decision picked search 23 of 24 times, so it cost 0.7 s for nothing.
Keep that check conservative: a false "lookup" searches for "who are u?",
a false "not a lookup" only costs the decision. A Laya
checkpoint (src-tauri/src/laya/) makes the same choice natively: a small
encoder trained for typed decisions scores every option at its marker in one
forward pass, about 8–20 ms instead of ~0.7 s, and returns calibrated
probabilities. It only runs where MLX does (Apple Silicon); elsewhere the
letter readout stays. Read-only
tools default to *Auto* (a product decision, 2026-09-26: they only read the
KB the user selected, and every call is traced); anything with another risk
level defaults to *Ask*.

### Hard-won platform facts (don't re-learn these)

The full list, with measurements and dates, is
**[docs/platform-facts.md](docs/platform-facts.md)**, grouped by area (webview
and Tauri, Windows, web deployment, ug, model downloads, wllama, MLX, Laya,
agent behavior and evals, dev tooling). **Read the section for the area you're
changing before you change it**, and add new measurements there. A code
comment citing "AGENTS.md §2" for a platform fact means that file. The ones
that break a build, a release or the machine:

- Every app command goes in `build.rs` **and** `capabilities/default.json`, or
  it fails only in release (`Command X not allowed by ACL`).
- WKWebView needs wllama's bundled compat build; COOP/COEP only work from the
  loopback origin (`localhost:14230`) in release and from Vite in dev. Don't
  remove either.
- GUI apps don't inherit the shell PATH (`ug_path()` probes the usual dirs).
- Windows is untested end to end (WebView2 is Chromium; the e2e drives
  WKWebView only). Say so wherever Windows support is claimed.
- Andai talks to ug only through its CLI, never its data folder
  (`ug-interface.test.ts`); ug has no `--` separator (`search_query()`).
- Model URLs pin a 40-hex commit; sha256 is the LFS oid.
- MLX needs `mlx.metallib` beside the binary, and the arm64 build needs
  `minimumSystemVersion` 14.0.
- Agent changes are measured with `bun run eval:agent`, never argued; don't
  edit `src/` while it runs (hot reload breaks it).
- `rtk` rewrites `curl` output: use `rtk proxy curl` to parse JSON.

---

## 3. What is real and what is simulated

| Area | Status | Where |
|---|---|---|
| Chat, streaming, stop, think folding | Real | `llm/engine.ts`, `agent/turn.ts` |
| Reasoning chips, Execution Trace, stats | Real (actual step timings, tokens, tok/s; every decision and tool call) | `agent/turn.ts`, `agent/loop.ts`, `screens/AgentTrace.tsx` |
| Knowledge bases: every ug project (`ug list`, no metadata file), create, ingest, index, search, delete; view a source's details, content and structure; read-only for projects Andai didn't create; listed `offline` without ug | Real (ug CLI) | `src-tauri/src/ug.rs`, `state/kb.ts`, `screens/SourceDialog.tsx` |
| Install ug from inside Andai (launch prompt, Knowledge, Settings; release lookup, download progress, sha256, install; macOS/Linux, website on Windows) | Real | `src-tauri/src/ug_install.rs`, `state/ugInstall.ts`, `screens/UgSetup.tsx` |
| Sample knowledge bases (Tidewater Ferries: documents, code, both), suggested questions | Real (bundled files, indexed by the user's ug) | `src-tauri/src/samples.rs`, `kb/samples.ts`, `screens/Knowledge.tsx` |
| Laya decision model (download, verify, load, decide, stop question, relevance check, search scope, claim check; Apple Silicon) | Real | `src-tauri/src/laya/`, `llm/laya.ts`, `llm/decide.ts` |
| Agent tool loop: decisions, 8 ug tools, per-tool policy, approvals, decision model, Tools screen | Real | `agent/loop.ts`, `agent/tools/`, `llm/decide.ts`, `state/tools.ts`, `screens/Tools.tsx`, `src-tauri/src/tools.rs` |
| Persona, auto-optimize | Real | `screens/Persona.tsx` |
| Models: download, load, unload, evict | Real | `llm/engine.ts` |
| Native MLX chat models (download, verify, load, stream, stop, decide; Apple Silicon) | Real | `src-tauri/src/llm/`, `src-tauri/src/mlx.rs`, `llm/native.ts` |
| Add a model from Hugging Face (search, compatibility check, pin, verified download; GGUF anywhere, MLX Qwen3 and Qwen3.5 on Apple Silicon) | Real | `llm/hub.ts`, `llm/custom.ts`, `screens/HubModels.tsx`, `src-tauri/src/llm/custom.rs` |
| Appearance (light / dark; first run follows the OS) | Real | `state/theme.ts` |
| Activity log (off by default; each step, argument writer call, tool call, context and answer, plus model and KB events, as JSONL with a one-line summary and level, 7 days) and its screen (⌘L: by question, filters, search, raw data) | Real | `state/activity.ts`, `lib/activityText.ts`, `src-tauri/src/activity.rs`, `screens/Logs.tsx`, Settings |
| Layout: collapsible nav (⌘B), Execution Trace on/off (⌘J) | Real, persisted | `state/layout.ts` |
| Workflows, approvals, tool library, node editor, run | **Simulated** | `mock/workflows.ts`, `screens/Workflow*.tsx` |
| "Choose tool" menu: the tools offered for the selected KB, with their policy | Real | `screens/CommandCenter.tsx` (registry) |

Promoting a simulated feature to real requires: a design note in this file,
tests at the same level as the real features, and removal of the "simulated"
copy **everywhere**: app UI, `docs/features.md`, and the website's Preview
section and FAQ (see §8).

---

## 4. Conventions

### Code
- TypeScript strict, React function components, zustand for state.
  Components read stores with selectors; I/O lives in `state/*` actions or
  `agent/*`, not in components.
- **Pure logic goes in pure modules** (like `agent/prompt.ts`) and gets unit
  tests. Orchestration (`agent/turn.ts`) is tested with the engine and ug
  mocked.
- Match the surrounding code's comment density. Comments explain *why*,
  especially platform workarounds; link the §2 fact they rely on.
- Rust: `cargo clippy --all-targets -- -D warnings` must be clean. Commands
  return `Result<_, String>` with messages a user can act on. Validate every
  path-like argument (`valid_slug`, `valid_source_name`); never join raw user
  input into paths.
- Run ug and other processes off the async runtime (`spawn_blocking`). Stream
  long-running progress as events (`kb-progress`), never block the UI.

### Design system
- **All colors come from `src/theme/tokens.css`.** Both themes define the same
  semantic tokens: `:root` is dark, `:root[data-theme='light']` is light. No hex
  literals anywhere else. A genuine exception (a theme preview swatch, a mask)
  is marked `/* theme-literal */` or wrapped in
  `/* theme-literal: start */ … /* theme-literal: end */`.
  `tests/unit/design-tokens.test.ts` enforces this, plus WCAG-AA contrast for
  light-theme text and accents.
- Adding a color token means adding it to **both** themes.
- Every UI change is checked in **both themes** (see §6, screenshots).
- Visual language: Space Grotesk for headings, Inter for body text,
  JetBrains Mono for uppercase tracked labels. 150–250 ms motion, used
  sparingly. The references are in `assets/`.
- Accessibility: interactive elements are real `<button>`/`<input>` with
  accessible names; toggles use `role="switch"`; radio groups use
  `role="radio"`. Component tests query by role, so broken semantics break
  tests.

### Tauri commands (checklist for adding one)
1. `#[tauri::command]` fn in `src-tauri/src/*.rs`
2. Register it in `generate_handler!` in `lib.rs`
3. Add it to `COMMANDS` in `build.rs`
4. Add `allow-<name-with-dashes>` to `capabilities/default.json`
5. Add a typed wrapper in `src/kb/api.ts` (or a sibling module)
6. `bun run test` then checks steps 2–4 (`tauri-acl.test.ts`), and
   `bun run test:e2e:release` proves it works in the shipped build.
7. Go through the security checklist in §9. The caller is the webview, and
   the webview is untrusted.

### Dependencies
- Pin exact versions in `package.json`. `@wllama/wllama` and
  `@wllama/wllama-compat` **must stay on the same version**. After bumping
  them, run `bun install` (re-copies the wasm) and both e2e runs.
- No new runtime network dependencies (§1.4). Anything with native code needs
  a human decision (§1.10). Exception recorded above: `firebase` (JS only,
  no install script needed — `@firebase/util`'s and `protobufjs`'s blocked
  postinstalls stay blocked), web-only lazy chunks.
- `overrides` pins `@grpc/grpc-js` to a patched release (GHSA-m9gg-hp2v-232j,
  2026-10-02): `@firebase/firestore` pins `~1.9.0`, which `bun audit` fails
  as high. Firestore is never imported (only `firebase/app` and
  `firebase/analytics`), so the override changes no shipped code. Drop it
  when a `firebase` release no longer pins a vulnerable version.

### Git: work on `main`, share it with other agents
- **`bun install` installs a pre-push hook** (`.githooks/pre-push` via
  `core.hooksPath`): every push runs `bun run check` first, so a broken tree
  fails locally instead of in CI. Bypass once with `git push --no-verify`
  (only when the user asked for it, like `--skip-checks`).
- **Commit on `main`.** Don't create a branch, worktree or PR unless a human
  asked for one or confirmed it first. This overrides any default of
  "branch first".
- **Other agents may be changing `main` at the same time**, in this checkout
  or another one. The working tree and the index may hold their uncommitted
  work, so:
  - Stage only the files you changed, by path (`git add <paths>`). Never use
    `git add -A`, `git add .` or `git commit -a`, and never `git reset`,
    `git stash`, `git checkout -- <file>` or `git clean` on changes you didn't
    make.
  - Run `git status` before committing. If a file you need already has
    someone else's changes in it, commit only your hunks or ask the human.
    Don't revert or "fix" their work.
  - Before pushing, run `git pull --rebase`, then re-run the checks your change
    needs. Never force-push `main`. If the rebase conflicts with another
    agent's work, resolve it only when the intent of both sides is clear.
    Otherwise stop and ask.
  - Keep commits small and self-contained (§1.9), so concurrent work
    rebases cleanly.

---

## 5. Commands

```bash
bun install              # deps + copies wllama wasm into public/wllama/
bun run tauri dev        # run the app (hot reload)
bun run tauri build      # → src-tauri/target/release/bundle/macos/Andai.app

bun run check            # typecheck + unit/component tests + Rust tests + clippy  ← before every change is "done"
bun run test             # Vitest: unit, component, guard tests (~2 s)
bun run test:coverage    # same, with a coverage report in coverage/
bun run test:rust        # Rust unit tests
bun run test:ug          # Rust ↔ real ug integration (needs ug)
bun run test:laya        # Laya port vs laya-mlx goldens + < 100 ms per decision (needs the checkpoints in the HF cache)
bun run test:llm         # native Qwen3 port vs transformers/mlx-lm goldens + tok/s (needs the MLX checkpoints in the HF cache)
bun run build:mac-arm64  # Apple Silicon release bundle with mlx.metallib (release.yml uses it)
bun run deploy:web       # build + firebase deploy → https://andai-agent.web.app (web app, no knowledge feature; needs the firebase CLI)
bun run test:e2e         # full app in WKWebView: ingest → retrieve → generate (needs ug; downloads model once)
bun run audit            # bun audit (JS deps) + cargo audit (RustSec); CI and releases run it
bun run test:e2e:release # same against the release binary (localhost origin + ACL + Finder-like PATH)
bun run perf             # bundle size (CI too) + micro-benchmarks vs. perf/baseline.json
bun run bench:engine     # engine probe: GPU layers, threads, prompt and generation tok/s per wllama setting (needs a downloaded model); BENCH_MODEL=qwen3-1.7b-mlx for native MLX
bun run eval:agent       # agent eval: 100 questions (4 knowledge bases, one of long documents) through the real agent → scorecard vs. perf/baseline.json (needs ug + model); read and compare reports with bun run eval:view
```

**Run the e2e tests with the default model, `qwen3-0.6b`** (don't set
`E2E_MODEL`). Only a pass with it counts toward the definition of done (§6).
`E2E_MODEL=stories-260k` is a quick plumbing check, not a result. Its context
is only 1024 tokens, so a grounded prompt overflows it (measured: 1033 tokens,
"exceeds the available context size"). It also skips the answer-grounding
check, so it fails or passes for reasons that say nothing about the app.

**Bun is the package manager and script runner** (`packageManager` in
`package.json`, lockfile `bun.lock`). Don't use npm, and don't commit a
`package-lock.json`. Things to know:
- Run the test suite with `bun run test`. Plain `bun test` starts Bun's own
  test runner instead of Vitest and fails.
- **Bun is also the runtime** for everything except Vitest: `scripts/*.mjs`,
  `tsc`, Vite and the Tauri CLI run with `bun` / `bun --bun` (see the
  `package.json` scripts). Don't add new `node` or `npx` calls. When a
  package script has the same name as a binary, call the binary with
  `bunx --bun <bin>`. Inside `"tauri": "bun --bun tauri"`, `tauri` resolves to
  the script itself and bun keeps launching itself (seen once).
- **Vitest stays on Node 22+.** Under `bun --bun vitest`, jsdom's `window`
  isn't a valid `EventTarget` and every DOM test worker crashes (measured with
  Bun 1.3.14 and Vitest 5). `bun run test` starts Vitest on Node through its
  shebang. Re-measure before moving it.
- Dependency lifecycle scripts only run for packages listed in
  `trustedDependencies`. If a new dependency needs its install script, add it
  there on purpose (`bun pm untrusted` lists the blocked ones).

The RTK shell proxy on this machine can mangle test-runner output. Prefix a
command with `rtk proxy` to see the raw output.

---

## 6. Testing strategy

| Layer | Tool | Location | Covers |
|---|---|---|---|
| Unit | Vitest | `src/**/*.test.ts` | pure logic: prompt assembly, agent state, think split, hit dedupe, decision readout, schema validation + GBNF, ug output readers, theme resolution, formatting, model catalog |
| Orchestration | Vitest + `vi.mock` | `src/agent/turn.test.ts`, `src/state/kb.test.ts` | turn step transitions, failure paths, abort, history; the agent loop with a scripted decider (fallbacks, approval, denial, repeat guard, step limit); KB actions against a mocked Rust bridge |
| Component | Testing Library (jsdom) | `src/**/*.test.tsx` | user-visible behavior, queried by role/text |
| Guards | Vitest | `tests/unit/` | design-token and Tauri ACL invariants |
| Rust unit | `cargo test` | `src-tauri/src/ug.rs` | ingestion, validation, status derivation, serialization |
| Rust ↔ ug | `cargo test -- --ignored` | same | real `ug gen` + `search` round trip |
| End-to-end | `scripts/e2e.mjs` + `src/smoke.ts` | real app | isolation, threads, ingest, retrieval, grounded answer; dev and release |
| Performance | `scripts/perf.mjs`, `tests/perf/`, e2e runner | `perf/baseline.json` | bundle sizes; hot-path timings; ingest, search, load, first token, tok/s ([docs/performance.md](docs/performance.md)) |
| Agent eval | `scripts/eval-agent.mjs` + `src/eval.ts` | `tests/fixtures/eval/`, `perf/baseline.json` | what the agent does: first-action accuracy, answer facts, grounding, wasted calls, decisions and time per question ([docs/performance.md](docs/performance.md#agent-eval-bun-run-evalagent-section-agent-eval)) |

Rules:
- **New behavior ships with a test at the lowest layer that can catch its
  regression.** Bug fixes start with a failing test.
- Test behavior, not implementation: assert on what the user or the next
  module sees.
- A guard test must be shown to fail. When you add one, break the invariant
  once and watch it go red.
- The e2e harness **must not** touch user data: `ANDAI_DATA_DIR` isolates KB
  files, and the harness snapshots and restores chat and KB selection. The ug
  project it creates (`andai-e2e-docs`) is removed before it reports `OK`,
  and it lives in the run's own `UG_HOME`, so the user's ug projects (which
  Andai lists) never appear in a run. The eval and bench runners do the same.
- **Performance is a tested behavior.** A change that makes a baselined
  metric worse than its tolerance fails `bun run perf` or the e2e run. If the
  cost is intended, re-record the baseline with `--update` in the same change
  and give the reason in the commit. Don't raise a tolerance to get a change
  through. Timing baselines are machine-bound: they're enforced only on the
  machine that recorded them (docs/performance.md).
- **Heavy runs, one at a time.** An agent eval, e2e run, `bench:engine` or
  `test:llm` starts the app with models on the GPU. About twenty evals back
  to back (2026-09-28) overheated a MacBook to a black screen and a forced
  power-off. Run one, pause, check `pgrep -fl "tauri dev|andai|vite"` for
  leftovers, prefer `--only` for a few questions, and stop when runs slow
  down unexpectedly (a sign the machine is throttling). Never queue a batch
  of them in the background.
- **Agent changes are measured, not argued.** A change to the agent loop,
  decision options, tool offers, argument filling or observations runs
  `bun run eval:agent` before and after, and the commit says what moved
  (`--against` lists the questions that changed). A worse scorecard needs
  a reason, like any other regression.
- Visual checks: headless Chrome against `bun run dev` renders every screen
  (`/#command`, `/#knowledge`, …). Seed `localStorage` (`andai.theme`,
  `andai.chat`) to check both themes and populated states. Look at the images;
  don't just generate them.

### Definition of done
- [ ] `bun run check` is green
- [ ] New or changed behavior has tests (§6 rules)
- [ ] UI changes looked at in light **and** dark
- [ ] Touching engine, ug, Tauri config or commands: `bun run test:e2e` and
      `bun run test:e2e:release` pass with the default model, `qwen3-0.6b` (§5)
- [ ] Touching a hot path (prompt, retrieval, chat store, Markdown, engine,
      dependencies): `bun run perf` passes, or the baseline was re-recorded
      with a reason (docs/performance.md)
- [ ] Real vs. simulated table (§3) and platform facts (§2) still true, or updated
- [ ] README (users), `docs/development.md` (build, test, release) and this file updated if commands, setup or behavior changed
- [ ] **Docs and website updated** for any user-visible change (§8), in the same change
- [ ] Security checklist (§9) holds, and `tests/unit/security.test.ts` passes without loosening an allowlist

---

## 7. Releases

Releases are **tag-driven**. Pushing a `vX.Y.Z` tag runs
`.github/workflows/release.yml`, which does three things:
1. **verify:** the tag must equal the version in every manifest, the full
   check suite must pass, and `bun run audit` must find no known
   vulnerability (§9).
2. **build:** `Andai.app` + `.dmg` for Apple Silicon and Intel, and an NSIS
   installer (`Andai_X.Y.Z_x64-setup.exe`) for Windows, with `.sha256`
   checksums. It asserts that the built `Info.plist` and the installer name
   report the tag's version.
3. **publish:** a GitHub Release whose notes are the annotated tag's message.

**Always cut releases with the script.** Never hand-edit versions or push tags
manually:

```bash
bun run release --dry-run             # ALWAYS first: shows next version + grouped notes; changes nothing
bun run release                       # patch bump (0.1.0 → 0.1.1), asks before pushing
bun run release minor                 # or: major, or an explicit 0.3.0
bun run release minor --yes --watch   # non-interactive, then follows the workflow to the published release
```

What the script does, in order:
1. Preflight: a branch (not detached), a clean tree, not behind origin, and
   the tag free locally and on origin.
2. `bun run check`.
3. `bun scripts/version.mjs set X.Y.Z` on all four manifests.
4. Confirm.
5. Commit `release: vX.Y.Z`, create an annotated tag with the notes, and push
   branch and tag atomically.

If it stops before the commit, it leaves nothing behind: a declined
confirmation reverts the bump.

Rules for agents:
- A release is **outward-facing**. Only cut one when the user asked for a
  release (that request covers `--yes`), and run `--dry-run` first so the
  version and notes can be reported back.
- Release notes come from commit subjects since the last tag, grouped by
  Conventional Commit prefix (`feat:`, `fix:`, `perf:`, `refactor:`; anything
  else goes under *Other*). Write commit subjects accordingly.
- Never pass `--skip-checks` or `--allow-dirty` unless the user explicitly
  asked for it.
- If the workflow fails, don't delete or re-push the tag blindly. Read
  `gh run view <id> --log-failed`, fix the problem on `main`, then cut the
  next patch version. To rebuild an existing tag unchanged, use the
  workflow's *Run workflow* button with the tag (`workflow_dispatch`).
- `bun run version` prints the version, and `bun run version check` checks
  that the manifests agree (also enforced by `tests/unit/version.test.ts` and
  CI).

The Apple Silicon build uses `bun run build:mac-arm64`, which bundles MLX's
`mlx.metallib` for the models that run on MLX, Laya and the native chat
models (about 58 MB more in the DMG; the workflow checks it's there). Intel
and Windows builds have no MLX: no Laya and no MLX models, only wllama.

Code signing: builds are **unsigned** until the repository has the secrets
`APPLE_CERTIFICATE`, `APPLE_CERTIFICATE_PASSWORD` and `APPLE_SIGNING_IDENTITY`
(add `APPLE_ID`, `APPLE_PASSWORD` and `APPLE_TEAM_ID` for notarization). The
workflow detects them automatically, and the release notes say whether a
build is unsigned. Windows builds are always unsigned for now (no certificate
is configured); the release notes say so.

Before announcing a release, also run `bun run test:e2e:release` locally
(CI can't run it: it needs ug and a model). Then launch the downloaded `.dmg`
build, ask a grounded question and switch themes. Do the same with the
Windows installer on a Windows PC when one is available, and record the
result in §2 and `docs/features.md`.

Known constraints to keep in mind: the release UI needs port **14230** free.
If another process holds it, Andai refuses to start and shows why, and never
loads the other process's page (§9). Also, dev (`localhost:1420`) and
release (`localhost:14230`) have separate webview storage, including separate
model caches.

---

## 8. Docs and website: keep them in sync with the code

> **Reminder for AI coding agents:** when you change what a user can see or
> do, update `docs/` and `docs/andai-website/` **in the same change**, before
> you call it done. Stale docs are bugs. A website that promises something the
> app doesn't do breaks rule §1.2 (report truthfully) in public.

| You changed… | Update |
|---|---|
| A feature's behavior, a new feature, or a removed one | `docs/features.md` (right section, status *Available*/*Preview*) and the website's feature cards or showcase rows |
| A screen's look (layout, theme, copy) | Re-capture the affected screenshots in `docs/andai-website/img/` (see below) |
| Something simulated became real (or the reverse) | `docs/features.md` status, website Preview section + FAQ, §3 table here |
| Shortcuts | `docs/features.md` → Workspace, website "Focus mode" row, README, About dialog |
| Models, sizes, supported file types, requirements, ports | `docs/features.md` → Models / Requirements, website FAQ + proof strip |
| Performance numbers | Only publish numbers you **measured**, and say on what (model, Mac). Re-measure before changing them |
| Install / release / signing | Website download CTA + FAQ, `docs/features.md` → Requirements, README → Get started |
| A new doc | Link it from `docs/README.md` |

Rules:
- **No claim without evidence.** Every number, platform and capability on the
  site or in the docs must be something that was run and observed. Say
  "tested on …" rather than implying broader support. Don't invent
  testimonials, user counts or benchmarks.
- **Screenshots are real captures** of the current app, never mock-ups. Use
  2× WebP at 1440×920 viewport, named `<screen>-<theme>.webp`. Capture with
  headless Chrome against `bun run dev`, seeding `localStorage`
  (`andai.theme`, `andai.chat`, `andai.layout`). Look at every image before
  committing it. Don't capture screens that show browser-only states (e.g. the
  Knowledge screen's "desktop runtime required" notice) as if they were the app.
- The website is **self-contained static pages** (`index.html`, plus
  `agent-loop.html`, which explains the agent loop with diagrams and real
  eval traces; each page linked from the home page) and `img/`: no build step
  and no JS framework. When the loop changes (decisions, tools, relevance
  check), update `agent-loop.html` in the same change; its examples and
  numbers come from `bun run eval:agent` reports. Its tokens mirror
  `src/theme/tokens.css`; if the app's palette changes, update the site's
  `:root` blocks to match. It must work in light and dark and at phone width
  (no horizontal scroll at 390 px).
- `tests/unit/docs.test.ts` guards the mechanical part: local links and images
  resolve, anchors exist, screenshots have alt text, and the Workflows preview
  stays labeled simulated. It can't judge whether prose is still *true*.
  That part is on you.

---

## 9. Security

Security is part of the product promise ("your documents never leave the
machine"), not a feature on top of it. Every rule below is enforced by a test.
When you change one, change its test and record the reason here.

### Threat model

| Asset | Where |
|---|---|
| The user's documents | KB copies in app data, ug graphs in `~/.ug/` (Andai's are `andai-*`), and the folders the user's own ug projects index |
| Chats, persona, settings | webview storage (localStorage) |
| Anything else readable by the user | the filesystem the Rust side can reach |

| Adversary | Can | Mitigation |
|---|---|---|
| **A malicious document** (prompt injection) | steer what the model writes, which is then rendered in the webview | Markdown loads no images and links are copy-only (`components/ui.tsx`); the CSP blocks egress; the navigation lock |
| **Script running in the webview** (the result of any of the above, or a compromised dependency) | call every app command | Rust validates every argument; file grants; argument hardening for ug; harness commands gated |
| **Another local process** | squat port 14230 to serve its own page with our IPC | `ui_server.rs` binds first on both loopback families or refuses to start; Host check against DNS rebinding |
| **A web page in the user's browser** | send requests to `localhost:14230` | Host must be `localhost:14230`; only static assets are served; the IPC bridge exists only in the app's webview |

Out of scope for now: an attacker with the user's login or root, and physical
access (rely on FileVault). Encryption at rest is planned (below).

### Principles

1. **The webview is untrusted. Rust is the trust boundary.** Validate every
   command argument as if an attacker sent it. The UI's checks are for UX,
   not safety.
2. **Least privilege.** A command, plugin permission, CSP source or origin
   must justify its presence. The allowlists live in
   `tests/unit/security.test.ts`, and growing one is a product decision (§1.10).
3. **Egress is deny-by-default.** CSP `connect-src` is `self`, IPC and Hugging
   Face. Nothing in `src/` besides `llm/models.ts` names a remote URL, only
   `llm/laya.ts` calls `fetch` (a Laya or MLX checkpoint, from pinned commit
   URLs), and the Rust side has no HTTP client; its one request path is the
   user-started ug install, through the system curl (§1.4).
4. **Users choose files, never the webview.** `kb_add_files` only accepts a
   path the user granted by drag-and-drop or the dialog opened by Rust
   (`grants.rs`). Each grant is one file, canonicalized and consumed once.
5. **Fail closed.** If the port can't be owned, there is no window. If a
   path isn't granted, it isn't read. If a query looks like a flag, it's
   neutralized. If a model doesn't match its sha256, it isn't loaded.
6. **Don't rely on the model behaving.** Fencing passages makes injection
   harder, but the real guarantees are that output is inert (no images, no
   live links, no HTML) and that egress is blocked.

### What enforces it

| Control | Code | Test |
|---|---|---|
| CSP (tauri.conf.json, mirrored by Vite in dev) | `tauri.conf.json`, `vite.config.ts` | `security.test.ts`; e2e: eval blocked, egress canary gets no requests |
| Safe Markdown rendering | `components/ui.tsx` | `components/ui.test.tsx` |
| Navigation lock, no new windows | `lib.rs`, `ui_server::is_app_url` | Rust unit test; e2e |
| Owned loopback server, Host check, hardening headers | `ui_server.rs` | Rust unit tests (squatted port, 403, headers, real sockets) |
| File grants | `grants.rs`, `ug::add_sources` | Rust unit tests (ungranted, consumed once, symlink); e2e |
| ug argument hardening, budgets clamped | `ug::search_query`, `ug::search_limits` | Rust unit tests |
| Private files (0700 dirs / 0600 files; Unix only, see §2), 100 MB cap, absolute PATH only | `ug.rs` | Rust unit tests |
| Windows installer makes no network request (WebView2 not bootstrapped) | `tauri.conf.json` | `security.test.ts` |
| Activity log: a closed set of event kinds and levels, turn ids `[A-Za-z0-9_-]{1,64}`, a summary of one line ≤ 1 KB, ≤ 64 events a call, ≤ 256 KB an event, 20 MB a day; file names and times from Rust's clock; 0700/0600; only `agent-YYYY-MM-DD.jsonl` files are read, pruned (7 days) or deleted; `activity_read` takes only such a name (a regular file, not a link; the newest 5,000 lines), and the Logs screen shows it as plain text; `activity_open` takes no argument. Off by default (a product decision, 2026-09-30); the webview decides when to write, so the caps are what bound it | `activity.rs` | Rust unit tests |
| `dev_log` / `dev_exit` need `ANDAI_SMOKE=1` | `lib.rs` | `security.test.ts` |
| `ug_install` takes no argument: fixed GitHub URLs (asset URL must be `…/releases/download/<tag>/<exact asset name>`), system curl with `-q`, HTTPS-only redirects and a scrubbed env; release JSON ≤ 2 MB, archive ≤ 512 MB, size + sha256 (the release's `digest`, else its `.sha256` file) checked before unpacking; unpacked beside the old folder first; never replaces an existing ug or a non-link `~/.local/bin/ug`; one install at a time | `ug_install.rs` | Rust unit tests (asset picking, checksum, install, bad archive); `cargo test installs_the_latest_release_for_real -- --ignored` (network); `security.test.ts` |
| `open_ug_website` takes no argument: it opens only `ug::UG_WEBSITE` (the ug install page, shown when ug is missing) in the system browser; the app makes no request and `src/` names no URL | `ug.rs` | `tauri-acl.test.ts`, `security.test.ts` |
| Webview holds no fs/shell/http/opener/dialog permission | `capabilities/default.json` | `security.test.ts` |
| Model downloads pinned to a commit and verified (size + sha256) before load; mismatch → removed | `llm/models.ts`, `llm/integrity.ts`, `engine.loadModel` | `integrity.test.ts` (FIPS vectors, tamper), `engine.test.ts` (gate), `models.test.ts` (pinning); e2e logs the check |
| Laya checkpoints: closed catalog in Rust (commit, sizes, sha256); chunks land in `.part` files and only `laya_finish` moves them into place, after every size and hash matches, else all are deleted; `load` reads only a verified folder | `laya/catalog.rs`, `laya/store.rs` | Rust unit tests (mismatch, order, oversize, tampering, 0600/0700); e2e downloads and verifies once |
| Laya decisions: state ≤ 64 KB, question ≤ 2 KB, 2–16 options with `[a-z0-9_]` ids, text ≤ 1 KB; mask tokens stripped from all input | `laya/mod.rs` `validate`, `laya/prompt.rs` | Rust unit tests |
| Laya relevance and claim checks: a fixed statement in Rust (`RELEVANT`, `SUPPORTS`), never text from the webview; 1–24 passages or claims, request ≤ 4 KB, statement ≤ 2 KB, passage ≤ 16 KB, source ≤ 512 bytes | `laya/mod.rs` `validate_passages`, `validate_claims` | Rust unit tests |
| MLX chat models: closed catalog in Rust (commit, sizes, sha256), the same verified store as Laya; generation bounded (1–512 messages ≤ 512 KB, no NUL, max_tokens ≤ 8192, sampling ranges, grammar ≤ 8 KB, context 512–32768); roles are system/user/assistant only | `llm/catalog.rs`, `laya/store.rs`, `llm/mod.rs` `validate` | Rust unit tests; `test:llm` |
| Source dialog (`kb_source`): the file must be one the KB lists (Andai's `docs/` copies, or the repo-relative paths `ug files` reports; normal components only, exact match), read from the KB's root (app data, or the `repoRoot` ug recorded), never through a symlink nor resolving outside the root, at most 512 KB; its outline comes from `ug file_context file:<name>` (a node id, so a name can't parse as a flag) under `tools::run` (scrubbed env, 20 s, 1 MB cap). Text is shown through `<Markdown>` or as plain text | `ug.rs` `kb_source`, `screens/SourceDialog.tsx` | Rust unit tests; `test:ug` checks the outline against real ug; `SourceDialog.test.tsx` |
| Sample knowledge bases: a closed list of ids; files only from the app's resource folder, copied like a user's | `samples.rs` | Rust unit tests |
| Models added from Hugging Face: only public, ungated repos; pinned to the commit seen when picked; GGUF by its LFS sha256 (the webview's integrity gate); MLX through Rust: repo/commit syntax, a closed set of file names (config, tokenizer, template, safetensors; no pickle, no code), sizes and caps (32 GB a file, 64 GB a model, 16 MB inline, 32 models), config and template checked before any download, manifest written and re-validated by Rust, inline files re-hashed on every load. Search results and model data are shown as text, never Markdown or HTML; no model card is rendered | `llm/hub.ts`, `llm/custom.ts`, `src-tauri/src/llm/custom.rs`, `config.rs` | `hub.test.ts`, `custom.test.ts`, Rust unit tests, `security.test.ts`; e2e (search, inspect, add, remove) |
| Only `llm/laya.ts` may `fetch`, and only `pinnedFileUrl(...)` (pinned HF commits); tokenizers built without its `http` feature | `llm/laya.ts`, `llm/models.ts`, `Cargo.toml` | `security.test.ts`, `models.test.ts` |
| Pre-pinning model copies removed only after the user confirms | `engine.removeLegacyCopies`, Settings | `Settings.test.tsx` |
| Retrieved passages fenced as untrusted data; a passage can't close its fence | `agent/prompt.ts` | `prompt.test.ts` |
| Agent tools: closed enum, no unknown fields, flag-like and KB-escaping args rejected (incl. symlinks), scrubbed env, 20 s kill, 256 KB cap, one KB's project only. A KB id is a project name (`valid_project`) that resolves to Andai's folder or a project `ug list` reports; its root comes from app data or ug's registry, never the webview | `tools.rs` | Rust unit tests; `test:ug` runs every tool against real ug; e2e |
| Tools registered in code only; read-only by default Auto, other risks default Ask; Off is never offered; Ask waits for approval; every call traced | `agent/tools/registry.ts`, `agent/loop.ts`, `state/tools.ts` | `tools.test.ts`, `turn.test.ts` |
| No known vulnerabilities in shipped dependencies | `bun run audit`, `ci.yml` (audit job), `release.yml` (verify) | CI |

`ANDAI_SMOKE` and `ANDAI_E2E_FILES` are read from the environment by Rust. The
webview can't set them. Only the e2e runner does.

### Security checklist (every change)

- [ ] New command: every argument is validated in Rust (slugs, names, sizes,
      ranges). Paths come from a grant, never from the webview directly.
- [ ] Anything passed to a CLI can't be parsed as a flag, and each argument
      is its own `arg()`.
- [ ] No new network destination: no CSP source, URL literal or HTTP client
      without a product decision recorded in §1.4.
- [ ] Model or document text is rendered only through `<Markdown>`, never as
      raw HTML, and never as a live link or image.
- [ ] New plugin permission or capability: justified here, and added to the
      allowlist in `security.test.ts`.
- [ ] New files under app data are created with private permissions
      (`create_private_dir` / `private_file`).
- [ ] Anything that accepts a model from outside the catalog goes through
      `llm/hub.ts` and, for MLX, `llm/custom.rs`: never a URL, path or hash the
      webview made up without those checks.
- [ ] New or updated catalog model: `url` pinned to a commit, with `bytes` and
      `sha256` taken from Hugging Face's `paths-info` (download it once and
      check). Move the old URL into `legacyUrls`.
- [ ] Retrieved or user-supplied text goes to the model through
      `buildSystem`'s fences, never spliced into instructions.
- [ ] New agent tool: a `ToolCall` variant validated in `tools.rs`, a `risk`
      level, a policy default from `defaultPolicy`, and an `observe` that
      tolerates any output shape (`tools.test.ts` feeds it junk).

### Known advisories

`cargo audit` reports no vulnerabilities, but it does report warnings (checked
2026-09-26), all transitive through Tauri: unmaintained `unic-*`
(RUSTSEC-2025-0075/0080/0081/0098/0100), `proc-macro-error`
(RUSTSEC-2024-0370), and `glib` unsoundness (RUSTSEC-2024-0429, Linux-only
GTK stack, not in the macOS build). Warnings don't fail the audit. Re-check
them when bumping Tauri.

### Planned

- **Encryption at rest** (needs a human decision, §1.10): a Keychain-held key,
  chats moved to an encrypted file owned by Rust, and encrypted KB copies, with
  a one-time migration behind a confirm step.
- **Gate for new tools and workflows (§3):** every model-initiated action has
  a per-tool policy (*Auto / Ask / Off*) and an entry in the Execution Trace
  (done for the ug tools). The read-only ug tools default to *Auto* by
  product decision (2026-09-26). Any tool that writes, or reaches beyond the
  selected knowledge base, must default to *Ask* (`registry.defaultPolicy`)
  and needs its own §9 review before it ships.
- **Code signing and notarization:** the workflow is ready (§7); it needs the
  Apple secrets.

The user-facing version of this section is [docs/security.md](docs/security.md).
Keep the two in sync (§8).

