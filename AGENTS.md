# AGENTS.md — Andai engineering guide

Andai is a **local-first AI agent desktop app**. It is a product, not a demo:
every change ships to users who trust it with their documents. This file is the
contract for anyone changing the code, human or AI agent. Read it before
editing. When this file and your instincts disagree, this file wins. When it
is wrong, fix it in the same change.

---

## 1. Grounding rules (non-negotiable)

These exist because each one was either violated once or would silently break
the product.

1. **Verify, don't assume.** Before claiming something works, run it. "It
   compiles" is not "it works". Use the smallest check that proves the claim:
   a unit test, `npm run test:e2e`, or a screenshot you actually looked at.
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
5. **User data is sacred.** Never delete, overwrite, or migrate user data
   (`~/Library/Application Support/dev.andai.agent/`, `~/.ug/andai-*`, webview
   storage) without an explicit user action and a confirm step. Tests must
   never touch it: the e2e harness uses `ANDAI_DATA_DIR` and restores webview
   state (see §6).
6. **Don't guess model or library behavior. Measure it.** wllama, ug, and
   WebKit have surprised us (Memory64, custom-scheme isolation, ACL). When
   behavior matters, probe it and write down the result here or in a test.
7. **Answers must be grounded.** The agent cites retrieved passages as `[n]`,
   and `n` must match the n-th source in the UI (`agent/prompt.ts`). When
   retrieval finds nothing, the prompt says so. Never inject content the user
   did not provide.
8. **Keep the tree green.** `npm run check` passes before any change is
   considered done. Do not disable, skip, or loosen a test to make a change
   pass. Fix the code, or fix the test with a written reason if the test was
   wrong.
9. **Small, reviewable changes.** One concern per change. Don't refactor
   unrelated code or reformat files you didn't otherwise touch.
10. **Ask when the decision is the user's.** Product scope, new dependencies
    with native code, data-format changes, and anything touching §1.4 or §1.5
    need a human decision. Everything else: make a sensible choice and state it.

---

## 2. Architecture

```
Andai/
├─ src/                      React 19 + Vite UI (runs in the Tauri webview)
│  ├─ llm/engine.ts          wllama: load / cache / stream. Compat build on WKWebView
│  ├─ llm/models.ts          model catalog (single-file GGUF, < 2 GB each)
│  ├─ agent/prompt.ts        PURE prompt assembly: keywords, system prompt, history, budgets
│  ├─ agent/turn.ts          one turn: analyze → retrieve (ug) → build → generate; writes the trace
│  ├─ kb/api.ts              typed wrappers over the Rust ug bridge + hit dedupe
│  ├─ state/                 zustand stores: chat, kb, persona, theme, layout, ui (persisted where noted)
│  ├─ screens/, shell/, components/
│  ├─ theme/tokens.css       ALL colors, both themes
│  ├─ mock/workflows.ts      data for the simulated Workflows screens
│  └─ smoke.ts               in-webview test harness (VITE_SMOKE)
├─ src-tauri/
│  ├─ src/lib.rs             app setup; release UI served from http://localhost:14230
│  ├─ src/ug.rs              knowledge bases → `ug gen/search/list/remove` CLI
│  ├─ build.rs               app command manifest (ACL)
│  └─ capabilities/default.json
├─ tests/                    setup, guard tests, e2e fixtures
└─ scripts/                  copy-wllama (postinstall), e2e runner
```

**Data flow of a turn:** `runTurn` → `kbSearch` (Rust → `ug search --json`) →
`dedupeHits` → `buildSystem` + `buildHistory` → `engine.chat` (wllama stream) →
the message trace in the chat store drives the reasoning chips and the
Execution Trace.

### Hard-won platform facts (don't re-learn these)

- **WKWebView lacks Memory64 and JSPI.** wllama's default wasm fails, so
  `engine.ts` always calls `setCompat()` with the bundled
  `public/wllama/compat/*` files, never the CDN. wllama only uses compat when
  `needCompat()` says so, so Chromium still gets the fast build.
- **WebKit ignores COOP/COEP on custom schemes.** `tauri://` never becomes
  `crossOriginIsolated`, and without `SharedArrayBuffer` wllama runs
  single-threaded. Release builds therefore serve the UI from
  `http://localhost:14230` via `tauri-plugin-localhost`, which adds the headers.
  Dev gets the same headers from Vite. Don't remove either.
- **That loopback origin is "remote" to Tauri's ACL.** Every app command must
  be listed in `build.rs` **and** granted in `capabilities/default.json`
  (`allow-<name>`). Missing it fails **only in release**, with
  `Command X not allowed by ACL`. `tests/unit/tauri-acl.test.ts` enforces this.
- **COEP blocks cross-origin subresources without CORP.** Fonts are bundled
  with `@fontsource`, and nothing loads from a CDN at runtime.
- **GUI apps don't inherit the shell PATH.** `ug_path()` probes PATH, then
  `~/.local/bin`, `~/.cargo/bin`, `~/.ug/bin`, `/opt/homebrew/bin` and
  `/usr/local/bin`. The release e2e launches with `PATH=/usr/bin:/bin` to prove
  this.
- **ug's `search` returns a document node and its sections side by side.**
  `dedupeHits` keeps the most specific one, so the prompt doesn't repeat text.
- **Qwen3 emits `<think></think>` even with thinking off.** `splitThink` treats
  an empty pair as no reasoning. Reasoning is never replayed to the model in
  history.
- **wllama loads single GGUF files up to 2 GB.** Larger models need gguf-split
  shards; `models.test.ts` enforces the limit.

---

## 3. What is real and what is simulated

| Area | Status | Where |
|---|---|---|
| Chat, streaming, stop, think folding | Real | `llm/engine.ts`, `agent/turn.ts` |
| Reasoning chips, Execution Trace, stats | Real (actual step timings, tokens, tok/s) | `agent/turn.ts` |
| Knowledge bases: create, ingest, index, search, delete | Real (ug CLI) | `src-tauri/src/ug.rs`, `state/kb.ts` |
| Persona, auto-optimize | Real | `screens/Persona.tsx` |
| Models: download, load, unload, evict | Real | `llm/engine.ts` |
| Appearance (system / light / dark) | Real | `state/theme.ts` |
| Layout: collapsible nav (⌘B), Execution Trace on/off (⌘J) | Real, persisted | `state/layout.ts` |
| Workflows, approvals, tool library, node editor, run | **Simulated** | `mock/workflows.ts`, `screens/Workflow*.tsx` |
| "Choose tool" (except Knowledge search), "Connect S3" | **Simulated** | toasts say so |

Promoting a simulated feature to real requires: a design note in this file,
tests at the same level as the real features, and removal of the "simulated"
copy.

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
6. `npm test` then checks steps 2–4 (`tauri-acl.test.ts`), and
   `npm run test:e2e:release` proves it works in the shipped build.

### Dependencies
- Pin exact versions in `package.json`. `@wllama/wllama` and
  `@wllama/wllama-compat` **must stay on the same version**. After bumping
  them, run `npm install` (re-copies the wasm) and both e2e runs.
- No new runtime network dependencies (§1.4). Anything with native code needs
  a human decision (§1.10).

---

## 5. Commands

```bash
npm install              # deps + copies wllama wasm into public/wllama/
npm run tauri dev        # run the app (hot reload)
npm run tauri build      # → src-tauri/target/release/bundle/macos/Andai.app

npm run check            # typecheck + unit/component tests + Rust tests + clippy  ← before every change is "done"
npm test                 # Vitest: unit, component, guard tests (~2 s)
npm run test:coverage    # same, with a coverage report in coverage/
npm run test:rust        # Rust unit tests
npm run test:ug          # Rust ↔ real ug integration (needs ug)
npm run test:e2e         # full app in WKWebView: ingest → retrieve → generate (needs ug; downloads model once)
npm run test:e2e:release # same against the release binary (localhost origin + ACL + Finder-like PATH)
```

`E2E_MODEL=stories-260k npm run test:e2e` checks the plumbing in seconds; the
answer-grounding assertion only runs with a real model.

The RTK shell proxy on this machine can mangle test-runner output. Prefix a
command with `rtk proxy` to see the raw output.

---

## 6. Testing strategy

| Layer | Tool | Location | Covers |
|---|---|---|---|
| Unit | Vitest | `src/**/*.test.ts` | pure logic: prompt assembly, think split, hit dedupe, theme resolution, formatting, model catalog |
| Orchestration | Vitest + `vi.mock` | `src/agent/turn.test.ts`, `src/state/kb.test.ts` | turn step transitions, failure paths, abort, history; KB actions against a mocked Rust bridge |
| Component | Testing Library (jsdom) | `src/**/*.test.tsx` | user-visible behavior, queried by role/text |
| Guards | Vitest | `tests/unit/` | design-token and Tauri ACL invariants |
| Rust unit | `cargo test` | `src-tauri/src/ug.rs` | ingestion, validation, status derivation, serialization |
| Rust ↔ ug | `cargo test -- --ignored` | same | real `ug gen` + `search` round trip |
| End-to-end | `scripts/e2e.mjs` + `src/smoke.ts` | real app | isolation, threads, ingest, retrieval, grounded answer; dev and release |

Rules:
- **New behavior ships with a test at the lowest layer that can catch its
  regression.** Bug fixes start with a failing test.
- Test behavior, not implementation: assert on what the user or the next
  module sees.
- A guard test must be shown to fail. When you add one, break the invariant
  once and watch it go red.
- The e2e harness **must not** touch user data: `ANDAI_DATA_DIR` isolates KB
  files, and the harness snapshots and restores chat and KB selection. The ug
  project it creates (`andai-e2e-docs`) is removed before it reports `OK`.
- Visual checks: headless Chrome against `npm run dev` renders every screen
  (`/#command`, `/#knowledge`, …). Seed `localStorage` (`andai.theme`,
  `andai.chat`) to check both themes and populated states. Look at the images;
  don't just generate them.

### Definition of done
- [ ] `npm run check` is green
- [ ] New or changed behavior has tests (§6 rules)
- [ ] UI changes looked at in light **and** dark
- [ ] Touching engine, ug, Tauri config or commands: `npm run test:e2e` and
      `npm run test:e2e:release` pass
- [ ] Real vs. simulated table (§3) and platform facts (§2) still true, or updated
- [ ] README / this file updated if commands, setup or behavior changed

---

## 7. Release checklist

1. `npm run check`, then `npm run test:e2e:release`.
2. Bump `version` in `package.json` and `src-tauri/tauri.conf.json` (keep them
   equal).
3. `npm run tauri build`. Launch the `.app` from Finder, load a cached model,
   ask a grounded question, and switch themes.
4. Note user-visible changes and anything simulated → real in the release
   notes.

Known constraints to keep in mind: the release UI needs port **14230** free
(the app currently can't start without it), and dev (`localhost:1420`) and
release (`localhost:14230`) have separate webview storage, including separate
model caches.
