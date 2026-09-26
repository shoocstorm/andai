# AGENTS.md — Andai engineering guide

Andai is a **local-first AI agent desktop app**. It is a product, not a demo:
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
│  ├─ llm/engine.ts          wllama: load / cache / stream. Compat build on WKWebView
│  ├─ llm/models.ts          model catalog (single-file GGUF, < 2 GB each, pinned commit + sha256)
│  ├─ llm/integrity.ts       incremental SHA-256 + download verification
│  ├─ agent/prompt.ts        PURE prompt assembly: keywords, system prompt, history, budgets
│  ├─ agent/turn.ts          one turn: analyze → retrieve (ug) → build → generate; writes the trace
│  ├─ kb/api.ts              typed wrappers over the Rust ug bridge + hit dedupe
│  ├─ state/                 zustand stores: chat, kb, persona, theme, layout, ui (persisted where noted)
│  ├─ screens/, shell/, components/
│  ├─ theme/tokens.css       ALL colors, both themes
│  ├─ mock/workflows.ts      data for the simulated Workflows screens
│  └─ smoke.ts               in-webview test harness (VITE_SMOKE)
├─ src-tauri/
│  ├─ src/lib.rs             app setup, navigation lock, drop → file grants
│  ├─ src/ui_server.rs       loopback server for the release UI (http://localhost:14230)
│  ├─ src/grants.rs          which files the webview may ingest (drop / Rust dialog only)
│  ├─ src/ug.rs              knowledge bases → `ug gen/search/list/remove` CLI
│  ├─ build.rs               app command manifest (ACL)
│  └─ capabilities/default.json
├─ docs/                     user-facing docs (features.md, …) — index in docs/README.md
│  └─ andai-website/         static product site: index.html + img/ (real app screenshots)
├─ tests/                    setup, guard tests, e2e fixtures, perf/ micro-benchmarks
├─ perf/baseline.json        performance baselines (docs/performance.md)
└─ scripts/                  copy-wllama (postinstall), e2e runner, perf runner
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
- **Whoever answers on port 14230 gets full IPC.** `ui_server.rs` binds
  `127.0.0.1` and `[::1]` before any window exists; if either is taken, Andai
  shows an error and opens no window. (`tauri-plugin-localhost` bound in a
  background thread and opened the window anyway.) See §9.
- **ug has no `--` separator.** `ug search -- "q"` fails with "missing query
  argument", so `search_query()` prefixes a query that starts with `-` with a
  space, which keeps it positional (ug 0.1.21). Without that, a query could pass
  `--base-url` and send itself to a remote embedder.
- **Model downloads go to `huggingface.co`, then redirect to a regional
  `*.cdn.hf.co` host** (measured: `us.aws.cdn.hf.co`). That's why `connect-src`
  allows `https://*.hf.co`.
- **Vite answers a revalidated `index.html` with a bare 304**, and WebKit
  keeps the cached headers, so a webview that cached the page before a CSP
  change never enforced it (measured: `new Function` ran). The
  `freshDocuments` plugin in `vite.config.ts` strips `If-None-Match` for
  documents. The release server sends no ETag.
- **WKWebView rejects a CSP-blocked fetch with a bare "Load failed"** and
  doesn't fire `securitypolicyviolation` for `connect-src`. The e2e run proves
  the block with a canary server instead (§9).
- **wllama needs `'wasm-unsafe-eval'` and `worker-src blob:`**, and no eval.
  Inline `style` attributes need `style-src 'unsafe-inline'`, so
  `dangerousDisableAssetCspModification: ["style-src"]` keeps Tauri from adding
  a nonce there. A nonce would make browsers ignore `'unsafe-inline'`.
- **Hugging Face commit URLs are immutable; `resolve/main` isn't.** The
  catalog pins `resolve/<40-hex sha>/…`. The file's sha256 is the LFS oid
  (`POST /api/models/<repo>/paths-info/<sha>`, and also the `X-Linked-ETag`
  header on the resolve URL). `ggml-org/models` was renamed to
  `ggml-org/models-moved` and 307-redirects, so the pin uses the new name.
- **Web Crypto has no streaming digest.** `integrity.ts` has an incremental
  SHA-256. Measured: 305 MB/s in V8 and about 93 MB/s in WKWebView (release
  build, Qwen3 0.6B, 639 MB in 6.9 s on an Apple Silicon Mac). It runs once per
  download, and `andai.verifiedModels` in localStorage skips re-hashing. A
  fresh download is always verified.
- **RTK rewrites `curl` output** and can mangle JSON. Use `rtk proxy curl`
  when you parse an API response.
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
  a human decision (§1.10).

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
bun run test:e2e         # full app in WKWebView: ingest → retrieve → generate (needs ug; downloads model once)
bun run audit            # bun audit (JS deps) + cargo audit (RustSec); CI and releases run it
bun run test:e2e:release # same against the release binary (localhost origin + ACL + Finder-like PATH)
bun run perf             # bundle size (CI too) + micro-benchmarks vs. perf/baseline.json
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
| Unit | Vitest | `src/**/*.test.ts` | pure logic: prompt assembly, think split, hit dedupe, theme resolution, formatting, model catalog |
| Orchestration | Vitest + `vi.mock` | `src/agent/turn.test.ts`, `src/state/kb.test.ts` | turn step transitions, failure paths, abort, history; KB actions against a mocked Rust bridge |
| Component | Testing Library (jsdom) | `src/**/*.test.tsx` | user-visible behavior, queried by role/text |
| Guards | Vitest | `tests/unit/` | design-token and Tauri ACL invariants |
| Rust unit | `cargo test` | `src-tauri/src/ug.rs` | ingestion, validation, status derivation, serialization |
| Rust ↔ ug | `cargo test -- --ignored` | same | real `ug gen` + `search` round trip |
| End-to-end | `scripts/e2e.mjs` + `src/smoke.ts` | real app | isolation, threads, ingest, retrieval, grounded answer; dev and release |
| Performance | `scripts/perf.mjs`, `tests/perf/`, e2e runner | `perf/baseline.json` | bundle sizes; hot-path timings; ingest, search, load, first token, tok/s ([docs/performance.md](docs/performance.md)) |

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
- **Performance is a tested behavior.** A change that makes a baselined
  metric worse than its tolerance fails `bun run perf` or the e2e run. If the
  cost is intended, re-record the baseline with `--update` in the same change
  and give the reason in the commit. Don't raise a tolerance to get a change
  through. Timing baselines are machine-bound: they're enforced only on the
  machine that recorded them (docs/performance.md).
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
- [ ] README / this file updated if commands, setup or behavior changed
- [ ] **Docs and website updated** for any user-visible change (§8), in the same change
- [ ] Security checklist (§9) holds, and `tests/unit/security.test.ts` passes without loosening an allowlist

---

## 7. Releases

Releases are **tag-driven**. Pushing a `vX.Y.Z` tag runs
`.github/workflows/release.yml`, which does three things:
1. **verify:** the tag must equal the version in every manifest, the full
   check suite must pass, and `bun run audit` must find no known
   vulnerability (§9).
2. **build:** `Andai.app` + `.dmg` for Apple Silicon and Intel, with
   `.sha256` checksums. It also asserts that the built `Info.plist` reports the
   tag's version.
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

Code signing: builds are **unsigned** until the repository has the secrets
`APPLE_CERTIFICATE`, `APPLE_CERTIFICATE_PASSWORD` and `APPLE_SIGNING_IDENTITY`
(add `APPLE_ID`, `APPLE_PASSWORD` and `APPLE_TEAM_ID` for notarization). The
workflow detects them automatically, and the release notes say whether a
build is unsigned.

Before announcing a release, also run `bun run test:e2e:release` locally
(CI can't run it: it needs ug and a model). Then launch the downloaded `.dmg`
build, ask a grounded question and switch themes.

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
| Install / release / signing | Website download CTA + FAQ, `docs/features.md` → Requirements |
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
- The website is **one self-contained static page** (`index.html` + `img/`):
  no build step and no JS framework. Its tokens mirror
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
| The user's documents | KB copies in app data, ug graphs in `~/.ug/andai-*` |
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
   Face. Nothing in `src/` besides `llm/models.ts` names a remote URL, and the
   Rust side has no HTTP client.
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
| Private files (0700 dirs / 0600 files), 100 MB cap, absolute PATH only | `ug.rs` | Rust unit tests |
| `dev_log` / `dev_exit` need `ANDAI_SMOKE=1` | `lib.rs` | `security.test.ts` |
| Webview holds no fs/shell/http/opener/dialog permission | `capabilities/default.json` | `security.test.ts` |
| Model downloads pinned to a commit and verified (size + sha256) before load; mismatch → removed | `llm/models.ts`, `llm/integrity.ts`, `engine.loadModel` | `integrity.test.ts` (FIPS vectors, tamper), `engine.test.ts` (gate), `models.test.ts` (pinning); e2e logs the check |
| Pre-pinning model copies removed only after the user confirms | `engine.removeLegacyCopies`, Settings | `Settings.test.tsx` |
| Retrieved passages fenced as untrusted data; a passage can't close its fence | `agent/prompt.ts` | `prompt.test.ts` |
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
- [ ] New or updated catalog model: `url` pinned to a commit, with `bytes` and
      `sha256` taken from Hugging Face's `paths-info` (download it once and
      check). Move the old URL into `legacyUrls`.
- [ ] Retrieved or user-supplied text goes to the model through
      `buildSystem`'s fences, never spliced into instructions.

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
- **Gate for promoting simulated features (§3):** before any tool or workflow
  becomes real, every model-initiated action needs explicit user approval in
  the UI, a per-tool capability, and an entry in the Execution Trace.
- **Code signing and notarization:** the workflow is ready (§7); it needs the
  Apple secrets.

The user-facing version of this section is [docs/security.md](docs/security.md).
Keep the two in sync (§8).

