# Andai — a local-first agentic RAG agent, driven by the Laya decision model

> Contributors and coding agents: read **[AGENTS.md](AGENTS.md)** first. It has the grounding rules, architecture, conventions and testing strategy.
> Users: see **[docs/features.md](docs/features.md)** and the product site in **[docs/andai-website/](docs/andai-website/index.html)**.

Andai is a desktop AI agent that answers from **your own knowledge bases**, and the
model runs on your computer (macOS or Windows). It is more than agentic RAG: every
step the agent takes is a measured choice. Instead of pasting the top-k
chunks into a prompt, the agent works its knowledge base with tools. It searches,
reads the lines around a hit, outlines a file, or follows a code symbol to its
callers until it has enough evidence. Then it writes an answer that cites every
passage it used.

- **Laya decides; the LLM writes.** Laya is a small decision model, trained for
  typed choices (SemIf-style): it scores every option in a single forward pass —
  about 8–20 ms per decision on Apple Silicon, with calibrated probabilities shown
  in the Execution Trace — and filters retrieved passages that don't help before
  the prompt. Without it, the chat model itself decides through the same one-pass
  letter readout.
- **The on-device LLM** writes the answers. On Apple Silicon Macs it runs
  natively on the GPU with **MLX**, in Rust (Qwen3 1.7B: about 350 tokens/s on
  an M5 Max); everywhere else, and for the GGUF models, with **wllama**
  (llama.cpp compiled to WebAssembly, 30–65 tokens/s for the same model).
- **[ug (UltraGraph)](https://ultra-graph.web.app)** ([source](https://github.com/shoocstorm/ug)) is the knowledge
  engine and the agent's toolbox: a local-first Rust engine that turns documents and code into an
  interactive, queryable semantic knowledge graph. Its commands (`search`,
  `get_code`, `file_context`, `context`, `find_usages`, …) are the tools the
  agent calls.
- **Tauri** hosts the UI. The UX ideas come from the earlier gpuix demo (see *Why Tauri*).
- **Also in the browser:** an initial web build is live at
  **[andai-agent.web.app](https://andai-agent.web.app)** (deployed with
  `bun run deploy:web`). Same UI, minus the knowledge feature: the ug engine
  runs natively, so knowledge bases are desktop-only.

Nothing leaves the machine — in the desktop app. The one exception is the
first model download from Hugging Face. (The web deployment additionally
loads Google Analytics; the desktop app does not.)

### How a turn works

With a knowledge base selected, each question runs the **agent loop**
(`agent/loop.ts`):

1. **Decide.** The model picks the next action (a ug tool, *answer now* or
   *ask a clarifying question*) from a lettered list. It doesn't write the
   choice: one forward pass is read out as a probability per option
   (SemIf-style, `llm/decide.ts`). An optional, separate *decision model* can
   do this: Qwen3 0.6B deciding while Qwen3 1.7B writes answered best in the
   agent eval.
2. **Fill arguments.** The model writes the tool's arguments as JSON, held to
   the tool's schema by a grammar. This is where the question becomes a search
   phrase (or a symbol name, a file, a line range).
3. **Gate and run.** The tool's policy applies (*Auto / Ask / Off*); *Ask*
   shows an approval card. Rust re-validates the call and runs ug.
4. **Observe**, then decide again, for up to *N* calls. The answer is then
   written from everything the tools returned, cited as `[n]`.

| | |
|---|---|
| Tools | 8 read-only ug tools, offered by knowledge-base kind (documents, code, mixed) |
| Control | Per-tool policy, step limit, minimum confidence, one plain search as the fallback when a decision can't be trusted, and agent mode off → the fixed pipeline (one `ug search` with the question, then answer) |
| Transparency | Every decision (all options and their probabilities) and every tool call (arguments, approval, ug command, timing, output) in the Execution Trace, copyable as JSON |

Tools are registered in code only: no plugins, no MCP, nothing the webview can
add. Rust re-validates every call and runs ug read-only, scoped to one
knowledge base, time-boxed and output-capped (see [AGENTS.md §9](AGENTS.md#9-security)).

```
Andai (Tauri 2)
├─ src/                 React 19 + Vite UI
│  ├─ llm/engine.ts     load / cache / stream: native MLX (Apple Silicon) or wllama in the webview
│  ├─ llm/native.ts     the native MLX models: Rust commands, streaming channel
│  ├─ llm/decide.ts     choice-based decisions: lettered options → one-pass probability readout
│  ├─ agent/turn.ts     one turn: agent loop (or the fixed search) → prompt assembly → generate
│  ├─ agent/loop.ts     decide → fill arguments → gate → run tool → observe
│  ├─ agent/tools/      tool registry, the 8 ug tools, argument filling, schema validation
│  ├─ state/            zustand stores: chat, knowledge bases, persona, ui
│  └─ screens/          Command Center, Knowledge, Tools, Persona, Settings, Workflows (mock), Workflow Detail (mock)
└─ src-tauri/
   ├─ src/ug.rs         knowledge bases: shells out to `ug gen / search / list / remove`
   ├─ src/tools.rs      agent tool calls: validated, read-only ug commands, time-boxed and capped
   ├─ src/llm/          native chat models on MLX: Qwen3 port, KV cache, grammar-held generation
   ├─ src/laya/         Laya decision models on MLX
   ├─ src/samples.rs    the bundled sample knowledge bases
   └─ src/lib.rs        app setup; release builds serve the UI from http://localhost:14230
```

## Run

```bash
bun install              # also copies the wllama wasm builds into public/wllama/
                         # and installs the pre-push hook (.githooks/: bun run check before every push;
                         # bypass once with git push --no-verify)
bun run tauri dev        # desktop app with hot reload
bun run tauri build      # macOS → src-tauri/target/release/bundle/macos/Andai.app
                         # Windows: bun run tauri build --bundles nsis → bundle/nsis/*-setup.exe
bun run deploy:web       # build + Firebase Hosting deploy → https://andai-agent.web.app
                         # (same UI in the browser, minus the knowledge feature; needs the firebase CLI)
```

Requirements:
- **Rust**, plus Xcode command-line tools (macOS) or the MSVC build tools (Windows).
- **Apple Silicon only:** CMake and the Xcode **Metal Toolchain** component (`xcodebuild -downloadComponent MetalToolchain`). The MLX models (native chat models and Laya) run on MLX, whose Metal kernels are compiled from source during the build.
- **Windows only:** the Microsoft Edge **WebView2** runtime (preinstalled on Windows 11). The installer doesn't download it, because Andai makes no network requests besides model downloads.
- **Bun 1.3+**, which is the package manager and script runner. Plain `bun test` is Bun's own runner, so use `bun run test`.
- **Node 22+**, only for Vitest (jsdom doesn't run on Bun's runtime). Everything else runs on Bun.
- **`ug`**, on `PATH` or in `~/.local/bin`, `~/.cargo/bin` or `~/.ug/bin` (on Windows: `ug.exe` in the same folders under `%USERPROFILE%`), or on macOS in `/opt/homebrew/bin` or `/usr/local/bin`. Finder-launched apps don't inherit the shell PATH, so `ug.rs` probes these locations.

First launch:
1. In **Settings → Models**, load a model. On an Apple Silicon Mac, pick **Qwen3 1.7B · MLX** (980 MB, runs natively, about 350 tokens/s); elsewhere **Qwen3 0.6B** (639 MB, cached in the webview's OPFS). A downloaded model loads in about a second.
2. In **Knowledge**, create a knowledge base and drop in PDFs, Markdown, TXT, CSV or source files, or click **Try a sample** for a ready-made one (a fictional ferry operator's documents and code) with suggested questions.
3. Chat in **Command Center**. The knowledge-base chip in the composer picks which KB the agent works from.
4. Optional: **Tools** (`⌘5`, `Ctrl+5` on Windows) shows every tool and its policy; **Settings → Decision model** loads a second model to make the agent's choices: with Qwen3 1.7B as the chat model, pick Qwen3 0.6B here (best in the agent eval).

## What's real and what's simulated

| Area | Status |
|---|---|
| Chat, streaming, `<think>` folding, stop | **Real**: native MLX in Rust (`src-tauri/src/llm/`) or wllama `createChatCompletion` |
| Reasoning chips and Execution Trace | **Real**: each step of the turn, every decision and tool call, with timings, hits and tok/s |
| Knowledge bases (create, ingest, re-index, remove, delete) | **Real**: `ug gen --with-embed`, with progress streamed from ug |
| Sample knowledge bases (documents, code, both) | **Real**: bundled files, indexed by your ug like your own |
| RAG retrieval (K, context budget) | **Real**: `ug search --snippets --json`; hits are cited as `[n]` |
| Agentic tool loop (model-chosen ug tools, per-tool policy, approvals) | **Real**: `agent/loop.ts`, `llm/decide.ts`, `src-tauri/src/tools.rs` |
| Persona (prompt, tone, temperature, max tokens, reasoning) | **Real**, persisted. *Auto-optimize* rewrites the prompt with the local model |
| Model registry (download, load, unload, evict) | **Real**: Rust's verified store (MLX) or wllama `ModelManager` |
| Add a model from Hugging Face (search, compatibility, pinned + verified download) | **Real**: `llm/hub.ts`, Rust checks MLX models in `src-tauri/src/llm/custom.rs` |
| Workflows, approvals, tool library, node editor, run simulation | **Simulated**: mock data in `src/mock/workflows.ts` |

## Security

Andai treats its own webview as untrusted, because model output can be steered
by a poisoned document. A content security policy limits the network to
Hugging Face. Answers never load images or open links. Rust ingests only files
you dropped or picked, and the app refuses to start if another process holds
its UI port. Retrieved passages reach the model as fenced, untrusted data.
Model downloads are pinned to a Hugging Face commit and verified by sha256
before loading, and `bun run audit` (JS + RustSec) gates CI and releases.

- What each protection means for users: [docs/security.md](docs/security.md)
- Threat model and the checklist for every change: [AGENTS.md §9](AGENTS.md#9-security)
- Reporting a vulnerability: [SECURITY.md](SECURITY.md)

## Why Tauri, and the WebKit details that matter

gpuix renders with Bun. wllama's default wasm needs **Memory64 + JSPI**, which Bun's JavaScriptCore lacks, and its compat build crashed mid-generation under Bun. So the UI moved to a webview.

On macOS, Tauri uses **WKWebView**, which is the Safari engine. (On Windows it uses **WebView2**, which is Chromium: wllama runs its default build there, and the loopback server below provides isolation the same way.)

- **Compat build.** wllama detects the missing features (`needCompat()`), and `engine.ts` points `setCompat()` at the bundled `@wllama/wllama-compat` files. The compat build still runs on the GPU: llama.cpp offloads every layer through WebGPU. Measured on an Apple M5 Max, Qwen3 0.6B generates about **65 tok/s** and reads its prompt at about **520 tok/s** (`bun run bench:engine` shows the load log and speeds). On Apple Silicon the MLX models skip the webview entirely and run in Rust, about 5–10× faster (docs/performance.md, *Engine*).
- **Cross-origin isolation.** Multi-threading needs `SharedArrayBuffer`, which needs COOP/COEP. WebKit **ignores isolation on custom schemes**: `tauri://` sends the headers, but `crossOriginIsolated` stays false. Release builds therefore serve the UI from `http://localhost:14230` through Andai's own loopback server (`src-tauri/src/ui_server.rs`), which adds the headers. It binds the port before any window exists and refuses to start if another process holds it. Dev gets the same headers from Vite.
- **ACL.** That loopback origin counts as "remote", so every app command is declared in `build.rs` and granted in `capabilities/default.json`. When you add a command, add it in both places.
- Fonts are bundled with `@fontsource`, because COEP blocks cross-origin font CSS.

## Tests

```bash
bun run check            # typecheck + Vitest + Rust tests + clippy — run before calling anything done
bun run test:ug          # Rust ↔ real ug round trip
bun run test:e2e         # the real app in WKWebView (macOS only): ingest fixtures → retrieve → grounded answer
bun run test:e2e:release # the same against the release binary
bun run audit            # known vulnerabilities in JS + Rust dependencies (CI and releases run it)
bun run perf             # bundle size + hot-path benchmarks vs. perf/baseline.json (docs/performance.md)
```

The layers, the rules and the definition of done are in [AGENTS.md](AGENTS.md) §6.
Tests never touch your data: the e2e runner isolates knowledge-base files with
`ANDAI_DATA_DIR` and restores your chat afterwards.

## Releasing

```bash
bun run release --dry-run      # preview the next version and release notes
bun run release                # bump, check, tag, push → GitHub Actions builds and publishes
```

Pushing a `vX.Y.Z` tag triggers `.github/workflows/release.yml`, which verifies
the tag, builds `Andai.app` and a `.dmg` for Apple Silicon and Intel plus a
Windows x64 installer (`Andai_X.Y.Z_x64-setup.exe`), and publishes a GitHub Release with checksums. For details, see [AGENTS.md](AGENTS.md) §7.

## Appearance

Choose **System**, **Light** or **Dark** from the sun/moon button in the top bar or
from **Settings → Appearance**. Every color is a token in `src/theme/tokens.css`,
defined once per theme. A guard test fails the build on stray color literals and
checks WCAG-AA contrast for the light theme.

## Shortcuts

On Windows, use `Ctrl` wherever this says `⌘`.

`⌘K` focuses the composer · the top-bar sun/moon button cycles the theme · `⌘1–5` switch screens · `⌘B` collapses the sidebar · `⌘J` shows/hides the Execution Trace · `⌘,` opens Settings · `Enter` sends · `Shift+Enter` adds a newline · `Esc` stops generating.
