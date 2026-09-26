# Andai — a local-first AI agent

> Contributors and coding agents: read **[AGENTS.md](AGENTS.md)** first. It has the grounding rules, architecture, conventions and testing strategy.
> Users: see **[docs/features.md](docs/features.md)** and the product site in **[docs/andai-website/](docs/andai-website/index.html)**.

Andai merges two demos into one desktop app:

- **wllama-chat**, llama.cpp compiled to WebAssembly. Here it serves as the on-device LLM.
- **gpuix-demo**. Only its UX ideas carried over; the UI runtime is Tauri (see *Why Tauri*).
- **ug** is the knowledge engine. It turns your documents into a searchable knowledge graph used for GraphRAG.

Nothing leaves the machine. The one exception is the first model download from Hugging Face.

```
Andai (Tauri 2)
├─ src/                 React 19 + Vite UI
│  ├─ llm/engine.ts     wllama in the webview: load / cache / stream (compat build on WKWebView)
│  ├─ agent/turn.ts     one agent turn: analyze → ug retrieval → prompt assembly → generate
│  ├─ state/            zustand stores: chat, knowledge bases, persona, ui
│  └─ screens/          Command Center, Knowledge, Persona, Settings, Workflows (mock), Workflow Detail (mock)
└─ src-tauri/
   ├─ src/ug.rs         knowledge bases: shells out to `ug gen / search / list / remove`
   └─ src/lib.rs        app setup; release builds serve the UI from http://localhost:14230
```

## Run

```bash
npm install              # also copies the wllama wasm builds into public/wllama/
npm run tauri dev        # desktop app with hot reload
npm run tauri build      # → src-tauri/target/release/bundle/macos/Andai.app
```

Requirements:
- **Rust**, plus Xcode command-line tools.
- **Node 20+**.
- **`ug`**, on `PATH` or in `~/.local/bin`, `~/.cargo/bin`, `/opt/homebrew/bin` or `/usr/local/bin`. Finder-launched apps don't inherit the shell PATH, so `ug.rs` probes these locations.

First launch:
1. In **Settings → Models**, load **Qwen3 0.6B**. It is a 639 MB one-time download, cached in the webview's OPFS; a cached load takes about 1 s.
2. In **Knowledge**, create a knowledge base and drop in PDFs, Markdown, TXT, CSV or source files.
3. Chat in **Command Center**. The knowledge-base chip in the composer picks which KB grounds the answers.

## What's real and what's simulated

| Area | Status |
|---|---|
| Chat, streaming, `<think>` folding, stop | **Real**: wllama `createChatCompletion` |
| Reasoning chips and Execution Trace | **Real**: each step of `agent/turn.ts`, with timings, hits and tok/s |
| Knowledge bases (create, ingest, re-index, remove, delete) | **Real**: `ug gen --with-embed`, with progress streamed from ug |
| RAG retrieval (K, context budget) | **Real**: `ug search --snippets --json`; hits are cited as `[n]` |
| Persona (prompt, tone, temperature, max tokens, reasoning) | **Real**, persisted. *Auto-optimize* rewrites the prompt with the local model |
| Model registry (download, load, unload, evict) | **Real**: wllama `ModelManager` |
| Workflows, approvals, tool library, node editor, run simulation | **Simulated**: mock data in `src/mock/workflows.ts` |

## Why Tauri, and the WebKit details that matter

gpuix renders with Bun. wllama's default wasm needs **Memory64 + JSPI**, which Bun's JavaScriptCore lacks, and its compat build crashed mid-generation under Bun. So the UI moved to a webview.

On macOS, Tauri uses **WKWebView**, which is the Safari engine:

- **Compat build.** wllama detects the missing features (`needCompat()`), and `engine.ts` points `setCompat()` at the bundled `@wllama/wllama-compat` files. Measured on an M-series Mac, Qwen3 0.6B runs at about **31 tok/s** on the WebGPU backend with 4 threads.
- **Cross-origin isolation.** Multi-threading needs `SharedArrayBuffer`, which needs COOP/COEP. WebKit **ignores isolation on custom schemes**: `tauri://` sends the headers, but `crossOriginIsolated` stays false. Release builds therefore serve the UI from `http://localhost:14230` using `tauri-plugin-localhost`, which adds the headers. Dev gets the same headers from Vite.
- **ACL.** That loopback origin counts as "remote", so every app command is declared in `build.rs` and granted in `capabilities/default.json`. When you add a command, add it in both places.
- Fonts are bundled with `@fontsource`, because COEP blocks cross-origin font CSS.

## Tests

```bash
npm run check            # typecheck + Vitest + Rust tests + clippy — run before calling anything done
npm run test:ug          # Rust ↔ real ug round trip
npm run test:e2e         # the real app in WKWebView: ingest fixtures → retrieve → grounded answer
npm run test:e2e:release # the same against the release binary
```

The layers, the rules and the definition of done are in [AGENTS.md](AGENTS.md) §6.
Tests never touch your data: the e2e runner isolates knowledge-base files with
`ANDAI_DATA_DIR` and restores your chat afterwards.

## Releasing

```bash
npm run release -- --dry-run   # preview the next version and release notes
npm run release                # bump, check, tag, push → GitHub Actions builds and publishes
```

Pushing a `vX.Y.Z` tag triggers `.github/workflows/release.yml`, which verifies
the tag, builds `Andai.app` and a `.dmg` for Apple Silicon and Intel, and
publishes a GitHub Release with checksums. For details, see [AGENTS.md](AGENTS.md) §7.

## Appearance

Choose **System**, **Light** or **Dark** from the sun/moon button in the top bar or
from **Settings → Appearance**. Every color is a token in `src/theme/tokens.css`,
defined once per theme. A guard test fails the build on stray color literals and
checks WCAG-AA contrast for the light theme.

## Shortcuts

`⌘K` focuses the composer · the top-bar sun/moon button cycles the theme · `⌘1–4` switch screens · `⌘B` collapses the sidebar · `⌘J` shows/hides the Execution Trace · `⌘,` opens Settings · `Enter` sends · `Shift+Enter` adds a newline · `Esc` stops generating.
