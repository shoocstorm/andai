# Platform facts

Hard-won facts about the platforms, models and tools Andai runs on: what was
measured, when, and what the code does because of it. They used to live in
[AGENTS.md](../AGENTS.md) §2, so code comments citing "AGENTS.md §2" for a
platform fact point here. **Read the section for the area you're changing
before you change it, and don't re-learn these.** When you measure something
new, add it here, in the right section, with the date and the machine.

## Webview, Tauri and the UI

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
  this. On Windows it looks for `ug.exe`, and home is `USERPROFILE` (Windows
  usually has no `HOME`); the Homebrew paths are skipped (`ug_candidates`).
- **Whoever answers on port 14230 gets full IPC.** `ui_server.rs` binds
  `127.0.0.1` and `[::1]` before any window exists; if either is taken, Andai
  shows an error and opens no window. (`tauri-plugin-localhost` bound in a
  background thread and opened the window anyway.) See §9.
- **Vite answers a revalidated `index.html` with a bare 304**, and WebKit
  keeps the cached headers, so a webview that cached the page before a CSP
  change never enforced it (measured: `new Function` ran). The
  `freshDocuments` plugin in `vite.config.ts` strips `If-None-Match` for
  documents. The release server sends no ETag.
- **The window's theme overrides `prefers-color-scheme`.** After
  `getCurrentWindow().setTheme('dark')`, the webview's media query reports
  dark even with macOS in light mode (seen 2026-09-27: a "system" appearance
  stayed dark until relaunch). So there are only Light and Dark;
  `state/theme.ts` reads the OS preference once, for the first-run default,
  before anything sets the window's theme.
- **WKWebView rejects a CSP-blocked fetch with a bare "Load failed"** and
  doesn't fire `securitypolicyviolation` for `connect-src`. The e2e run proves
  the block with a canary server instead (§9).
- **wllama needs `'wasm-unsafe-eval'` and `worker-src blob:`**, and no eval.
  Inline `style` attributes need `style-src 'unsafe-inline'`, so
  `dangerousDisableAssetCspModification: ["style-src"]` keeps Tauri from adding
  a nonce there. A nonce would make browsers ignore `'unsafe-inline'`.
- **A Tauri channel can deliver its last messages after the command
  resolves.** `chatNative` (engine.ts) therefore takes any text that hadn't
  streamed yet from the command's result and drops later pieces.
- **A `Modal` renders into `document.body`** (portal). Opened from inside
  a transformed element (an answer card animates with `transform`), its
  `position: fixed` backdrop was laid out and clipped inside that card
  (seen 2026-09-27 with the claim dialog).

## Windows

- **Windows (WebView2) is Chromium**, so wllama runs its default build there,
  and the macOS-only WebKit facts in this list don't apply. It builds and its
  unit tests run in CI (`check-windows`), but the e2e harness drives
  WKWebView only, so **the Windows app is untested end to end** until someone
  runs it on a Windows PC. Say so wherever Windows support is claimed (§8).
- **Tool calls on Windows need `SystemRoot`.** `tools::scrubbed` clears the
  environment; on Windows it keeps `SystemRoot`, the profile variables and
  `TEMP`, and sets `PATH` to `System32`, or child processes may fail to start.
- **Windows file permissions aren't tightened.** `create_private_dir` /
  `private_file` set `0700`/`0600` on Unix only. On Windows, app data lives in
  `%APPDATA%` and inherits the profile ACL (user, SYSTEM, Administrators).
  Explicit ACLs would need a native-code dependency (§1.10).
- **The Windows installer doesn't fetch WebView2.** Tauri's default runs
  Microsoft's bootstrapper, an outbound request (§1.4), so
  `bundle.windows.webviewInstallMode` is `skip` (`security.test.ts`). Windows
  11 ships WebView2; the docs list it as a requirement.
- **Windows CI checks out CRLF and has no exec bits** (measured 2026-09-27:
  Git for Windows' `autocrlf`, NTFS). Guard tests broke three ways: paths from
  `readdirSync(…, { recursive: true })` come back with `\` and must be
  normalized before matching a forward-slash allowlist (`security.test.ts`);
  file content can carry `\r\n`, so manifest regexes use `\r?\n`
  (`scripts/version.mjs`); and `statSync().mode` has no `0o111` there — check
  exec-ness through `git ls-files -s` (`hooks.test.ts`).
- **Cross-checking Windows from a Mac:** `cargo clippy --target
  x86_64-pc-windows-msvc` needs `llvm-rc` for the app's resource file. A stub
  `llvm-rc` on `PATH` that touches its `/fo` output is enough for clippy (it
  doesn't link); it passed with `-D warnings` on 2026-09-26.

## Web deployment

- **The web deployment is plain static hosting** (`firebase.json`, site
  `andai-agent`, project `aldrick-ai`, `bun run deploy:web`): Vite's `dist/`
  with the desktop isolation headers (COOP/COEP, so Chromium gets
  SharedArrayBuffer and wllama's threaded build) and a CSP that mirrors
  `tauri.conf.json` minus the `ipc:` sources plus the Analytics hosts (gtag
  hosts, and `firebase.googleapis.com` + `firebaseinstallations.googleapis.com`,
  which the Firebase Analytics SDK fetches its config and instance ID from). Routing
  is hash-based, so no SPA rewrites are needed. `index.html` is served
  `no-cache` for the same reason as the 304 fact above; `/assets/**` are
  content-hashed and `immutable`. The nested `docs/andai-website/firebase.json`
  is a separate site — the CLI only reads the config in the working directory.

## ug

- **ug's `search` returns a document node and its sections side by side.**
  `dedupeHits` keeps the most specific one, so the prompt doesn't repeat text.
- **ug has no `--` separator.** `ug search -- "q"` fails with "missing query
  argument", so `search_query()` prefixes a query that starts with `-` with a
  space, which keeps it positional (ug 0.1.21). Without that, a query could pass
  `--base-url` and send itself to a remote embedder.
- **ug search reports how each item was found** (ug 0.1.22): `matched_by`
  is `semantic` (vector), `keyword` (full-text) or `graph` (walked `hop`
  edges from a match), and `distance` is its Personalized PageRank score
  negated, so lower ranks higher (items come sorted by it ascending; graph
  neighbours last). Its absolute value means little, so the UI shows it
  relative to the best match in the same list (`kb/match.ts`).
- **A knowledge base is a ug project** (2026-10-01, a product decision: no
  `kb.json`). **Andai talks to ug only through its command line**, never by
  reading ug's data folder (`tests/unit/ug-interface.test.ts` holds it): ug's
  file layout is ug's to change. `ug list --json` (ug 0.1.22) reports `name`,
  `repoRoot`, node/edge counts, `sizeBytes`, `createdAt`/`updatedAt`
  (seconds), `isStale`, `hasDb`, `repoMissing` and `kbKind`
  (`docs`/`code`/`mixed`; an empty project says `code`); the full list (with
  the staleness scan) took a few ms. With **no projects at all it exits 1 and
  prints a sentence, not JSON**, read as an empty list. A project's files come
  from **`ug files -n <project> --json -k 5000`** (added in ug 0.1.23,
  2026-10-01, for this; also the MCP `files` tool and `POST /api/tools/files`): per file `path`, `ext`,
  `language`, `kind`, `bytes`, `modified` and `status` (`fresh` · `changed` ·
  `missing`, the same per-file check `ug list` counts with), plus `total`
  and `counts` over every match. A copy reads as indexed when ug lists it
  `fresh`, otherwise pending. **ug 0.1.22 has no `ug files`**: it ends stderr
  with `error: unknown command: files`, and Andai shows "update ug"
  (`UG_TOO_OLD`) on each project instead of its files. `kb_list` runs one
  `ug files` per project, in parallel. `ug gen` on an empty folder registers
  a 0-file project, so Andai's new KBs show in ug at once. `UG_HOME`
  relocates every project (the embedder cache stays in
  `~/Library/Caches/ug/models`), so the e2e, eval and bench runners set it
  and never see the user's projects. `ug uninstall` deletes **all**
  projects; Andai's copies in app data survive and list as never indexed.
  ug 0.1.22 overflowed its stack indexing this repo's `src/` (exit 134).
- **ug's lookups fail with the useful message in stdout JSON** (`"error":
  "No symbol named …, try find_symbols"`) and exit 1 with a bare `error:` on
  stderr. `tools::run` surfaces the JSON message.

## Model downloads and Hugging Face

- **Model downloads go to `huggingface.co`, then redirect to a regional
  `*.cdn.hf.co` host** (measured: `us.aws.cdn.hf.co`). That's why `connect-src`
  allows `https://*.hf.co`.
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
- **Models from Hugging Face** (2026-09-27; `llm/hub.ts`, `llm/custom.ts`,
  `src-tauri/src/llm/custom.rs`). The API answers the app's origin with CORS
  (`access-control-allow-origin` echoes it). `/api/models?…&expand[]=gated`
  gives gating in search results, and `/api/models/<repo>?expand[]=sha&expand[]=gguf`
  the current commit plus GGUF metadata (architecture, context length, chat
  template). The tree at a commit gives each file's LFS sha256; small files
  kept in git have only a git blob SHA-1, so an MLX model's JSON files go to
  Rust inline, and Rust hashes and keeps what it was sent (no SHA-1 crate).
  A small file's `resolve/<commit>` URL 307-redirects within huggingface.co
  (`/api/resolve-cache/…`). Some MLX repos ship
  `model.safetensors.index.json` that names only `model.safetensors`; the
  single file wins (`load_weights`). The native engine runs Qwen3 and dense
  Qwen3.5 (below), MLX-quantized (affine, 2–8 bits, mixed per weight), with
  a ChatML template; untied LM heads (Qwen3 8B) and sharded weights are
  supported. Checked end to end on
  Qwen3 8B 4-bit (`test:llm`): add, chunked download, sha256, load, same
  greedy text as mlx-lm, 116 tok/s on an M5 Max.

## wllama

- **WKWebView runs wllama's compat build on the GPU.** WebGPU works in the
  compat (Asyncify) worker: llama.cpp logs `offloaded 29/29 layers to GPU`
  for Qwen3 1.7B (e2e checks every layer is on the GPU, from that log line;
  `navigator.gpu` existing proves nothing). CPU-only is 10× slower. WebKit
  reports 8 cores (`hardwareConcurrency`), so wllama uses 4 threads; 8
  threads made GPU generation 2–4× *slower*, 2 threads was no different.
  Flash attention isn't supported on WebGPU and is switched off by
  llama.cpp. Measured 2026-09-26 on an M5 Max with `bun run bench:engine`.
- **In wllama, prompt reading, not generation, is the wait.** Qwen3 1.7B
  reads about 185 prompt tok/s and generates 30–65 tok/s (0.6B: about 520
  and 65), so a 550-token grounded prompt costs 3 s before the first word.
  Generation speed is timed from the first token (`StreamEvent`); it used to
  include the prompt, which made 60 tok/s read as 15–20. The ceiling is the
  WebAssembly build, which is why Apple Silicon gets the native MLX engine
  (below).
- **wllama loads single GGUF files up to 2 GB.** Larger models need gguf-split
  shards; `models.test.ts` enforces the limit.
- **wllama's chat logprobs are the raw next-token distribution** (wllama
  3.6.1, measured in the app). `top_logprobs` lists every option letter when
  the prompt asks for one; a logit bias doesn't change the reported values;
  and `post_sampling_probs: true` makes the reply carry **no** `top_logprobs`.
  So `decide` sends a letter grammar and reads raw logprobs. A letter missing
  from the top 20 is bounded at the lowest listed value and flagged.
- **This wllama build can't turn a JSON Schema into a grammar.**
  `response_format: {type: 'json_schema'}` and `json_schema` both fail with
  "Failed to initialize samplers"; a `grammar` (GBNF) string works. Tool
  arguments therefore use `schemaGrammar()` (agent/tools/validate.ts).
- **Qwen3 emits `<think></think>` even with thinking off.** `splitThink` treats
  an empty pair as no reasoning. Reasoning is never replayed to the model in
  history.

## MLX and the native models

- **Laya runs on MLX through mlx-rs (pinned `=0.32.0`), Apple Silicon only.**
  `build.rs` sets `cfg(mlx)` for `aarch64-apple-darwin`; elsewhere MLX and
  tokenizers aren't in the dependency graph and the commands (Laya's and the
  native chat models') say "needs an Apple Silicon Mac". mlx-rs compiles MLX from source: it needs CMake and,
  on Xcode 27, the separate Metal Toolchain component
  (`xcodebuild -downloadComponent MetalToolchain`; CI installs it). A clean
  build of MLX took about 2 minutes on an M5 Max.
- **MLX needs its kernel library, `mlx.metallib` (183 MB).** It searches next
  to the binary (and a `Resources/` folder beside it) and at its build path,
  not the app bundle's `Contents/Resources`, so the
  release bundle ships it there (`tauri.laya.conf.json` + `beforeBundleCommand`,
  since `bundle.resources` is copied at build.rs time, before mlx-sys may have
  built it) and `mlx.rs` points MLX at it with `set_metallib_path`.
  Without it MLX compiles kernels at runtime: the first run measured 42 ms
  P50 instead of 9 ms. The release job fails if the arm64 app lacks it.
- **mlx-sys builds `mlx.metallib` outside every cargo cache.** Its build script
  writes it to `~/.mlx/lib/<hash>` (or `$MLX_RS_METAL_PATH`) and MLX bakes that
  path into every binary as its runtime fallback. CI's rust-cache restores
  `target/` and `~/.cargo` but never `~/.mlx`, so on a cache hit the build
  script doesn't rerun and the file is missing; the first test that evaluates
  an MLX array then aborts the whole test process ("Failed to load the default
  metallib", exit 255; reproduced 2026-09-27). The macOS CI jobs therefore set
  `MLX_RS_METAL_PATH` to `src-tauri/target/mlx`, which rust-cache saves, and
  the one unignored test that evaluates arrays
  (`llm::model::tests::the_kv_cache_grows_in_steps_and_trims`) is `#[ignore]`d
   as a second guard. Release bundling is unaffected: `scripts/mlx-metallib.mjs`
   reads the CMake install tree under `target/`, produced either way.
- **`minimumSystemVersion` is the arm64 MLX build's deployment target**
  (measured 2026-09-27): the Tauri CLI exports `MACOSX_DEPLOYMENT_TARGET`
  from `bundle.macOS.minimumSystemVersion` (default `10.13`), CMake
  initializes `CMAKE_OSX_DEPLOYMENT_TARGET` from that env var, and MLX's
  CMakeLists refuses anything below 14.0 — the v0.1.2 arm64 release failed
  with "MLX requires macOS >= 14.0" while the local build (env var unset)
  passed. A failed configure poisons the CMake cache: reruns fail even with
  the env var fixed until the build dir is deleted. The arm64 overlay
  (`tauri.laya.conf.json`) therefore pins `minimumSystemVersion: "14.0"`
  (every Apple Silicon Mac supports 14; Intel and Windows builds have no MLX
  and keep the default), guarded by `tests/unit/arm64-release.test.ts`.
- **One thread owns MLX** (`mlx.rs`): the Laya checkpoint and both native
  LLM slots live on it, and commands send it closures. Decisions queue behind
  a streaming answer, which the agent never overlaps anyway.
- **Native chat models run Qwen3 on MLX in Rust** (`src-tauri/src/llm/`,
  2026-09-27). Picked over a `llama-server` sidecar (a second binary to
  sign, plus a loopback socket) and llama.cpp through a crate (a second
  Metal runtime) because MLX was already a dependency, and it measured
  fastest. Qwen3 1.7B on an M5 Max, 550-token prompt: mlx-lm 4-bit 347 tok/s
  generation and 12,200 tok/s prompt; llama.cpp (b11205, Metal, flash
  attention) Q4_K_M 285 and 8,770; wllama on WebGPU 30–65 and 185. In the
  app (`bun run bench:engine`, dev build): **343 tok/s generation, 9,900
  tok/s prompt, first token 55 ms** against about 3 s in wllama.
- **The Qwen3 port matches mlx-lm** (`bun run test:llm`): the chat template
  matches transformers on every golden case, token ids match, greedy output
  is identical, and the decision letters' log-probabilities match mlx-lm's
  float32 values exactly (Δ 0.000). The speed comes from mlx-lm's decode
  pipelining (queue step n+1 with `async_eval` before reading token n back);
  the model isn't compiled.
- **A 4-bit model's logprobs move with how the prompt is split.** Qwen3
  1.7B 4-bit on one decision prompt, all in mlx-lm: C scored −5.75 in one
  pass, −3.05 after a 3-token cached prefix, −1.50 when the last token ran
  alone (what a generation does). The chosen letter held, but confidences
  aren't comparable across cache states, so parity tests start from a clean
  cache, and decisions on the chat slot keep `cache_prompt: false` as with
  wllama.
- **Native generation reuses the KV cache's shared prefix** (`cache_prompt`,
  on by default): the cache keeps the token ids it holds, trims to the common
  prefix, and runs only the rest. The first-token distribution then matches a
  cold run within bf16 noise (tested: drift 0.08–0.46, as in mlx-lm, where a
  greedy tie can flip a reply either way).
- **Native grammars** (`llm/grammar.rs`): a choice (`root ::= "A" | "B"`)
  restricts the reply to those strings; any other GBNF without recursion
  (every `schemaGrammar`, fixture-checked) is compiled to a regex and a lazy
  DFA (`regex-automata`, already in the tree through Tauri), and each step
  picks among the tokens whose raw bytes keep it alive (80–90 ms for a
  17-token argument fill). Tokens are spelled from the byte-level BPE
  alphabet, not `decode`, which turns partial UTF-8 into U+FFFD.
- **Qwen3.5 on MLX** (`model_type` `qwen3_5`, 2026-09-30; `llm/model.rs`,
  `llm/delta.rs`, golden35.py). A hybrid: three of every four layers are
  Gated DeltaNet (depthwise causal conv + a gated linear recurrence with a
  fixed `[Hv, Dv, Dk]` float32 state), the fourth attention with a sigmoid
  output gate and RoPE on a quarter of each head; config under
  `text_config`, weights under `language_model.`, vision tower skipped.
  Dense only (`qwen3_5_moe` refused). The recurrence is mlx-lm's Metal
  kernel through mlx-sys's C API (`mlx_fast_metal_kernel_*`; mlx-rs 0.32
  doesn't wrap it), checked against an ops version. Its template differs
  from Qwen3's (content trimmed, thinking on ends the prompt with
  `<think>\n`, which `llm_generate` puts back in the reply). The state can't
  be trimmed, so prefill splits at each `<|im_start|>` and snapshots the
  states there (`Cache::snapshot`, ≤ 256 MB); a new prompt resumes from the
  last shared snapshot. Qwen3.5 0.8B OptiQ (mixed 4/8-bit) on an M5 Max
  (`test:llm`): same token ids and greedy text as mlx-lm, decision letters
  Δ 0.002, 307 tok/s generation, 4,164 tok/s prompt.
- **mlx-lm's Python wheel and the MLX that mlx-sys builds round
  differently on an M5** (2026-09-30). mlx-sys 0.6.0 builds MLX v0.32.2
  from source without the neural-accelerator (NAX) kernels the wheel's
  metallib has (6,055 NAX symbols vs 1); for a quantized matmul over 64+
  rows the wheel takes them, and a bf16 logit can land a step apart
  (identical inputs, `in_proj_qkv` sums −4136.06 vs −4136.14). Goldens
  therefore come from `mlx==0.32.2` on the engine's own prompt split. The
  Qwen3 goldens predate this and fail on `HEAD` (0.6B letters Δ 0.21, 1.7B
  greedy text differs after two words) while the Qwen3.5 ones pass.
- **1-bit MLX checkpoints don't load** (e.g. prism-ml Bonsai 27B): MLX 0.32
  quantizes 2, 3, 4, 5, 6 or 8 bits (`mx.quantize` refuses 1); Bonsai needs
  PrismML's MLX fork. `config.rs` refuses them with that reason.
- **The catalog carries Qwen's own MLX builds, 4B to 32B** (`Qwen/Qwen3-*-MLX-4bit`,
  2026-09-27): 4-bit in groups of 128, the same tokenizer as the smaller
  ones, untied from 8B up, and sharded at 14B and 32B, so their
  `model.safetensors.index.json` is pinned with the shards (4B and 8B ship
  an index naming the single file; it isn't pinned). They're chat models,
  not deciders (item 8: larger deciders chose worse). `test:llm`
  (`llm_qwen3_big_match_mlx_lm`) loads each from the HF cache and matches
  mlx-lm's greedy text; on an M5 Max, generation 184 / 113 / 64 / 28 tok/s
  (mlx-lm: 188 / 114 / 64 / 29), prompt 2,900 / 2,800 / 1,550 / 640 tok/s,
  peak memory 3.0 / 5.0 / 8.4 / 18.2 GB (mlx-lm), load 0.2–0.5 s. Settings warns (`memoryFit`) when a model
  likely exceeds ~75% of the Mac's memory, read by Rust from `hw.memsize`.
- **mlx-sys builds MLX as CMake "Debug" in dev builds** (its build script
  follows `debug_assertions`), which made Laya about 4× slower in `tauri dev`
  and the agent eval (53 ms vs 13 ms per decision). `Cargo.toml` builds
  mlx-sys, mlx-rs and tokenizers optimized in dev. `[profile.dev.build-override]`
  would reach it too, but it also changes Tauri's codegen and breaks the
  build (`missing field referenced_by`).

## Laya

- **The Laya port matches laya-mlx** (`bun run test:laya`, 2026-09-26, M5
  Max): identical token ids and markers, max |Δp| 0.0002 (multilingual) and
  0.0007 (English) against laya-mlx FP32, P50 8.0 ms and 18.5 ms per decision
  (laya-mlx in Python: 6.0 and 11.4 ms; the port has no graph compilation yet).
  mmBERT's config sets `rope_parameters.sliding_attention.rope_theta` to
  160000: the first port used the flat 10000 default and got every
  probability wrong while looking plausible, so parity tests are mandatory.
- **Laya's input budget:** options share `head_max_len` tokens (256
  multilingual, 192 English; each option at most 48), and the state gets the
  rest of `max_len` (1024 / 512), cut from the end, which is where the newest
  results are. So the decision state is fitted in TS instead (`agentState`'s
  `budget`, from `LayaDef.input`): the request (up to half, a long one keeps
  its start and end), the knowledge base and the step count always fit, then
  the newest results, then the conversation. Laya's tokenizers read 4.1–4.3
  characters per token of prose and 3.2–3.4 of paths and result lines; a
  993-character state budgeted at 3.2 still overflowed Laya English, so
  budgets use 2.9 (`LAYA_CHARS_PER_TOKEN`). `truncated` in the trace says
  when something was cut anyway (an option longer than its share: 3
  decisions in the item 15 eval).
- **Score a short follow-up with the question before it.** Against "And the
  Osprey?" alone, an accessibility section scored 96% and the dry-dock
  schedule 1% (`scoringRequest`).
- **Laya batches questions, but each is its own row:** the state is encoded
  again per question (laya-mlx does the same). Choice alone vs choice +
  stop in one pass: 7.9 → 12.7 ms (multilingual), 18.3 → 30.8 ms (English)
  in release; separate calls would cost about twice the single time.
- **Laya as a relevance filter** (`agent/relevance.ts`, `laya_relevance`: one
  `noul` row per passage, its own state `User request … Passage from …`).
  Probed on the eval fixtures (47 passages × 29 questions, a passage counts
  as relevant when it holds the question's expected fact), the statement
  "This passage contains information that helps answer the user's request."
  separated them with AUC 0.76 (multilingual) and 0.85 (English); dropping at
  0.2 would lose 17–30% of relevant passages, so the policy keeps the top 2
  search results and drops only scores < 0.10. Agent eval (Laya deciding,
  2026-09-26): multilingual dropped 21 of 168 passages, prompt 500 → 460
  tokens, facts 82.8% → 82.8%, grounded 48.5% → 54.5%, check 36 ms median;
  English dropped 43 of 158, prompt 432 → 333 tokens, first token 2990 →
  2406 ms median, facts 79.3% → 79.3% (one question gained, one lost), 57 ms.
  The eval's knowledge bases are small (7 files); larger ones retrieve more
  noise to drop.
- **Laya picks enum arguments** (`ToolDef.choices`, today `kb_search`'s
  `scope`), 2026-09-27. The chat model wrote `broad` for 13 of 16 code and
  mixed searches, bare identifiers included (`computeFare`, `withRetry`), in
  every MLX precision. Laya asks each as a `choice` row in the decision's
  batch (next + stop + choices ≤ `MAX_QUESTIONS`), or in its own pass when
  the tool came without a decision (search first), and `schemaFor` narrows
  the argument's enum to the picked value so the grammar holds it. A failed
  pick leaves the argument to the chat model. Agent eval, Qwen3 1.7B MLX
  answering (`eval/laya10-*`): Laya Multilingual facts 75.9% → 89.7% (4
  gained, none lost), grounded 75.8% → 84.8%; Laya English 82.8% → 89.7%
  (3 gained; lost `mixed-followup-surcharge`, where Laya chose `focused` for
  the topical query "vehicle features" and the answer named the constant
  without 18.5). Laya English: 36 → 37 ms per decision with the extra row.
- **Laya as a claim check** (`agent/claims.ts`, `laya_support`, 2026-09-27):
  after the answer, one `noul` row per cited sentence and source ("The
  passage supports this statement.", state `Statement: … Passage from …`).
  On 60 cited sentences from eval answers, own passage vs. one cited for
  another question: AUC 0.67 (multilingual), 0.82 (English). Below 0.10,
  1–2 of 60 own citations were clear false alarms and the rest real gaps,
  so it flags "may not be supported" and never changes the answer. In the
  eval it flagged 3 of 42 (multilingual, 14 ms) and 8 of 41 (English, 27 ms)
  cited sentences, including two real miscitations (right fact, wrong
  `[n]`) (tracker item 13). Models often cite after the full stop ("…
  terminal. [6]"), which split into a word-less "[6]": about a third of
  cited answers were skipped until `citedClaims` let a bare citation reach
  back to the ≤ 3 uncited sentences before it (reported from the app,
  2026-09-27). Sentences about what the sources lack (`aboutMissing`) are
  not checked. The probe's labels are a proxy (own
  passage vs. a random one, not hand-checked), and random negatives are
  easier than real miscitations, so treat its numbers as an upper bound;
  `claims.ts` `MEASURED` shows them in the claim dialog. Its input is
  `claimState` in TS and `claim_state` in Rust: keep them identical (both
  tested).

## Agent behavior and evals

- **Qwen3 0.6B's choices hinge on the option wording** (5 requests × 3
  orders): an *answer* option mentioning "general knowledge" won 9/9
  knowledge questions; the narrower wording in `loop.ts` scored 15/15 with
  *answer* as option A, 12/15 as the last option, 11/15 shuffled. After a
  search that found passages, it re-picked search 9/9 times while search was
  still offered and answered 9/9 once it wasn't, hence `offered()` in
  `loop.ts`. Removing options moves its choices too: hiding the three
  symbol tools until a symbol was seen made it answer without searching on
  12 of 16 code questions (agent eval, 2026-09-26), so a symbol tool chosen
  too early is redirected to Find symbols instead. Measure with
  `bun run eval:agent` before rewording or removing options.
- **Agent eval baseline (Qwen3 0.6B, 2026-09-26, `bun run eval:agent`):**
  the first action was `kb_search` for 23 of the 24 lookup questions,
  including every code question: the code tools were never chosen first.
  The miss was a pronoun follow-up ("What does it add for a vehicle?"),
  answered from history without a lookup. Small talk got `answer_now` 3/3.
  Only 56% of answers with sources cited them. Two seeded runs agreed on
  every question.
- **Native agent eval (2026-09-27, 34 questions, M5 Max):**

  | Setup (answers + decisions) | First action | Facts | Grounded | ms / decision | s / q |
  |---|---|---|---|---|---|
  | MLX 1.7B + MLX 0.6B | 100% | 82.8% | 86.2% | 23 | 0.58 |
  | MLX 1.7B alone | 85.3% | 79.3% | 93.1% | 38 | 0.58 |
  | MLX 0.6B alone | 100% | 82.8% | 55.2% | 24 | 0.46 |
  | wllama 1.7B + 0.6B (`eval/item7fix-1.7b+0.6b.json`) | 100% | 96.6% | 86.2% | 899 | 7.65 |

  10–13× faster per question, but fewer facts than wllama's 1.7B. The gap is
  mostly one argument: for "Which function implements the group discount…"
  the search `scope` came out `broad` with MLX 4-bit, 8-bit **and bf16**, and
  `focused` only with GGUF Q4_K_M; broad search returns one-line fragments
  and the answer misses the function. MLX 5-bit scored 89.7% at 302 tok/s
  and DWQ 4-bit 82.8%; 4-bit ships (349 tok/s, a human decision,
  2026-09-27). With a Laya decider, Laya now picks `scope` (tracker item 10,
  below); with a Qwen decider the chat model still writes it.
- **The decision never reads the results' text; each passage is scored
  whole in its own row** (item 15, 2026-09-28). A state can't hold the
  results (a search returns up to 6,000 characters, a read up to 8,000), and
  putting the best passages' text in it made Laya's stop question say
  "enough" sooner and cost facts. Instead `PassageScorer` asks Laya's
  relevance question per passage as each tool returns (a passage longer than
  a row is split into overlapping pieces, scored by the best), the result
  line says which passage is most useful and how likely it helps, and the
  loop overrules a "results suffice" when no passage scored ≥ 0.5
  (`STOP_EVIDENCE`). The relevance check before the answer reuses the scores.
- **ug search clips each passage to a share of the result budget** (about
  750 characters with `k` 8 and 6,000 characters), so a long section's end,
  and the fact in it, never reaches the decision or the answer: the item 15
  eval's large-document misses were mostly this, not Laya's window. Before
  answering from a clipped search passage (fewer than 80% of its node's
  lines, scored ≥ 0.3 with Laya, or the first one without), the loop reads
  that node whole with Read lines, once a turn (`read-whole` in the trace).
- **Fetch a symbol the request names before answering** (code and mixed,
  2026-09-29): a search often returns the passages around `cancelBooking`
  but not its code. When the request names an identifier (`namedIdentifiers`)
  whose code isn't among the results whole, the loop reads its source (Read
  symbol source), or its usages when the request asks who calls it, once a
  turn, without a decision or argument writer (`named-symbol` in the trace).
  100-question eval, Qwen3 1.7B MLX answering: Laya English facts 79.6% →
  80.6% (`mixed-cancel` gained, none lost), Laya Multilingual unchanged
  (three "who calls" answers, already right, gained a Find usages call).
- **Telling the answer that nothing found clearly helps didn't help**
  (2026-09-29). With Laya English, when no passage scored ≥ 0.25 (every
  unanswerable question scored 0.04–0.22), the prompt said so and asked for
  "the knowledge base doesn't say" over a guess: 1 unanswerable question
  gained, 1 lost, and the 1.7B still invented Wi-Fi on the Kestrel.
  Grounded fell 86.2% → 81.9%, but only because honest "not stated"
  answers cite nothing (`grounded` counts an answer with sources and no
  `[n]` as ungrounded). Laya Multilingual can't gate it: its unanswerable
  questions scored up to 0.80 and two answered-right ones 0.18 and 0.23.
  Not shipped.
- **Rules can plan the follow-ups, but didn't add facts** (tracker item 16,
  2026-09-28). `agent/plan.ts` reads clipped passages, fetches named symbols
  and answers from Laya's scores without a decision; it cut tool calls by a
  third and time per question by 25–45%, but lost 1–3 of 80 answers across
  the two Laya checkpoints. It ships off (`useTools` `plan`), as does
  `searchAgain` (search once more with the question as written when nothing
  scores ≥ 0.5).
- **Laya agent eval (2026-09-26, `bun run eval:agent`, Qwen3 0.6B answering,
  34 questions, seed fixed; `eval/laya-*.json`):**

  | Decider | First action | Facts | Decisions/q | ms/decision | s/question |
  |---|---|---|---|---|---|
  | Qwen3 0.6B | 100% | 86.2% | 1.09 | 756 | 4.16 |
  | Laya Multilingual, no stop question | 85.3% | 93.1% | 4.62 | 48 | 8.13 |
  | Laya Multilingual + stop question | 85.3% | 82.8% | 1.97 | 22 | 3.40 |
  | Laya English + stop question | 94.1% | 79.3% | 1.62 | 74 | 5.11 |

  Without the stop question the loop almost never picked *answer* after
  results and took every other tool until `offered()` ran out; with it
  (`STOP`, a `noul` asked in the same batch, answer at ≥ 0.5) that's gone.
  Laya still misses small talk that `needsLookup` lets through (it searches
  on "hi", clarifies on "thanks"), and a probe of a Laya "needs the
  knowledge base" `noul` scored the ferry question 0.0003 and "hi" 0.53, so
  it doesn't replace `needsLookup` yet. A four-way intent `choice`
  (`small_talk`, `about_assistant`, `kb_content`, `follow_up`; probe in
  `laya/engine.rs`, 2026-09-27) fared no better: Multilingual put 19 of 29
  lookups at ≥ 0.5 "no lookup", English put "hi" in `kb_content`
  (tracker item 12).
- **Search first holds on code knowledge bases too** (tracker item 14,
  2026-09-27). Letting the decision model pick the first step on code and
  mixed questions lost answer facts on every setup (−3.4 to −7 points), even
  only for questions naming a symbol: Find symbols returns names and lines,
  not code, and the deciders answered from that. Without a name, the
  argument writer guessed symbols that don't exist.

## Dev tooling and test harnesses

- **RTK rewrites `curl` output** and can mangle JSON. Use `rtk proxy curl`
  when you parse an API response.
- **Don't edit `src/` while `eval:agent` runs from the same checkout.** The
  harness serves the dev UI with hot reload, and a reload mid-run failed it
  ("eval-docs did not index: empty", 2026-09-27). A second run failed with
  "62 of 34 questions reported" while only `AGENTS.md` and `scripts/e2e.mjs`
  were being edited (cause not found), so edit nothing while it runs. Measure a "before" from an
  export of `HEAD` (`git archive HEAD | tar -x`, `node_modules` symlinked,
  `public/wllama` copied, `CARGO_TARGET_DIR` pointed at this checkout's
  `src-tauri/target`).
- **The e2e and eval runners serve the dev UI on port 1431**
  (`scripts/dev-port.mjs`, `ANDAI_DEV_PORT`), not 1420. "localhost" reaches
  both 127.0.0.1 and [::1]: with a developer's own `tauri dev` on 1420, a
  harness run loaded that app instead of its own and hung until its
  timeout. Their webview storage is therefore separate from the dev app's
  (the model downloads once more for the harness).
