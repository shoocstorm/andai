# Andai — features

Andai is a local-first **agentic RAG** agent for macOS and Windows. It answers
from **your own knowledge bases**, and the language model runs **inside the app
on your computer**. Rather than pasting a few matching chunks into a prompt, the agent
uses tools on your knowledge base: it searches, reads around a hit, outlines a
file or follows a code symbol, then answers with citations. After a one-time
model download nothing leaves the machine: no account, no API key, no cloud.

> Status legend: **Available**: works today. **Preview**: the interface is
> there, but the underlying execution is simulated.

---

## Command Center — talk to your agent · Available

- **Streaming answers.** Replies render as they're generated, with Markdown,
  code blocks and tables. **Stop** (or `Esc`) halts generation at any point.
- **Visible reasoning.** Each answer shows its steps as they happen:
  *planning and using tools → assembling context → generating*, with a chip
  for each tool call. Qwen3 models can also show their step-by-step thinking, folded
  away until you open it.
- **Cited sources.** When a knowledge base is selected, answers cite passages
  as `[1]`, `[2]`. Each citation matches a source listed under the answer, with
  its file name and line range.
- **Execution Trace.** A side panel with each step's status and timing,
  each decision and tool call (see below), the sources used, live throughput (tokens/second) and how much of the model's
  context window the turn used. Hide it for a focused view (`⌘J`).
- **Conversation memory.** Earlier turns are carried into follow-up questions
  (within the model's context budget), and your history is kept between launches.

## Agentic retrieval — tools over your knowledge · Available

With a knowledge base selected, the agent works it with tools instead of
pasting in whatever one search returns:

- **The agent decides what to do next.** At each step the local model picks
  one action: use a tool, answer now, or ask you a clarifying question. It
  chooses from a short lettered list and doesn't write free text, so every
  choice is valid, and the probability of each option is shown in the trace.
  When the model isn't sure enough, or a decision fails, Andai does one plain
  knowledge search instead of acting on a guess. About 0.6 s per decision
  with Qwen3 0.6B in our tests on an Apple Silicon Mac.
- **It writes its own search.** Before a search, the model turns your
  question into a search phrase (for example *"What HTTP headers does wllama
  need for multi-threading?"* became *wllama multi-threading headers*); for
  code tools it picks the symbol or file. Arguments are held to each tool's
  schema, and a file, symbol or line range must be one the knowledge base or
  earlier results showed (Read lines reads around a passage already found,
  with 20 lines either side); if it picks such a tool before anything has
  turned up, it looks symbols up or searches first, and the trace says so.
- **Tools built on [ug](https://github.com/shoocstorm/ug)**, offered to
  match what the knowledge base holds (documents, code or both; set
  automatically from your files, and you can change it):

  | Tool | What it does | For |
  |---|---|---|
  | Knowledge search | Finds relevant passages, broad (follow related sections) or focused | All |
  | Read lines | Reads up to 400 lines of a file around a result | All |
  | File outline | A file's headings or symbols, and what it connects to | All |
  | Overview | What the knowledge base contains: size, file types, largest files | All |
  | Find symbols | Looks up functions, classes and so on by name or wildcard | Code |
  | Symbol context | One symbol's source, callers, tests and dependencies | Code |
  | Read symbol source | The full source of a named symbol | Code |
  | Find usages | Who calls, imports or references a symbol | Code |

- **You stay in control.** The **Tools** screen (`⌘5`) lists every tool, the
  exact ug command it runs and how often it ran. Each tool is *Auto* (runs
  without asking; the default for these read-only tools), *Ask* (an approval
  card appears in the chat) or *Off* (never offered). You also set how many
  tool calls a question may use and the minimum confidence, or turn agent mode
  off to go back to one search per question.
- **An optional decision model.** *Settings → Decision model* loads a second
  model just for choosing the next step, so a larger model can decide while a
  smaller one writes. Without it, the chat model decides.
- **Every step is visible.** The Execution Trace lists each decision with
  its options and probabilities, and each tool call with its arguments, the
  ug command that ran, timing and what it returned. **Copy trace** exports a
  turn as JSON.
- **Read-only and on your computer.** Every tool only reads the selected
  knowledge base. Andai re-checks each call before running it, limits its
  time and output, and never lets a tool reach the network.

## Knowledge — your documents, as a knowledge graph · Available

- **Drop files to ingest them:** PDF, Markdown, plain text, CSV, and source code
  (TypeScript, JavaScript, Python, Java, Rust). Drag them from Finder or File
  Explorer onto the window or use **Upload**.
- **Local files only.** Knowledge bases are built from files on your computer, and
  the agent searches only those. There are no remote or cloud sources.
- **Indexed on your computer by [ug](https://github.com/shoocstorm/ug)**, which
  splits documents along their structure (headings, pages, symbols), links the
  sections into a graph and embeds them locally. Live indexing progress is
  shown as it runs.
- **Multiple knowledge bases.** Keep separate collections (e.g. *Specs*,
  *Research*) and choose which one grounds the chat from the composer.
- **Kind: documents, code or mixed**, set automatically from your files and
  changeable next to the source list. It decides which agent tools apply:
  the code tools (symbols, callers) are offered only for code.
- **Retrieval controls:** how many passages to retrieve (4 / 8 / 16 / 32) and
  how much text to give the model per question.
- Remove a source or delete a whole knowledge base at any time. Your original
  files are never modified.

## Persona — shape how the agent behaves · Available

- **System prompt** editor, plus **Auto-optimize**: the local model rewrites
  your prompt for clarity, and **Undo** restores the previous version.
- **Tone of voice:** Professional, Friendly or Creative.
- **Temperature** (deterministic ↔ creative) and **maximum response length**.
- **Verbose reasoning:** show the model's thinking steps.
- Rename the agent. Everything is remembered between launches.

## Models — on-device inference · Available

- Built-in catalog: **Qwen3 0.6B** (default, 639 MB), **Qwen3 1.7B**
  (1.1 GB, smarter and slower) and a tiny test model.
- One-time download from Hugging Face with progress. After that the model loads
  from the local cache: about 1.3 s for Qwen3 0.6B in our tests on an Apple Silicon Mac.
- **Verified downloads.** Each model comes from a fixed Hugging Face commit and
  must match its recorded sha256 before it loads. This takes about 7 s for
  Qwen3 0.6B, once per download, shown as *Verifying…*. After upgrading, models
  download once more; **Settings → Models** offers to remove the older copy,
  after asking you to confirm.
- Runs with [wllama](https://github.com/ngxson/wllama) (llama.cpp compiled to
  WebAssembly) on the GPU through WebGPU: every layer of the model runs on
  the GPU. On an Apple M5 Max we measured Qwen3 0.6B writing about 65
  tokens/second after reading its prompt at about 520 tokens/second; Qwen3
  1.7B writes 30–65 tokens/second (it varies with how busy the Mac is) and
  reads about 185 tokens/second, so a grounded answer starts after about
  3 s. The answer footer shows both speeds.
- Unload or delete cached models from **Settings**.

## Workspace

- **Appearance:** Light, Dark, or follow the system (`Settings → Appearance`, or the
  sun/moon button in the top bar).
- **Collapsible sidebar** (`⌘B`) and **hideable Execution Trace** (`⌘J`),
  both remembered.
- **Keyboard shortcuts:** `⌘K` focus the input · `⌘1–5` switch screens ·
  `⌘,` Settings · `Enter` send · `Shift+Enter` new line · `Esc` stop. On
  Windows, use `Ctrl` instead of `⌘`; the app shows the right key.

## Workflows & tools · Preview

A glimpse of where Andai is heading. The screens work, but **nothing is
executed**; every action says it's simulated.

- **Tool library**: web search, data analysis, email, SQL, S3, all simulated.
  The real knowledge-base tools are on the **Tools** screen (see *Agentic
  retrieval* above).
- **Human-in-the-loop approvals**: approve or reject actions an agent wants to take.
- **Visual workflow editor**: a pannable, zoomable node canvas with node
  configuration and a simulated run that pauses for your approval.

---

## Privacy & security

- Inference and indexing run locally. The only network use is downloading a
  model you chose, from Hugging Face. A content security policy enforces this:
  the app can't reach any other host.
- **Answers can't phone home.** Images in a model's answer are never loaded,
  and links show their real host and can only be copied, not opened. A
  document crafted to trick the model can't use either to send your text out.
- **Only files you choose are read.** Andai ingests exactly the files you
  drop or pick with **Upload**, nothing else on your computer. Files over 100 MB are
  refused.
- Knowledge-base copies are readable only by your user account (on macOS,
  owner-only permissions; on Windows, the permissions of your user profile
  folder, which also let administrators read them).
- **Documents can't take over the model.** Retrieved passages are sent as
  fenced, untrusted data that the model is told not to follow as
  instructions. The protections above hold even if a document fools the
  model anyway.
- **Model files are verified** (sha256) before they load, and dependencies are
  checked for known vulnerabilities on every build and release.
- The full picture, including what isn't protected yet, is in
  [security.md](security.md).
- No telemetry, analytics or accounts.
- Your data lives in the app data folder (knowledge-base copies:
  `~/Library/Application Support/dev.andai.agent/` on macOS,
  `%APPDATA%\dev.andai.agent\` on Windows), in `.ug/andai-*` in your home
  folder (knowledge graphs) and in the app's own storage (chats, settings,
  cached models).

## Requirements & known limits

- **macOS**, tested on macOS 26 (Apple Silicon). The release pipeline also
  builds for Intel Macs; that build hasn't been tested on Intel hardware yet.
- **Windows 10/11 (x64)**: the release pipeline builds an installer, and the
  test suite runs on Windows in CI. It hasn't been tested on a Windows PC yet.
  Needs the Microsoft Edge WebView2 runtime (preinstalled on Windows 11); the
  installer doesn't download it for you.
- Knowledge bases need the **ug** CLI installed (on your PATH, or in
  `.local/bin`, `.cargo/bin` or `.ug/bin` in your home folder).
- Builds are currently **unsigned**: on first launch on macOS, right-click
  Andai.app → Open; on Windows, choose *More info* → *Run anyway* if
  SmartScreen warns.
- Andai uses local port **14230** internally. If another app holds it, Andai
  shows an error and doesn't start, rather than loading something it doesn't
  own. Quit the other app (`lsof -i :14230` on macOS or
  `netstat -ano | findstr :14230` on Windows shows which one) and reopen Andai.
- Small on-device models are fast and private but less capable than large
  cloud models. Grounding answers in a knowledge base helps a lot.
