# Andai — features

Andai is a private AI agent for macOS. The language model runs **inside the
app on your Mac**, and it answers from **your own documents**. After a one-time
model download nothing leaves the machine: no account, no API key, no cloud.

> Status legend: **Available**: works today. **Preview**: the interface is
> there, but the underlying execution is simulated.

---

## Command Center — talk to your agent · Available

- **Streaming answers.** Replies render as they're generated, with Markdown,
  code blocks and tables. **Stop** (or `Esc`) halts generation at any point.
- **Visible reasoning.** Each answer shows its steps as they happen:
  *analyzing the query → searching your knowledge → assembling context →
  generating*. Qwen3 models can also show their step-by-step thinking, folded
  away until you open it.
- **Cited sources.** When a knowledge base is selected, answers cite passages
  as `[1]`, `[2]`. Each citation matches a source listed under the answer, with
  its file name and line range.
- **Execution Trace.** A side panel with each step's status and timing, the
  sources used, live throughput (tokens/second) and how much of the model's
  context window the turn used. Hide it for a focused view (`⌘J`).
- **Conversation memory.** Earlier turns are carried into follow-up questions
  (within the model's context budget), and your history is kept between launches.

## Knowledge — your documents, as a knowledge graph · Available

- **Drop files to ingest them:** PDF, Markdown, plain text, CSV, and source code
  (TypeScript, JavaScript, Python, Java, Rust). Drag them from Finder onto the
  window or use **Upload**.
- **Indexed on your Mac by [ug](https://github.com/shoocstorm/ug)**, which
  splits documents along their structure (headings, pages, symbols), links the
  sections into a graph and embeds them locally. Live indexing progress is
  shown as it runs.
- **Multiple knowledge bases.** Keep separate collections (e.g. *Specs*,
  *Research*) and choose which one grounds the chat from the composer.
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
  from the Mac's local cache: about 1.3 s for Qwen3 0.6B in our tests on an Apple Silicon Mac.
- Runs with [wllama](https://github.com/ngxson/wllama) (llama.cpp compiled to
  WebAssembly) on the GPU through WebGPU, multi-threaded. We measured ~30
  tokens/second for Qwen3 0.6B on an Apple Silicon Mac.
- Unload or delete cached models from **Settings**.

## Workspace

- **Appearance:** Light, Dark, or follow macOS (`Settings → Appearance`, or the
  sun/moon button in the top bar).
- **Collapsible sidebar** (`⌘B`) and **hideable Execution Trace** (`⌘J`),
  both remembered.
- **Keyboard shortcuts:** `⌘K` focus the input · `⌘1–4` switch screens ·
  `⌘,` Settings · `Enter` send · `Shift+Enter` new line · `Esc` stop.

## Workflows & tools · Preview

A glimpse of where Andai is heading. The screens work, but **nothing is
executed**; every action says it's simulated.

- **Tool library**: web search, data analysis, email, SQL, S3. Only
  **Knowledge Search** is live today.
- **Human-in-the-loop approvals**: approve or reject actions an agent wants to take.
- **Visual workflow editor**: a pannable, zoomable node canvas with node
  configuration and a simulated run that pauses for your approval.

---

## Privacy

- Inference and indexing run locally. The only network use is downloading a
  model you chose, from Hugging Face.
- No telemetry, analytics or accounts.
- Your data lives in `~/Library/Application Support/dev.andai.agent/`
  (knowledge-base copies), `~/.ug/andai-*` (knowledge graphs) and the app's
  own storage (chats, settings, cached models).

## Requirements & known limits

- **macOS**, tested on macOS 26 (Apple Silicon). The release pipeline also
  builds for Intel Macs; that build hasn't been tested on Intel hardware yet.
- Knowledge bases need the **ug** CLI installed (`~/.local/bin/ug` or on your
  PATH).
- Builds are currently **unsigned**: on first launch, right-click Andai.app → Open.
- Andai uses local port **14230** internally. If another app holds it, Andai
  can't start.
- Small on-device models are fast and private but less capable than large
  cloud models. Grounding answers in a knowledge base helps a lot.
