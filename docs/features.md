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
- **Visible reasoning.** While Andai works, one line above the answer shows
  a dot per step (*planning and using tools → assembling context →
  generating*) and what the current one is doing, such as the tool call it's
  running. Once the answer is in, it folds to a single line (how long it
  took, the tool calls and passages it used, and a warning if a step failed
  or a claim needs checking) that opens to every step. Each step carries the
  same number and icon as its card in the Execution Trace; click one to
  scroll the trace to that card and highlight it. Qwen3 models can also
  show their step-by-step thinking, folded away until you open it.
- **Cited sources.** When a knowledge base is selected, answers cite passages
  as `[1]`, `[2]`. Each citation matches a source listed under the answer, with
  its file name and line range, in *Passages found*, where each also shows
  how the search found it (by meaning, by keyword, or by following the
  knowledge graph) and how strongly it matched. Click a citation, or a
  passage in the list, to read it, with what the relevance and claim
  checks said about it.
- **What the model read.** *What went in?* on the trace's *Assemble context*
  step shows how the model's context window was split between the reply,
  retrieved passages and earlier conversation; which passages went in whole,
  were cut to fit or were left out; how much conversation was kept; and the
  exact system prompt.
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
- **It searches first.** With a knowledge base selected, a question about
  your content starts with a search, which saves a decision; the trace says
  so. Greetings, thanks and questions to the assistant itself ("who are
  you?") go to the model, which answers them without searching. After the
  first step, the model decides each step.
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
  model just for choosing the next step. Without it, the chat model decides.
  In our agent eval (32 questions, Apple M5 Max) the best setup was **Qwen3
  1.7B as the chat model with Qwen3 0.6B deciding**: 96.6% of answers had the
  expected facts, against 86.2% with 0.6B alone, at about 7 s per question
  instead of 3.6 s. A 1.7B decision model did worse: it often asked a
  clarifying question instead of answering. The MLX models can decide too
  (Apple Silicon): with Qwen3 1.7B · MLX answering, Qwen3 0.6B · MLX deciding
  takes about 25 ms per decision.
- **Laya decision models (Apple Silicon Macs).** *Settings → Decision model*
  lists them first and offers **Laya Multilingual** (322M parameters, 614 MB) and **Laya
  English** (421M, 804 MB): small encoders built for exactly this kind of
  choice, run natively on the Mac's GPU. A decision takes about 10 ms with
  Laya Multilingual and 20 ms with Laya English, against about 0.6 s for a
  Qwen3 0.6B decision (measured on an Apple M5 Max). Each downloads once from
  a fixed Hugging Face commit and is checked against its recorded sha256
  before it's kept. Not available on Intel Macs or Windows. Once a tool has
  found something, Laya also answers a yes/no question in the same pass, *do
  the results already cover the request?*, and the agent answers when it
  says yes and at least one passage found so far scored as helping (below). In our agent eval (34 questions, Qwen3 0.6B writing the answers)
  Laya Multilingual made questions faster overall, 3.4 s against 4.2 s with
  Qwen3 0.6B deciding, but its answers had the expected facts slightly less
  often (82.8% against 86.2%) and it searched on greetings and small talk.
  Laya English was slower overall (5.1 s). Qwen3 0.6B stays the most
  accurate decision model; Laya Multilingual is the fastest.
- **Every passage read whole (with a Laya decision model).** Laya reads at
  most 512 or 1,024 tokens, far less than a search returns, so it doesn't
  decide from the results' text. As each tool returns, Laya scores every
  passage on its own, together with the question (a long passage is split
  into overlapping pieces and scored by its best one). The next decision
  sees which passage is most useful and how likely it helps, and a
  "the results suffice" is overruled while no passage scores at least 50%.
  The Execution Trace shows the scores under each tool call (*Scored by*).
  A short follow-up ("And the Osprey?") is scored together with the
  question before it.
- **Clipped passages read whole.** A search shows each passage only up to
  its share of the result size, so the end of a long section can be cut
  off. Before answering from such a passage, the agent reads its whole
  section first (*Read lines*, once per answer; the trace says why). This
  works with every decision model. In our agent eval's long-document
  questions (11 of 45), answers had the expected facts in 81.8% of them with
  Laya English (36.4% before), 72.7% with Laya Multilingual (63.6%) and
  54.5% with Qwen3 0.6B deciding (27.3%), with Qwen3 1.7B answering on MLX,
  on an Apple M5 Max. With Laya it costs about 0.05–0.2 s more per
  question, mostly from the extra read.
- **Named code read before answering.** On a code or mixed knowledge base,
  when your question names a function or other symbol (`cancelBooking`,
  `refundFraction()`) and the results don't show its code whole, the agent
  reads its source before answering (*Read symbol source*), or finds its
  callers when you ask who calls it (*Find usages*). Once per answer, with
  any decision model; the trace says why. In our agent eval (100 questions,
  Qwen3 1.7B answering on MLX, Apple M5 Max) it gained one answer with Laya
  English and changed none with Laya Multilingual.
- **Relevance check (with a Laya decision model).** Before retrieved
  passages go into the chat model's prompt, Laya scores how likely each one
  helps answer the question and drops the clear misses (below 10%). The top
  two search results are always kept, and if the check fails nothing is
  dropped. The Execution Trace lists every passage with its score, what was
  kept (numbered as the answer cites it) and what was dropped, folded to one
  line until you open it; click a passage to see the request, its text,
  Laya's answer and why it was kept or dropped. In our agent
  eval it made the prompt 23% shorter with Laya English (the first word came
  about 0.6 s sooner) and 8% shorter with Laya Multilingual, with no loss of
  answer facts, for 36–57 ms per question.
- **Claim check (with a Laya decision model).** After the answer is
  written, Laya checks each sentence that cites a source (`[n]`) against the
  passage it cites. When one probably isn't supported, a short note under
  the answer lists it, and the Execution Trace shows every cited sentence
  with its score (folded to one line). Click a sentence to open the check: the
  sentence, the passage it was compared with, Laya's answer, and why it
  counts as a flag. The answer itself is never changed. It takes about
  15–30 ms per answer (median 14 ms with Laya Multilingual, 27 ms with
  Laya English).
  *How far to trust it:* treat a flag as "read this source", not "this is
  wrong". On our small test set (60 cited sentences, not hand-checked),
  Laya ranked a sentence's own passage above an unrelated one 67%
  (Multilingual) or 82% (English) of the time, flagged about 40% or 60% of
  the unrelated pairings, and wrongly flagged 1 or 2 of 60 correct ones.
  Real miscitations are usually subtler than an unrelated passage, so
  expect it to miss more. It also checks only sentences that carry a `[n]`,
  checks a sentence that cites several sources against each one alone (so
  a sentence that combines them can be flagged), and reads the stored
  passage, which may be shorter than what the chat model saw.
- **Search scope (with a Laya decision model).** A knowledge search is either
  *focused* (direct matches for a name or exact term) or *broad* (related
  passages too). With Laya loaded, Laya picks the scope as a typed choice in
  the same pass as the next step, and the chat model writes only the search
  phrase; the tool call dialog says who picked it and how sure it was. Small
  chat models chose *broad* even for function names, and a broad search
  returns one-line fragments instead of the function. In our agent eval
  (Qwen3 1.7B on MLX answering) answers found the expected facts in 89.7% of
  questions, up from 75.9% with Laya Multilingual and 82.8% with Laya English.
- **Every step is visible.** The Execution Trace lists each decision with
  its options and probabilities, and each tool call with its arguments, the
  ug command that ran, timing and what it returned. Each step shows how long
  its decision took and the turn shows the total. **Why this step?** opens a
  dialog that explains the decision in plain sentences (what the model was
  asked, how it scored each option, any override, what the agent did), shows
  what the model saw (the state, the question and the options) and what it
  returned, with the exact prompt, parameters and raw reply folded away and a
  copy button. A decision that failed shows the same. **Tool call** opens a
  second dialog that says what the call did in plain sentences (who wrote
  the arguments, whether it ran on its own or waited for your approval, how
  long ug took, how many passages went into the answer), then each argument
  with what it means and who set it. **How the arguments were written**
  unfolds what the chat model was sent to write them, attempt by attempt:
  the system prompt, the state it read, the tool's schema, the parameters
  (temperature, token limit, the grammar the reply was held to), its raw
  JSON reply and why a reply was rejected, with **Copy argument call**. Then
  the ug command, and the passages it found by file and line, with the raw
  output folded away. **Copy trace** exports a turn as JSON.
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
- **Sample knowledge bases** to try the agent before adding your own files:
  *Tidewater Ferries*, a made-up ferry operator, as **documents** (a
  handbook, a refund policy, release notes), as **code** (its booking service
  in TypeScript), or **both**. One click adds a sample, indexes it on your
  computer in a few seconds and grounds the chat in it, and the Command
  Center then suggests questions the sample can answer. Offered on the
  Knowledge screen, in *New knowledge base*, and as *Try a sample* when you
  have none. A sample is an ordinary knowledge base: delete it any time.
- **Kind: documents, code or mixed**, set automatically from your files and
  changeable next to the source list. It decides which agent tools apply:
  the code tools (symbols, callers) are offered only for code.
- **Look inside a source.** Click a file's name in the source list to open
  it: **Overview** (size, tokens, lines, where it was added from, when it was
  added and indexed, language), **Content** (Markdown rendered or as numbered
  text, code with line numbers; a PDF shows the text ug indexed from each
  page) and **Structure** (the sections, pages or code symbols ug split it
  into, with their lines, plus related files such as imports and files in the
  same folder). Clicking an outline entry shows its lines; clicking a related
  file in the same knowledge base opens it. Files over 512 KB show their
  first 512 KB.
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
  (1.1 GB, smarter and slower) and a tiny test model, on every computer; and
  on Apple Silicon Macs, **Qwen3 1.7B · MLX** (980 MB) and **Qwen3 0.6B ·
  MLX** (645 MB), listed first there, followed by Qwen's own larger MLX
  builds: **Qwen3 4B** (2.1 GB), **8B** (4.4 GB), **14B** (7.9 GB) and
  **32B** (17.4 GB). Larger models answer better and write more slowly
  (measured on an Apple M5 Max: about 185, 115, 65 and 28 tokens/second; they need about 3, 5, 9 and 19 GB of memory). A model that probably won't fit your Mac's memory says so on its
  card before you download it.
- **What's loaded is marked.** The chat model in use has a green check and a
  green outline in *Settings → Models*, and the decision model in use has a
  green check and an *In use* label.
- **MLX models are much faster on a Mac.** They run natively on the Mac's
  GPU with Apple's MLX, instead of inside the app's web view. Measured on an
  Apple M5 Max with a grounded-size prompt: Qwen3 1.7B · MLX writes about
  **350 tokens/second** and starts answering after about **60 ms**, where
  Qwen3 1.7B in the web view writes 30–65 tokens/second and starts after
  about 3 s; Qwen3 0.6B · MLX writes about 450 tokens/second. A whole agent
  question (search, decide, answer) took 0.6 s instead of 7.7 s. They're
  4-bit and 8-bit versions of the same models, so answers can differ: in our
  agent eval, Qwen3 1.7B · MLX with Qwen3 0.6B · MLX deciding found the
  expected facts in 82.8% of answers, against 96.6% for the web-view Qwen3
  1.7B with 0.6B deciding (we're working on the gap; it's mostly one search
  setting). Not available on Intel Macs or Windows. Settings marks which
  engine runs each model.
- One-time download from Hugging Face with progress. After that the model loads
  from the local cache: about 1.3 s for Qwen3 0.6B in our tests on an Apple Silicon Mac.
- **Verified downloads.** Each model comes from a fixed Hugging Face commit and
  must match its recorded sha256 before it loads (for MLX models, every file
  of it). This takes about 7 s for
  Qwen3 0.6B, once per download, shown as *Verifying…*. After upgrading, models
  download once more; **Settings → Models** offers to remove the older copy,
  after asking you to confirm.
- The other models run with [wllama](https://github.com/ngxson/wllama)
  (llama.cpp compiled to WebAssembly) on the GPU through WebGPU: every layer
  of the model runs on the GPU. On an Apple M5 Max we measured Qwen3 0.6B writing about 65
  tokens/second after reading its prompt at about 520 tokens/second; Qwen3
  1.7B writes 30–65 tokens/second (it varies with how busy the Mac is) and
  reads about 185 tokens/second, so a grounded answer starts after about
  3 s. The answer footer shows both speeds.
- **Add a model from Hugging Face** (*Settings → Models → Add from Hugging
  Face*). Search public models by name; pick one to see whether Andai can
  run it and why not, its license, downloads and the exact version that will
  be used. Two kinds are offered:
  - **GGUF**, on any computer: one file up to 2 GB (Andai recommends a
    4-bit K-quant when there's a choice), run like the built-in models.
  - **MLX**, on Apple Silicon Macs: Qwen3 models quantized for MLX (4- to
    8-bit, including larger ones like Qwen3 8B), run natively on the GPU.
    Tested with Qwen3 8B 4-bit: about 115 tokens/second on an Apple M5 Max.
  The model is pinned to the version you saw and each file is checked
  against its sha256 before it loads. Models that need a Hugging Face login
  (gated) or are private can't be added. These are third-party models Andai
  hasn't reviewed: the files can't run code, but a model can still give
  wrong or harmful answers, and its license is yours to check. Added models
  show where they came from and can be removed from Settings (their files
  are deleted after you confirm).
- Unload or delete cached models from **Settings**.

## Workspace

- **Appearance:** Light or Dark (`Settings → Appearance`, or the sun/moon
  button in the top bar). The first launch starts in whichever your system
  uses; after that, Andai keeps your choice.
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
- No telemetry, analytics or accounts. (This is about the desktop app; the
  browser deployment loads Google Analytics — see *Run in the browser*.)
- Your data lives in the app data folder (knowledge-base copies:
  `~/Library/Application Support/dev.andai.agent/` on macOS,
  `%APPDATA%\dev.andai.agent\` on Windows), in `.ug/andai-*` in your home
  folder (knowledge graphs) and in the app's own storage (chats, settings,
  cached models).

## Requirements & known limits

- **macOS**, tested on macOS 26 (Apple Silicon). The Apple Silicon build
  requires macOS 14 or newer (the native MLX models need it, and the app
  refuses to install on older versions). The release pipeline also
  builds for Intel Macs; that build hasn't been tested on Intel hardware yet. The Laya
  decision models need Apple Silicon.
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

## Run in the browser · Preview

The same UI also runs as a web app at
[andai-agent.web.app](https://andai-agent.web.app), deployed from the repo
with `bun run deploy:web` (Firebase Hosting).

- **Chat, persona, models and settings work** — inference runs in the browser,
  on the same engines as the desktop app (WebGPU where available, a
  WebAssembly fallback elsewhere).
- **Knowledge bases are desktop-only**: indexing uses the local **ug** CLI,
  which a web page can't run. The web app has no knowledge feature; the
  Knowledge screens say so.
- **Desktop models are desktop-only**: the native MLX engine and the Laya
  decision models need Apple Silicon. In the browser, decisions fall back to
  the on-device model itself.
- **The desktop app stays analytics-free.** The web deployment loads Google
  Analytics; the desktop app does not, by design (its content security policy
  names no analytics host).

## Requirements & known limits (web app)

- A current **Chrome or Edge** (WebGPU, multi-threaded inference) or **Safari
  16+** (slower WebAssembly fallback). Not tested on Firefox.
- The model (~0.6–1.2 GB) is downloaded once per browser and stored in the
  browser's cache; clearing site data removes it.
- Everything a web page can offer: your chats live in this browser's storage
  and follow its private-mode and quota rules.

