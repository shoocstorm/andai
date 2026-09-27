// The local LLM, on one of two engines (llm/models.ts `engine`):
// - wllama (llama.cpp → WASM) running inside the app's webview, everywhere;
// - MLX, natively in Rust on Apple Silicon (llm/native.ts), 5–10× faster.
// Callers don't care which: `chat` streams and `complete` answers the same
// way on both. The rest of this note is about wllama.
//
// On macOS, Tauri renders with WKWebView (Safari's engine), which lacks the
// Memory64 and JSPI features wllama's default build needs. wllama detects that
// itself (`needCompat()`) and switches to the compat build we point it at via
// `setCompat` — served from public/wllama/, never the CDN. In Chromium (plain
// `vite` in a browser, and WebView2 on Windows) the default build runs and
// `setCompat` is ignored.
//
// Models are cached in OPFS by wllama's ModelManager, so each downloads once.
// Every download is checked against the catalog's pinned size and sha256
// before it is loaded (integrity.ts, AGENTS.md §9).

import { ModelManager, Wllama, WllamaAbortError, type ChatCompletionParams, type ChatCompletionResponse } from '@wllama/wllama';
import { create } from 'zustand';
import { verifyBlobs } from './integrity';
import { downloadCheckpoint, downloadLaya, layaLoad, layaRemove, layaStatus, layaUnload, type LayaStatus } from './laya';
import { isMlx, layaById, MODELS, modelById, type LayaDef, type MlxDef, type ModelDef, type WllamaDef } from './models';
import {
  asCompletion,
  genTokPerSec,
  nativeGenerate,
  nativeLoad,
  nativeRemove,
  nativeStatus,
  nativeUnload,
  promptTokPerSec,
  type Generated,
  type NativeLoaded,
  type NativeSlot,
  type NativeStatus,
} from './native';

const asset = (path: string) => new URL(path, window.location.href).href;

export type ChatMessage = { role: 'system' | 'user' | 'assistant'; content: string };

export type EngineInfo = {
  backend: string;
  threads: string;
  context: string;
  layers: number;
  arch: string;
  libllama: string;
  compat: boolean;
};

export type Caps = {
  isolated: boolean;
  sharedArrayBuffer: boolean;
  opfs: boolean;
  webgpu: boolean;
  cores: number;
};

type EngineState = {
  status: 'idle' | 'loading' | 'ready' | 'error';
  loadingId: string | null;
  loadedId: string | null;
  progress: { loaded: number; total: number; speed: number; phase: string } | null;
  error: string | null;
  info: EngineInfo | null;
  cached: Record<string, number>; // url -> bytes
  /** Copies cached under a model's pre-pinning URL, by model id -> bytes. */
  legacy: Record<string, number>;
  caps: Caps;
  generating: boolean;
  tokPerSec: number | null;
  lastLoadMs: number | null;
  /** How long the last sha256 check took; null when a verified copy was reused. */
  lastVerifyMs: number | null;
  /** Laya checkpoints (Rust, Apple Silicon only): which are downloaded, which is loaded. */
  laya: LayaStatus;
  /** Native MLX chat models (Rust, Apple Silicon only): supported here, downloaded, loaded per slot. */
  native: NativeStatus;
  /** The optional decision model (loadDecider): a wllama model or a Laya checkpoint. */
  decider: {
    status: 'idle' | 'loading' | 'ready' | 'error';
    loadingId: string | null;
    loadedId: string | null;
    progress: { loaded: number; total: number; speed: number; phase: string } | null;
    error: string | null;
  };
};

export const useEngine = create<EngineState>(() => ({
  status: 'idle',
  loadingId: null,
  loadedId: null,
  progress: null,
  error: null,
  info: null,
  cached: {},
  legacy: {},
  caps: {
    isolated: window.crossOriginIsolated === true,
    sharedArrayBuffer: typeof SharedArrayBuffer !== 'undefined',
    opfs: !!navigator.storage?.getDirectory,
    webgpu: 'gpu' in navigator,
    cores: navigator.hardwareConcurrency || 0,
  },
  generating: false,
  tokPerSec: null,
  lastLoadMs: null,
  lastVerifyMs: null,
  laya: { supported: false, loaded: null, checkpoints: [] },
  native: { supported: false, chat: null, decider: null, checkpoints: [] },
  decider: { status: 'idle', loadingId: null, loadedId: null, progress: null, error: null },
}));

let wllama: Wllama | null = null;
let modelManager: ModelManager | null = null;

function manager(): ModelManager | null {
  if (modelManager) return modelManager;
  try {
    modelManager = new ModelManager();
  } catch (e) {
    console.warn('[engine] model cache unavailable', e);
  }
  return modelManager;
}

/**
 * What llama.cpp's load log says about the GPU: "offloaded 29/29 layers to
 * GPU" when WebGPU took the layers, or "Failed to get an adapter" when it
 * ran on the CPU. `navigator.gpu` existing says neither (AGENTS.md §2).
 */
export function gpuFromLog(line: string): { layers: number; total: number } | null {
  const m = /offloaded (\d+)\/(\d+) layers to GPU/.exec(line);
  if (m) return { layers: Number(m[1]), total: Number(m[2]) };
  return /ggml_webgpu: Failed to get an adapter/.test(line) ? { layers: 0, total: 0 } : null;
}

function newWllama(onGpu?: (gpu: { layers: number; total: number }) => void): Wllama {
  // Native logs still reach the console; the GPU lines are also kept for EngineInfo.
  const watch =
    (fn: (...a: unknown[]) => void) =>
    (...a: unknown[]) => {
      const gpu = onGpu && gpuFromLog(a.map(String).join(' '));
      if (gpu) onGpu(gpu);
      fn(...a);
    };
  const logger = { debug: watch(console.debug), log: watch(console.log), warn: watch(console.warn), error: watch(console.error) };
  const w = new Wllama({ default: asset('/wllama/default/wllama.wasm') }, { parallelDownloads: 3, logger });
  w.setCompat({ worker: asset('/wllama/compat/wllama.js'), wasm: asset('/wllama/compat/wllama.wasm') });
  return w;
}

export async function refreshCache(): Promise<void> {
  const mm = manager();
  const cached: Record<string, number> = {};
  try {
    // Without OPFS there are no GGUF copies, but MLX models (in Rust's store) still count.
    for (const m of (await mm?.getModels()) ?? []) if (m.size > 0) cached[m.url] = m.size;
  } catch (e) {
    console.warn('[engine] cache listing failed', e);
  }
  const legacy: Record<string, number> = {};
  for (const def of MODELS) {
    if (isMlx(def)) continue;
    const bytes = def.legacyUrls.reduce((n, url) => n + (cached[url] ?? 0), 0);
    if (bytes) legacy[def.id] = bytes;
  }
  // MLX models live in Rust's verified store, keyed here by their pinned repo URL.
  const native = await refreshNative();
  for (const def of MODELS) {
    if (isMlx(def) && native.checkpoints.some((c) => c.id === def.native && c.downloaded)) cached[def.url] = def.bytes;
  }
  useEngine.setState({ cached, legacy });
}

export async function refreshNative(): Promise<NativeStatus> {
  const native = await nativeStatus().catch(() => useEngine.getState().native);
  useEngine.setState({ native });
  return native;
}

/** Downloads an MLX model into Rust's store when it isn't there yet (Rust verifies every file). */
async function ensureNative(def: MlxDef, onProgress: (p: Progress) => void): Promise<void> {
  const c = (await refreshNative()).checkpoints.find((x) => x.id === def.native);
  if (!c) throw new Error(`${def.name} runs on MLX, which needs an Apple Silicon Mac.`);
  if (!c.downloaded) await downloadCheckpoint('llm', c, onProgress);
  onProgress({ loaded: def.bytes, total: def.bytes, speed: 0, phase: 'Loading…' });
}

function nativeInfo(loaded: NativeLoaded): EngineInfo {
  return {
    backend: 'MLX · Metal (native)',
    threads: '— (GPU)',
    context: `${loaded.nCtx}`,
    layers: loaded.layers,
    arch: `qwen3 · ${loaded.bits}-bit`,
    libllama: 'MLX 0.32 (mlx-rs, in Rust)',
    compat: false,
  };
}

// Models already verified, url -> sha256, so a cached copy isn't re-hashed on
// every launch. A fresh download is always verified regardless.
const VERIFIED_KEY = 'andai.verifiedModels';
const verified = (): Record<string, string> => {
  try {
    return JSON.parse(localStorage.getItem(VERIFIED_KEY) ?? '{}');
  } catch {
    return {};
  }
};
const setVerified = (url: string, sha256: string | null) => {
  const v = verified();
  if (sha256) v[url] = sha256;
  else delete v[url];
  localStorage.setItem(VERIFIED_KEY, JSON.stringify(v));
};

type Progress = NonNullable<EngineState['progress']>;

/**
 * Downloads (or reuses the OPFS copy of) a catalog model and verifies it
 * against its pinned sha256 before anything may load it (AGENTS.md §9).
 */
async function openVerified(def: WllamaDef, onProgress: (p: Progress) => void) {
  const started = performance.now();
  const mm = manager();
  if (!mm) throw new Error('The model cache is unavailable in this webview.');
  const wasCached = (await mm.getModels()).some((m) => m.url === def.url && m.size > 0);
  const model = await mm.getModelOrDownload(
    { url: def.url },
    {
      progressCallback: ({ loaded, total }: { loaded: number; total: number }) => {
        const secs = (performance.now() - started) / 1000;
        onProgress({ loaded, total: total || def.bytes, speed: secs > 0 ? loaded / secs : 0, phase: 'Downloading' });
      },
    },
  );
  let verifyMs: number | null = null;
  if (!wasCached || verified()[def.url] !== def.sha256) {
    const t = performance.now();
    const setDone = (loaded: number) => onProgress({ loaded, total: def.bytes, speed: 0, phase: 'Verifying…' });
    setDone(0);
    const verdict = await verifyBlobs(await model.open(), def, setDone);
    if (!verdict.ok) {
      await model.remove();
      setVerified(def.url, null);
      throw new Error(
        `The downloaded ${def.name} failed its integrity check (${verdict.reason}), so it was removed and not loaded. Try again; if it keeps failing, something is altering the download.`,
      );
    }
    setVerified(def.url, def.sha256);
    verifyMs = performance.now() - t;
  }
  onProgress({ loaded: def.bytes, total: def.bytes, speed: 0, phase: 'Warming up…' });
  return { model, verifyMs };
}

export async function loadModel(id: string): Promise<void> {
  const def = modelById(id);
  if (!def) throw new Error(`unknown model ${id}`);
  const { status, loadedId } = useEngine.getState();
  if (status === 'loading') return;
  if (loadedId === id) return;
  if (loadedId) await unloadModel();

  useEngine.setState({
    status: 'loading',
    loadingId: id,
    error: null,
    progress: { loaded: 0, total: def.bytes, speed: 0, phase: 'Connecting…' },
  });
  const started = performance.now();
  try {
    if (isMlx(def)) {
      await ensureNative(def, (progress) => useEngine.setState({ progress }));
      const loaded = await nativeLoad('chat', def.native, def.n_ctx);
      useEngine.setState({
        status: 'ready',
        loadingId: null,
        loadedId: id,
        progress: null,
        info: nativeInfo(loaded),
        lastLoadMs: performance.now() - started,
        lastVerifyMs: null,
      });
      localStorage.setItem('andai.lastModel', id);
      return;
    }
    const { model, verifyMs } = await openVerified(def, (progress) => useEngine.setState({ progress }));
    useEngine.setState({ lastVerifyMs: verifyMs });
    let gpu: { layers: number; total: number } | null = null;
    const w = newWllama((g) => (gpu = g));
    await w.loadModel(model, {
      n_ctx: def.n_ctx,
      // keep <think> blocks in the raw stream so the UI can fold them itself
      reasoning_format: 'none',
    });
    wllama = w;
    useEngine.setState({
      status: 'ready',
      loadingId: null,
      loadedId: id,
      progress: null,
      info: readInfo(w, gpu),
      lastLoadMs: performance.now() - started,
    });
    localStorage.setItem('andai.lastModel', id);
  } catch (e) {
    console.error('[engine] load failed', e);
    useEngine.setState({
      status: 'error',
      loadingId: null,
      progress: null,
      error: e instanceof Error ? e.message : String(e),
    });
  } finally {
    await refreshCache();
  }
}

export async function unloadModel(): Promise<void> {
  const w = wllama;
  const native = isMlx(loadedModel());
  wllama = null;
  useEngine.setState({ status: 'idle', loadedId: null, info: null, tokPerSec: null });
  await w?.exit().catch(() => {});
  if (native) await nativeUnload('chat').catch(() => {});
}

// ── decision model ──────────────────────────────────────────────────────
// A second, optional model that only scores tool choices (llm/decide.ts): a
// wllama instance, an MLX model in Rust's decider slot, or a Laya checkpoint.
// Without one, decisions run on the chat model.

let deciderWllama: Wllama | null = null;

export async function refreshLaya(): Promise<LayaStatus> {
  const laya = await layaStatus().catch(() => useEngine.getState().laya);
  useEngine.setState({ laya });
  return laya;
}

/** The loaded decision model when it's a Laya checkpoint (decide.ts routes to Rust then). */
export function deciderLaya(): LayaDef | null {
  const { decider } = useEngine.getState();
  return decider.status === 'ready' ? (layaById(decider.loadedId) ?? null) : null;
}

export async function loadDecider(id: string): Promise<void> {
  const laya = layaById(id);
  const def = modelById(id);
  if (!laya && !def?.decider) throw new Error(`${def?.name ?? id} can't be used as a decision model`);
  const { decider } = useEngine.getState();
  if (decider.status === 'loading' || decider.loadedId === id) return;
  if (decider.loadedId) await unloadDecider();
  const setDecider = (patch: Partial<EngineState['decider']>) =>
    useEngine.setState((s) => ({ decider: { ...s.decider, ...patch } }));
  setDecider({ status: 'loading', loadingId: id, error: null, progress: { loaded: 0, total: def?.bytes ?? 0, speed: 0, phase: 'Connecting…' } });
  try {
    if (laya) {
      const c = (await refreshLaya()).checkpoints.find((x) => x.id === id);
      if (!c) throw new Error(`${laya.name} needs an Apple Silicon Mac.`);
      if (!c.downloaded) await downloadLaya(c, (progress) => setDecider({ progress }));
      setDecider({ progress: { loaded: c.bytes, total: c.bytes, speed: 0, phase: 'Loading…' } });
      await layaLoad(id);
    } else if (isMlx(def)) {
      await ensureNative(def, (progress) => setDecider({ progress }));
      await nativeLoad('decider', def.native, Math.min(def.n_ctx, 4096));
    } else {
      const { model } = await openVerified(def!, (progress) => setDecider({ progress }));
      const w = newWllama();
      // Decisions are short prompts scored in one forward pass.
      await w.loadModel(model, { n_ctx: Math.min(def!.n_ctx, 4096), reasoning_format: 'none' });
      deciderWllama = w;
    }
    setDecider({ status: 'ready', loadingId: null, loadedId: id, progress: null });
    localStorage.setItem('andai.lastDecider', id);
  } catch (e) {
    console.error('[engine] decider load failed', e);
    setDecider({ status: 'error', loadingId: null, progress: null, error: e instanceof Error ? e.message : String(e) });
  } finally {
    await (laya ? refreshLaya() : refreshCache());
  }
}

export async function unloadDecider(): Promise<void> {
  const w = deciderWllama;
  const { loadedId } = useEngine.getState().decider;
  const laya = layaById(loadedId);
  const native = isMlx(modelById(loadedId));
  deciderWllama = null;
  useEngine.setState((s) => ({ decider: { ...s.decider, status: 'idle', loadedId: null } }));
  localStorage.removeItem('andai.lastDecider');
  await w?.exit().catch(() => {});
  if (laya) await layaUnload().catch(() => {});
  if (native) await nativeUnload('decider').catch(() => {});
}

/** Deletes a downloaded Laya checkpoint (Settings confirms first), unloading it if in use. */
export async function removeLaya(id: string): Promise<void> {
  if (useEngine.getState().decider.loadedId === id) await unloadDecider();
  await layaRemove(id);
  await refreshLaya();
}

export type Slot = 'chat' | 'decider';
/**
 * wllama hands these to llama.cpp's server-side parser as JSON, so the
 * llama.cpp-only `grammar` (GBNF) works here too. Measured: this build can't
 * turn a JSON Schema into a grammar (`response_format: json_schema` fails with
 * "Failed to initialize samplers"), so callers pass GBNF (tools/validate.ts).
 */
export type CompletionParams = ChatCompletionParams & { grammar?: string; top_k?: number; top_p?: number };
export type Completion = ChatCompletionResponse;

/**
 * The instance a request runs on: `decider` falls back to the chat model when
 * no decision model is loaded, and the result says which one answered.
 */
export function slotFor(prefer: Slot): { slot: Slot; def: ModelDef } | null {
  const { decider, loadedId } = useEngine.getState();
  const d = modelById(decider.loadedId);
  if (prefer === 'decider' && d && (deciderWllama || isMlx(d))) return { slot: 'decider', def: d };
  const c = modelById(loadedId);
  return c && (wllama || isMlx(c)) ? { slot: 'chat', def: c } : null;
}

/**
 * wllama's request as native parameters, with llama.cpp's defaults for what
 * it leaves out (temperature 0.8, top-k 40, top-p 0.95, thinking on).
 */
function nativeParams(p: CompletionParams) {
  const kwargs = (p as { chat_template_kwargs?: { enable_thinking?: boolean } }).chat_template_kwargs;
  return {
    maxTokens: p.max_tokens ?? 512,
    temperature: p.temperature ?? 0.8,
    topK: p.top_k ?? 40,
    topP: p.top_p ?? 0.95,
    seed: typeof p.seed === 'number' ? p.seed : undefined,
    thinking: kwargs?.enable_thinking ?? true,
    topLogprobs: p.logprobs ? (p.top_logprobs ?? 0) : 0,
    grammar: p.grammar,
    cachePrompt: (p as { cache_prompt?: boolean }).cache_prompt ?? true,
  };
}

/** One non-streaming completion (decisions, argument filling). */
export async function complete(prefer: Slot, params: CompletionParams): Promise<{ response: Completion; slot: Slot; def: ModelDef }> {
  const target = slotFor(prefer);
  if (target && isMlx(target.def)) {
    const out = await nativeGenerate(target.slot as NativeSlot, params.messages as ChatMessage[], nativeParams(params), () => {}, params.abortSignal);
    return { response: asCompletion(out) as unknown as Completion, ...target };
  }
  const w = target?.slot === 'decider' ? deciderWllama : wllama;
  if (!target || !w) throw new Error('No model loaded — open Settings → Models to load one.');
  const response = await w.createChatCompletion({ ...(params as ChatCompletionParams), stream: false });
  return { response, ...target };
}

export async function evictModel(id: string): Promise<void> {
  const def = modelById(id);
  if (!def) return;
  if (useEngine.getState().loadedId === id) await unloadModel();
  if (useEngine.getState().decider.loadedId === id) await unloadDecider();
  if (isMlx(def)) {
    await nativeRemove(def.native);
    await refreshCache();
    return;
  }
  const mm = manager();
  if (!mm) return;
  for (const m of await mm.getModels()) if (m.url === def.url) await m.remove();
  setVerified(def.url, null);
  await refreshCache();
}

/**
 * Removes copies of a model cached under its pre-pinning URLs (AGENTS.md §9).
 * Only called after the user confirms in Settings (user data, §1.5).
 */
export async function removeLegacyCopies(id: string): Promise<void> {
  const def = modelById(id);
  const mm = manager();
  if (!def || isMlx(def) || !mm) return;
  for (const m of await mm.getModels()) if (def.legacyUrls.includes(m.url)) await m.remove();
  await refreshCache();
}

function readInfo(w: Wllama, gpu: { layers: number; total: number } | null): EngineInfo {
  const i = w.getLoadedContextInfo();
  const meta = w.getModelMetadata().meta;
  return {
    backend: gpu
      ? gpu.layers > 0
        ? `WebGPU · ${gpu.layers}/${gpu.total} layers`
        : 'WASM · CPU (no GPU adapter)'
      : w.isSupportWebGPU()
        ? 'WebGPU (unconfirmed)'
        : 'WASM · CPU',
    threads: w.isMultithread() ? `${w.getNumThreads()} (multi)` : '1 (single)',
    context: `${i.n_ctx} / ${i.n_ctx_train}`,
    layers: i.n_layer,
    arch: meta['general.architecture'] ?? '—',
    libllama: Wllama.getLibllamaVersion?.() ?? '—',
    compat: w.getWorkerResources().compat,
  };
}

export const loadedModel = (): ModelDef | undefined => modelById(useEngine.getState().loadedId);
export const isAbort = (e: unknown) =>
  e instanceof WllamaAbortError || (e instanceof Error && e.name === 'AbortError');

/**
 * `tokPerSec` is generation speed, timed from the first token: reading the
 * prompt comes first and is reported apart, as `promptTokPerSec`. Counted
 * together, a 3 s prompt made a 60 tok/s answer look like 15 tok/s
 * (Qwen3 1.7B on WebGPU, AGENTS.md §2).
 */
export type StreamEvent =
  | { type: 'delta'; text: string; tokens: number; tokPerSec: number }
  | { type: 'done'; promptTokens: number | null; completionTokens: number; tokPerSec: number | null; promptTokPerSec: number | null };

/** Streams a chat completion. Throws `WllamaAbortError` when `signal` aborts. */
export async function* chat(
  messages: ChatMessage[],
  opts: { temperature: number; maxTokens: number; thinking: boolean; signal: AbortSignal },
): AsyncGenerator<StreamEvent> {
  const w = wllama;
  const def = loadedModel();
  if (isMlx(def)) {
    yield* chatNative(def, messages, opts);
    return;
  }
  if (!w || !def) throw new Error('No model loaded — open Settings → Models to load one.');

  useEngine.setState({ generating: true });
  const started = performance.now();
  let firstAt: number | null = null;
  let tokens = 0;
  let promptTokens: number | null = null;
  let timings: { prompt_per_second?: number; predicted_per_second?: number } | undefined;
  try {
    const stream = await w.createChatCompletion({
      messages,
      stream: true,
      abortSignal: opts.signal,
      max_tokens: opts.maxTokens,
      temperature: opts.temperature,
      top_p: 0.95,
      top_k: 40,
      ...(def.thinking ? { chat_template_kwargs: { enable_thinking: opts.thinking } } : {}),
    } as Parameters<Wllama['createChatCompletion']>[0] & { stream: true });

    for await (const chunk of stream as AsyncIterable<{
      choices?: { delta?: { content?: string | null } }[];
      usage?: { prompt_tokens: number } | null;
      timings?: { prompt_n: number; prompt_per_second?: number; predicted_per_second?: number };
    }>) {
      if (chunk.usage?.prompt_tokens) promptTokens = chunk.usage.prompt_tokens;
      else if (chunk.timings?.prompt_n) promptTokens = chunk.timings.prompt_n;
      if (chunk.timings) timings = chunk.timings;
      const delta = chunk.choices?.[0]?.delta?.content;
      if (!delta) continue;
      tokens++;
      const now = performance.now();
      firstAt ??= now;
      // The first token only ends the prompt; the rate starts from it.
      const secs = (now - firstAt) / 1000;
      const tokPerSec = tokens > 1 && secs > 0 ? (tokens - 1) / secs : 0;
      if (tokens > 1) useEngine.setState({ tokPerSec });
      yield { type: 'delta', text: delta, tokens, tokPerSec };
    }
    // llama.cpp's own timings when the stream carries them, else the wall clock.
    const rate = (v: number | undefined) => (typeof v === 'number' && Number.isFinite(v) && v > 0 ? v : null);
    const genSecs = firstAt != null ? (performance.now() - firstAt) / 1000 : 0;
    const tokPerSec = rate(timings?.predicted_per_second) ?? (tokens > 1 && genSecs > 0 ? (tokens - 1) / genSecs : null);
    const promptSecs = firstAt != null ? (firstAt - started) / 1000 : 0;
    const promptTokPerSec = rate(timings?.prompt_per_second) ?? (promptTokens && promptSecs > 0 ? promptTokens / promptSecs : null);
    if (tokPerSec != null) useEngine.setState({ tokPerSec });
    yield { type: 'done', promptTokens, completionTokens: tokens, tokPerSec, promptTokPerSec };
  } finally {
    useEngine.setState({ generating: false });
  }
}

/**
 * `chat` on a native model: Rust sends text through a channel as it's
 * written, and the result carries the whole text and Rust's own timings.
 * Channel messages can land after the command resolves, so text that hasn't
 * streamed by then is taken from the result, and later pieces are dropped.
 */
async function* chatNative(
  def: MlxDef,
  messages: ChatMessage[],
  opts: { temperature: number; maxTokens: number; thinking: boolean; signal: AbortSignal },
): AsyncGenerator<StreamEvent> {
  useEngine.setState({ generating: true });
  const queue: string[] = [];
  let wake: (() => void) | null = null;
  let closed = false;
  let settled = false;
  let result: Generated | null = null;
  let failure: unknown = null;
  const onText = (piece: string) => {
    if (closed) return;
    queue.push(piece);
    wake?.();
  };
  const params = { maxTokens: opts.maxTokens, temperature: opts.temperature, topP: 0.95, topK: 40, thinking: def.thinking && opts.thinking };
  const run = nativeGenerate('chat', messages, params, onText, opts.signal).then(
    (r) => void (result = r),
    (e: unknown) => void (failure = e),
  );
  void run.finally(() => {
    settled = true;
    wake?.();
  });
  let tokens = 0;
  let firstAt: number | null = null;
  let streamed = '';
  const delta = (text: string): StreamEvent => {
    tokens++;
    streamed += text;
    const now = performance.now();
    firstAt ??= now;
    const secs = (now - firstAt) / 1000;
    const tokPerSec = tokens > 1 && secs > 0 ? (tokens - 1) / secs : 0;
    if (tokens > 1) useEngine.setState({ tokPerSec });
    return { type: 'delta', text, tokens, tokPerSec };
  };
  try {
    for (;;) {
      if (queue.length) {
        yield delta(queue.shift()!);
        continue;
      }
      if (settled) break;
      await new Promise<void>((r) => (wake = r));
      wake = null;
    }
    await run;
    closed = true;
    if (failure) throw failure;
    const g = result as Generated | null;
    if (!g) throw new Error('The model returned no result.');
    if (g.text.length > streamed.length && g.text.startsWith(streamed)) yield delta(g.text.slice(streamed.length));
    const tokPerSec = genTokPerSec(g);
    if (tokPerSec != null) useEngine.setState({ tokPerSec });
    yield { type: 'done', promptTokens: g.promptTokens, completionTokens: g.completionTokens, tokPerSec, promptTokPerSec: promptTokPerSec(g) };
  } finally {
    closed = true;
    useEngine.setState({ generating: false });
  }
}

/** Load whatever model the user had last, if it's still cached. */
export async function autoload(): Promise<void> {
  await refreshCache();
  const last = localStorage.getItem('andai.lastModel');
  const def = modelById(last);
  if (def && useEngine.getState().cached[def.url]) await loadModel(def.id);
  const lastDecider = localStorage.getItem('andai.lastDecider');
  const decider = modelById(lastDecider);
  if (decider && useEngine.getState().cached[decider.url]) await loadDecider(decider.id);
  // A Laya checkpoint loads in well under a second, but only once downloaded.
  const laya = (await refreshLaya()).checkpoints.find((c) => c.id === lastDecider && c.downloaded);
  if (laya) await loadDecider(laya.id);
}

export { MODELS };
