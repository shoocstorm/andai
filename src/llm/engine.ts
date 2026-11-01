// The local LLM, on one of two engines (llm/models.ts `engine`):
// - wllama (llama.cpp → WASM) running inside the app's webview, everywhere;
// - MLX, natively in Rust on Apple Silicon (llm/native.ts), 5–10× faster.
// Callers don't care which: `chat` streams and `complete` answers the same
// way on both. The rest of this note is about wllama.
//
// On macOS, Tauri renders with WKWebView (Safari's engine), which lacks the
// Memory64 the default build needs (JSPI it has since macOS/iOS 27). wllama's
// own check is fooled there, so `AndaiWllama` (wasm.ts) decides by compiling a
// Memory64 module and switches to the compat build we point it at via
// `setCompat` — served from public/wllama/, never the CDN. In Chromium (plain
// `vite` in a browser, and WebView2 on Windows) the default build runs and
// `setCompat` is ignored.
//
// Models are cached in OPFS by wllama's ModelManager, so each downloads once.
// Every download is checked against the catalog's pinned size and sha256
// before it is loaded (integrity.ts, AGENTS.md §9).

import { ModelManager, Wllama, WllamaAbortError, type ChatCompletionParams, type ChatCompletionResponse } from '@wllama/wllama';
import { create } from 'zustand';
import { logApp } from '../state/activity';
import { verifyBlobs } from './integrity';
import { AndaiWllama } from './wasm';
import { crashMessage, describeLoadError, detectDevice, isFatalLog, loadPlan, MAX_LEVEL, planText, startLevel, type LoadPlan } from './device';
import { downloadCheckpoint, downloadLaya, layaLoad, layaRemove, layaStatus, layaUnload, type LayaStatus } from './laya';
import { allModels, isMlx, layaById, MODELS, modelById, type LayaDef, type MlxDef, type ModelDef, type WllamaDef } from './models';
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

/** The activity log's record of a model event (state/activity.ts; written only while the log is on). */
const logModel = (action: 'load' | 'unload' | 'load-failed' | 'remove', slot: Slot | null, id: string, extra: Record<string, unknown> = {}) => {
  const def = modelById(id) ?? layaById(id);
  logApp('model', { action, slot, id, model: def?.name ?? id, engine: layaById(id) ? 'Laya' : isMlx(modelById(id)) ? 'MLX' : 'wllama', ...extra });
};

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
  /** The model the error is about, so its card can show it. */
  errorId: string | null;
  /** The context the chat model was loaded with (phones load less than the catalog's n_ctx). */
  nCtx: number | null;
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
  errorId: null,
  nCtx: null,
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
  native: { supported: false, chat: null, decider: null, checkpoints: [], memory: null },
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

const device = detectDevice(typeof navigator === 'undefined' ? undefined : navigator);

// A load in progress, so that a page iOS killed mid-load (no error to catch)
// is recognized on the next launch (autoload → checkCrashedLoad). The level
// a model last failed at is kept per model, so the next try asks for less.
const ATTEMPT_KEY = 'andai.loadAttempt';
const LEVEL_KEY = 'andai.loadLevel';
type Attempt = { id: string; level: number; at: number };
const readJson = <T,>(key: string, fallback: T): T => {
  try {
    return (JSON.parse(localStorage.getItem(key) ?? 'null') as T) ?? fallback;
  } catch {
    return fallback;
  }
};
const failedLevel = (id: string): number | undefined => readJson<Record<string, number>>(LEVEL_KEY, {})[id];
const setFailedLevel = (id: string, level: number) => localStorage.setItem(LEVEL_KEY, JSON.stringify({ ...readJson<Record<string, number>>(LEVEL_KEY, {}), [id]: level }));

/** How a wllama model will load here next: the device's start, or lighter after a failure. */
export const plannedLoad = (def: { id: string; n_ctx: number }): LoadPlan => loadPlan(def.n_ctx, startLevel(device, failedLevel(def.id)));

/**
 * A load that never finished because the page was closed under it: says so
 * and makes the next try lighter. Returns the model's id so autoload doesn't
 * start it again (that would close the page on every launch).
 */
export function checkCrashedLoad(): string | null {
  const a = readJson<Attempt | null>(ATTEMPT_KEY, null);
  localStorage.removeItem(ATTEMPT_KEY);
  // Only a recent attempt counts: an old marker is from a session the user ended.
  if (!a || Date.now() - a.at > 15 * 60_000) return null;
  const def = modelById(a.id);
  const next = a.level < MAX_LEVEL ? loadPlan(def?.n_ctx ?? 4096, a.level + 1) : null;
  setFailedLevel(a.id, Math.min(MAX_LEVEL, a.level + 1));
  useEngine.setState({ status: 'error', error: crashMessage(def?.name ?? a.id, loadPlan(def?.n_ctx ?? 4096, a.level), next, device), errorId: a.id });
  logModel('load-failed', 'chat', a.id, { error: 'page closed while loading', level: a.level });
  return a.id;
}

/**
 * Fails a warm-up that hangs (a worker that died without a word), instead of
 * spinning forever: at the time limit, or as soon as the native log says a
 * worker died (`isFatalLog`), since wllama's promise never settles then.
 */
const WARMUP_LIMIT_MS = 180_000;
function withinWarmup<T>(p: Promise<T>, watch: NativeWatch, onStop: () => void): Promise<T> {
  let timer: ReturnType<typeof setTimeout>;
  return Promise.race([
    p.finally(() => clearTimeout(timer)),
    new Promise<T>((_, reject) => {
      const stop = (e: Error) => {
        clearTimeout(timer);
        onStop();
        reject(e);
      };
      timer = setTimeout(() => stop(new Error(`timed out warming up${watch.last ? ` (last log: ${watch.last})` : ''}`)), WARMUP_LIMIT_MS);
      // a worker can die before this hook is set; the line is kept for that
      if (watch.fatal) stop(new Error(watch.fatal));
      else watch.onFatal = (line) => stop(new Error(line));
    }),
  ]);
}

/** Loads a verified model into a new wllama at `plan`; the attempt is marked until it settles. */
async function warmUp(id: string, model: Awaited<ReturnType<typeof openVerified>>['model'], plan: LoadPlan, onGpu: (g: { layers: number; total: number }) => void): Promise<Wllama> {
  const watch: NativeWatch = { last: '' };
  const w = newWllama(onGpu, watch);
  localStorage.setItem(ATTEMPT_KEY, JSON.stringify({ id, level: plan.level, at: Date.now() } satisfies Attempt));
  try {
    await withinWarmup(w.loadModel(model, wllamaParams(plan)), watch, () => void w.exit().catch(() => {}));
    return w;
  } catch (e) {
    await w.exit().catch(() => {});
    throw e;
  } finally {
    watch.onFatal = undefined;
    localStorage.removeItem(ATTEMPT_KEY);
  }
}

/** wllama's load options for a plan. */
function wllamaParams(plan: LoadPlan) {
  return {
    n_ctx: plan.n_ctx,
    // CPU only: wllama then doesn't initialize WebGPU at all
    ...(plan.gpu ? {} : { n_gpu_layers: 0 }),
    // keep <think> blocks in the raw stream so the UI can fold them itself
    reasoning_format: 'none' as const,
  };
}

/** What a warm-up reads from the native log: the last line, and a hook for a fatal one. */
type NativeWatch = { last: string; fatal?: string; onFatal?: (line: string) => void };

/** A worker's `error` event, which wllama hands its logger as is (and nothing else). */
const isEvent = (v: unknown): v is Event => typeof Event !== 'undefined' && v instanceof Event;
const logText = (v: unknown): string => (isEvent(v) ? `worker error: ${(v as ErrorEvent).message || v.type}` : String(v));

function newWllama(onGpu?: (gpu: { layers: number; total: number }) => void, native?: NativeWatch): Wllama {
  // Native logs still reach the console; the GPU lines are also kept for EngineInfo,
  // and a warm-up watches them for the last step and for a worker that died.
  const watch =
    (fn: (...a: unknown[]) => void) =>
    (...a: unknown[]) => {
      const line = a.map(logText).join(' ');
      const gpu = onGpu && gpuFromLog(line);
      if (gpu) onGpu(gpu);
      if (native) {
        native.last = line.replace(/\s+/g, ' ').trim().slice(0, 160);
        // any channel: wllama logs an abort (e.g. a build that can't compile) with console.log
        if (!native.fatal && (a.some(isEvent) || isFatalLog(line))) {
          native.fatal = native.last;
          native.onFatal?.(native.fatal);
        }
      }
      fn(...a);
    };
  const logger = { debug: watch(console.debug), log: watch(console.log), warn: watch(console.warn), error: watch(console.error) };
  const w = new AndaiWllama({ default: asset('/wllama/default/wllama.wasm') }, { parallelDownloads: 3, logger });
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
  for (const def of allModels()) {
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
    arch: `${loaded.arch} · ${loaded.bits}-bit`,
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
    errorId: null,
    progress: { loaded: 0, total: def.bytes, speed: 0, phase: 'Connecting…' },
  });
  const started = performance.now();
  let plan: LoadPlan | null = null;
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
        nCtx: loaded.nCtx,
        lastLoadMs: performance.now() - started,
        lastVerifyMs: null,
      });
      logModel('load', 'chat', id, { ms: Math.round(performance.now() - started), backend: nativeInfo(loaded).backend, context: loaded.nCtx });
      localStorage.setItem('andai.lastModel', id);
      return;
    }
    const { model, verifyMs } = await openVerified(def, (progress) => useEngine.setState({ progress }));
    useEngine.setState({ lastVerifyMs: verifyMs });
    let gpu: { layers: number; total: number } | null = null;
    // Phones start lighter; a failure that more memory or the GPU caused steps down a level.
    let w: Wllama | null = null;
    for (let level = startLevel(device, failedLevel(id)); !w; level++) {
      plan = loadPlan(def.n_ctx, level);
      useEngine.setState({ progress: { loaded: def.bytes, total: def.bytes, speed: 0, phase: level > startLevel(device, failedLevel(id)) ? `Retrying with less memory (${planText(plan)})…` : 'Warming up…' } });
      try {
        w = await warmUp(id, model, plan, (g) => (gpu = g));
      } catch (e) {
        const failure = describeLoadError(e instanceof Error ? e.message : String(e), def.name, plan, device);
        if (!failure.retryLighter || level >= MAX_LEVEL) throw e;
        console.warn(`[engine] ${def.name} failed at ${planText(plan)}; trying less`, e);
        setFailedLevel(id, level + 1);
      }
    }
    wllama = w;
    useEngine.setState({
      status: 'ready',
      loadingId: null,
      loadedId: id,
      progress: null,
      info: readInfo(w!, gpu),
      nCtx: plan?.n_ctx ?? def.n_ctx,
      lastLoadMs: performance.now() - started,
    });
    const info = useEngine.getState().info;
    logModel('load', 'chat', id, { ms: Math.round(performance.now() - started), verifyMs: verifyMs === null ? null : Math.round(verifyMs), backend: info?.backend, threads: info?.threads, context: def.n_ctx });
    localStorage.setItem('andai.lastModel', id);
  } catch (e) {
    console.error('[engine] load failed', e);
    const raw = e instanceof Error ? e.message : String(e);
    logModel('load-failed', 'chat', id, { ms: Math.round(performance.now() - started), error: raw, plan: plan ? planText(plan) : null });
    useEngine.setState({
      status: 'error',
      loadingId: null,
      progress: null,
      error: describeLoadError(raw, def.name, plan, device).message,
      errorId: id,
    });
  } finally {
    await refreshCache();
  }
}

export async function unloadModel(): Promise<void> {
  const w = wllama;
  const native = isMlx(loadedModel());
  const was = useEngine.getState().loadedId;
  if (was) logModel('unload', 'chat', was);
  wllama = null;
  useEngine.setState({ status: 'idle', loadedId: null, info: null, nCtx: null, tokPerSec: null });
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
  const started = performance.now();
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
      // On a phone the decision model shares the page's memory with the chat model: the same lighter plan.
      const plan = loadPlan(Math.min(def!.n_ctx, 4096), startLevel(device, undefined));
      await w.loadModel(model, wllamaParams(plan));
      deciderWllama = w;
    }
    setDecider({ status: 'ready', loadingId: null, loadedId: id, progress: null });
    localStorage.setItem('andai.lastDecider', id);
    logModel('load', 'decider', id, { ms: Math.round(performance.now() - started) });
  } catch (e) {
    console.error('[engine] decider load failed', e);
    logModel('load-failed', 'decider', id, { ms: Math.round(performance.now() - started), error: e instanceof Error ? e.message : String(e) });
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
  if (loadedId) logModel('unload', 'decider', loadedId);
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
  logModel('remove', null, id);
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
    logModel('remove', null, id);
    await refreshCache();
    return;
  }
  const mm = manager();
  if (!mm) return;
  for (const m of await mm.getModels()) if (m.url === def.url) await m.remove();
  setVerified(def.url, null);
  logModel('remove', null, id);
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
  // A load the page didn't survive is reported, and not started again by itself.
  const crashed = checkCrashedLoad();
  await refreshCache();
  const last = localStorage.getItem('andai.lastModel');
  const def = modelById(last);
  if (def && def.id !== crashed && useEngine.getState().cached[def.url]) await loadModel(def.id);
  const lastDecider = localStorage.getItem('andai.lastDecider');
  const decider = modelById(lastDecider);
  if (decider && useEngine.getState().cached[decider.url]) await loadDecider(decider.id);
  // A Laya checkpoint loads in well under a second, but only once downloaded.
  const laya = (await refreshLaya()).checkpoints.find((c) => c.id === lastDecider && c.downloaded);
  if (laya) await loadDecider(laya.id);
}

export { MODELS };
