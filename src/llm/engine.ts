// The local LLM: wllama (llama.cpp → WASM) running inside the app's webview.
//
// On macOS, Tauri renders with WKWebView (Safari's engine), which lacks the
// Memory64 and JSPI features wllama's default build needs. wllama detects that
// itself (`needCompat()`) and switches to the compat build we point it at via
// `setCompat` — served from public/wllama/, never the CDN. In Chromium (plain
// `vite` in a browser) the default build runs and `setCompat` is ignored.
//
// Models are cached in OPFS by wllama's ModelManager, so each downloads once.
// Every download is checked against the catalog's pinned size and sha256
// before it is loaded (integrity.ts, AGENTS.md §9).

import { ModelManager, Wllama, WllamaAbortError } from '@wllama/wllama';
import { create } from 'zustand';
import { verifyBlobs } from './integrity';
import { MODELS, modelById, type ModelDef } from './models';

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

function newWllama(): Wllama {
  const w = new Wllama({ default: asset('/wllama/default/wllama.wasm') }, { parallelDownloads: 3 });
  w.setCompat({ worker: asset('/wllama/compat/wllama.js'), wasm: asset('/wllama/compat/wllama.wasm') });
  return w;
}

export async function refreshCache(): Promise<void> {
  const mm = manager();
  if (!mm) return;
  const cached: Record<string, number> = {};
  try {
    for (const m of await mm.getModels()) if (m.size > 0) cached[m.url] = m.size;
  } catch (e) {
    console.warn('[engine] cache listing failed', e);
  }
  const legacy: Record<string, number> = {};
  for (const def of MODELS) {
    const bytes = def.legacyUrls.reduce((n, url) => n + (cached[url] ?? 0), 0);
    if (bytes) legacy[def.id] = bytes;
  }
  useEngine.setState({ cached, legacy });
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

export async function loadModel(id: string): Promise<void> {
  const def = modelById(id);
  if (!def) throw new Error(`unknown model ${id}`);
  const { status, loadedId } = useEngine.getState();
  if (status === 'loading') return;
  if (loadedId === id) return;
  if (wllama) await unloadModel();

  useEngine.setState({
    status: 'loading',
    loadingId: id,
    error: null,
    progress: { loaded: 0, total: def.bytes, speed: 0, phase: 'Connecting…' },
  });
  const started = performance.now();
  try {
    const mm = manager();
    if (!mm) throw new Error('The model cache is unavailable in this webview.');
    const wasCached = (await mm.getModels()).some((m) => m.url === def.url && m.size > 0);
    const model = await mm.getModelOrDownload(
      { url: def.url },
      {
        progressCallback: ({ loaded, total }: { loaded: number; total: number }) => {
          const secs = (performance.now() - started) / 1000;
          useEngine.setState({
            progress: { loaded, total: total || def.bytes, speed: secs > 0 ? loaded / secs : 0, phase: 'Downloading' },
          });
        },
      },
    );
    useEngine.setState({ lastVerifyMs: null });
    if (!wasCached || verified()[def.url] !== def.sha256) {
      const t = performance.now();
      const setDone = (loaded: number) =>
        useEngine.setState({ progress: { loaded, total: def.bytes, speed: 0, phase: 'Verifying…' } });
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
      useEngine.setState({ lastVerifyMs: performance.now() - t });
    }
    useEngine.setState({ progress: { loaded: def.bytes, total: def.bytes, speed: 0, phase: 'Warming up…' } });
    const w = newWllama();
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
      info: readInfo(w),
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
  wllama = null;
  useEngine.setState({ status: 'idle', loadedId: null, info: null, tokPerSec: null });
  await w?.exit().catch(() => {});
}

export async function evictModel(id: string): Promise<void> {
  const def = modelById(id);
  const mm = manager();
  if (!def || !mm) return;
  if (useEngine.getState().loadedId === id) await unloadModel();
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
  if (!def || !mm) return;
  for (const m of await mm.getModels()) if (def.legacyUrls.includes(m.url)) await m.remove();
  await refreshCache();
}

function readInfo(w: Wllama): EngineInfo {
  const i = w.getLoadedContextInfo();
  const meta = w.getModelMetadata().meta;
  return {
    backend: w.isSupportWebGPU() ? 'WebGPU' : 'WASM · CPU',
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

export type StreamEvent =
  | { type: 'delta'; text: string; tokens: number; tokPerSec: number }
  | { type: 'done'; promptTokens: number | null; completionTokens: number };

/** Streams a chat completion. Throws `WllamaAbortError` when `signal` aborts. */
export async function* chat(
  messages: ChatMessage[],
  opts: { temperature: number; maxTokens: number; thinking: boolean; signal: AbortSignal },
): AsyncGenerator<StreamEvent> {
  const w = wllama;
  const def = loadedModel();
  if (!w || !def) throw new Error('No model loaded — open Settings → Models to load one.');

  useEngine.setState({ generating: true });
  const started = performance.now();
  let tokens = 0;
  let promptTokens: number | null = null;
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
      timings?: { prompt_n: number };
    }>) {
      if (chunk.usage?.prompt_tokens) promptTokens = chunk.usage.prompt_tokens;
      else if (chunk.timings?.prompt_n) promptTokens = chunk.timings.prompt_n;
      const delta = chunk.choices?.[0]?.delta?.content;
      if (!delta) continue;
      tokens++;
      const secs = (performance.now() - started) / 1000;
      const tokPerSec = secs > 0 ? tokens / secs : 0;
      useEngine.setState({ tokPerSec });
      yield { type: 'delta', text: delta, tokens, tokPerSec };
    }
    yield { type: 'done', promptTokens, completionTokens: tokens };
  } finally {
    useEngine.setState({ generating: false });
  }
}

/** Load whatever model the user had last, if it's still cached. */
export async function autoload(): Promise<void> {
  await refreshCache();
  const last = localStorage.getItem('andai.lastModel');
  const def = modelById(last);
  if (def && useEngine.getState().cached[def.url]) await loadModel(def.id);
}

export { MODELS };
