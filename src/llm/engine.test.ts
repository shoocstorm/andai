// loadModel's integrity gate (AGENTS.md §9), with wllama mocked: a download
// is loaded only if it matches the catalog's pinned size and sha256.
import { createHash } from 'node:crypto';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const bytes = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8]);
const state = vi.hoisted(() => ({
  cached: [] as { url: string; size: number; remove: () => Promise<void> }[],
  blob: null as Blob | null,
  removed: [] as string[],
  loaded: 0,
  opened: 0,
  gpuLine: null as string | null,
  /** Errors the next loads throw, in order; `'hang'` never settles, the other two log a dead worker first. */
  loadFails: [] as (Error | 'hang' | 'dead-thread' | 'worker-event' | 'compile-abort')[],
  loadParams: [] as Record<string, unknown>[],
  exited: 0,
  stream: { promptMs: 0, tokens: 0, gapMs: 0, timings: null as Record<string, number> | null },
}));

vi.mock('./models', () => {
  const def = {
    id: 'tiny',
    engine: 'wllama',
    name: 'Tiny',
    bytes: 8,
    url: 'https://huggingface.co/o/r/resolve/0000000000000000000000000000000000000000/t.gguf',
    sha256: createHash('sha256').update(new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8])).digest('hex'),
    legacyUrls: ['https://huggingface.co/o/r/resolve/main/t.gguf'],
    n_ctx: 1024,
    thinking: false,
  };
  return {
    MODELS: [def],
    modelById: (id: string) => (id === 'tiny' ? def : undefined),
    allModels: () => [def],
    layaById: () => undefined,
    isMlx: (d?: { engine?: string }) => d?.engine === 'mlx',
  };
});

vi.mock('@wllama/wllama', () => {
  const model = (url: string) => ({
    url,
    size: 8,
    open: async () => {
      state.opened++;
      return [state.blob!];
    },
    remove: async () => {
      state.removed.push(url);
      state.cached = state.cached.filter((m) => m.url !== url);
    },
  });
  class ModelManager {
    async getModels() {
      return state.cached;
    }
    async getModelOrDownload({ url }: { url: string }) {
      const m = model(url);
      if (!state.cached.some((c) => c.url === url)) state.cached.push(m);
      return m;
    }
  }
  class Wllama {
    logger?: { log: (...a: unknown[]) => void; error: (...a: unknown[]) => void };
    getWorkerResources() {
      return { compat: false, wasmPath: '' };
    }
    constructor(_paths: unknown, config: { logger?: { log: (...a: unknown[]) => void; error: (...a: unknown[]) => void } }) {
      this.logger = config.logger;
    }
    setCompat() {}
    async loadModel(_model: unknown, params: Record<string, unknown>) {
      state.loadParams.push(params);
      const fail = state.loadFails.shift();
      if (fail === 'hang') return new Promise(() => {});
      // what wllama does when a pthread worker dies: it logs, and its promise never settles
      if (fail === 'dead-thread') {
        this.logger?.log('llama_model_load: loading tensors');
        this.logger?.error('worker sent an error! blob:x:1: RuntimeError: unreachable');
        return new Promise(() => {});
      }
      // 0.1.7 in WebKit 27: the default build can't compile, and wllama logs the abort with console.log
      if (fail === 'compile-abort') {
        this.logger?.log("Aborted(CompileError: WebAssembly.Module doesn't parse at byte 4617: Memory64 is not enabled)");
        return new Promise(() => {});
      }
      if (fail === 'worker-event') {
        this.logger?.error(new Event('error'));
        return new Promise(() => {});
      }
      if (fail) throw fail;
      state.loaded++;
      if (state.gpuLine) this.logger?.log(state.gpuLine);
    }
    async createChatCompletion() {
      const { promptMs, tokens, gapMs, timings } = state.stream;
      return (async function* () {
        await new Promise((r) => setTimeout(r, promptMs));
        for (let i = 0; i < tokens; i++) {
          if (i) await new Promise((r) => setTimeout(r, gapMs));
          yield { choices: [{ delta: { content: 't' } }] };
        }
        yield { choices: [{ delta: {} }], usage: { prompt_tokens: 500 }, ...(timings ? { timings } : {}) };
      })();
    }
    getLoadedContextInfo() {
      return { n_ctx: 1024, n_ctx_train: 1024, n_layer: 1 };
    }
    getModelMetadata() {
      return { meta: {} };
    }
    isSupportWebGPU() {
      return false;
    }
    isMultithread() {
      return true;
    }
    getNumThreads() {
      return 4;
    }
    async exit() {
      state.exited++;
    }
  }
  return { ModelManager, Wllama, WllamaAbortError: class extends Error {} };
});

const { autoload, chat, checkCrashedLoad, gpuFromLog, loadModel, unloadModel, refreshCache, removeLegacyCopies, useEngine } = await import('./engine');

beforeEach(async () => {
  await unloadModel();
  Object.assign(state, { cached: [], blob: new Blob([bytes]), removed: [], loaded: 0, opened: 0, gpuLine: null, loadFails: [], loadParams: [], exited: 0 });
  localStorage.clear();
  useEngine.setState({ status: 'idle', error: null });
});

/** A failed load is logged too; catch it, so the test says it happened instead of printing it. */
const catchLoadError = () => vi.spyOn(console, 'error').mockImplementation(() => {});

describe('loadModel integrity gate', () => {
  it('verifies a fresh download, then loads it', async () => {
    await loadModel('tiny');
    expect(useEngine.getState().status).toBe('ready');
    expect(state.opened).toBe(1);
    expect(state.loaded).toBe(1);
  });

  it('removes and refuses a download whose bytes differ from the pin', async () => {
    state.blob = new Blob([new Uint8Array([1, 2, 3, 4, 5, 6, 7, 9])]);
    const logged = catchLoadError();
    await loadModel('tiny');
    expect(logged).toHaveBeenCalledWith('[engine] load failed', expect.objectContaining({ message: expect.stringMatching(/integrity check/) }));
    logged.mockRestore();
    const s = useEngine.getState();
    expect(s.status).toBe('error');
    expect(s.error).toMatch(/integrity check.*sha256/);
    expect(state.loaded).toBe(0);
    expect(state.removed).toHaveLength(1);
  });

  it('does not re-hash a cached copy it already verified', async () => {
    await loadModel('tiny');
    await unloadModel();
    await loadModel('tiny');
    expect(state.opened).toBe(1);
    expect(state.loaded).toBe(2);
  });

  it('always re-verifies after a new download, even with a stale "verified" mark', async () => {
    await loadModel('tiny');
    await unloadModel();
    state.cached = []; // cache cleared behind our back; the mark survives in localStorage
    state.blob = new Blob([new Uint8Array(8)]);
    const logged = catchLoadError();
    await loadModel('tiny');
    expect(logged).toHaveBeenCalledWith('[engine] load failed', expect.objectContaining({ message: expect.stringMatching(/integrity check/) }));
    logged.mockRestore();
    expect(useEngine.getState().status).toBe('error');
    expect(state.loaded).toBe(1);
  });
});

describe('a load that fails on this device', () => {
  it('steps down to less memory (CPU only) after an out-of-memory error, and remembers it', async () => {
    state.loadFails = [new Error('RangeError: Out of memory')];
    const warned = vi.spyOn(console, 'warn').mockImplementation(() => {});
    await loadModel('tiny');
    warned.mockRestore();
    const s = useEngine.getState();
    expect(s.status).toBe('ready');
    expect(state.loadParams[0]).not.toHaveProperty('n_gpu_layers');
    expect(state.loadParams[1]).toMatchObject({ n_gpu_layers: 0, n_ctx: 1024 });
    expect(s.nCtx).toBe(1024);
    expect(JSON.parse(localStorage.getItem('andai.loadLevel')!)).toEqual({ tiny: 1 });
    expect(localStorage.getItem('andai.loadAttempt')).toBeNull();
    // the next load starts at the level that worked
    await unloadModel();
    await loadModel('tiny');
    expect(state.loadParams[2]).toMatchObject({ n_gpu_layers: 0 });
  });

  it('says plainly when nothing fits, on the model it was about', async () => {
    state.loadFails = [new Error('Out of memory'), new Error('Out of memory'), new Error('Out of memory')];
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    const logged = catchLoadError();
    await loadModel('tiny');
    logged.mockRestore();
    vi.mocked(console.warn).mockRestore();
    const s = useEngine.getState();
    expect(s.status).toBe('error');
    expect(s.errorId).toBe('tiny');
    expect(s.error).toMatch(/^Tiny needs more memory than the browser gives this page.*CPU only, 1,024-token context.*smaller model/);
    expect(state.loadParams).toHaveLength(3);
    expect(state.exited).toBe(3);
  });

  it("doesn't retry what less memory can't fix", async () => {
    state.loadFails = [new Error('QuotaExceededError: the quota has been exceeded')];
    const logged = catchLoadError();
    await loadModel('tiny');
    logged.mockRestore();
    expect(state.loadParams).toHaveLength(1);
    expect(useEngine.getState().error).toMatch(/enough storage space/);
  });

  it('stops a warm-up that never finishes, instead of spinning forever', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    state.loadFails = ['hang', 'hang', 'hang'];
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    const logged = catchLoadError();
    const done = loadModel('tiny');
    for (let i = 0; i < 3; i++) {
      await vi.waitFor(() => expect(state.loadParams).toHaveLength(i + 1));
      await vi.advanceTimersByTimeAsync(180_000);
    }
    await done;
    vi.useRealTimers();
    logged.mockRestore();
    vi.mocked(console.warn).mockRestore();
    expect(useEngine.getState().error).toMatch(/didn't finish warming up in 3 minutes/);
    expect(state.exited).toBeGreaterThanOrEqual(3);
  });

  it('stops at once when a worker dies under wllama, and tries again lighter', async () => {
    for (const fail of ['dead-thread', 'worker-event'] as const) {
      await unloadModel();
      localStorage.clear();
      state.loadParams = [];
      state.loadFails = ['dead-thread', fail];
      vi.spyOn(console, 'warn').mockImplementation(() => {});
      vi.spyOn(console, 'log').mockImplementation(() => {});
      const native = catchLoadError();
      // no timers advanced: without the fatal-log check this would wait out the limit
      await loadModel('tiny');
      native.mockRestore();
      vi.mocked(console.warn).mockRestore();
      vi.mocked(console.log).mockRestore();
      expect(useEngine.getState().status).toBe('ready');
      expect(state.loadParams).toHaveLength(3);
      expect(state.loadParams[2]).toMatchObject({ n_gpu_layers: 0, n_ctx: 1024 });
      expect(JSON.parse(localStorage.getItem('andai.loadLevel')!)).toEqual({ tiny: 2 });
    }
  });

  it('says so at once when the engine build cannot compile, instead of hanging or blaming memory', async () => {
    state.loadFails = ['compile-abort'];
    vi.spyOn(console, 'log').mockImplementation(() => {});
    const logged = catchLoadError();
    await loadModel('tiny');
    logged.mockRestore();
    vi.mocked(console.log).mockRestore();
    expect(state.loadParams).toHaveLength(1);
    expect(useEngine.getState().error).toMatch(/^This browser can't run the engine build Andai picked for Tiny .*Memory64 is not enabled/);
  });

  it('reports a page closed mid-load on the next launch, asks for less, and does not start it again', async () => {
    localStorage.setItem('andai.loadAttempt', JSON.stringify({ id: 'tiny', level: 0, at: Date.now() - 5_000 }));
    localStorage.setItem('andai.lastModel', 'tiny');
    state.cached = [{ url: 'https://huggingface.co/o/r/resolve/0000000000000000000000000000000000000000/t.gguf', size: 8, remove: async () => {} }];
    await autoload();
    const s = useEngine.getState();
    expect(s.status).toBe('error');
    expect(s.errorId).toBe('tiny');
    expect(s.error).toMatch(/closed while Tiny was warming up \(GPU, 1,024-token context\).*ran out of memory.*next try uses less memory \(CPU only/);
    expect(state.loadParams).toHaveLength(0);
    expect(localStorage.getItem('andai.loadAttempt')).toBeNull();
    expect(JSON.parse(localStorage.getItem('andai.loadLevel')!)).toEqual({ tiny: 1 });
  });

  it('ignores a stale marker from an old session', () => {
    localStorage.setItem('andai.loadAttempt', JSON.stringify({ id: 'tiny', level: 0, at: Date.now() - 60 * 60_000 }));
    expect(checkCrashedLoad()).toBeNull();
    expect(useEngine.getState().error).toBeNull();
    expect(localStorage.getItem('andai.loadAttempt')).toBeNull();
  });

  it('marks the attempt only while warming up', async () => {
    let during: string | null = null;
    state.loadFails = [];
    const spy = vi.spyOn(Storage.prototype, 'setItem');
    await loadModel('tiny');
    during = spy.mock.calls.find(([k]) => k === 'andai.loadAttempt')?.[1] ?? null;
    spy.mockRestore();
    expect(JSON.parse(during!)).toMatchObject({ id: 'tiny', level: 0 });
    expect(localStorage.getItem('andai.loadAttempt')).toBeNull();
  });
});

describe('pre-pinning copies', () => {
  it('are reported and removed only on request', async () => {
    const legacy = 'https://huggingface.co/o/r/resolve/main/t.gguf';
    state.cached = [{ url: legacy, size: 8, remove: async () => void state.removed.push(legacy) }];
    await refreshCache();
    expect(useEngine.getState().legacy).toEqual({ tiny: 8 });
    expect(state.removed).toEqual([]);
    await removeLegacyCopies('tiny');
    expect(state.removed).toEqual([legacy]);
  });
});

describe('what the engine reports', () => {
  const drain = async () => {
    const events = [];
    for await (const ev of chat([{ role: 'user', content: 'hi' }], { temperature: 0, maxTokens: 64, thinking: false, signal: new AbortController().signal })) events.push(ev);
    return events;
  };

  it('times generation from the first token, and reports the prompt apart', async () => {
    // Counting the prompt made a 60 tok/s answer look like 15 tok/s (AGENTS.md §2).
    await loadModel('tiny');
    state.stream = { promptMs: 200, tokens: 5, gapMs: 10, timings: null };
    const done = (await drain()).at(-1) as { type: 'done'; tokPerSec: number; promptTokPerSec: number; promptTokens: number };
    expect(done.promptTokens).toBe(500);
    expect(done.tokPerSec).toBeGreaterThan(40); // 4 tokens in ~40 ms, not 5 in ~240 ms (~21)
    expect(done.promptTokPerSec).toBeGreaterThan(1000);
    expect(done.promptTokPerSec).toBeLessThan(3000); // 500 tokens in ~200 ms
  });

  it("prefers llama.cpp's own timings when the stream has them", async () => {
    await loadModel('tiny');
    state.stream = { promptMs: 0, tokens: 3, gapMs: 1, timings: { prompt_n: 500, prompt_per_second: 185, predicted_per_second: 63 } };
    expect((await drain()).at(-1)).toMatchObject({ type: 'done', tokPerSec: 63, promptTokPerSec: 185 });
    expect(useEngine.getState().tokPerSec).toBe(63);
  });

  it('reports the GPU layers llama.cpp offloaded, not just that WebGPU exists', async () => {
    // The engine reads these lines and still passes them on to the console.
    const logged = vi.spyOn(console, 'log').mockImplementation(() => {});
    state.gpuLine = 'load_tensors: offloaded 29/29 layers to GPU';
    await loadModel('tiny');
    expect(useEngine.getState().info?.backend).toBe('WebGPU · 29/29 layers');
    await unloadModel();
    state.gpuLine = 'ggml_webgpu: Failed to get an adapter: WebGPU not available on this browser (requestAdapter returned null)';
    await loadModel('tiny');
    expect(useEngine.getState().info?.backend).toBe('WASM · CPU (no GPU adapter)');
    expect(gpuFromLog('llama_context: n_ctx = 4096')).toBeNull();
    expect(logged.mock.calls.map((c) => c[0])).toEqual([expect.stringMatching(/^load_tensors/), expect.stringMatching(/^ggml_webgpu/)]);
    logged.mockRestore();
  });
});
