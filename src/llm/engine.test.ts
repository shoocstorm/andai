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
}));

vi.mock('./models', () => {
  const def = {
    id: 'tiny',
    name: 'Tiny',
    bytes: 8,
    url: 'https://huggingface.co/o/r/resolve/0000000000000000000000000000000000000000/t.gguf',
    sha256: createHash('sha256').update(new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8])).digest('hex'),
    legacyUrls: ['https://huggingface.co/o/r/resolve/main/t.gguf'],
    n_ctx: 1024,
    thinking: false,
  };
  return { MODELS: [def], modelById: (id: string) => (id === 'tiny' ? def : undefined) };
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
    setCompat() {}
    async loadModel() {
      state.loaded++;
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
    getWorkerResources() {
      return { compat: false };
    }
    async exit() {}
  }
  return { ModelManager, Wllama, WllamaAbortError: class extends Error {} };
});

const { loadModel, unloadModel, refreshCache, removeLegacyCopies, useEngine } = await import('./engine');

beforeEach(async () => {
  await unloadModel();
  Object.assign(state, { cached: [], blob: new Blob([bytes]), removed: [], loaded: 0, opened: 0 });
  localStorage.clear();
  useEngine.setState({ status: 'idle', error: null });
});

describe('loadModel integrity gate', () => {
  it('verifies a fresh download, then loads it', async () => {
    await loadModel('tiny');
    expect(useEngine.getState().status).toBe('ready');
    expect(state.opened).toBe(1);
    expect(state.loaded).toBe(1);
  });

  it('removes and refuses a download whose bytes differ from the pin', async () => {
    state.blob = new Blob([new Uint8Array([1, 2, 3, 4, 5, 6, 7, 9])]);
    await loadModel('tiny');
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
    await loadModel('tiny');
    expect(useEngine.getState().status).toBe('error');
    expect(state.loaded).toBe(1);
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
