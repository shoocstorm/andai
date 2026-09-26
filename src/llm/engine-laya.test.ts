// A Laya checkpoint as the decision model: loading downloads it first when
// needed, decide.ts is routed to it, and removing it unloads it first.
import { beforeEach, describe, expect, it, vi } from 'vitest';

const laya = vi.hoisted(() => ({
  downloaded: false,
  supported: true,
  log: [] as string[],
  failLoad: null as Error | null,
}));

vi.mock('./laya', () => ({
  layaStatus: async () => ({
    supported: laya.supported,
    loaded: null,
    checkpoints: laya.supported ? [{ id: 'laya-multilingual', repo: 'r/x', commit: 'c', bytes: 10, files: [], downloaded: laya.downloaded }] : [],
  }),
  downloadLaya: async (c: { id: string }, onProgress: (p: unknown) => void) => {
    laya.log.push(`download ${c.id}`);
    onProgress({ loaded: 5, total: 10, speed: 1, phase: 'Downloading…' });
    laya.downloaded = true;
  },
  layaLoad: async (id: string) => {
    if (laya.failLoad) throw laya.failLoad;
    laya.log.push(`load ${id}`);
    return 60;
  },
  layaUnload: async () => void laya.log.push('unload'),
  layaRemove: async (id: string) => {
    laya.log.push(`remove ${id}`);
    laya.downloaded = false;
  },
}));

const { deciderLaya, loadDecider, removeLaya, unloadDecider, useEngine, autoload } = await import('./engine');

beforeEach(async () => {
  await unloadDecider();
  laya.log = [];
  laya.downloaded = false;
  laya.supported = true;
  laya.failLoad = null;
  useEngine.setState((s) => ({ decider: { ...s.decider, status: 'idle', error: null } }));
});

describe('Laya as the decision model', () => {
  it('downloads a missing checkpoint, loads it and routes decisions to it', async () => {
    await loadDecider('laya-multilingual');
    expect(laya.log).toEqual(['download laya-multilingual', 'load laya-multilingual']);
    expect(useEngine.getState().decider).toMatchObject({ status: 'ready', loadedId: 'laya-multilingual', error: null });
    expect(deciderLaya()?.name).toBe('Laya Multilingual');
    expect(useEngine.getState().laya.checkpoints[0].downloaded).toBe(true);
    expect(localStorage.getItem('andai.lastDecider')).toBe('laya-multilingual');
  });

  it('skips the download when the checkpoint is already verified on disk', async () => {
    laya.downloaded = true;
    await loadDecider('laya-multilingual');
    expect(laya.log).toEqual(['load laya-multilingual']);
  });

  it('reports an unsupported Mac and a failed load as the decider error', async () => {
    laya.supported = false;
    await loadDecider('laya-multilingual');
    expect(useEngine.getState().decider).toMatchObject({ status: 'error', error: expect.stringMatching(/Apple Silicon/) });
    laya.supported = true;
    laya.failLoad = new Error('checkpoint is missing type_emb.weight');
    await loadDecider('laya-multilingual');
    expect(useEngine.getState().decider.error).toMatch(/type_emb/);
    expect(deciderLaya()).toBeNull();
  });

  it('unloads in Rust, and a removal unloads first', async () => {
    await loadDecider('laya-multilingual');
    await unloadDecider();
    expect(laya.log.at(-1)).toBe('unload');
    expect(deciderLaya()).toBeNull();
    await loadDecider('laya-multilingual');
    laya.log = [];
    await removeLaya('laya-multilingual');
    expect(laya.log).toEqual(['unload', 'remove laya-multilingual']);
    expect(useEngine.getState().laya.checkpoints[0].downloaded).toBe(false);
  });

  it('reloads the last Laya decider on launch only if it is downloaded', async () => {
    localStorage.setItem('andai.lastDecider', 'laya-multilingual');
    await autoload();
    expect(laya.log).toEqual([]);
    laya.downloaded = true;
    localStorage.setItem('andai.lastDecider', 'laya-multilingual');
    await autoload();
    expect(laya.log).toEqual(['load laya-multilingual']);
  });
});
