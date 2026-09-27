// Models the user added from Hugging Face: a GGUF one is kept in webview
// storage (malformed entries are dropped), an MLX one is added through Rust,
// both become loadable by id, and removing forgets them and their files.
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { HubModel } from './hub';

const calls = vi.hoisted(() => ({ invoke: [] as [string, unknown][], evicted: [] as string[], removed: [] as string[] }));

vi.mock('@tauri-apps/api/core', async (orig) => ({
  ...(await orig<typeof import('@tauri-apps/api/core')>()),
  invoke: vi.fn(async (cmd: string, args: unknown) => {
    calls.invoke.push([cmd, args]);
    return cmd === 'llm_add_custom' ? { id: 'hf-mlx-community--qwen3-4b-4bit-d7f544ee' } : undefined;
  }),
}));

vi.mock('./engine', async (orig) => ({
  ...(await orig<typeof import('./engine')>()),
  refreshCache: vi.fn(async () => {}),
  refreshNative: vi.fn(async () => {}),
  evictModel: vi.fn(async (id: string) => void calls.evicted.push(id)),
}));

vi.mock('./native', async (orig) => ({
  ...(await orig<typeof import('./native')>()),
  nativeRemove: vi.fn(async (id: string) => void calls.removed.push(id)),
}));

const { addHubModel, ggufDef, initCustomModels, mlxDefs, removeCustomModel, useCustomModels } = await import('./custom');
const { useEngine } = await import('./engine');
const { modelById } = await import('./models');

const COMMIT = 'd7f544eead698dbd1f15126ef60b45a1e1933222';
const gguf: HubModel = {
  repo: 'unsloth/Qwen3-1.7B-GGUF',
  commit: COMMIT,
  format: 'gguf',
  license: 'apache-2.0',
  downloads: 1,
  likes: 1,
  lastModified: null,
  checks: [],
  ok: true,
  gguf: {
    architecture: 'qwen3',
    contextLength: 40960,
    thinking: true,
    recommended: 'Qwen3-1.7B-Q4_K_M.gguf',
    variants: [{ path: 'Qwen3-1.7B-Q4_K_M.gguf', bytes: 1_107_409_472, sha256: 'b'.repeat(64), quant: 'Q4_K_M', fits: true }],
  },
};

beforeEach(() => {
  calls.invoke = [];
  calls.evicted = [];
  calls.removed = [];
  useCustomModels.setState({ gguf: [] });
  useEngine.setState({ native: { supported: true, chat: null, decider: null, checkpoints: [], memory: null }, loadedId: null });
  initCustomModels();
});

describe('custom GGUF models', () => {
  it('are pinned to the commit and file picked, with the file’s sha256', () => {
    const d = ggufDef(gguf, gguf.gguf!.variants[0]);
    expect(d).toMatchObject({
      engine: 'wllama',
      name: 'Qwen3-1.7B · Q4_K_M',
      url: `https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/${COMMIT}/Qwen3-1.7B-Q4_K_M.gguf`,
      sha256: 'b'.repeat(64),
      n_ctx: 4096,
      thinking: true,
      source: { repo: 'unsloth/Qwen3-1.7B-GGUF', commit: COMMIT, license: 'apache-2.0' },
    });
    expect(d.id).toMatch(/^hf-gguf-[a-z0-9.-]+$/);
  });

  it('become loadable by id once added, and are kept across launches', async () => {
    const id = await addHubModel(gguf, gguf.gguf!.variants[0]);
    expect(modelById(id)?.name).toBe('Qwen3-1.7B · Q4_K_M');
    expect(JSON.parse(localStorage.getItem('andai.customModels')!).state.gguf[0].id).toBe(id);
  });

  it('drop an entry from storage that isn’t pinned and hashed', async () => {
    const good = ggufDef(gguf, gguf.gguf!.variants[0]);
    const bad = [
      { ...good, id: 'hf-gguf-x', url: 'https://evil.example/x.gguf' },
      { ...good, id: 'hf-gguf-y', sha256: 'nope' },
      { ...good, id: 'qwen3-0.6b' },
    ];
    localStorage.setItem('andai.customModels', JSON.stringify({ state: { gguf: [good, ...bad] }, version: 0 }));
    await useCustomModels.persist.rehydrate();
    expect(useCustomModels.getState().gguf.map((d) => d.id)).toEqual([good.id]);
  });

  it('refuse a file over 2 GB or a model that can’t run', async () => {
    await expect(addHubModel(gguf, { ...gguf.gguf!.variants[0], fits: false })).rejects.toThrow(/2 GB/);
    await expect(addHubModel({ ...gguf, ok: false }, gguf.gguf!.variants[0])).rejects.toThrow(/can’t be added/);
  });

  it('are removed with their cached file', async () => {
    const id = await addHubModel(gguf, gguf.gguf!.variants[0]);
    await removeCustomModel(id);
    expect(calls.evicted).toEqual([id]);
    expect(useCustomModels.getState().gguf).toEqual([]);
    expect(modelById(id)).toBeUndefined();
  });
});

describe('custom MLX models', () => {
  const spec = { repo: 'mlx-community/Qwen3-4B-4bit', commit: COMMIT, files: [], inline: [] };
  const mlxModel: HubModel = { ...gguf, repo: spec.repo, format: 'mlx', gguf: undefined, mlx: { spec, bytes: 2e9, layers: 36, bits: 4, thinking: true } };

  it('are added through Rust, which checks them', async () => {
    const id = await addHubModel(mlxModel);
    expect(calls.invoke).toEqual([['llm_add_custom', { spec }]]);
    expect(id).toBe('hf-mlx-community--qwen3-4b-4bit-d7f544ee');
  });

  it('come from Rust’s status, and are loadable by id', () => {
    const c = { id: 'hf-mlx-community--qwen3-4b-4bit-d7f544ee', repo: spec.repo, commit: COMMIT, bytes: 2_300_000_000, files: [], downloaded: true, custom: { thinking: true, layers: 36, bits: 4, addedAt: 1 } };
    expect(mlxDefs({ supported: true, chat: null, decider: null, checkpoints: [{ ...c, custom: null }], memory: null })).toEqual([]);
    useEngine.setState({ native: { supported: true, chat: null, decider: null, checkpoints: [c], memory: null } });
    expect(modelById(c.id)).toMatchObject({ engine: 'mlx', native: c.id, name: 'Qwen3-4B-4bit · MLX', family: '4-bit · 36 layers · Hugging Face', size: '2.3 GB' });
  });

  it('are removed through Rust, which forgets them', async () => {
    const c = { id: 'hf-x--y-d7f544ee', repo: 'x/y', commit: COMMIT, bytes: 5, files: [], downloaded: true, custom: { thinking: false, layers: 1, bits: 4, addedAt: 1 } };
    useEngine.setState({ native: { supported: true, chat: null, decider: null, checkpoints: [c], memory: null } });
    await removeCustomModel(c.id);
    expect(calls.removed).toEqual([c.id]);
  });

  it('can’t remove a catalog model this way', async () => {
    await expect(removeCustomModel('qwen3-0.6b')).rejects.toThrow(/Only models you added/);
  });
});
