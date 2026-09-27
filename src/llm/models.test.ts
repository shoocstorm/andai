import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { availableModels, DEFAULT_MODEL, isMlx, LAYA_MODELS, layaById, MODELS, modelById, pinnedFileUrl, recommendedModel, type WllamaDef } from './models';

describe('model catalog', () => {
  it('has unique ids and a valid default', () => {
    expect(new Set(MODELS.map((m) => m.id)).size).toBe(MODELS.length);
    expect(modelById(DEFAULT_MODEL)).toBeDefined();
  });
  const gguf = MODELS.filter((m): m is WllamaDef => m.engine === 'wllama');
  it.each(gguf.map((m) => [m.id, m]))('%s points at a single HTTPS GGUF under wllama’s 2 GB limit', (_, m) => {
    expect(m.url).toMatch(/^https:\/\/huggingface\.co\/.+\.gguf$/);
    expect(m.bytes).toBeGreaterThan(0);
    expect(m.bytes).toBeLessThan(2 * 1024 ** 3);
    expect(m.n_ctx).toBeGreaterThanOrEqual(1024);
  });
  it.each(gguf.map((m) => [m.id, m]))('%s is pinned to an immutable commit with a sha256 (AGENTS.md §9)', (_, m) => {
    expect(m.url).toMatch(/^https:\/\/huggingface\.co\/[^/]+\/[^/]+\/resolve\/[0-9a-f]{40}\/.+\.gguf$/);
    expect(m.sha256).toMatch(/^[0-9a-f]{64}$/);
    expect(m.legacyUrls).not.toContain(m.url);
  });
  it('returns undefined for unknown ids', () => {
    expect(modelById('nope')).toBeUndefined();
    expect(modelById(null)).toBeUndefined();
  });

  it('offers MLX models only where MLX runs, and recommends them there', () => {
    expect(availableModels(false).some(isMlx)).toBe(false);
    expect(availableModels(true).filter(isMlx).map((m) => m.id)).toEqual(['qwen3-1.7b-mlx', 'qwen3-0.6b-mlx']);
    expect(recommendedModel(true).id).toBe('qwen3-1.7b-mlx');
    expect(recommendedModel(false).id).toBe(DEFAULT_MODEL);
    expect(DEFAULT_MODEL).toBe('qwen3-0.6b');
  });
});

describe('MLX catalog', () => {
  const rust = readFileSync(join(__dirname, '../../src-tauri/src/llm/catalog.rs'), 'utf8');
  const pins = [...rust.matchAll(/pinned\("([\w.-]+)", "([^"]+)", "([0-9a-f]{40})", &(\w+)\)/g)].map((m) => ({ id: m[1], repo: m[2], commit: m[3], files: m[4] }));

  it('names exactly the checkpoints the Rust catalog pins, at the same repo and commit', () => {
    const mlx = MODELS.filter(isMlx);
    expect(mlx.map((m) => m.native).sort()).toEqual(pins.map((p) => p.id).sort());
    for (const m of mlx) {
      const pin = pins.find((p) => p.id === m.native)!;
      expect(m.url).toBe(`https://huggingface.co/${pin.repo}/tree/${pin.commit}`);
    }
  });

  it('states the total size of the pinned files', () => {
    for (const m of MODELS.filter(isMlx)) {
      const pin = pins.find((p) => p.id === m.native)!;
      const block = rust.slice(rust.indexOf(`static ${pin.files}:`));
      const files = block.slice(0, block.indexOf('];'));
      const sizes = [...files.matchAll(/"[\w./-]+",\s*([\d_]+),/g)].map((x) => Number(x[1].replaceAll('_', '')));
      const tokenizer = files.includes('TOKENIZER') ? Number(/TOKENIZER: CheckpointFile = f\(\s*"tokenizer\.json",\s*([\d_]+)/.exec(rust)![1].replaceAll('_', '')) : 0;
      expect(sizes.reduce((a, b) => a + b, 0) + tokenizer).toBe(m.bytes);
    }
  });
});

describe('Laya catalog', () => {
  it('names the checkpoints the Rust catalog pins', () => {
    const rust = readFileSync(join(__dirname, '../../src-tauri/src/laya/catalog.rs'), 'utf8');
    const ids = [...rust.matchAll(/pinned\("([\w-]+)"/g)].map((m) => m[1]);
    expect(LAYA_MODELS.map((m) => m.id).sort()).toEqual(ids.sort());
    expect(layaById('laya-en')?.name).toBe('Laya English');
  });

  it('builds only pinned Hugging Face file URLs', () => {
    const sha = 'f2b4faf51023039425946074e2cf1361d2db11d5';
    expect(pinnedFileUrl('aac6fef/laya-mlx', sha, 'tokenizer/tokenizer.json')).toBe(
      `https://huggingface.co/aac6fef/laya-mlx/resolve/${sha}/tokenizer/tokenizer.json`,
    );
    expect(() => pinnedFileUrl('aac6fef/laya-mlx', 'main', 'model.safetensors')).toThrow(/pinned/);
    expect(() => pinnedFileUrl('aac6fef/laya-mlx', sha, '../x')).toThrow();
    expect(() => pinnedFileUrl('evil.com/../x', sha, 'a')).toThrow();
    expect(() => pinnedFileUrl('a/b', sha, 'a?x=1')).toThrow();
  });
});
