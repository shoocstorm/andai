import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { DEFAULT_MODEL, LAYA_MODELS, layaById, layaFileUrl, MODELS, modelById } from './models';

describe('model catalog', () => {
  it('has unique ids and a valid default', () => {
    expect(new Set(MODELS.map((m) => m.id)).size).toBe(MODELS.length);
    expect(modelById(DEFAULT_MODEL)).toBeDefined();
  });
  it.each(MODELS.map((m) => [m.id, m]))('%s points at a single HTTPS GGUF under wllama’s 2 GB limit', (_, m) => {
    expect(m.url).toMatch(/^https:\/\/huggingface\.co\/.+\.gguf$/);
    expect(m.bytes).toBeGreaterThan(0);
    expect(m.bytes).toBeLessThan(2 * 1024 ** 3);
    expect(m.n_ctx).toBeGreaterThanOrEqual(1024);
  });
  it.each(MODELS.map((m) => [m.id, m]))('%s is pinned to an immutable commit with a sha256 (AGENTS.md §9)', (_, m) => {
    expect(m.url).toMatch(/^https:\/\/huggingface\.co\/[^/]+\/[^/]+\/resolve\/[0-9a-f]{40}\/.+\.gguf$/);
    expect(m.sha256).toMatch(/^[0-9a-f]{64}$/);
    expect(m.legacyUrls).not.toContain(m.url);
  });
  it('returns undefined for unknown ids', () => {
    expect(modelById('nope')).toBeUndefined();
    expect(modelById(null)).toBeUndefined();
  });
});

describe('Laya catalog', () => {
  it('names the checkpoints the Rust catalog pins', () => {
    const rust = readFileSync(join(__dirname, '../../src-tauri/src/laya/catalog.rs'), 'utf8');
    const ids = [...rust.matchAll(/id: "([\w-]+)"/g)].map((m) => m[1]);
    expect(LAYA_MODELS.map((m) => m.id).sort()).toEqual(ids.sort());
    expect(layaById('laya-en')?.name).toBe('Laya English');
  });

  it('builds only pinned Hugging Face file URLs', () => {
    const sha = 'f2b4faf51023039425946074e2cf1361d2db11d5';
    expect(layaFileUrl('aac6fef/laya-mlx', sha, 'tokenizer/tokenizer.json')).toBe(
      `https://huggingface.co/aac6fef/laya-mlx/resolve/${sha}/tokenizer/tokenizer.json`,
    );
    expect(() => layaFileUrl('aac6fef/laya-mlx', 'main', 'model.safetensors')).toThrow(/pinned/);
    expect(() => layaFileUrl('aac6fef/laya-mlx', sha, '../x')).toThrow();
    expect(() => layaFileUrl('evil.com/../x', sha, 'a')).toThrow();
    expect(() => layaFileUrl('a/b', sha, 'a?x=1')).toThrow();
  });
});
