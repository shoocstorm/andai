import { describe, expect, it } from 'vitest';
import { DEFAULT_MODEL, MODELS, modelById } from './models';

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
