// The runners call these for every run, for the chat model and the decider,
// whatever they are.
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { homedir, tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
// @ts-expect-error — plain ESM script without type declarations
import { keepCheckpoint, seedCheckpoint } from '../../scripts/checkpoint-cache.mjs';

describe('checkpoint test cache', () => {
  it('does nothing for a run without a decider, a wllama model, or an id that is not a checkpoint', () => {
    // a null decider crashed the eval runner after a full run (join(…, null))
    for (const id of [null, undefined, 'qwen3-0.6b', '../x', '../../x-mlx', 'laya-../x']) {
      expect(() => keepCheckpoint('/nonexistent', id)).not.toThrow();
      expect(seedCheckpoint('/nonexistent', id)).toBe(false);
    }
  });

  it('keeps only a verified MLX checkpoint, under its kind, and seeds it back', () => {
    const id = `test-${process.pid}-${Date.now()}-mlx`;
    const cached = join(homedir(), '.cache/andai-test/llm', id);
    const app = mkdtempSync(join(tmpdir(), 'andai-cache-test-'));
    try {
      const dir = join(app, 'models/llm', id);
      mkdirSync(dir, { recursive: true });
      writeFileSync(join(dir, 'config.json'), '{}');
      keepCheckpoint(app, id);
      expect(seedCheckpoint(app, id)).toBe(false); // no `.verified`: never cached
      writeFileSync(join(dir, '.verified'), 'commit');
      keepCheckpoint(app, id);
      rmSync(join(app, 'models'), { recursive: true });
      expect(seedCheckpoint(app, id)).toBe(true);
      expect(readFileSync(join(app, 'models/llm', id, 'config.json'), 'utf8')).toBe('{}');
    } finally {
      rmSync(app, { recursive: true, force: true });
      rmSync(cached, { recursive: true, force: true });
    }
  });
});
