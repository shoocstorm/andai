// The runners call these for every run, with or without a Laya decider.
import { describe, expect, it } from 'vitest';
// @ts-expect-error — plain ESM script without type declarations
import { keepLaya, seedLaya } from '../../scripts/laya-cache.mjs';

describe('laya test cache', () => {
  it('does nothing for a run without a Laya decider, or an id that is not one', () => {
    // a null decider crashed the eval runner after a full run (join(…, null))
    for (const id of [null, undefined, 'qwen3-0.6b', '../x']) {
      expect(() => keepLaya('/nonexistent', id)).not.toThrow();
      expect(seedLaya('/nonexistent', id)).toBe(false);
    }
  });
});
