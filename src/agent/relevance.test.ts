// The relevance check's policy (keep the top results, drop clear misses) and
// its record, with the Rust command mocked.
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { SearchHit } from '../kb/api';

const rust = vi.hoisted(() => ({ scores: [] as number[], seen: [] as unknown[] }));
vi.mock('../llm/laya', () => ({
  layaRelevance: async (request: string, passages: unknown[]) => {
    rust.seen.push({ request, passages });
    return { scores: rust.scores, ms: 30, model: 'laya-multilingual' };
  },
}));

const { checkRelevance, choosePassages, DROP_BELOW, KEEP_TOP } = await import('./relevance');

const hit = (i: number, snippet = `passage ${i} `.repeat(10)): SearchHit => ({
  id: `h${i}`,
  name: `n${i}`,
  node_type: 'Section',
  file: `f${i}.md`,
  start_line: i,
  end_line: i + 4,
  snippet,
});

beforeEach(() => {
  rust.seen = [];
});

describe('choosePassages', () => {
  it('always keeps the top results, then drops only scores below the threshold', () => {
    expect(KEEP_TOP).toBe(2);
    expect(choosePassages([0.01, 0.02, 0.5, DROP_BELOW, 0.09])).toEqual([
      { kept: true, reason: 'top' },
      { kept: true, reason: 'top' },
      { kept: true, reason: 'score' },
      { kept: true, reason: 'score' },
      { kept: false, reason: 'low' },
    ]);
  });

  it('never drops everything', () => {
    expect(choosePassages([0, 0, 0]).filter((p) => p.kept).length).toBeGreaterThanOrEqual(1);
  });
});

describe('checkRelevance', () => {
  it('sends each passage with its source, keeps the order, and records what was dropped', async () => {
    rust.scores = [0.9, 0.05, 0.02, 0.7];
    const hits = [hit(1), hit(2), hit(3, 'x'.repeat(400)), hit(4)];
    const r = (await checkRelevance('What headers?', hits))!;
    expect(rust.seen).toEqual([{ request: 'What headers?', passages: hits.map((h) => ({ source: `${h.file}:${h.start_line}-${h.end_line}`, text: h.snippet!.trim() })) }]);
    expect(r.hits.map((h) => h.id)).toEqual(['h1', 'h2', 'h4']);
    expect(r.record).toMatchObject({ model: 'Laya Multilingual', modelMs: 30, keepTop: 2, dropBelow: DROP_BELOW, tokensSaved: Math.round(400 / 3.2) });
    expect(r.record.items.map((x) => [x.file, x.kept, x.reason])).toEqual([
      ['f1.md', true, 'top'],
      ['f2.md', true, 'top'],
      ['f3.md', false, 'low'],
      ['f4.md', true, 'score'],
    ]);
  });

  it('does nothing when there are no more than the top results', async () => {
    expect(await checkRelevance('q', [hit(1), hit(2)])).toBeNull();
    expect(rust.seen).toEqual([]);
  });

  it('refuses a reply with the wrong number of scores', async () => {
    rust.scores = [0.5];
    await expect(checkRelevance('q', [hit(1), hit(2), hit(3)])).rejects.toThrow(/1 scores for 3 passages/);
  });
});
