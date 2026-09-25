import { describe, expect, it } from 'vitest';
import { dedupeHits, inTauri, kbList, kbSearch, ugStatus, type SearchHit } from './api';

const h = (id: string, file: string, start: number, end: number, snippet = 'text'): SearchHit => ({
  id,
  name: id,
  node_type: 'Concept',
  file,
  start_line: start,
  end_line: end,
  snippet,
});

describe('dedupeHits', () => {
  it('drops a whole-document hit when a section inside it is also returned', () => {
    const doc = h('doc', 'a.md', 1, 83);
    const section = h('sec', 'a.md', 11, 33);
    expect(dedupeHits([doc, section]).map((x) => x.id)).toEqual(['sec']);
  });
  it('keeps ug relevance order', () => {
    const a = h('a', 'x.md', 1, 5);
    const b = h('b', 'y.md', 1, 5);
    expect(dedupeHits([b, a]).map((x) => x.id)).toEqual(['b', 'a']);
  });
  it('keeps non-overlapping sections of the same file', () => {
    expect(dedupeHits([h('1', 'a.md', 1, 10), h('2', 'a.md', 11, 20)])).toHaveLength(2);
  });
  it('drops hits with no text to show the model', () => {
    expect(dedupeHits([h('empty', 'a.md', 1, 2, '  '), { ...h('d', 'b.md', 1, 2), snippet: null, description: 'ok' }]).map((x) => x.id)).toEqual(['d']);
  });
});

describe('outside the desktop runtime', () => {
  it('is detected as not-Tauri under jsdom', () => {
    expect(inTauri).toBe(false);
  });
  it('degrades read calls to empty results', async () => {
    await expect(kbList()).resolves.toEqual([]);
    await expect(ugStatus()).resolves.toMatchObject({ found: false });
  });
  it('rejects write/search calls with an actionable message', async () => {
    await expect(kbSearch('x', 'q', 4, 1000)).rejects.toThrow(/desktop app/);
  });
});
