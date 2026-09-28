// The relevance check's policy (keep the top results, drop clear misses) and
// its record, with the Rust command mocked.
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { SearchHit } from '../kb/api';

const rust = vi.hoisted(() => ({ scores: [] as number[], seen: [] as unknown[], score: null as null | ((text: string) => number) }));
vi.mock('../llm/laya', () => ({
  layaRelevance: async (request: string, passages: { source: string; text: string }[]) => {
    rust.seen.push({ request, passages });
    // Either fixed scores, or a score computed from each row's text.
    const scores = rust.score ? passages.map((p) => rust.score!(p.text)) : rust.scores;
    return { scores, inputTokens: scores.map(() => 120), truncated: scores.map((_, i) => i === 0 && !rust.score), ms: 30, model: 'laya-multilingual' };
  },
}));

const { checkRelevance, choosePassages, chunkText, DROP_BELOW, KEEP_TOP, PassageScorer, shownLines } = await import('./relevance');

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
  rust.score = null;
});

type Seen = { request: string; passages: { source: string; text: string }[] };
const seen = () => rust.seen as Seen[];

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
    expect(r.record).toMatchObject({ model: 'Laya Multilingual', modelId: 'laya-multilingual', request: 'What headers?', modelMs: 30, keepTop: 2, dropBelow: DROP_BELOW, tokensSaved: Math.round(400 / 3.2) });
    // what the trace's relevance dialog shows: the text Laya read, its size, and whether it was cut
    expect(r.record.items[2]).toMatchObject({ text: 'x'.repeat(400), inputTokens: 120, truncated: false });
    expect(r.record.items[0].truncated).toBe(true);
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

describe('chunkText', () => {
  it('keeps short text whole', () => {
    expect(chunkText('abc', 10)).toEqual(['abc']);
  });

  it('splits long text into pieces that fit, overlap, and cover all of it', () => {
    const lines = Array.from({ length: 60 }, (_, i) => `line ${i} ${'w'.repeat(30)}`);
    const text = lines.join('\n');
    const pieces = chunkText(text, 400);
    expect(pieces.length).toBeGreaterThan(1);
    for (const p of pieces) expect(p.length).toBeLessThanOrEqual(400);
    // every line is whole in some piece, and neighbours share text
    for (const l of lines) expect(pieces.some((p) => p.includes(l))).toBe(true);
    for (let i = 1; i < pieces.length; i++) expect(pieces[i - 1].includes(pieces[i].split('\n')[0])).toBe(true);
  });

  it('splits text without line breaks too', () => {
    const pieces = chunkText('z'.repeat(1000), 300);
    expect(pieces.every((p) => p.length <= 300)).toBe(true);
    expect(pieces.join('').length).toBeGreaterThanOrEqual(1000);
  });
});

describe('PassageScorer', () => {
  it('scores a long passage in pieces that each fit a row, by its best piece', async () => {
    const fact = 'The Kestrel is dry-docked every 30 months.';
    const long = hit(1, `${'Dry-docking is planned for the quiet months.\n'.repeat(80)}${fact}`);
    rust.score = (text) => (text.includes(fact) ? 0.9 : 0.1);
    const scorer = new PassageScorer('How often is the Kestrel dry-docked?', 512);
    const r = await scorer.score([long, hit(2)]);
    const rows = seen()[0].passages;
    expect(rows.length).toBe(r.rows);
    expect(r).toMatchObject({ passages: 2 });
    expect(rows.filter((p) => p.source === 'f1.md:1-5').length).toBeGreaterThan(1);
    // each row fits Laya English's 512 tokens with the request (3.2 characters a token, less the question's share)
    for (const p of rows) expect(p.text.length + seen()[0].request.length).toBeLessThanOrEqual((512 - 64) * 3.2);
    expect(scorer.get(long)).toMatchObject({ score: 0.9, truncated: false });
    expect(scorer.get(long)!.best).toContain(fact);
    expect(scorer.get(long)!.chunks).toBe(rows.filter((p) => p.source === 'f1.md:1-5').length);
  });

  it('scores each passage once, and again when its text grew', async () => {
    rust.score = () => 0.5;
    const scorer = new PassageScorer('q', 1024);
    await scorer.score([hit(1), hit(2)]);
    await scorer.score([hit(1), hit(2), hit(3)]);
    expect(seen().map((c) => c.passages.map((p) => p.source))).toEqual([['f1.md:1-5', 'f2.md:2-6'], ['f3.md:3-7']]);
    await scorer.score([hit(1, 'passage 1 with more text from a later call')]);
    expect(seen()).toHaveLength(3);
    expect(await scorer.score([hit(1), hit(2)])).toMatchObject({ rows: 0, passages: 0 });
    expect(seen()).toHaveLength(3);
  });

  it('sends at most 24 rows a call', async () => {
    rust.score = () => 0.5;
    const scorer = new PassageScorer('q', 1024);
    await scorer.score(Array.from({ length: 30 }, (_, i) => hit(i)));
    expect(seen().map((c) => c.passages.length)).toEqual([24, 6]);
  });

  it('squeezes a long request to its share of a row, keeping its end', async () => {
    rust.score = () => 0.5;
    const request = `${'Some background about the trip. '.repeat(60)}How many oxygen cylinders may he bring?`;
    await new PassageScorer(request, 512).score([hit(1)]);
    expect(seen()[0].request.length).toBeLessThanOrEqual(Math.floor((512 - 64) * 3.2 * 0.3));
    expect(seen()[0].request).toMatch(/How many oxygen cylinders may he bring\?$/);
  });

  it('caches nothing when Laya fails', async () => {
    const scorer = new PassageScorer('q', 512);
    rust.scores = [0.5];
    await expect(scorer.score([hit(1), hit(2)])).rejects.toThrow(/1 scores for 2 passages/);
    expect(scorer.get(hit(1))).toBeUndefined();
  });
});

describe('shownLines', () => {
  it('says how much of a node a clipped passage shows, and nothing for a whole one', () => {
    expect(shownLines({ ...hit(1), start_line: 10, end_line: 37, snippet: 'a\nb\nc' })).toEqual({ lines: 3, of: 28 });
    expect(shownLines({ ...hit(1), start_line: 10, end_line: 12, snippet: 'a\nb\nc' })).toBeUndefined();
    expect(shownLines({ ...hit(1), start_line: 0, end_line: 0 })).toBeUndefined();
  });
});

describe('checkRelevance with the loop’s scores', () => {
  it('scores only what the loop didn’t, and records the pieces', async () => {
    rust.score = (text) => (text.startsWith('passage 3') ? 0.02 : 0.8);
    const scorer = new PassageScorer('q', 1024);
    await scorer.score([hit(1), hit(2), hit(3)]);
    const r = (await checkRelevance('q', [hit(1), hit(2), hit(3), hit(4)], scorer))!;
    expect(seen().map((c) => c.passages.map((p) => p.source))).toEqual([['f1.md:1-5', 'f2.md:2-6', 'f3.md:3-7'], ['f4.md:4-8']]);
    expect(r.hits.map((h) => h.id)).toEqual(['h1', 'h2', 'h4']);
    expect(r.record.items.map((x) => x.chunks)).toEqual([1, 1, 1, 1]);
  });
});
