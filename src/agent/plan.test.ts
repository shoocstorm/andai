import { describe, expect, it } from 'vitest';
import type { SearchHit } from '../kb/api';
import { namedIdentifiers, planNext, PLAN_MAX_READS, searchAgainStep, type PlanInput, type Scored } from './plan';

const hit = (over: Partial<SearchHit> = {}): SearchHit => ({ id: 'h', name: 'Dry-dock schedule', node_type: 'Section', file: 'fleet.md', start_line: 10, end_line: 37, snippet: 'one line', ...over });
const found = (score: number | null, over: Partial<SearchHit> = {}, tool = 'kb_search'): Scored => ({ hit: hit(over), tool, score });
const input = (over: Partial<PlanInput> = {}): PlanInput => ({
  prompt: 'How often is the Kestrel dry-docked?',
  kind: 'document',
  found: [],
  can: new Set(['kb_search', 'kb_read_lines', 'kb_get_code', 'kb_find_usages']),
  done: () => false,
  searches: [{ query: 'kestrel dry dock', scope: 'focused' }],
  ...over,
});

describe('namedIdentifiers', () => {
  it('finds code names, not plain words', () => {
    expect(namedIdentifiers('Which functions call computeFare, and `withRetry`?')).toEqual(['computeFare', 'withRetry']);
    expect(namedIdentifiers('Where is refund_fraction set? What about BookingService and init()?')).toEqual(['refund_fraction', 'BookingService', 'init']);
    expect(namedIdentifiers('How often is the Kestrel dry-docked?')).toEqual([]);
  });
});

describe('planNext', () => {
  it('reads whole the best-scoring passage a search clipped, by its exact lines', () => {
    const step = planNext(input({ found: [found(0.35, { id: 'a', name: 'Hull', start_line: 40, end_line: 60 }), found(0.8)] }));
    expect(step).toMatchObject({ action: 'kb_read_lines', args: { range: 'fleet.md:10-37' } });
    expect(step!.note).toMatch(/“Dry-dock schedule” is likely to help \(80%\), but the search showed 1 of its 28 lines/);
  });

  it('leaves clear misses, whole passages and ranges already read alone, and reads at most twice', () => {
    expect(planNext(input({ found: [found(0.2)] }))).toBeNull();
    expect(planNext(input({ found: [found(0.8, { end_line: 11, snippet: 'a\nb' })] }))).toMatchObject({ action: 'answer_now' });
    expect(planNext(input({ found: [found(0.8)], done: (t, a) => t === 'kb_read_lines' && a.range === 'fleet.md:10-37' }))).toMatchObject({ action: 'answer_now' });
    const read = (i: number) => found(0.9, { id: `r${i}`, start_line: i, end_line: i + 1, snippet: 'x\ny' }, 'kb_read_lines');
    const reads = Array.from({ length: PLAN_MAX_READS }, (_, i) => read(i + 100));
    expect(planNext(input({ found: [...reads, found(0.8)] }))).toMatchObject({ action: 'answer_now' });
  });

  it('never plans a tool that is switched off or denied', () => {
    expect(planNext(input({ found: [found(0.8)], can: new Set(['kb_search']) }))).toMatchObject({ action: 'answer_now' });
  });

  it('on code, reads the source of a symbol the request names once the results show it', () => {
    const fn = found(0.2, { name: 'computeFare', node_type: 'Function', start_line: 3, end_line: 4, snippet: 'function computeFare(…)\n…' });
    expect(planNext(input({ kind: 'code', prompt: 'How does computeFare work out the price?', found: [fn] }))).toMatchObject({ action: 'kb_get_code', args: { symbol: 'computeFare' } });
    // not before the results show it, not in a document knowledge base, not twice
    expect(planNext(input({ kind: 'code', prompt: 'How does computeFare work?', found: [found(0.2)] }))).toBeNull();
    expect(planNext(input({ kind: 'document', prompt: 'How does computeFare work?', found: [fn] }))).toBeNull();
    expect(planNext(input({ kind: 'code', prompt: 'How does computeFare work?', found: [fn], done: (t) => t === 'kb_get_code' }))).toBeNull();
  });

  it('finds the usages instead when the request asks who calls the symbol', () => {
    const fn = found(0.2, { name: 'refundFraction', node_type: 'Function', start_line: 3, end_line: 4, snippet: 'a\nb' });
    for (const prompt of ['Who calls refundFraction, and on which line?', 'Which functions call refundFraction?', 'Where is refundFraction used?']) {
      expect(planNext(input({ kind: 'mixed', prompt, found: [fn] }))).toMatchObject({ action: 'kb_find_usages', args: { symbol: 'refundFraction' } });
    }
  });

  it('answers once a whole passage scores as helping, and has no rule otherwise', () => {
    expect(planNext(input({ found: [found(0.55, { end_line: 11, snippet: 'a\nb' })] }))!.note).toMatch(/likely to help \(55%\), and nothing clipped is left to read, so answering/);
    expect(planNext(input({ found: [found(0.45, { end_line: 11, snippet: 'a\nb' })] }))).toBeNull();
    expect(planNext(input({ found: [found(null)] }))).toBeNull();
  });
});

describe('searchAgainStep', () => {
  const whole = { end_line: 11, snippet: 'a\nb' };
  it('searches once more with the question as written and the other scope when nothing scores as helping', () => {
    const step = searchAgainStep(input({ found: [found(0.08, whole)] }));
    expect(step).toMatchObject({ action: 'kb_search', args: { query: 'How often is the Kestrel dry-docked?', scope: 'broad' } });
    expect(step!.note).toMatch(/the best scored 8%\), so searching again with the question as written and a broad scope/);
    expect(searchAgainStep(input({ found: [found(0.08, whole)], searches: [{ query: 'q', scope: 'broad' }] }))!.args).toMatchObject({ scope: 'focused' });
  });

  it('stays out when a passage helps, after two searches, when search is off, or when it already ran', () => {
    expect(searchAgainStep(input({ found: [found(0.6, whole)] }))).toBeNull();
    expect(searchAgainStep(input({ found: [found(0.1, whole)], searches: [{ scope: 'focused' }, { scope: 'broad' }] }))).toBeNull();
    expect(searchAgainStep(input({ found: [found(0.1, whole)], can: new Set(['kb_read_lines']) }))).toBeNull();
    expect(searchAgainStep(input({ found: [found(0.1, whole)], done: (t) => t === 'kb_search' }))).toBeNull();
  });

  it('is the plan’s last rule when switched on', () => {
    expect(planNext(input({ found: [found(0.1, whole)] }))).toBeNull();
    expect(planNext(input({ found: [found(0.1, whole)], searchAgain: true }))).toMatchObject({ action: 'kb_search' });
  });
});
