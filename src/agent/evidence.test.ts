import { describe, expect, it } from 'vitest';
import type { SearchHit } from '../kb/api';
import { addEvidence, mergeEvidence, type Found } from './evidence';

const hit = (id: string, file: string, start: number, end: number, snippet: string): SearchHit => ({ id, name: id, node_type: 'Function', file, start_line: start, end_line: end, snippet });

describe('addEvidence', () => {
  it('keeps a node found again in its place, with the text the new passage adds', () => {
    // Find usages returned the call site of refundFraction in cancelBooking,
    // and it was thrown away because a search had found cancelBooking first.
    const found: Found[] = [];
    expect(addEvidence(found, 'kb_search', [hit('f:cancel', 'booking.ts', 35, 40, 'export async function cancelBooking(…)'), hit('f:x', 'x.ts', 1, 2, 'x')])).toBe(2);
    expect(addEvidence(found, 'kb_find_usages', [hit('f:cancel', 'booking.ts', 35, 40, 'Function cancelBooking → Calls refundFraction\n  37: const amount = …')])).toBe(1);
    expect(found.map((f) => f.hit.id)).toEqual(['f:cancel', 'f:x']);
    expect(found[0].hit.snippet).toBe('export async function cancelBooking(…)\nFunction cancelBooking → Calls refundFraction\n  37: const amount = …');
    expect(found[0].tool).toBe('kb_search');
  });

  it('adds nothing for a passage it already has, and takes the longer text when one contains the other', () => {
    const found: Found[] = [];
    addEvidence(found, 'kb_search', [hit('a', 'a.md', 1, 5, 'short')]);
    expect(addEvidence(found, 'kb_search', [hit('a', 'a.md', 1, 5, 'short')])).toBe(0);
    expect(addEvidence(found, 'kb_symbol_context', [hit('a', 'a.md', 1, 5, 'short, and longer')])).toBe(1);
    expect(found[0].hit.snippet).toBe('short, and longer');
  });
});

describe('mergeEvidence', () => {
  it('drops passages inside a range read on purpose, and puts reads first, then lookups, then search hits', () => {
    const found: Found[] = [
      { tool: 'kb_search', hit: hit('s1', 'booking.ts', 35, 40, 'cancelBooking') },
      { tool: 'kb_search', hit: hit('s2', 'fares.ts', 32, 38, 'computeFare') },
      { tool: 'kb_find_usages', hit: hit('u1', 'booking.ts', 22, 32, 'createBooking → Calls computeFare') },
      { tool: 'kb_read_lines', hit: hit('r1', 'booking.ts', 30, 41, 'lines 30–41') },
    ];
    expect(mergeEvidence(found).map((h) => h.id)).toEqual(['r1', 'u1', 's2']);
  });

  it('drops a search passage when the same lines were read whole afterwards (item 15)', () => {
    const found: Found[] = [
      { tool: 'kb_search', hit: hit('s1', 'crew.md', 40, 54, 'Minimum crew (clipped)') },
      { tool: 'kb_read_lines', hit: hit('r1', 'crew.md', 40, 54, 'Minimum crew, the whole section') },
    ];
    expect(mergeEvidence(found).map((h) => h.id)).toEqual(['r1']);
  });

  it('keeps passages without line numbers, and the earlier of two reads of the same range', () => {
    const found: Found[] = [
      { tool: 'kb_overview', hit: hit('o', '(overview)', 0, 0, 'Kind: code') },
      { tool: 'kb_get_code', hit: hit('c1', 'a.ts', 1, 9, 'code') },
      { tool: 'kb_read_lines', hit: hit('c2', 'a.ts', 1, 9, 'code') },
      { tool: 'kb_read_lines', hit: hit('c3', 'a.ts', 1, 20, 'more code') },
    ];
    expect(mergeEvidence(found).map((h) => h.id)).toEqual(['c3', 'o']);
    expect(mergeEvidence(found.slice(0, 3)).map((h) => h.id)).toEqual(['c1', 'o']);
  });
});
