// How ug found a passage, from its search JSON (a real `ug search` result:
// the refund question against the Tidewater documents).
import { describe, expect, it } from 'vitest';
import type { SearchHit } from './api';
import { countMatches, foundLine, matchOf, strengthOf } from './match';

const hit = (distance: number | undefined, matched_by: string | undefined, hop?: number): SearchHit => ({
  id: `h${distance}`,
  name: 'n',
  node_type: 'Concept',
  file: 'f.md',
  start_line: 1,
  end_line: 2,
  distance,
  matched_by,
  hop,
});

describe('strengthOf', () => {
  it('gives ug’s negated PageRank scores as a share of the best one', () => {
    const peers = [-0.08497894, -0.06472097, -0.052159287, -0.04433533];
    expect(strengthOf(-0.08497894, peers)).toBe(1);
    expect(strengthOf(-0.04433533, peers)).toBeCloseTo(0.522, 3);
  });

  it('ranks positive distances lowest-first, and is 1 when there is nothing to compare', () => {
    expect(strengthOf(0.2, [0.2, 0.6])).toBe(1);
    expect(strengthOf(0.6, [0.2, 0.6])).toBe(0);
    expect(strengthOf(0.4, [0.4])).toBe(1);
    expect(strengthOf(-0.1, [])).toBe(1);
  });
});

describe('countMatches and foundLine', () => {
  it('counts a search’s passages by channel, in a fixed order, and says so in one line', () => {
    const hits = [hit(-0.08, 'semantic'), hit(-0.07, 'graph', 1), hit(-0.06, 'semantic'), hit(-0.05, 'keyword'), hit(-0.04, undefined)];
    const found = countMatches(hits);
    expect(found).toEqual({ total: 5, by: { semantic: 2, graph: 1, keyword: 1 } });
    expect(foundLine(found)).toBe('5 passages · 2 semantic · 1 keyword · 1 graph');
    expect(foundLine(countMatches([hit(-0.1, 'semantic')]))).toBe('1 passage · 1 semantic');
    expect(foundLine(countMatches([]))).toBe('0 passages');
  });
});

describe('matchOf', () => {
  const peers = [hit(-0.08497894, 'semantic'), hit(-0.04433533, 'graph', 1)];

  it('labels the channel, with the hops of a graph walk, and explains it', () => {
    expect(matchOf(peers[0], peers)).toMatchObject({ how: 'semantic', label: 'semantic', score: -0.08497894, strength: 1 });
    expect(matchOf(peers[1], peers)).toMatchObject({ how: 'graph', label: 'graph · 1 hop', help: expect.stringMatching(/following links/) });
    expect(matchOf(hit(-0.05, 'keyword'), peers)!.help).toMatch(/full-text/);
    expect(matchOf(hit(-0.05, 'graph', 2), peers)!.label).toBe('graph · 2 hops');
  });

  it('is null for a passage no search returned, and has no channel when ug didn’t say', () => {
    expect(matchOf(hit(undefined, undefined))).toBeNull();
    expect(matchOf(hit(-0.05, undefined))).toMatchObject({ how: null, label: '', strength: 1 });
  });
});
