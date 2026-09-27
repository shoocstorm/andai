// The claim check: which sentences are checked against which source, and the
// record, with the Rust command mocked.
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { SearchHit } from '../kb/api';

const rust = vi.hoisted(() => ({ scores: [] as number[], seen: [] as unknown[] }));
vi.mock('../llm/laya', () => ({
  layaSupport: async (claims: unknown[]) => {
    rust.seen.push(claims);
    return { scores: rust.scores, inputTokens: rust.scores.map(() => 40), truncated: rust.scores.map((_, i) => i === 1), ms: 12, model: 'laya-en' };
  },
}));

const { checkClaims, citedClaims, claimState, FLAG_BELOW, MAX_CLAIMS } = await import('./claims');

const hit = (i: number): SearchHit => ({ id: `h${i}`, name: `n${i}`, node_type: 'Section', file: `f${i}.md`, start_line: i, end_line: i + 4, snippet: ` passage ${i} ` });

beforeEach(() => {
  rust.seen = [];
});

describe('citedClaims', () => {
  it('pairs each cited sentence with every source it cites, without Markdown or citation marks', () => {
    const answer = 'The peak multiplier is **1.25** [3]. It applies on Friday and Sunday [1][2], per `PEAK_DAYS`.\n- Cars pay a surcharge [1, 2].';
    expect(citedClaims(answer, 3)).toEqual([
      { sentence: 'The peak multiplier is 1.25.', n: 3, cites: [3] },
      { sentence: 'It applies on Friday and Sunday, per PEAK_DAYS.', n: 1, cites: [1, 2] },
      { sentence: 'It applies on Friday and Sunday, per PEAK_DAYS.', n: 2, cites: [1, 2] },
      { sentence: 'Cars pay a surcharge.', n: 1, cites: [1, 2] },
      { sentence: 'Cars pay a surcharge.', n: 2, cites: [1, 2] },
    ]);
  });

  it('skips uncited sentences, code blocks, sources that don’t exist, repeats and sentences with no words left', () => {
    const answer = 'Hello there. See this:\n```ts\nconst x = 1; // [1]\n```\nIt is 48 hours [1]. It is 48 hours [1]. Wrong source [4]. , , and [2].';
    expect(citedClaims(answer, 3)).toEqual([{ sentence: 'It is 48 hours.', n: 1, cites: [1] }]);
  });

  it('checks at most MAX_CLAIMS, and cuts a very long sentence', () => {
    const many = Array.from({ length: MAX_CLAIMS + 5 }, (_, i) => `Fact number ${i} holds [1].`).join(' ');
    expect(citedClaims(many, 1)).toHaveLength(MAX_CLAIMS);
    expect(citedClaims(`${'word '.repeat(300)}[1].`, 1)[0].sentence.length).toBeLessThanOrEqual(500);
  });
});

describe('checkClaims', () => {
  it('sends each claim with the passage it cites, and flags low scores', async () => {
    rust.scores = [0.9, FLAG_BELOW / 2];
    const r = (await checkClaims('It is 48 hours [1]. Cars pay 18.5 [2].', [hit(1), hit(2)]))!;
    expect(rust.seen).toEqual([
      [
        { statement: 'It is 48 hours.', source: 'f1.md:1-5', text: 'passage 1' },
        { statement: 'Cars pay 18.5.', source: 'f2.md:2-6', text: 'passage 2' },
      ],
    ]);
    expect(r).toMatchObject({ model: 'Laya English', modelId: 'laya-en', modelMs: 12, flagBelow: FLAG_BELOW });
    expect(r.items.map((x) => [x.n, x.source, x.flagged, x.inputTokens, x.truncated])).toEqual([
      [1, 'f1.md:1-5', false, 40, false],
      [2, 'f2.md:2-6', true, 40, true],
    ]);
  });

  it('shows the input exactly as Rust builds it (mod.rs `claim_state`)', () => {
    expect(claimState(' It is 48. ', 'a.md:1-3', ' body ')).toBe('Statement:\nIt is 48.\n\nPassage from a.md:1-3:\nbody');
  });

  it('asks nothing when the answer cites nothing', async () => {
    expect(await checkClaims('No sources here.', [hit(1)])).toBeNull();
    expect(rust.seen).toEqual([]);
  });

  it('refuses a reply with the wrong number of scores', async () => {
    rust.scores = [0.5];
    await expect(checkClaims('It is 48 [1]. Cars pay 18.5 [1].', [hit(1)])).rejects.toThrow(/1 scores for 2 claims/);
  });
});
