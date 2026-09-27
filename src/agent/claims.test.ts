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

const { checkClaims, citedClaims, claimState, FLAG_BELOW, MAX_CLAIMS, supportSummary } = await import('./claims');

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

  it('gives a citation standing after the full stop to the sentence before it (reported: the check skipped)', () => {
    const answer = 'No, refunds only go to the original payment method. Tidewater never refunds in cash at the terminal. [6]';
    expect(citedClaims(answer, 10)).toEqual([
      { sentence: 'No, refunds only go to the original payment method.', n: 6, cites: [6] },
      { sentence: 'Tidewater never refunds in cash at the terminal.', n: 6, cites: [6] },
    ]);
  });

  it('gives a line of citations after a list to the list’s last few items, not to sentences already cited', () => {
    const answer = 'The fare is set in code [1].\n\n- Cars pay 18.5.\n- Bikes pay 2.\n- Dogs ride free.\n- Peak days cost more.\n\n[2], [3]';
    expect(citedClaims(answer, 3).map((c) => [c.sentence, c.n])).toEqual([
      ['The fare is set in code.', 1],
      ['Bikes pay 2.', 2],
      ['Bikes pay 2.', 3],
      ['Dogs ride free.', 2],
      ['Dogs ride free.', 3],
      ['Peak days cost more.', 2],
      ['Peak days cost more.', 3],
    ]);
  });

  it('doesn’t check a sentence about what the sources lack, which no passage can support', () => {
    const answer = 'The information provided does not mention the CEO of Tidewater Ferries [1][2]. The context focuses on refunds and has no details about leadership [1]. Refunds take 5 days [1].';
    expect(citedClaims(answer, 2)).toEqual([{ sentence: 'Refunds take 5 days.', n: 1, cites: [1] }]);
    // a claim that merely uses one of the words is still checked
    expect(citedClaims('Canceling less than 6 hours before departure results in no refund [1].', 1)).toHaveLength(1);
  });

  it('leaves a cited sentence’s own citation alone, and a bare citation with nothing uncited before it', () => {
    expect(citedClaims('It is 48 hours [1]. [2]', 2)).toEqual([{ sentence: 'It is 48 hours.', n: 1, cites: [1] }]);
  });

  it('checks at most MAX_CLAIMS, and cuts a very long sentence', () => {
    const many = Array.from({ length: MAX_CLAIMS + 5 }, (_, i) => `Fact number ${i} holds [1].`).join(' ');
    expect(citedClaims(many, 1)).toHaveLength(MAX_CLAIMS);
    expect(citedClaims(`${'word '.repeat(300)}[1].`, 1)[0].sentence.length).toBeLessThanOrEqual(500);
  });
});

describe('supportSummary', () => {
  const item = (sentence: string, flagged: boolean) => ({ sentence, n: 1, cites: [1], source: 'a:1-2', score: flagged ? 0.01 : 0.9, flagged, inputTokens: 9, truncated: false });

  it('counts sentences, not sentence–source pairs, and reads right for one', () => {
    expect(supportSummary([item('A b.', false)])).toBe('The cited sentence looks supported by its source');
    expect(supportSummary([item('A b.', false), item('C d.', false)])).toBe('All 2 cited sentences look supported by their sources');
    expect(supportSummary([item('A b.', true), { ...item('A b.', false), n: 2 }, item('C d.', false)])).toBe('1 of 2 cited sentences may not be supported by its source');
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
