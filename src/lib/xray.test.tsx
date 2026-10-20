// The answer X-ray finds the claim check's verdict by the sentence as Markdown
// renders it, so the two must split and normalize sentences the same way.
import { render } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { citedClaims, FLAG_BELOW, type SupportRecord } from '../agent/claims';
import { Markdown } from '../components/ui';
import { xrayIndex } from './xray';

const record = (answer: string, sources: number, score = (_i: number) => 0.9): SupportRecord => ({
  model: 'Laya English',
  modelId: 'laya-en',
  ms: 10,
  modelMs: 8,
  flagBelow: FLAG_BELOW,
  items: citedClaims(answer, sources).map((c, i) => ({
    ...c,
    source: `f${c.n}.md:1-4`,
    score: score(i),
    flagged: score(i) < FLAG_BELOW,
    inputTokens: 40,
    truncated: false,
  })),
});

/** The text of every sentence `Markdown` hands its `sentence` renderer. */
function renderedSentences(answer: string, sources: number): string[] {
  const seen: string[] = [];
  render(
    <Markdown
      text={answer}
      sources={sources}
      onCite={() => {}}
      sentence={(text, children) => {
        seen.push(text);
        return children;
      }}
    />,
  );
  return seen;
}

describe('answer X-ray', () => {
  it('finds every claim the check scored among the rendered sentences', () => {
    const answer = [
      'The peak multiplier is **1.25** [3]. It applies on Friday and Sunday [1][2], per `PEAK_DAYS`.',
      '',
      '- Cars pay a surcharge [1, 2].',
      '- Bikes ride free on the',
      '  *morning* boats [2].',
      '',
      'Boarding closes ten minutes before departure. [1]',
      '',
      '| Fare | Note |',
      '|---|---|',
      '| Adult | Pays the full fare [3] |',
      '',
      '```ts',
      'const x = 1; // not a claim [1]',
      '```',
    ].join('\n');
    const r = record(answer, 3);
    const lookup = xrayIndex(r);
    const checked = renderedSentences(answer, 3)
      .map(lookup)
      .flatMap((v) => (v?.kind === 'checked' ? v.items.map((x) => `${x.n} ${x.sentence}`) : []));
    expect(new Set(checked)).toEqual(new Set(r.items.map((x) => `${x.n} ${x.sentence}`)));
  });

  it('marks a sentence by its weakest source: flagged when any source may not support it', () => {
    const answer = 'Cars pay 18.5 [1][2]. Use COOP [1].';
    const lookup = xrayIndex(record(answer, 2, (i) => [0.8, 0.04, 0.93][i]));
    expect(lookup('Cars pay 18.5 [1][2].')).toMatchObject({
      kind: 'checked',
      flagged: true,
      worst: { n: 2, score: 0.04 },
    });
    expect(lookup('Use COOP [1].')).toMatchObject({
      kind: 'checked',
      flagged: false,
      worst: { n: 1, score: 0.93 },
    });
  });

  it('tells a cited sentence the check has no score for from a statement with no citation', () => {
    const lookup = xrayIndex(record('Use COOP [1].', 1));
    expect(lookup('Cars pay 18.5 [2].')).toEqual({ kind: 'cited' });
    expect(lookup('Cars pay a surcharge on weekends.')).toEqual({
      kind: 'uncited',
    });
  });

  it('marks nothing on a fragment, a bare citation, or a sentence about what the sources lack', () => {
    const lookup = xrayIndex(record('Use COOP [1].', 1));
    expect(lookup('[1]')).toBeNull();
    expect(lookup('Yes.')).toBeNull();
    expect(lookup('The provided sources do not mention the CEO.')).toBeNull();
  });

  it('splits only text blocks: headings and code are left whole', () => {
    expect(renderedSentences('## Fares and rules [1]\n\nOne. Two [1].\n\n```\nA. B.\n```', 1)).toEqual(['One.', 'Two [1].']);
  });
});
