// explainCall: the plain sentences the tool call dialog shows.
import { describe, expect, it } from 'vitest';
import type { ToolCallRecord } from '../state/chat';
import { explainCall, explainClaim } from './AgentTrace';

const call = (p: Partial<ToolCallRecord> = {}): ToolCallRecord => ({
  tool: 'kb_search',
  title: 'Knowledge search',
  args: { query: 'group discount', scope: 'focused' },
  argsRaw: null,
  argModel: 'Qwen3 1.7B',
  argAttempts: 1,
  policy: 'auto',
  startedAt: 0,
  status: 'done',
  ...p,
});

describe('explainCall', () => {
  it('says which argument Laya picked, and with what probability', () => {
    const out = explainCall(call({ argChoices: [{ arg: 'scope', value: 'focused', probability: 0.87, model: 'Laya Multilingual' }] }));
    expect(out).toContain('Laya Multilingual picked scope “focused” (87%) as a typed choice, and the chat model kept it.');
  });

  it('says nothing about picks when the chat model wrote every argument', () => {
    expect(explainCall(call()).some((l) => l.includes('typed choice'))).toBe(false);
  });
});

describe('explainClaim', () => {
  const r = { model: 'Laya Multilingual', modelId: 'laya-multilingual', ms: 20, modelMs: 15, flagBelow: 0.1, items: [] };
  const x = { sentence: 'It is 48 hours.', n: 1, cites: [1], source: 'refund-policy.md:3-9', score: 0.62, flagged: false, inputTokens: 80, truncated: false };

  it('says what was asked, the verdict against the cut, and how far the check can be trusted', () => {
    expect(explainClaim(x, r)).toEqual([
      'The agent asked Laya Multilingual whether passage [1] (refund-policy.md:3-9) supports the sentence, as a yes/no question: “The passage supports this statement.”',
      'It answered 62% yes, at or above the 10% cut, so nothing is marked.',
      'How far to trust it: on our test set Laya Multilingual scored a sentence’s own passage above a passage from another answer 67% of the time; at this cut it flagged about 40% of those wrong pairings and wrongly flagged 1 of 60 right ones. It’s a hint to read the source, not a verdict.',
    ]);
  });

  it('says so when the checkpoint was never measured on the claim check', () => {
    expect(explainClaim(x, { ...r, modelId: 'laya-typed-decisions' }).at(-1)).toMatch(/wasn’t measured on the claim check/);
  });
});
