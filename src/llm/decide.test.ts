// The SemIf-style readout with the engine mocked: prompt shape, the request
// that pins the model to the option letters, and the fail-closed checks.
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { MODELS } from './models';

const eng = vi.hoisted(() => ({
  slot: 'decider' as 'chat' | 'decider' | null,
  modelIndex: 1,
  top: [] as { token: string; logprob: number; bytes?: number[] | null }[],
  seen: [] as Record<string, unknown>[],
  fail: null as Error | null,
}));

vi.mock('./engine', async () => {
  const { MODELS } = await import('./models');
  const target = () => (eng.slot ? { slot: eng.slot, def: MODELS[eng.modelIndex] } : null);
  return {
    slotFor: target,
    complete: async (_: string, params: Record<string, unknown>) => {
      eng.seen.push(params);
      if (eng.fail) throw eng.fail;
      return {
        ...target()!,
        response: { choices: [{ logprobs: { content: [{ token: 'A', logprob: 0, top_logprobs: eng.top }] } }], usage: { prompt_tokens: 321 } },
      };
    },
  };
});

const { decide, DecisionError, decisionMessages, labelsFor, optionLogprobs, seededShuffle, softmax } = await import('./decide');

const opts = [
  { id: 'answer_now', text: 'Answer now' },
  { id: 'kb_search', text: 'Search the knowledge base' },
  { id: 'kb_overview', text: 'Overview' },
];

beforeEach(() => {
  eng.slot = 'decider';
  eng.modelIndex = 1;
  eng.seen = [];
  eng.fail = null;
  eng.top = [
    { token: 'B', logprob: Math.log(0.6) },
    { token: 'A', logprob: Math.log(0.3) },
    { token: 'C', logprob: Math.log(0.1) },
  ];
});

describe('pure helpers', () => {
  it('labels options A, B, C…', () => {
    expect(labelsFor(3)).toEqual(['A', 'B', 'C']);
  });

  it('lists every option under its letter and asks for exactly one letter', () => {
    const [sys, user] = decisionMessages('state here', 'Which?', opts);
    expect(sys.role).toBe('system');
    expect(user.content).toContain('State:\nstate here');
    expect(user.content).toContain('A. Answer now\nB. Search the knowledge base\nC. Overview');
    expect(user.content).toContain('Reply with exactly one option letter from: A, B, C.');
  });

  it('softmax is normalized and stable for large logits', () => {
    const p = softmax([1000, 1000, 998]);
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1);
    expect(p[0]).toBeCloseTo(p[1]);
    expect(p[2]).toBeLessThan(p[0]);
  });

  it('matches letters by token text or single byte', () => {
    const res = { choices: [{ logprobs: { content: [{ top_logprobs: [{ token: 'x', logprob: -9, bytes: [65] }, { token: 'B', logprob: -1 }] }] } }] };
    expect(optionLogprobs(res as never, ['A', 'B'])).toEqual({ values: [-9, -1], bounded: [] });
  });

  it('bounds a letter missing from the readout at the lowest listed logprob, and flags it', () => {
    const res = { choices: [{ logprobs: { content: [{ top_logprobs: [{ token: 'A', logprob: -1 }, { token: 'x', logprob: -7 }] }] } }] };
    expect(optionLogprobs(res as never, ['A', 'B'])).toEqual({ values: [-1, -7], bounded: [1] });
  });

  it('fails closed when no option letter was scored', () => {
    const res = { choices: [{ logprobs: { content: [{ top_logprobs: [{ token: 'x', logprob: -1 }] }] } }] };
    expect(() => optionLogprobs(res as never, ['A', 'B'])).toThrow(/none of the options/);
    expect(() => optionLogprobs({ choices: [] } as never, ['A'])).toThrow();
  });

  it('shuffles deterministically per seed and keeps every item', () => {
    const items = Array.from({ length: 8 }, (_, i) => i);
    expect(seededShuffle(items, 42)).toEqual(seededShuffle(items, 42));
    expect(seededShuffle(items, 42)).not.toEqual(seededShuffle(items, 43));
    expect([...seededShuffle(items, 7)].sort()).toEqual(items);
  });
});

describe('decide', () => {
  it('returns every option with its probability and the argmax', async () => {
    const d = await decide('s', 'q', opts);
    expect(d.chosen).toBe('kb_search');
    expect(d.confidence).toBeCloseTo(0.6);
    expect(d.options.map((o) => [o.id, o.label])).toEqual([
      ['answer_now', 'A'],
      ['kb_search', 'B'],
      ['kb_overview', 'C'],
    ]);
    expect(d.options.reduce((a, o) => a + o.probability, 0)).toBeCloseTo(1);
    expect(d).toMatchObject({ model: MODELS[1].name, slot: 'decider', promptTokens: 321 });
  });

  it('asks for one token, restricted to the letters, with neutral sampling', async () => {
    await decide('s', 'q', opts);
    const p = eng.seen[0];
    expect(p).toMatchObject({ max_tokens: 1, temperature: 1, top_k: 0, top_p: 1, logprobs: true });
    expect(p.grammar).toBe('root ::= "A" | "B" | "C"');
    // measured: post_sampling_probs drops top_logprobs from wllama's reply
    expect(p).not.toHaveProperty('post_sampling_probs');
    expect(p.top_logprobs).toBeGreaterThanOrEqual(3);
    expect(p.chat_template_kwargs).toEqual({ enable_thinking: false });
  });

  it('reuses the prompt cache only on a separate decision model', async () => {
    await decide('s', 'q', opts);
    expect(eng.seen[0].cache_prompt).toBe(true);
    eng.slot = 'chat';
    await decide('s', 'q', opts);
    expect(eng.seen[1].cache_prompt).toBe(false);
  });

  it('refuses a model that cannot decide, and a run with no model', async () => {
    eng.modelIndex = MODELS.findIndex((m) => !m.decider);
    await expect(decide('s', 'q', opts)).rejects.toThrow(/can't make decisions/);
    eng.slot = null;
    await expect(decide('s', 'q', opts)).rejects.toThrow(/No model loaded/);
  });

  it('needs between 2 and 16 options', async () => {
    await expect(decide('s', 'q', opts.slice(0, 1))).rejects.toThrow(/2–16/);
    const many = Array.from({ length: 17 }, (_, i) => ({ id: `o${i}`, text: `o${i}` }));
    await expect(decide('s', 'q', many)).rejects.toThrow(/2–16/);
  });

  it('returns the call as sent and the raw readout, for the trace', async () => {
    const d = await decide('state here', 'Which?', opts);
    expect(d.io.request.messages).toEqual(decisionMessages('state here', 'Which?', opts));
    expect(d.io.request.params).toMatchObject({ max_tokens: 1, grammar: 'root ::= "A" | "B" | "C"', top_logprobs: 20 });
    // the abort signal is plumbing, not part of what the model saw
    expect(d.io.request.params).not.toHaveProperty('abortSignal');
    expect(d.io.response).toEqual({ sampled: 'A', topLogprobs: eng.top.map(({ token, logprob }) => ({ token, logprob })) });
  });

  it('keeps the call on the error when the readout fails, so the trace can show it', async () => {
    eng.top = [{ token: 'x', logprob: -1 }];
    const e = await decide('s', 'q', opts).catch((err: unknown) => err);
    expect(e).toBeInstanceOf(DecisionError);
    expect((e as InstanceType<typeof DecisionError>).message).toMatch(/none of the options/);
    expect((e as InstanceType<typeof DecisionError>).io.response?.topLogprobs).toEqual([{ token: 'x', logprob: -1 }]);
    expect((e as InstanceType<typeof DecisionError>).ms).toBeGreaterThanOrEqual(0);
  });

  it('keeps the request when the model call itself fails, with no reply', async () => {
    eng.fail = new Error('context overflow');
    const e = (await decide('s', 'q', opts).catch((err: unknown) => err)) as InstanceType<typeof DecisionError>;
    expect(e).toBeInstanceOf(DecisionError);
    expect(e.message).toBe('context overflow');
    expect(e.io.request.messages).toHaveLength(2);
    expect(e.io.response).toBeNull();
  });
});
