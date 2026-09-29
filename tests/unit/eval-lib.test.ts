// The agent eval's scoring (scripts/eval-lib.mjs) and its question set. The
// eval itself needs a model and ug (bun run eval:agent); what it counts must
// be right without one.
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
// @ts-expect-error — plain ESM script without type declarations
import { diffCases, factRegexes, loadCases, scoreCase, scorecard, scorecardByKb } from '../../scripts/eval-lib.mjs';

const { notFound, cases } = loadCases(resolve(import.meta.dirname, '../fixtures/eval/cases.json'));

type Step = { action: string; fallback?: string | null; decision?: { ms: number; promptTokens: number | null; truncated?: boolean } | null; call?: Record<string, unknown> | null };
const record = (over: { steps?: Step[]; answer?: string; sources?: string[]; error?: string | null; ms?: number; relevance?: unknown } = {}) => ({
  id: 'q',
  ms: 2000,
  error: null,
  sources: ['a.md:1-5'],
  answer: 'It is 48 hours [1].',
  ...over,
  steps: (over.steps ?? [{ action: 'kb_search' }, { action: 'answer_now' }]).map((s) => ({
    fallback: null,
    decision: { ms: 600, promptTokens: 300 },
    call: null,
    ...s,
  })),
});
const q = { id: 'q', kb: 'docs', prompt: '?', first: ['kb_search'], facts: ['48'] };
const call = (over: Record<string, unknown> = {}) => ({ tool: 'kb_search', status: 'done', args: { query: 'x' }, argsFallback: false, hits: 2, error: null, observation: '', ...over });

describe('agent eval question set', () => {
  it('has 20–110 questions with unique ids, a known KB, expected first actions and valid regexes', () => {
    // Items add questions that exercise what they change (docs/agentic-rag-improvements.md);
    // 100 questions tell a 1–2 question change from noise better than 45 did,
    // at about 2 minutes on the native engine with Laya (about 1 s a question)
    // and 9 minutes with wllama Qwen3 0.6B; past 110 it stops being run before
    // every change. Use --only for a quick look.
    expect(cases.length).toBeGreaterThanOrEqual(20);
    expect(cases.length).toBeLessThanOrEqual(110);
    expect(new Set(cases.map((c: { id: string }) => c.id)).size).toBe(cases.length);
    for (const c of cases) {
      expect(['docs', 'code', 'mixed', 'large']).toContain(c.kb);
      expect(c.first.length).toBeGreaterThan(0);
      expect(() => factRegexes(c, notFound)).not.toThrow();
    }
  });

  it('covers documents, code, mixed, long documents, small talk, follow-ups and an unanswerable question', () => {
    const kbs = new Set(cases.map((c: { kb: string }) => c.kb));
    expect([...kbs].sort()).toEqual(['code', 'docs', 'large', 'mixed']);
    expect(cases.some((c: { first: string[] }) => c.first.includes('answer_now'))).toBe(true);
    expect(cases.some((c: { history?: unknown[] }) => c.history?.length)).toBe(true);
    expect(cases.some((c: { facts: string[] }) => c.facts.includes('$notFound'))).toBe(true);
  });

  it('recognizes an answer that says the knowledge base lacks it', () => {
    const re = new RegExp(notFound, 'i');
    expect(re.test('The knowledge base does not mention the CEO.')).toBe(true);
    expect(re.test('The passages don’t say; there is no information about the CEO.')).toBe(true);
    expect(re.test('The CEO is Maria Lind.')).toBe(false);
  });
});

describe('scoreCase', () => {
  it('scores the first action taken and the facts in the answer, ignoring the think block', () => {
    const s = scoreCase(q, record({ answer: '<think>48 hours</think>You need to cancel 2 days early [1].' }), notFound);
    expect(s).toMatchObject({ first: 'kb_search', firstOk: true, factsOk: false, missing: ['48'] });
    expect(scoreCase(q, record(), notFound)).toMatchObject({ factsOk: true, grounded: true });
  });

  it('has no fact verdict when a question has no facts, and fails the facts when the turn errored', () => {
    expect(scoreCase({ ...q, facts: [] }, record(), notFound).factsOk).toBeNull();
    expect(scoreCase(q, record({ error: 'No model is loaded.' }), notFound)).toMatchObject({ factsOk: false, error: 'No model is loaded.' });
  });

  it('counts a citation outside the listed sources as ungrounded, and no sources as not applicable', () => {
    expect(scoreCase(q, record({ answer: '48 hours [3].' }), notFound).grounded).toBe(false);
    expect(scoreCase(q, record({ answer: '48 hours.' }), notFound).grounded).toBe(false);
    expect(scoreCase(q, record({ sources: [] }), notFound).grounded).toBeNull();
  });

  it('counts wasted calls, invalid arguments, unknown symbols and fallbacks', () => {
    const s = scoreCase(
      q,
      record({
        steps: [
          { action: 'kb_search', fallback: 'low-confidence', call: call({ argsFallback: true }) },
          { action: 'kb_get_code', call: call({ tool: 'kb_get_code', status: 'error', error: 'No symbol named “fare”, try find_symbols' }) },
          { action: 'kb_file_context', call: call({ tool: 'kb_file_context', status: 'error', args: null, error: 'Could not produce valid arguments' }) },
          { action: 'kb_overview', call: call({ tool: 'kb_overview', hits: 0 }) },
          { action: 'kb_overview', call: call({ tool: 'kb_overview', status: 'skipped' }) },
          { action: 'answer_now', decision: null },
        ],
      }),
      notFound,
    );
    expect(s).toMatchObject({
      calls: 5,
      wasted: 4,
      wastedBy: { errors: 2, empty: 1, skipped: 1 },
      argsInvalid: 2,
      noSymbol: 1,
      fallbacks: 1,
      decisions: 5,
    });
  });
});

describe('scorecard', () => {
  it('averages over the questions each rate applies to', () => {
    const a = scoreCase(q, record(), notFound);
    const b = scoreCase({ ...q, id: 'b', first: ['answer_now'], facts: [] }, record({ steps: [{ action: 'answer_now' }], sources: [], ms: 1000 }), notFound);
    const c = scoreCase({ ...q, id: 'c', first: ['kb_overview'] }, record({ answer: 'no idea [1]' }), notFound);
    expect(scorecard([a, b, c])).toMatchObject({
      questions: 3,
      firstActionAccuracy: 2 / 3,
      factHitRate: 1 / 2,
      groundedRate: 1,
      decisionsPerQuestion: 5 / 3,
      secondsPerQuestion: 5 / 3,
      msPerDecision: 600,
      promptTokensPerDecision: 300,
    });
  });
});

describe('Laya input cuts and the per-KB scorecard', () => {
  it('counts decisions and scored passages that were cut to fit, and splits the scorecard by knowledge base', () => {
    const cut = { truncated: true };
    const a = scoreCase(
      q,
      record({
        steps: [
          { action: 'kb_search', decision: { ms: 10, promptTokens: 512, ...cut }, call: { tool: 'kb_search', status: 'done', hits: 2, scored: { items: [cut, { truncated: false }] } } },
          { action: 'answer_now', decision: { ms: 10, promptTokens: 300 } },
        ],
        relevance: { items: [cut] },
      }),
      notFound,
    );
    expect(a).toMatchObject({ decisionsCut: 1, passagesCut: 2 });
    const b = scoreCase({ ...q, id: 'b', kb: 'large' }, record(), notFound);
    const card = scorecard([a, b]);
    expect(card).toMatchObject({ decisionsCut: 1, passagesCut: 2, maxPromptTokensPerDecision: 512 });
    const by = scorecardByKb([a, b]);
    expect(Object.keys(by)).toEqual(['docs', 'large']);
    expect(by.large).toMatchObject({ questions: 1, decisionsCut: 0 });
  });
});

describe('diffCases', () => {
  it('lists the questions whose first action, facts or actions changed', () => {
    const a = scoreCase(q, record(), notFound);
    const b = scoreCase(q, record({ steps: [{ action: 'answer_now' }], answer: 'no' }), notFound);
    expect(diffCases([a], [a])).toEqual([]);
    expect(diffCases([a], [b])[0].change).toMatch(/first kb_search → answer_now; facts true → false/);
  });
});
