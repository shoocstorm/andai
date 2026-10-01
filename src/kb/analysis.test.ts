import { describe, expect, it } from 'vitest';
import {
  analysisError,
  analysisText,
  insightsReport,
  onboardingQuestions,
  parseNodeId,
  readAnalysis,
  rowText,
  type Insights,
} from './analysis';

// Envelopes as ug 0.1.22 printed them for the Tidewater code fixtures (tests/fixtures/eval/code).
const untested = {
  title: 'untested_symbols',
  description: 'Source functions no test reaches within 2 hops, most depended-upon first.',
  columns: ['id', 'depended_on_by', 'loc'],
  rows: [
    ['function_declaration:retry.ts:withRetry', 3, 12],
    ['function_declaration:refunds.ts:refundFraction', 2, 5],
  ],
  rowsTotal: 6,
  truncated: false,
  unindexed: [],
  targetNotIndexed: [],
};
const impactMissing = {
  title: 'impact',
  columns: ['file', 'dependents', 'tests'],
  rows: [],
  rowsTotal: 0,
  unindexed: [],
  targetNotIndexed: ['src/fares.ts'],
};
const deadOld = { title: 'dead_code', columns: ['id', 'loc'], rows: [], rowsTotal: 0, unindexed: ['name_mentions'], targetNotIndexed: [] };
const biggest = {
  title: 'biggest_files',
  columns: ['file', 'symbols', 'code_lines'],
  rows: [
    ['fares.ts', 9, 20],
    ['booking.ts', 4, 29],
  ],
  rowsTotal: 5,
};
const fanin = {
  title: 'dependency_fanin',
  columns: ['id', 'depended_on_by', 'loc'],
  rows: [
    ['type:fares.ts:Route', 3, 1],
    ['function_declaration:retry.ts:withRetry', 3, 12],
  ],
};

describe('readAnalysis', () => {
  it('reads the table and counts the rows the limit cut', () => {
    const a = readAnalysis(untested);
    expect(a.columns).toEqual(['id', 'depended_on_by', 'loc']);
    expect(a.rows).toHaveLength(2);
    expect(a.total).toBe(6);
    expect(a.caveats).toEqual([]);
  });

  it('says when an empty result means nothing: a target or a property the index lacks', () => {
    expect(readAnalysis(impactMissing).caveats.join(' ')).toMatch(/src\/fares\.ts isn't in the index/);
    expect(readAnalysis(deadOld).caveats.join(' ')).toMatch(/no name_mentions.*re-index/);
    expect(readAnalysis('{"title": "imp').caveats.join(' ')).toMatch(/cut short/);
  });

  it('tolerates any output shape', () => {
    for (const junk of [null, 42, [], { rows: 'x', columns: [1, {}] }, { rows: [[{}, NaN, 'a']] }])
      expect(() => readAnalysis(junk)).not.toThrow();
    expect(readAnalysis({ rows: [[{}, NaN, 'a']] }).rows[0]).toEqual([null, null, 'a']);
  });
});

describe('parseNodeId', () => {
  it('splits kind, file and name; the file may hold colons', () => {
    expect(parseNodeId('function_declaration:src/a.ts:foo')).toEqual({ kind: 'Function', file: 'src/a.ts', name: 'foo' });
    expect(parseNodeId('method_definition:src/m.ts:SkillManager.index')).toMatchObject({ kind: 'Method', name: 'SkillManager.index' });
    expect(parseNodeId('class:c:/x.py:Thing')).toMatchObject({ kind: 'Class', file: 'c:/x.py' });
    expect(parseNodeId('type:fares.ts:Route')?.kind).toBe('Type');
    expect(parseNodeId('weird:x:y')?.kind).toBe('Node');
    for (const bad of ['', 'foo', ':a:b', 'a::', 'a:b:']) expect(parseNodeId(bad)).toBeNull();
  });
});

describe('analysis as text', () => {
  it('names symbols, labels numbers, and carries the caveats into the passage', () => {
    const a = readAnalysis(untested);
    expect(rowText(a, a.rows[0])).toBe('withRetry (Function, retry.ts) · depended on by 3 · loc 12');
    const text = analysisText(a);
    expect(text).toMatch(/^Analysis: untested_symbols\. Source functions/);
    expect(text).toMatch(/Showing 2 of 6 rows\./);
    expect(analysisText(readAnalysis(impactMissing))).toMatch(/No rows matched\.\nNote: src\/fares\.ts isn't in the index/);
  });

  it('turns a missing `ug analyze` into an update hint', () => {
    expect(analysisError(new Error('unknown command: analyze'))).toMatch(/ug upgrade/);
    expect(analysisError('No project named x')).toBe('No project named x');
  });
});

describe('onboardingQuestions', () => {
  it('asks about the reading order, the most-used function and the biggest file, without repeats', () => {
    const ins: Insights = {
      where_to_start: readAnalysis({ columns: ['id'], rows: [['function_declaration:p.ts:parseSkillMd'], ['class:r.ts:Retriever']] }),
      dependency_fanin: readAnalysis(fanin),
      biggest_files: readAnalysis(biggest),
    };
    expect(onboardingQuestions(ins)).toEqual([
      'How does parseSkillMd work?',
      'How does Retriever work?',
      'Who calls withRetry?',
      'What does fares.ts do?',
    ]);
    expect(onboardingQuestions(ins, 10)).toContain('What would break if I changed fares.ts?');
  });

  it('is empty when nothing qualifies', () => {
    expect(onboardingQuestions({})).toEqual([]);
    expect(onboardingQuestions({ where_to_start: readAnalysis({ columns: ['id'], rows: [] }) })).toEqual([]);
  });
});

describe('insightsReport', () => {
  it('writes each section with its rows and caveats', () => {
    const md = insightsReport('Ferries', { biggest_files: readAnalysis(biggest), dead_code: readAnalysis(deadOld) });
    expect(md).toMatch(/^# Ferries: code insights/);
    expect(md).toMatch(/## biggest files\n\n- fares\.ts · symbols 9 · code lines 20/);
    expect(md).toMatch(/… 3 more/);
    expect(md).toMatch(/## dead code[\s\S]*- none\n> This index has no name_mentions/);
  });
});
