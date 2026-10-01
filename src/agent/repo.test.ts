import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { namedFile, wholeRepoStep } from './repo';

const files = ['booking.ts', 'fares.ts', 'fares.test.ts', 'refunds.ts', 'retry.ts', 'refund-policy.md'];
const both = new Set(['kb_analyze', 'kb_impact']);
const step = (q: string, offered = both) => wholeRepoStep(q, files, offered);

describe('wholeRepoStep', () => {
  it('runs the analysis a whole-repo question plainly asks for', () => {
    expect(step('Which functions have no tests?')).toEqual({ tool: 'kb_analyze', args: { question: 'untested_symbols' } });
    expect(step('Which file in this codebase defines the most symbols?')?.args).toEqual({ question: 'biggest_files' });
    expect(step('What are the largest files?')?.args).toEqual({ question: 'biggest_files' });
    expect(step('What is the most used function?')?.args).toEqual({ question: 'dependency_fanin' });
    expect(step('What programming language is the booking code written in?')?.args).toEqual({ question: 'language_breakdown' });
    expect(step('Where should I start reading this code?')?.args).toEqual({ question: 'where_to_start' });
    expect(step('Show me the longest functions')?.args).toEqual({ question: 'long_functions' });
    expect(step('Which modules depend on each other?')?.args).toEqual({ question: 'coupling_matrix' });
  });

  it('traces a named file’s blast radius', () => {
    expect(step('What would be affected if I changed fares.ts?')).toEqual({ tool: 'kb_impact', args: { file: 'fares.ts' } });
    expect(step('Which tests should I re-run after changing fares.ts?')?.args).toEqual({ file: 'fares.ts' });
    expect(step('Which files depend on retry.ts?')?.args).toEqual({ file: 'retry.ts' });
    // No file named: not an impact question it can run.
    expect(step('What would break if I changed the fare rules?')).toBeNull();
  });

  it('runs only tools that are offered', () => {
    expect(step('Which functions have no tests?', new Set(['kb_search']))).toBeNull();
    expect(step('Which files depend on retry.ts?', new Set(['kb_analyze']))).toBeNull();
  });

  it('leaves every other eval question to search and the decision', () => {
    const { cases } = JSON.parse(readFileSync(join(__dirname, '../../tests/fixtures/eval/cases.json'), 'utf8')) as {
      cases: { id: string; prompt: string }[];
    };
    const caught = cases.filter((c) => !/-repo-/.test(c.id) && step(c.prompt)).map((c) => c.prompt);
    expect(caught).toEqual([]);
    for (const q of ['hi', 'Which functions call computeFare?', 'Show me the full source code of withRetry.', 'What does the refund policy say?'])
      expect(step(q), q).toBeNull();
  });
});

describe('namedFile', () => {
  it('finds a path or a unique basename, not part of a longer name', () => {
    expect(namedFile('what uses src/util/a.ts?', ['src/util/a.ts', 'lib/a.ts'])).toBe('src/util/a.ts');
    expect(namedFile('what uses a.ts?', ['src/util/a.ts', 'lib/a.ts'])).toBeNull();
    expect(namedFile('what uses db.ts?', ['src/modules/db.ts'])).toBe('src/modules/db.ts');
    expect(namedFile('changes to fares.test.ts', files)).toBe('fares.test.ts');
    expect(namedFile('changes to myfares.ts', files)).toBeNull();
  });
});
