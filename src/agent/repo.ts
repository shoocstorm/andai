// Whole-repo questions: the request plainly asks something only `ug analyze`
// answers (which functions lack tests, what a file's change would reach, the
// biggest files, the most used functions, the languages, a reading order).
// Pure. The loop runs the analysis as its first step without a decision, as
// it searches first for a content question (loop.ts `searchFirst`): with
// search first, Laya's stop question took the search results as enough and
// the analysis was never chosen (agent eval, 2026-10-01, AGENTS.md §2).
//
// Conservative like `needsLookup`: a false match runs one analysis for
// nothing (fast, and the loop goes on), but the rules must not catch a
// question about content, so `repo.test.ts` runs every other eval question
// through them.
import type { AnalyzePreset } from '../kb/api';

export type RepoStep = { tool: 'kb_analyze'; args: { question: AnalyzePreset } } | { tool: 'kb_impact'; args: { file: string } };

const ANALYSES: [RegExp, AnalyzePreset][] = [
  [/\b(untested|no tests?|not (been )?tested|without (any )?tests?|lacks? tests?|missing tests?|test coverage|not covered by (any )?tests?)\b/, 'untested_symbols'],
  [/\b(start reading|where (do|should|can) i (start|begin)|entry points?)\b/, 'where_to_start'],
  [/\b(longest|biggest|largest) (functions?|methods?)\b/, 'long_functions'],
  [
    /\b(biggest|largest) (source )?files?\b|\bfiles?\b[^?.!]{0,30}\b(has|have|holds?|defines?|contains?|declares?) the most (code|symbols|lines|functions|classes)\b/,
    'biggest_files',
  ],
  [/\bmost (used|called|depended[- ]upon|depended on|referenced|imported)\b/, 'dependency_fanin'],
  [/\b(what|which) (programming )?languages?\b/, 'language_breakdown'],
  [/\b(which|what) (folders|modules|packages|directories) depend on (which|each other)\b|\bcoupling between\b/, 'coupling_matrix'],
];

/** "What would change/break/be affected", "who depends on", "which tests to re-run": about a file it names. */
const IMPACT = /\b(affect(s|ed)?|break(s)?|impact(ed)?|blast radius|depends? on|dependents|re-?run|retest)\b/;

/** The knowledge base file the request names: its path, or a basename only one file has. Longest first, so `a/b.ts` beats `b.ts`. */
export function namedFile(text: string, files: string[]): string | null {
  const t = text.toLowerCase();
  const has = (name: string) => new RegExp(`(^|[\\s"'\`(/])${name.toLowerCase().replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}($|[\\s"'\`),.?!:;])`).test(t);
  for (const f of [...files].sort((a, b) => b.length - a.length)) if (has(f)) return f;
  const byBase = new Map<string, string[]>();
  for (const f of files) {
    const base = f.split('/').pop()!;
    byBase.set(base, [...(byBase.get(base) ?? []), f]);
  }
  for (const [base, all] of byBase) if (all.length === 1 && has(base)) return all[0];
  return null;
}

/** The analysis the request plainly asks for, among the tools offered; null when it doesn't. */
export function wholeRepoStep(text: string, files: string[], offered: Set<string>): RepoStep | null {
  const t = text.toLowerCase();
  if (offered.has('kb_impact') && IMPACT.test(t)) {
    const file = namedFile(text, files);
    if (file) return { tool: 'kb_impact', args: { file } };
  }
  if (!offered.has('kb_analyze')) return null;
  const hit = ANALYSES.find(([re]) => re.test(t));
  return hit ? { tool: 'kb_analyze', args: { question: hit[1] } } : null;
}
