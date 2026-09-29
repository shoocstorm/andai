// The evidence plan (docs/agentic-rag-improvements.md, item 16): with Laya
// scoring every passage, most next steps don't need a model to choose them.
// Search is the only tool whose output is uncertain; the others return what
// they're asked for. So code plans the follow-ups from what was found and
// how it scored, Laya judges (its per-passage scores), and the chat model
// only writes free text (search queries, the answer). Pure, so every rule is
// unit-tested (plan.test.ts); loop.ts runs the steps.

import type { SearchHit } from '../kb/api';
import { shownLines } from './relevance';
import { SYMBOL_TYPES } from './tools/ug';

/** A passage found so far, the tool that found it, and Laya's score for it. */
export type Scored = { hit: SearchHit; tool: string; score: number | null };

export type PlanInput = {
  prompt: string;
  kind: 'document' | 'code' | 'mixed';
  found: Scored[];
  /** Tool ids that can run now (policy not Off, not denied). */
  can: Set<string>;
  /** Calls already made this turn, as `tool args` keys (loop.ts `argKey`), so a step is never repeated. */
  done: (tool: string, args: Record<string, unknown>) => boolean;
  /** The arguments of the knowledge searches made this turn, in order. */
  searches: Record<string, unknown>[];
  /** Search once more when nothing found scores as helping (`searchAgainStep`). */
  searchAgain?: boolean;
};

/** A step the plan takes: a tool with its arguments, or answer; and why, for the trace. */
export type PlanStep = { action: string; args?: Record<string, unknown>; note: string };

/** A clipped passage scoring at least this is read whole. */
export const PLAN_READ_AT = 0.3;
/** Answer once some passage scores at least this and nothing clipped is left to read. */
export const PLAN_ENOUGH_AT = 0.5;
/** Passages read whole in one turn, at most. */
export const PLAN_MAX_READS = 2;
/** Knowledge searches in one turn, at most, counting the one that searched again. */
export const PLAN_MAX_SEARCHES = 2;

/** The request asks who calls or uses something. */
export const CALLERS = /\b(who|which|what)\b[^?]*\b(calls?|callers?|uses|usages?|references?|invokes?)\b|\bcallers? of\b|\bwhere is\b[^?]*\b(called|used)\b/i;

/**
 * Identifiers the request names, in order: backticked text, camelCase,
 * PascalCase with an inner capital, snake_case, or a name followed by `()`
 * (`computeFare`, `BookingService`, `refund_fraction`, `withRetry()`).
 */
export function namedIdentifiers(text: string): string[] {
  const out: string[] = [];
  for (const m of text.matchAll(/`([^`\s]+)`|([A-Za-z_][A-Za-z0-9_]*)(\(\))?/g)) {
    const word = (m[1] ?? m[2]).replace(/\(\)$/, '');
    if (m[1] || m[3] || /[a-z][A-Z]|^[A-Z][a-z0-9]+[A-Z]|[A-Za-z0-9]_[A-Za-z0-9]/.test(word)) out.push(word);
  }
  return [...new Set(out)];
}

const pct = (x: number) => `${Math.round(x * 100)}%`;
const range = (h: SearchHit) => `${h.file}:${h.start_line}-${h.end_line}`;

/**
 * The next step the evidence calls for, or null when no rule applies (the
 * loop then asks the decision model, or answers). In order:
 * 1. read whole the best-scoring passage a tool clipped (≥ `PLAN_READ_AT`);
 * 2. on code, fetch a symbol the request names once the results show it:
 *    its callers when the request asks who calls it, else its source;
 * 3. answer when a passage scores ≥ `PLAN_ENOUGH_AT`;
 * 4. with `searchAgain`, when nothing does: search once more, with the
 *    question as written and the other scope (a rewritten query or the wrong
 *    scope is the likeliest reason a search missed).
 */
export function planNext(p: PlanInput): PlanStep | null {
  const reads = p.found.filter((f) => f.tool === 'kb_read_lines').length;
  const clipped = p.found
    .filter((f) => f.tool !== 'kb_read_lines' && f.score != null && f.score >= PLAN_READ_AT && shownLines(f.hit) && f.hit.end_line - f.hit.start_line < 400)
    .sort((a, b) => b.score! - a.score!)
    .find((f) => !p.done('kb_read_lines', { range: range(f.hit) }));
  if (clipped && reads < PLAN_MAX_READS && p.can.has('kb_read_lines')) {
    const shows = shownLines(clipped.hit)!;
    return {
      action: 'kb_read_lines',
      args: { range: range(clipped.hit) },
      note: `Planned: “${clipped.hit.name || clipped.hit.file}” is likely to help (${pct(clipped.score!)}), but the search showed ${shows.lines} of its ${shows.of} lines, so reading it whole.`,
    };
  }

  if (p.kind !== 'document') {
    const seen = new Set(p.found.filter((f) => SYMBOL_TYPES.has(f.hit.node_type) && f.hit.name).map((f) => f.hit.name));
    const callers = CALLERS.test(p.prompt);
    for (const name of namedIdentifiers(p.prompt).filter((n) => seen.has(n))) {
      const tool = callers ? 'kb_find_usages' : 'kb_get_code';
      if (!p.can.has(tool) || p.done(tool, { symbol: name })) continue;
      return {
        action: tool,
        args: { symbol: name },
        note: callers
          ? `Planned: the request asks who uses \`${name}\`, and the results show it, so finding its usages.`
          : `Planned: the request names \`${name}\`, and the results show it, so reading its source.`,
      };
    }
  }

  const best = p.found.reduce<Scored | null>((a, f) => (f.score != null && (!a || f.score > a.score!) ? f : a), null);
  if (best && best.score! >= PLAN_ENOUGH_AT) {
    return { action: 'answer_now', note: `Planned: “${best.hit.name || best.hit.file}” is likely to help (${pct(best.score!)}), and nothing clipped is left to read, so answering.` };
  }
  return p.searchAgain ? searchAgainStep(p) : null;
}

/**
 * Search once more when nothing found so far scores ≥ `PLAN_ENOUGH_AT`: with
 * the question as written and the other scope, the likeliest reasons a
 * search missed being a rewritten query or the wrong scope. Once a turn
 * (`PLAN_MAX_SEARCHES`), and never the same call twice. Also used without the
 * plan (loop.ts, the `searchAgain` setting).
 */
export function searchAgainStep(p: PlanInput): PlanStep | null {
  const scores = p.found.flatMap((f) => (f.score == null ? [] : [f.score]));
  const best = scores.length ? Math.max(...scores) : null;
  if (best != null && best >= PLAN_ENOUGH_AT) return null;
  if (!p.searches.length || p.searches.length >= PLAN_MAX_SEARCHES || !p.can.has('kb_search')) return null;
  const last = p.searches.at(-1)!;
  const args = { query: p.prompt.trim().slice(0, 300), scope: last.scope === 'broad' ? 'focused' : 'broad' };
  if (p.done('kb_search', args)) return null;
  return {
    action: 'kb_search',
    args,
    note: `Nothing found so far clearly helps (the best scored ${best != null ? pct(best) : 'nothing'}), so searching again with the question as written and a ${args.scope} scope.`,
  };
}
