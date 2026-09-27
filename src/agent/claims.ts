// The claim check: after the answer is written, the Laya decision model scores
// whether each cited sentence is supported by the passage it cites (a yes/no
// question per sentence and citation, in Rust), and sentences that probably
// aren't are shown under the answer. It never changes or holds up the answer.
//
// Probed on the item 10 eval answers (60 cited sentences, each against its own
// passage and one cited for another question; docs/agentic-rag-improvements.md,
// item 13): AUC 0.67 multilingual, 0.82 English. Below 0.10, 1–2 of 60 own
// citations were clear false alarms; the other flags were real gaps (a release
// note cited for a function name it never mentions). So a flag says "may not
// be supported", never "wrong".

import type { SearchHit } from '../kb/api';
import { layaSupport } from '../llm/laya';
import { layaById } from '../llm/models';
import { passageText } from './relevance';

/** Below this likelihood of support, a cited sentence is flagged. */
export const FLAG_BELOW = 0.1;
/** Claims per check; Rust accepts up to 24 (src-tauri/src/laya/mod.rs `MAX_CLAIMS`). */
export const MAX_CLAIMS = 24;
/** Longer sentences are cut: 500 characters are at most 2 KB in UTF-8, Rust's bound (`MAX_STATEMENT`). */
const MAX_SENTENCE = 500;

export type Claim = {
  /** The sentence as the model wrote it, without Markdown or citation marks. */
  sentence: string;
  /** The source number it cites, `[n]`. */
  n: number;
};

export type SupportItem = Claim & {
  /** P(the cited passage supports the sentence). */
  score: number;
  flagged: boolean;
};

export type SupportRecord = {
  model: string;
  /** Wall time, IPC included; `modelMs` is Rust's own. */
  ms: number;
  modelMs: number;
  flagBelow: number;
  items: SupportItem[];
};

const CITE = /\[(\d+(?:\s*,\s*\d+)*)\]/g;

/**
 * The cited sentences of an answer, one per sentence and source it cites, in
 * order, for sources 1…`sources`. Code blocks are skipped, and Markdown and
 * inline-code marks removed. Pure.
 */
export function citedClaims(answer: string, sources: number): Claim[] {
  const text = answer.replace(/```[\s\S]*?(```|$)/g, '\n');
  const out: Claim[] = [];
  const seen = new Set<string>();
  for (const raw of text.split(/(?<=[.!?])\s+|\n+/)) {
    const ns = [...raw.matchAll(CITE)].flatMap((m) => m[1].split(',').map((x) => Number(x.trim())));
    if (!ns.length) continue;
    const sentence = raw
      .replace(CITE, '')
      .replace(/`([^`]*)`/g, '$1')
      .replace(/^\s*(?:[-*+]|\d+\.|#+|>)\s+/, '')
      .replace(/[*_]{1,3}([^*_]+)[*_]{1,3}/g, '$1')
      .replace(/\s+([.,;:!?])/g, '$1')
      .replace(/\s+/g, ' ')
      .trim()
      .slice(0, MAX_SENTENCE);
    // A sentence left with fewer than two words once the citations are removed (", , and [2].") can't be checked.
    if ((sentence.match(/[\p{L}\p{N}]+/gu) ?? []).length < 2) continue;
    for (const n of ns) {
      const key = `${n} ${sentence}`;
      if (n < 1 || n > sources || seen.has(key)) continue;
      seen.add(key);
      out.push({ sentence, n });
    }
  }
  return out.slice(0, MAX_CLAIMS);
}

/**
 * Scores the answer's cited sentences against `sources` (numbered as the
 * answer cites them). Null when nothing is cited.
 */
export async function checkClaims(answer: string, sources: SearchHit[]): Promise<SupportRecord | null> {
  const claims = citedClaims(answer, sources.length);
  if (!claims.length) return null;
  const started = performance.now();
  const r = await layaSupport(
    claims.map((c) => {
      const h = sources[c.n - 1];
      return { statement: c.sentence, source: `${h.file}:${h.start_line}-${h.end_line}`, text: passageText(h) };
    }),
  );
  if (r.scores.length !== claims.length || !r.scores.every(Number.isFinite)) throw new Error(`${r.model} returned ${r.scores.length} scores for ${claims.length} claims.`);
  return {
    model: layaById(r.model)?.name ?? r.model,
    ms: Math.round(performance.now() - started),
    modelMs: r.ms,
    flagBelow: FLAG_BELOW,
    items: claims.map((c, i) => ({ ...c, score: r.scores[i], flagged: r.scores[i] < FLAG_BELOW })),
  };
}
