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
/** What Laya is asked of each claim; a Rust constant (src-tauri/src/laya/mod.rs `SUPPORTS`), repeated here for the trace. */
export const SUPPORTS = 'The passage supports this statement.';

/**
 * What the probe measured per checkpoint (docs/agentic-rag-improvements.md,
 * item 13), shown with each verdict so a user can weigh it: how often the
 * sentence's own passage outscored another question's (AUC), the share of
 * such wrong pairings flagged at `FLAG_BELOW`, and the clear false alarms.
 */
export const MEASURED: Record<string, { auc: number; caught: number; falseAlarms: string }> = {
  'laya-multilingual': { auc: 0.67, caught: 0.4, falseAlarms: '1 of 60' },
  'laya-en': { auc: 0.82, caught: 0.6, falseAlarms: '2 of 60' },
};

/** The input one claim is judged in, as Rust builds it (`claim_state`). */
export const claimState = (statement: string, source: string, text: string) => `Statement:\n${statement.trim()}\n\nPassage from ${source}:\n${text.trim()}`;
/** Claims per check; Rust accepts up to 24 (src-tauri/src/laya/mod.rs `MAX_CLAIMS`). */
export const MAX_CLAIMS = 24;
/** Longer sentences are cut: 500 characters are at most 2 KB in UTF-8, Rust's bound (`MAX_STATEMENT`). */
const MAX_SENTENCE = 500;

export type Claim = {
  /** The sentence as the model wrote it, without Markdown or citation marks. */
  sentence: string;
  /** The source number checked here, `[n]`. */
  n: number;
  /** Every source the sentence cites; each is checked on its own. */
  cites: number[];
};

export type SupportItem = Claim & {
  /** `file:start-end` of the passage it was checked against. */
  source: string;
  /** P(the cited passage supports the sentence). */
  score: number;
  flagged: boolean;
  /** Laya's input for this claim, in tokens, and whether it was cut to fit. */
  inputTokens: number;
  truncated: boolean;
};

export type SupportRecord = {
  model: string;
  /** Checkpoint id, for what was measured of it (`MEASURED`). */
  modelId: string;
  /** Wall time, IPC included; `modelMs` is Rust's own. */
  ms: number;
  modelMs: number;
  flagBelow: number;
  items: SupportItem[];
};

const CITE = /\[(\d+(?:\s*,\s*\d+)*)\]/g;

/**
 * Uncited sentences a citation-only fragment reaches back to: "… terminal. [6]"
 * splits into a sentence and a bare "[6]", and a list often ends in a line of
 * citations; either way they cite what comes just before them.
 */
const REACH_BACK = 3;

/** A sentence without Markdown, inline-code or citation marks. */
const plain = (raw: string) =>
  raw
    .replace(CITE, '')
    .replace(/`([^`]*)`/g, '$1')
    .replace(/^\s*(?:[-*+]|\d+\.|#+|>)\s+/, '')
    .replace(/[*_]{1,3}([^*_]+)[*_]{1,3}/g, '$1')
    .replace(/\s+([.,;:!?])/g, '$1')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, MAX_SENTENCE);

/**
 * A sentence about what the sources lack ("The information provided does not
 * mention the CEO [1][2]") can't be supported by any one passage, so it isn't
 * checked: in the eval every such sentence scored ~0 and read as a false
 * alarm, although the answer was right to say so.
 */
export const aboutMissing = (sentence: string) =>
  /\b(the|these|this|provided|given|available)\s+(information|context|passages?|sources?|documents?|knowledge base|search results)\b/i.test(sentence) &&
  /\b(not|no|nothing|doesn['’]t|don['’]t|isn['’]t|aren['’]t|lacks?|without)\b/i.test(sentence);

/** Two words at least: ", , and [2]." or a bare "[6]" leaves nothing to check. */
const checkable = (sentence: string) => (sentence.match(/[\p{L}\p{N}]+/gu) ?? []).length >= 2;

/**
 * The cited sentences of an answer, one per sentence and source it cites, in
 * order, for sources 1…`sources`. A citation standing on its own after a
 * sentence or list ("… terminal. [6]") cites the uncited sentences just
 * before it. Code blocks are skipped. Pure.
 */
export function citedClaims(answer: string, sources: number): Claim[] {
  const text = answer.replace(/```[\s\S]*?(```|$)/g, '\n');
  const cited: { sentence: string; ns: number[] }[] = [];
  let pending: { sentence: string; ns: number[] }[] = [];
  for (const raw of text.split(/(?<=[.!?])\s+|\n+/)) {
    const ns = [...raw.matchAll(CITE)].flatMap((m) => m[1].split(',').map((x) => Number(x.trim())));
    const sentence = plain(raw);
    if (checkable(sentence)) {
      const item = { sentence, ns };
      cited.push(item);
      pending = ns.length ? [] : [...pending, item].slice(-REACH_BACK);
    } else if (ns.length) {
      for (const p of pending) p.ns.push(...ns);
      pending = [];
    }
  }
  const out: Claim[] = [];
  const seen = new Set<string>();
  for (const { sentence, ns } of cited) {
    if (aboutMissing(sentence)) continue;
    const cites = [...new Set(ns.filter((n) => n >= 1 && n <= sources))];
    for (const n of cites) {
      const key = `${n} ${sentence}`;
      if (seen.has(key)) continue;
      seen.add(key);
      out.push({ sentence, n, cites });
    }
  }
  return out.slice(0, MAX_CLAIMS);
}

/** One line for the trace step: how many cited sentences were checked, and how many may not be supported. */
export function supportSummary(items: SupportItem[]): string {
  // A sentence citing two sources is two items; count sentences.
  const n = new Set(items.map((x) => x.sentence)).size;
  const flagged = new Set(items.filter((x) => x.flagged).map((x) => x.sentence)).size;
  if (flagged) return `${flagged} of ${n} cited sentence${n === 1 ? '' : 's'} may not be supported by ${flagged === 1 ? 'its' : 'their'} source`;
  return n === 1 ? 'The cited sentence looks supported by its source' : `All ${n} cited sentences look supported by their sources`;
}

export const sourceOf = (h: SearchHit) => `${h.file}:${h.start_line}-${h.end_line}`;

/**
 * Scores the answer's cited sentences against `sources` (numbered as the
 * answer cites them). Null when nothing is cited.
 */
export async function checkClaims(answer: string, sources: SearchHit[]): Promise<SupportRecord | null> {
  const claims = citedClaims(answer, sources.length);
  if (!claims.length) return null;
  const started = performance.now();
  const sent = claims.map((c) => {
    const h = sources[c.n - 1];
    return { statement: c.sentence, source: sourceOf(h), text: passageText(h) };
  });
  const r = await layaSupport(sent);
  if (r.scores.length !== claims.length || !r.scores.every(Number.isFinite)) throw new Error(`${r.model} returned ${r.scores.length} scores for ${claims.length} claims.`);
  return {
    model: layaById(r.model)?.name ?? r.model,
    modelId: r.model,
    ms: Math.round(performance.now() - started),
    modelMs: r.ms,
    flagBelow: FLAG_BELOW,
    items: claims.map((c, i) => ({
      ...c,
      source: sent[i].source,
      score: r.scores[i],
      flagged: r.scores[i] < FLAG_BELOW,
      inputTokens: r.inputTokens[i],
      truncated: r.truncated[i],
    })),
  };
}
