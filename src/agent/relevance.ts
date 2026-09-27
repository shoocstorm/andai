// The relevance check: before retrieved passages go into the chat model's
// prompt, the Laya decision model scores how likely each one helps answer the
// request (a yes/no question per passage, in Rust), and clearly unhelpful
// ones are dropped. A shorter prompt is read faster (reading, not writing, is
// the wait; AGENTS.md §2), and a small model is less distracted by noise.
//
// It's conservative on purpose. Probed on the eval fixtures, Laya separates
// passages that hold the expected fact from the rest only moderately (AUC
// 0.76 multilingual, 0.85 English), so a dropped passage can be a lost fact:
// the top-ranked search results are always kept, only low scores are dropped,
// and a failed check keeps everything. Only with Laya: on the chat model a
// score would cost ~0.7 s per passage.

import type { SearchHit } from '../kb/api';
import { layaRelevance } from '../llm/laya';
import { layaById } from '../llm/models';
import { CHARS_PER_TOKEN } from './prompt';

/** The first `KEEP_TOP` passages are kept whatever they score: search ranking already vouches for them. */
export const KEEP_TOP = 2;
/** Below this likelihood of helping, a passage past the top ones is dropped. */
export const DROP_BELOW = 0.1;
/** What Laya is asked of each passage; a Rust constant (src-tauri/src/laya/mod.rs `RELEVANT`), repeated here for the trace. */
export const RELEVANT = 'This passage contains information that helps answer the user’s request.';
/** What was measured per checkpoint (AGENTS.md §2), shown in the relevance dialog: AUC on the eval fixtures. */
export const MEASURED_RELEVANCE: Record<string, { auc: number }> = { 'laya-multilingual': { auc: 0.76 }, 'laya-en': { auc: 0.85 } };

/** The input one passage is judged in, as Rust builds it (`passage_state`). */
export const passageState = (request: string, source: string, text: string) => `User request:\n${request.trim()}\n\nPassage from ${source}:\n${text.trim()}`;

export type RelevanceItem = {
  file: string;
  start_line: number;
  end_line: number;
  name: string;
  /** P(the passage helps answer the request). */
  score: number;
  kept: boolean;
  /** Why it was kept or dropped. */
  reason: 'top' | 'score' | 'low';
  chars: number;
  /** The passage text Laya read, its input size in tokens, and whether it was cut to fit. */
  text: string;
  inputTokens: number;
  truncated: boolean;
};

export type RelevanceRecord = {
  model: string;
  /** Checkpoint id, for what was measured of it (`MEASURED_RELEVANCE`). */
  modelId: string;
  /** The request each passage was judged against. */
  request: string;
  /** Wall time, IPC included; `modelMs` is Rust's own. */
  ms: number;
  modelMs: number;
  keepTop: number;
  dropBelow: number;
  items: RelevanceItem[];
  /** Approximate prompt tokens the dropped passages would have taken. */
  tokensSaved: number;
};

export const passageText = (h: SearchHit) => (h.snippet ?? h.description ?? '').trim();

/** Which passages to keep, in order, given their scores. Pure: the policy above. */
export function choosePassages(scores: number[], keepTop = KEEP_TOP, dropBelow = DROP_BELOW): { kept: boolean; reason: RelevanceItem['reason'] }[] {
  return scores.map((s, i) => (i < keepTop ? { kept: true, reason: 'top' } : s >= dropBelow ? { kept: true, reason: 'score' } : { kept: false, reason: 'low' }));
}

/**
 * Scores `hits` for `request` and returns the ones to keep, in their order,
 * with the record the trace shows. Nothing to decide with `KEEP_TOP` or fewer.
 */
export async function checkRelevance(request: string, hits: SearchHit[]): Promise<{ hits: SearchHit[]; record: RelevanceRecord } | null> {
  if (hits.length <= KEEP_TOP) return null;
  const started = performance.now();
  const texts = hits.map(passageText);
  const r = await layaRelevance(
    request,
    hits.map((h, i) => ({ source: `${h.file}:${h.start_line}-${h.end_line}`, text: texts[i] })),
  );
  if (r.scores.length !== hits.length || !r.scores.every(Number.isFinite)) throw new Error(`${r.model} returned ${r.scores.length} scores for ${hits.length} passages.`);
  const picks = choosePassages(r.scores);
  const items: RelevanceItem[] = hits.map((h, i) => ({
    file: h.file,
    start_line: h.start_line,
    end_line: h.end_line,
    name: h.name,
    score: r.scores[i],
    kept: picks[i].kept,
    reason: picks[i].reason,
    chars: texts[i].length,
    text: texts[i],
    inputTokens: r.inputTokens[i],
    truncated: r.truncated[i],
  }));
  const dropped = items.filter((x) => !x.kept).reduce((n, x) => n + x.chars, 0);
  return {
    hits: hits.filter((_, i) => picks[i].kept),
    record: {
      model: layaById(r.model)?.name ?? r.model,
      modelId: r.model,
      request,
      ms: Math.round(performance.now() - started),
      modelMs: r.ms,
      keepTop: KEEP_TOP,
      dropBelow: DROP_BELOW,
      items,
      tokensSaved: Math.round(dropped / CHARS_PER_TOKEN),
    },
  };
}
