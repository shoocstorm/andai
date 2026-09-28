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
import { CHARS_PER_TOKEN, LAYA_CHARS_PER_TOKEN, squeeze } from './prompt';

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
  /** Pieces it was scored in, when it was too long for one row; its score is the best piece's. */
  chunks?: number;
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

// ── Scoring passages as they arrive ─────────────────────────────────────
// The agent loop scores each passage when a tool returns it, so the next
// decision can read the most useful ones (loop.ts), and the check before
// the answer reuses the scores. Each passage is its own row with the
// request, so every one is read whole, however much the tools returned in
// all: a passage longer than a row is split into overlapping pieces and
// scored by its best one, which is also the text the decision reads.

/** A passage's score: its best piece's, and that piece. */
export type PassageScore = { score: number; best: string; chunks: number; inputTokens: number; truncated: boolean };

/** Rows per `laya_relevance` call (src-tauri/src/laya/mod.rs `MAX_PASSAGES`). */
const MAX_ROWS = 24;
/** Tokens a relevance row's question and its yes/no options take, with room to spare (about 40 measured). */
const ROW_HEAD_TOKENS = 64;
/** At most this share of a row goes to the request; a long one is squeezed (`squeeze`). */
const ROW_REQUEST_SHARE = 0.3;

/**
 * Splits `text` into pieces of at most `size` characters that overlap by
 * about a sixth, at line breaks where there are any, so a sentence cut by
 * one piece is whole in the next.
 */
export function chunkText(text: string, size: number): string[] {
  if (text.length <= size) return [text];
  const overlap = Math.floor(size / 6);
  const out: string[] = [];
  let start = 0;
  while (start < text.length) {
    let end = Math.min(text.length, start + size);
    if (end < text.length) {
      const nl = text.lastIndexOf('\n', end);
      if (nl > start + size / 2) end = nl;
    }
    out.push(text.slice(start, end));
    if (end >= text.length) break;
    let next = Math.max(end - overlap, start + 1);
    const nl = text.indexOf('\n', next);
    if (nl !== -1 && nl < end) next = nl + 1;
    start = next;
  }
  return out;
}

const sourceOf = (h: SearchHit) => `${h.file}:${h.start_line}-${h.end_line}`;
const keyOf = (h: SearchHit) => `${h.id}\n${passageText(h)}`;

/** Scores passages for one request, each once (a passage whose text grew is scored again). */
export class PassageScorer {
  private scores = new Map<string, PassageScore>();
  readonly request: string;
  private readonly rowRequest: string;
  private readonly chunkChars: number;
  /** Checkpoint that scored, once one has. */
  model = '';

  constructor(request: string, inputTokens: number) {
    const rowChars = Math.floor((inputTokens - ROW_HEAD_TOKENS) * LAYA_CHARS_PER_TOKEN);
    this.request = request;
    this.rowRequest = squeeze(request.trim(), Math.floor(rowChars * ROW_REQUEST_SHARE));
    // What's left of a row for the passage, less its source and the labels around it.
    this.chunkChars = Math.max(400, rowChars - this.rowRequest.length - 120);
  }

  get(h: SearchHit): PassageScore | undefined {
    return this.scores.get(keyOf(h));
  }

  /** Scores the passages not scored yet. Throws when Laya fails; nothing is cached then. */
  async score(hits: SearchHit[]): Promise<{ ms: number; modelMs: number; rows: number; passages: number }> {
    const todo = [...new Map(hits.filter((h) => !this.scores.has(keyOf(h)) && passageText(h)).map((h) => [keyOf(h), h])).values()];
    const rows = todo.flatMap((h, i) => chunkText(passageText(h), this.chunkChars).map((text) => ({ i, source: sourceOf(h), text })));
    const started = performance.now();
    let modelMs = 0;
    const out: { score: number; inputTokens: number; truncated: boolean }[] = [];
    for (let at = 0; at < rows.length; at += MAX_ROWS) {
      const batch = rows.slice(at, at + MAX_ROWS);
      const r = await layaRelevance(
        this.rowRequest,
        batch.map(({ source, text }) => ({ source, text })),
      );
      if (r.scores.length !== batch.length || !r.scores.every(Number.isFinite)) throw new Error(`${r.model} returned ${r.scores.length} scores for ${batch.length} passages.`);
      this.model = r.model;
      modelMs += r.ms;
      out.push(...r.scores.map((score, k) => ({ score, inputTokens: r.inputTokens[k] ?? 0, truncated: !!r.truncated[k] })));
    }
    todo.forEach((h, i) => {
      const mine = rows.map((row, k) => ({ row, s: out[k] })).filter(({ row }) => row.i === i);
      const best = mine.reduce((a, b) => (b.s.score > a.s.score ? b : a));
      this.scores.set(keyOf(h), {
        score: best.s.score,
        best: best.row.text,
        chunks: mine.length,
        inputTokens: Math.max(...mine.map(({ s }) => s.inputTokens)),
        truncated: mine.some(({ s }) => s.truncated),
      });
    });
    return { ms: Math.round(performance.now() - started), modelMs, rows: rows.length, passages: todo.length };
  }
}

/** A request this short, after earlier turns, is scored with the question before it. */
const FOLLOW_UP_CHARS = 80;

/**
 * What passages are scored against. A short follow-up ("And the Osprey?")
 * says too little alone: scored against it, an accessibility section beat
 * the dry-dock schedule the question was about (item 15), so it goes with
 * the user's previous question.
 */
export function scoringRequest(prompt: string, history: { role: string; content: string }[]): string {
  const text = prompt.trim();
  const before = [...history].reverse().find((m) => m.role === 'user')?.content.trim();
  if (!before || text.length > FOLLOW_UP_CHARS) return text;
  return `${before.length > 300 ? `${before.slice(0, 300)}…` : before}\nFollow-up: ${text}`;
}

/** Lines a hit's node spans, and how many its text shows, when a tool clipped it (a search passage cut to its share). */
export function shownLines(h: SearchHit): { lines: number; of: number } | undefined {
  if (!(h.start_line > 0 && h.end_line >= h.start_line)) return undefined;
  const of = h.end_line - h.start_line + 1;
  const lines = passageText(h).split('\n').length;
  return of >= 4 && lines < of * 0.8 ? { lines, of } : undefined;
}

/**
 * Scores `hits` for `request` and returns the ones to keep, in their order,
 * with the record the trace shows. Nothing to decide with `KEEP_TOP` or fewer.
 * Passages `scorer` already scored (the agent loop's) aren't scored again.
 */
export async function checkRelevance(request: string, hits: SearchHit[], scorer?: PassageScorer | null, inputTokens = 512): Promise<{ hits: SearchHit[]; record: RelevanceRecord } | null> {
  if (hits.length <= KEEP_TOP) return null;
  const started = performance.now();
  const s = scorer ?? new PassageScorer(request, inputTokens);
  const r = await s.score(hits);
  const scored = hits.map((h) => s.get(h));
  if (scored.some((x) => !x)) throw new Error(`${s.model || 'Laya'} left ${scored.filter((x) => !x).length} of ${hits.length} passages unscored.`);
  const picks = choosePassages(scored.map((x) => x!.score));
  const items: RelevanceItem[] = hits.map((h, i) => ({
    file: h.file,
    start_line: h.start_line,
    end_line: h.end_line,
    name: h.name,
    score: scored[i]!.score,
    kept: picks[i].kept,
    reason: picks[i].reason,
    chars: passageText(h).length,
    text: passageText(h),
    inputTokens: scored[i]!.inputTokens,
    truncated: scored[i]!.truncated,
    chunks: scored[i]!.chunks,
  }));
  const dropped = items.filter((x) => !x.kept).reduce((n, x) => n + x.chars, 0);
  return {
    hits: hits.filter((_, i) => picks[i].kept),
    record: {
      model: layaById(s.model)?.name ?? s.model,
      modelId: s.model,
      // What the passages were judged against (a follow-up goes with the question before it).
      request: s.request,
      ms: Math.round(performance.now() - started),
      modelMs: r.modelMs,
      keepTop: KEEP_TOP,
      dropBelow: DROP_BELOW,
      items,
      tokensSaved: Math.round(dropped / CHARS_PER_TOKEN),
    },
  };
}
