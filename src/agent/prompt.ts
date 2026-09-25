// Pure prompt assembly for an agent turn — no stores, no I/O, so every rule
// here is unit-tested (prompt.test.ts). turn.ts wires these to live state.

import type { SearchHit } from '../kb/api';
import type { ChatMessage } from '../llm/engine';
import { splitThink, type Message } from '../state/chat';
import { TONES, type Tone } from '../state/persona';

/** Rough chars-per-token for budgeting; Qwen3's tokenizer averages ~3.2 on English prose. */
export const CHARS_PER_TOKEN = 3.2;
/** Below this much room a passage is dropped rather than clipped to a useless stub. */
export const MIN_PASSAGE_CHARS = 200;

const STOP = new Set(
  'a an and are as at be but by can could did do does for from had has have how i if in into is it its me my of on or our should so than that the their them then there these they this to was we were what when where which who why will with would you your about please tell explain give show'.split(
    ' ',
  ),
);

/** The longest distinct non-stopwords, for the "Analyzing query" step. */
export function keywords(text: string, n = 3): string[] {
  const seen = new Set<string>();
  const words = text.toLowerCase().match(/[a-z0-9][a-z0-9_.\-/]{2,}/g) ?? [];
  const out: string[] = [];
  for (const w of words.sort((a, b) => b.length - a.length)) {
    if (STOP.has(w) || seen.has(w)) continue;
    seen.add(w);
    out.push(w);
    if (out.length === n) break;
  }
  return out;
}

export type PersonaInput = { systemPrompt: string; tone: Tone };

/**
 * System prompt = persona + tone clause + (optionally) retrieved passages.
 * Passage numbers are the hit's index + 1, so `[n]` in the answer matches the
 * n-th source shown in the UI even when a later passage is dropped for budget.
 */
export function buildSystem(
  persona: PersonaInput,
  hits: SearchHit[],
  contextBudget: number,
  kbName: string | null,
): string {
  const parts = [persona.systemPrompt.trim(), TONES[persona.tone].clause];
  if (kbName) {
    if (!hits.length) {
      parts.push(`The knowledge base “${kbName}” had no relevant passages for this question. Say so if the answer depends on it.`);
    } else {
      let used = 0;
      const blocks: string[] = [];
      hits.forEach((h, i) => {
        const text = (h.snippet ?? h.description ?? '').trim();
        const room = contextBudget - used;
        if (room < MIN_PASSAGE_CHARS) return;
        const clipped = text.length > room ? `${text.slice(0, room)}…` : text;
        used += clipped.length;
        blocks.push(`[${i + 1}] ${h.file} (lines ${h.start_line}-${h.end_line})\n${clipped}`);
      });
      parts.push(
        `Knowledge base “${kbName}” — retrieved context. Answer from it and cite sources inline as [n].\n\n${blocks.join('\n\n---\n\n')}`,
      );
    }
  }
  return parts.join('\n\n');
}

/**
 * Most recent conversation that fits `budgetChars`, oldest first. Error
 * bubbles are UI-only and `<think>` blocks are never replayed to the model.
 */
export function buildHistory(messages: Message[], budgetChars: number, excludeId?: string): ChatMessage[] {
  const out: ChatMessage[] = [];
  let used = 0;
  const msgs = messages.filter((m) => m.id !== excludeId && m.role !== 'error');
  for (let i = msgs.length - 1; i >= 0; i--) {
    const m = msgs[i];
    const content = m.role === 'assistant' ? splitThink(m.content).answer : m.content;
    if (!content.trim()) continue;
    if (used + content.length > budgetChars) break;
    used += content.length;
    out.unshift({ role: m.role as 'user' | 'assistant', content });
  }
  return out;
}

/** Split the model's context window between retrieved passages and history. */
export function budgets(nCtx: number, maxTokens: number) {
  const tokenRoom = nCtx - maxTokens - 256;
  return {
    context: Math.max(800, Math.floor(tokenRoom * CHARS_PER_TOKEN * 0.6)),
    history: Math.max(400, Math.floor(tokenRoom * CHARS_PER_TOKEN * 0.25)),
  };
}
