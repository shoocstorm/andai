// What one agent turn found, merged into the passages the answer is built
// from (buildSystem) and the source list the UI shows; `[n]` is the n-th of
// both. Pure, so every rule is unit-tested (evidence.test.ts).
//
// Two things went wrong when calls were appended as they came: a later call
// that found the same ug node with more in it (Find usages with the call
// site) was dropped as a duplicate, and a passage inside a range the agent
// then read on purpose sent the same text twice (docs/agentic-rag-improvements.md,
// item 5).

import type { SearchHit } from '../kb/api';

/** One passage and the call that found it. */
export type Found = { hit: SearchHit; tool: string };

/** Tools that read text on purpose: an exact line range or a symbol's source. */
const READS = new Set(['kb_read_lines', 'kb_get_code']);

const text = (h: SearchHit) => (h.snippet ?? h.description ?? '').trim();
const hasLines = (h: SearchHit) => h.start_line > 0 && h.end_line >= h.start_line;
const inside = (a: SearchHit, b: SearchHit) => a.file === b.file && hasLines(a) && hasLines(b) && b.start_line <= a.start_line && a.end_line <= b.end_line;

/**
 * Adds one call's passages to `found` (in place). A node seen before keeps
 * its place and gains the text the new passage adds. Returns how many
 * passages were new or gained text: what the call contributed.
 */
export function addEvidence(found: Found[], tool: string, hits: SearchHit[]): number {
  let added = 0;
  for (const hit of hits) {
    const t = text(hit);
    const i = found.findIndex((f) => f.hit.id === hit.id);
    if (i === -1) {
      found.push({ hit, tool });
      added++;
      continue;
    }
    const old = text(found[i].hit);
    if (!t || old.includes(t)) continue;
    const merged = t.includes(old) ? t : `${old}\n${t}`;
    found[i] = { ...found[i], hit: { ...found[i].hit, snippet: merged } };
    added++;
  }
  return added;
}

/**
 * The passages for the answer: those inside a range read on purpose are
 * dropped (the read repeats their text), then reads come first, lookups
 * (symbols, context, usages, outlines, overview) next and search hits last,
 * each group in the order it arrived. `buildSystem` cuts from the end, so a
 * broad search hit is what gives way to budget.
 */
export function mergeEvidence(found: Found[]): SearchHit[] {
  const reads = found.filter((f) => READS.has(f.tool));
  const same = (a: SearchHit, b: SearchHit) => inside(a, b) && inside(b, a);
  // Covered by another read; of two reads of the same range, the earlier
  // stays. A search passage whose lines were then read is always covered: it's
  // the clipped copy of the read (the loop reads such passages whole, item 15).
  const covered = (f: Found) =>
    reads.some((r) => r !== f && inside(f.hit, r.hit) && !(READS.has(f.tool) && same(f.hit, r.hit) && found.indexOf(f) < found.indexOf(r)));
  const kept = found.filter((f) => !covered(f));
  const rank = (f: Found) => (READS.has(f.tool) ? 0 : f.tool === 'kb_search' ? 2 : 1);
  return kept
    .map((f, i) => ({ f, i }))
    .sort((a, b) => rank(a.f) - rank(b.f) || a.i - b.i)
    .map(({ f }) => f.hit);
}
