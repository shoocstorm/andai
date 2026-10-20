// Answer X-ray: what the claim check (agent/claims.ts) said about each sentence
// of an answer, looked up by the sentence as it renders. The check keys its
// items by the sentence without Markdown or citation marks (`plain`), and the
// rendered text of a sentence normalizes to the same string, so the lookup is
// an exact match. Pure, for the answer view and its tests.

import { aboutMissing, checkable, plain, type SupportItem, type SupportRecord } from '../agent/claims';

export type Verdict =
  /** Scored against every source it cites; `worst` is the lowest, the one a flag is about. */
  | {
      kind: 'checked';
      items: SupportItem[];
      worst: SupportItem;
      flagged: boolean;
    }
  /** Cites a source, but the check has no score for it (past its 24-claim cap, say). */
  | { kind: 'cited' }
  /** A statement with no citation: nothing ties it to a source. */
  | { kind: 'uncited' };

const HAS_CITE = /\[\d+(?:\s*,\s*\d+)*\]/;

/**
 * A lookup from a rendered sentence to its verdict, or null when there is
 * nothing to mark: a fragment, a bare citation, or a sentence about what the
 * sources lack (the check skips those on purpose).
 */
export function xrayIndex(r: SupportRecord): (text: string) => Verdict | null {
  const bySentence = new Map<string, SupportItem[]>();
  for (const x of r.items) bySentence.set(x.sentence, [...(bySentence.get(x.sentence) ?? []), x]);
  // The check splits by line, so a table row is one claim ("| Adult | Pays the
  // full fare |") while it renders as cells: a cell is the claim it sits in,
  // when it cites the source that claim was checked against.
  const within = (sentence: string, text: string) => {
    if (!checkable(sentence)) return undefined;
    const cites = new Set([...text.matchAll(/\[(\d+)\]/g)].map((m) => Number(m[1])));
    for (const [claim, items] of bySentence) {
      if (claim.includes(sentence) && items.some((x) => cites.has(x.n))) return items.filter((x) => cites.has(x.n));
    }
    return undefined;
  };
  return (text) => {
    const sentence = plain(text);
    const items = bySentence.get(sentence) ?? within(sentence, text);
    if (items) {
      const worst = items.reduce((a, b) => (b.score < a.score ? b : a));
      return {
        kind: 'checked',
        items,
        worst,
        flagged: items.some((x) => x.flagged),
      };
    }
    if (!checkable(sentence) || aboutMissing(sentence)) return null;
    return HAS_CITE.test(text) ? { kind: 'cited' } : { kind: 'uncited' };
  };
}
