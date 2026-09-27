// How ug search found a passage, for the passage lists: which channel matched
// it (`matched_by`: the vector search, the full-text search, or a walk along
// the graph from a match, `hop` edges away), and its ranking score.
//
// ug ranks by Personalized PageRank seeded by both searches, and reports the
// score negated as `distance`: lower ranks higher (measured: results come
// sorted by it ascending, graph neighbours last). The raw number means little
// on its own, so it's shown relative to the best match in the same list.

import type { SearchHit } from './api';

export type Match = {
  /** `semantic`, `keyword`, `graph`, or whatever else ug reports; null when it didn't say. */
  how: string | null;
  /** Short label: "semantic", "keyword", "graph · 1 hop"; empty when the channel is unknown. */
  label: string;
  /** What the channel means, for a tooltip. */
  help: string;
  /** ug's raw ranking score (`distance`; lower ranks higher). */
  score?: number;
  /** 0–1: the score relative to the best in the list (1 = the best). */
  strength?: number;
};

const HELP: Record<string, string> = {
  semantic: 'Found by meaning: close to the query in the vector search.',
  keyword: 'Found by its words: the full-text search matched the query’s terms.',
  graph: 'Reached by following links in the knowledge graph from a passage that matched.',
};

/** How many passages a search returned, and how many each channel found. */
export type Found = { total: number; by: Record<string, number> };

/** Counts `hits` by the channel ug matched them by (`matched_by`). Pure. */
export function countMatches(hits: SearchHit[]): Found {
  const by: Record<string, number> = {};
  for (const h of hits) {
    const how = h.matched_by?.trim().toLowerCase();
    if (how) by[how] = (by[how] ?? 0) + 1;
  }
  return { total: hits.length, by };
}

const CHANNEL_ORDER = ['semantic', 'keyword', 'graph'];

/** "10 passages · 7 semantic · 1 keyword · 2 graph": a search's result in one short line. Pure. */
export function foundLine(f: Found): string {
  const channels = Object.entries(f.by).sort(
    ([a], [b]) => (CHANNEL_ORDER.indexOf(a) + 1 || 99) - (CHANNEL_ORDER.indexOf(b) + 1 || 99) || a.localeCompare(b),
  );
  return [`${f.total} passage${f.total === 1 ? '' : 's'}`, ...channels.map(([how, n]) => `${n} ${how}`)].join(' · ');
}

/** Each hit's strength relative to the best-ranked one among `peers`. Pure. */
export function strengthOf(score: number, peers: number[]): number {
  const all = peers.filter(Number.isFinite);
  if (!all.length) return 1;
  const best = Math.min(...all);
  const worst = Math.max(...all);
  // ug's scores are negated PageRank: then a score's share of the best one is its relative weight.
  if (worst <= 0 && best < 0) return Math.max(0, Math.min(1, score / best));
  return worst === best ? 1 : Math.max(0, Math.min(1, (worst - score) / (worst - best)));
}

/** How `h` was found, with its strength among `peers`; null when ug said nothing about it (not a search result). Pure. */
export function matchOf(h: SearchHit, peers: SearchHit[] = [h]): Match | null {
  const how = h.matched_by?.trim().toLowerCase();
  const score = typeof h.distance === 'number' && Number.isFinite(h.distance) ? h.distance : undefined;
  if (!how && score == null) return null;
  const hops = how === 'graph' && typeof h.hop === 'number' && h.hop > 0 ? ` · ${h.hop} hop${h.hop === 1 ? '' : 's'}` : '';
  return {
    how: how || null,
    label: how ? `${how}${hops}` : '',
    help: (how && HELP[how]) || 'How ug search found this passage.',
    ...(score != null
      ? { score, strength: strengthOf(score, peers.map((p) => p.distance).filter((d): d is number => typeof d === 'number')) }
      : {}),
  };
}
