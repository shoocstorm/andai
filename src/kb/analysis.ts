// Reads `ug analyze --json` (whole-repo statistics over a code knowledge
// base) for the Insights view, the source dialog's blast radius, the agent's
// analysis tools and the onboarding questions. Pure.
//
// The envelope (ug 0.1.22, AGENTS.md §2): `title`, `description`, `columns`,
// `rows`, `rowsTotal`, `truncated`, `unindexed` (properties no node carries,
// so every predicate on them matched nothing) and `targetNotIndexed` (a
// target path the graph doesn't hold). An empty table is only "nothing" when
// neither is set, so both become caveats wherever a result is shown.
import type { AnalyzePreset } from './api';

export type Cell = string | number | null;

export type Analysis = {
  title: string;
  description: string;
  columns: string[];
  rows: Cell[][];
  /** Rows before the limit cut them. */
  total: number;
  /** What the result can't tell, in words: set whenever an empty or short table doesn't mean "none". */
  caveats: string[];
};

/** A symbol from an analysis row's node id (`function_declaration:src/a.ts:foo`). */
export type NodeRef = { kind: string; file: string; name: string };

const obj = (v: unknown): Record<string, unknown> => (v && typeof v === 'object' && !Array.isArray(v) ? (v as Record<string, unknown>) : {});
const arr = (v: unknown): unknown[] => (Array.isArray(v) ? v : []);
const str = (v: unknown) => (typeof v === 'string' ? v : '');
const cell = (v: unknown): Cell => (typeof v === 'string' ? v : typeof v === 'number' && Number.isFinite(v) ? v : null);

/** Any shape is tolerated: missing fields come back empty, never as an error. */
export function readAnalysis(output: unknown): Analysis {
  const o = obj(output);
  const columns = arr(o.columns).map((c) => str(c));
  const rows = arr(o.rows).map((r) => arr(r).map(cell));
  const total = typeof o.rowsTotal === 'number' && o.rowsTotal >= rows.length ? o.rowsTotal : rows.length;
  const caveats: string[] = [];
  const missing = arr(o.unindexed).map(str).filter(Boolean);
  if (missing.length)
    caveats.push(
      `This index has no ${missing.join(', ')} yet, so this result is incomplete: re-index with a newer ug to fill it in.`,
    );
  for (const t of arr(o.targetNotIndexed).map(str).filter(Boolean))
    caveats.push(`${t} isn't in the index, so an empty result doesn't mean nothing depends on it.`);
  if (typeof output === 'string') caveats.push('ug’s output was cut short and couldn’t be read.');
  return { title: str(o.title), description: str(o.description), columns, rows, total, caveats };
}

/** ug's tree-sitter kinds → the node types the symbol tools use (agent/tools/ug.ts `SYMBOL_TYPES`). */
const KINDS: [RegExp, string][] = [
  [/method/, 'Method'],
  [/function|^fn|arrow/, 'Function'],
  [/class/, 'Class'],
  [/interface/, 'Interface'],
  [/struct/, 'Struct'],
  [/enum/, 'Enum'],
  [/trait/, 'Trait'],
  [/type/, 'Type'],
  [/const/, 'Constant'],
  [/variable|lexical/, 'Variable'],
  [/module|namespace/, 'Module'],
];

/** `kind:file:name` → its parts; the file may hold colons, so it's what's between the first and the last. */
export function parseNodeId(id: string): NodeRef | null {
  const first = id.indexOf(':');
  const last = id.lastIndexOf(':');
  if (first <= 0 || last <= first + 1 || last === id.length - 1) return null;
  const raw = id.slice(0, first).toLowerCase();
  const kind = KINDS.find(([re]) => re.test(raw))?.[1] ?? 'Node';
  return { kind, file: id.slice(first + 1, last), name: id.slice(last + 1) };
}

/** Column `name` of a row, or null. */
export function col(a: Analysis, row: Cell[], name: string): Cell {
  const i = a.columns.indexOf(name);
  return i < 0 ? null : (row[i] ?? null);
}

const fmt = (v: Cell) => (v === null ? '–' : typeof v === 'number' ? String(Math.round(v * 100) / 100) : v);
const label = (c: string) => c.replace(/_/g, ' ');

/** One row as text: a symbol id as "name (Kind, file)", the other columns as `label value`. */
export function rowText(a: Analysis, row: Cell[]): string {
  return a.columns
    .map((c, i) => {
      const v = row[i] ?? null;
      const ref = c === 'id' && typeof v === 'string' ? parseNodeId(v) : null;
      if (ref) return `${ref.name} (${ref.kind}, ${ref.file})`;
      return c === 'id' || c === 'file' || c === 'folder' || c === 'language' ? fmt(v) : `${label(c)} ${fmt(v)}`;
    })
    .join(' · ');
}

/** The result as one passage: what was asked, the rows, how many there were, and what it can't tell. */
export function analysisText(a: Analysis, max = 20): string {
  const head = [a.title && `Analysis: ${a.title}`, a.description].filter(Boolean).join('. ');
  const rows = a.rows.slice(0, max).map((r, i) => `${i + 1}. ${rowText(a, r)}`);
  const shown = rows.length < a.total ? `Showing ${rows.length} of ${a.total} rows.` : '';
  return [head, ...(rows.length ? rows : ['No rows matched.']), shown, ...a.caveats.map((c) => `Note: ${c}`)]
    .filter(Boolean)
    .join('\n');
}

/** What to tell the user when `ug analyze` fails. ug before 0.1.22 had no `analyze`. */
export function analysisError(e: unknown): string {
  const msg = e instanceof Error ? e.message : String(e);
  return /unknown command/i.test(msg) ? 'This version of ug has no `ug analyze`. Update it with `ug upgrade`, then refresh.' : msg;
}

/** The presets the Insights view shows, in the order its cards read them. */
export const INSIGHT_PRESETS: AnalyzePreset[] = [
  'language_breakdown',
  'file_kinds',
  'size_histogram',
  'biggest_files',
  'where_to_start',
  'dependency_fanin',
  'risky_symbols',
  'coupling_matrix',
  'untested_symbols',
  'undocumented_hotspots',
  'long_functions',
  'dead_code',
];

export type Insights = Partial<Record<AnalyzePreset, Analysis>>;

const CALLABLE = new Set(['Function', 'Method']);

/** Symbol refs from a result's `id` column, in order. */
export function symbolsIn(a: Analysis | undefined): (NodeRef & { row: Cell[] })[] {
  if (!a) return [];
  return a.rows.flatMap((row) => {
    const id = col(a, row, 'id');
    const ref = typeof id === 'string' ? parseNodeId(id) : null;
    return ref ? [{ ...ref, row }] : [];
  });
}

/**
 * Questions a newcomer could ask about a codebase, from what its analysis
 * shows: the reading order's first entries, the most depended-upon function,
 * the biggest file and what depends on it. Empty when nothing qualifies.
 */
export function onboardingQuestions(ins: Insights, max = 4): string[] {
  const out: string[] = [];
  const add = (q: string) => {
    if (!out.includes(q)) out.push(q);
  };
  for (const s of symbolsIn(ins.where_to_start).slice(0, 2)) add(`How does ${s.name} work?`);
  const used = symbolsIn(ins.dependency_fanin).find((s) => CALLABLE.has(s.kind));
  if (used) add(`Who calls ${used.name}?`);
  const big = ins.biggest_files?.rows.map((r) => col(ins.biggest_files!, r, 'file')).find((f): f is string => typeof f === 'string' && !!f);
  if (big) {
    add(`What does ${big} do?`);
    add(`What would break if I changed ${big}?`);
  }
  return out.slice(0, max);
}

/** The Insights view as Markdown, for Copy report. Values are ug's, shown as text. */
export function insightsReport(kbName: string, ins: Insights, max = 10): string {
  const sections = INSIGHT_PRESETS.flatMap((p) => {
    const a = ins[p];
    if (!a) return [];
    const rows = a.rows.slice(0, max).map((r) => `- ${rowText(a, r)}`);
    const more = a.rows.length > max || a.total > a.rows.length ? [`- … ${a.total - Math.min(max, a.rows.length)} more`] : [];
    return [`## ${label(a.title || p)}`, ...(a.description ? [a.description] : []), '', ...(rows.length ? rows : ['- none']), ...more, ...a.caveats.map((c) => `> ${c}`), ''];
  });
  return [`# ${kbName}: code insights`, '', ...sections].join('\n').trim() + '\n';
}
