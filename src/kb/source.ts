// Reads ug's `file_context` report for the source dialog (screens/SourceDialog.tsx). Pure.
import type { Source } from './api';

/** One source as `kb_source` returns it (src-tauri/src/ug.rs `SourceView`). */
export type SourceView = {
  source: Source;
  /** Where the file is on disk. */
  path: string;
  /** The file's text; null for a PDF, whose text only ug's index holds. */
  text: string | null;
  textTruncated: boolean;
  /** ug's `file_context` JSON, or null with `structureError` saying why. */
  structure: unknown;
  structureError: string | null;
};

export type OutlineItem = {
  id: string;
  name: string;
  /** ug's node type: Concept for a heading or page, Function, Class, … for code. */
  nodeType: string;
  start: number;
  end: number;
  doc: string | null;
  /** How many earlier items enclose this one's lines (a section under its heading). */
  depth: number;
};

export type Related = { role: string; label: string; items: { name: string; file: string; why: string }[] };

export type Structure = {
  language: string | null;
  classification: string | null;
  lines: number | null;
  symbols: number | null;
  isTest: boolean;
  outline: OutlineItem[];
  related: Related[];
};

/** ug's relation roles, in the order its report ranks them, with what they mean for this file. */
const ROLES: [string, string][] = [
  ['importer', 'Imported by'],
  ['import', 'Imports'],
  ['test', 'Tested by'],
  ['dependent', 'Used by'],
  ['sibling', 'Same folder'],
];

const obj = (v: unknown): Record<string, unknown> => (v && typeof v === 'object' && !Array.isArray(v) ? (v as Record<string, unknown>) : {});
const str = (v: unknown) => (typeof v === 'string' ? v : null);
const num = (v: unknown) => (typeof v === 'number' && Number.isFinite(v) ? v : null);

/**
 * The outline, facts and related files from ug's report on one file. Any shape
 * is tolerated: missing fields come back empty, never as an error.
 */
export function readStructure(report: unknown): Structure {
  const files = obj(report).files;
  const file = obj(Array.isArray(files) ? files[0] : null);
  const facts = obj(file.facts);
  const items = (Array.isArray(file.items) ? file.items : []).map(obj);

  const flat = items
    .filter((i) => i.role === 'outline')
    .map((i) => {
      const start = num(i.start_line) ?? 1;
      return {
        id: str(i.id) ?? '',
        name: str(i.name) ?? '',
        nodeType: str(i.node_type) ?? '',
        start,
        end: Math.max(start, num(i.end_line) ?? start),
        doc: str(i.doc)?.replace(/\s+/g, ' ').trim() || null,
      };
    })
    .filter((i) => i.name)
    .sort((a, b) => a.start - b.start || b.end - a.end);

  // Nest by line containment: an item sits under every open one whose range holds it.
  const open: OutlineItem[] = [];
  const outline = flat.map((i) => {
    while (open.length && !(i.start >= open[open.length - 1].start && i.end <= open[open.length - 1].end)) open.pop();
    const item = { ...i, depth: open.length };
    open.push(item);
    return item;
  });

  const related = ROLES.map(([role, label]) => {
    const seen = new Set<string>();
    const list = items
      .filter((i) => i.role === role)
      .map((i) => ({ name: str(i.name) ?? str(i.file) ?? '', file: str(i.file) ?? '', why: str(i.why) ?? '' }))
      .filter((i) => i.name && !seen.has(i.name) && seen.add(i.name));
    return { role, label, items: list };
  }).filter((r) => r.items.length);

  return {
    language: str(facts.language),
    classification: str(facts.classification),
    lines: num(facts.lines),
    symbols: num(facts.symbols),
    isTest: facts.is_test === true,
    outline,
    related,
  };
}

/** A PDF's text as ug indexed it: one entry per page outline item (`p.3 · Heading`). */
export function pdfPages(outline: OutlineItem[]): { title: string; text: string }[] {
  return outline.filter((i) => i.doc).map((i) => ({ title: i.name, text: i.doc! }));
}
