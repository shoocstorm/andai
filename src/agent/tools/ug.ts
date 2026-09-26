// The built-in tools: read-only ug queries over the grounding knowledge base.
// Output shapes were probed against ug 0.1.21 (AGENTS.md §2); every reader
// here treats the output as untrusted and tolerates missing fields.

import { dedupeHits, type KbKind, type SearchHit } from '../../kb/api';
import type { Evidence, ToolDef } from './types';

const NODE_TYPES = ['any', 'Function', 'Method', 'Class', 'Interface', 'Struct', 'Enum', 'Trait', 'Type', 'Constant', 'Variable', 'Module', 'File'];
/** Node types that are code symbols, which the symbol tools accept. ug calls document sections `Concept`. */
export const SYMBOL_TYPES = new Set(NODE_TYPES.filter((t) => t !== 'any' && t !== 'File'));
/** Tools that take a `symbol`: only worth offering once one has been seen (loop.ts `offered`). */
export const needsSymbol = (t: ToolDef) => !!t.schema?.properties.symbol;

type Node = {
  id?: string;
  name?: string;
  node_type?: string;
  file?: string;
  start_line?: number;
  end_line?: number;
  doc?: string;
  description?: string;
  code?: string;
  snippet?: string;
  role?: string;
  why?: string;
  error?: string;
};

const obj = (v: unknown): Record<string, unknown> => (v && typeof v === 'object' && !Array.isArray(v) ? (v as Record<string, unknown>) : {});
const arr = (v: unknown): unknown[] => (Array.isArray(v) ? v : []);
const str = (v: unknown) => (typeof v === 'string' ? v : '');
const num = (v: unknown) => (typeof v === 'number' && Number.isFinite(v) ? v : 0);

function node(v: unknown): Node {
  const o = obj(v);
  return {
    id: str(o.id),
    name: str(o.name),
    node_type: str(o.node_type),
    file: str(o.file),
    start_line: num(o.start_line),
    end_line: num(o.end_line),
    doc: str(o.doc),
    description: str(o.description),
    code: str(o.code),
    snippet: str(o.snippet),
    role: str(o.role),
    why: str(o.why),
    error: str(o.error),
  };
}

/** A graph node as a passage: `snippet` is what the model reads. */
function hit(n: Node, snippet: string, source: string): SearchHit {
  const file = n.file || n.name || '(knowledge base)';
  return {
    id: n.id || `${source}:${file}:${n.start_line}-${n.end_line}`,
    name: n.name || file,
    node_type: n.node_type || 'Node',
    file,
    start_line: n.start_line ?? 0,
    end_line: n.end_line ?? 0,
    snippet: snippet.trim(),
  };
}

const loc = (h: SearchHit) => (h.start_line ? `${h.file}:${h.start_line}-${h.end_line}` : h.file);
const list = (hits: SearchHit[], max = 5) =>
  hits
    .slice(0, max)
    .map((h) => `${h.name && h.name !== h.file ? `${h.name} @ ` : ''}${loc(h)}`)
    .join('; ') + (hits.length > max ? `; +${hits.length - max} more` : '');

function errorsIn(items: Node[]): string[] {
  return items.map((n) => n.error ?? '').filter(Boolean);
}

function evidence(hits: SearchHit[], what: string, errors: string[] = []): Evidence {
  if (!hits.length) return { hits, summary: errors[0] ?? `No ${what} found.` };
  return { hits, summary: `${hits.length} ${what}: ${list(hits)}` };
}

const QUERY_GUIDE: Record<KbKind, string> = {
  document:
    'Write `query` as a short search phrase for semantic + full-text retrieval over documents: the key concepts, names and likely synonyms, not the user’s literal wording or a question.',
  code: 'Write `query` as a short search phrase for retrieval over source code: likely identifiers (function, class, file names) and domain terms, not a question.',
  mixed:
    'Write `query` as a short search phrase for retrieval over documents and source code: key concepts plus likely identifiers, not the user’s literal wording or a question.',
};

export const UG_TOOLS: ToolDef[] = [
  {
    id: 'kb_search',
    title: 'Knowledge search',
    option: 'Search the knowledge base for passages relevant to the request',
    description:
      'GraphRAG search: semantic + full-text seeds, then a walk along the knowledge graph. The model writes the search phrase itself and picks broad (with graph expansion) or focused (direct matches only).',
    kinds: ['document', 'code', 'mixed'],
    risk: 'read',
    command: 'ug search <query> -k <k> --max-chars <n> [--no-expand] --snippets --json',
    schema: {
      type: 'object',
      properties: {
        query: { type: 'string', minLength: 2, maxLength: 300, description: 'search phrase' },
        scope: { type: 'string', enum: ['broad', 'focused'], description: 'broad follows related nodes; focused returns direct matches only' },
      },
      required: ['query', 'scope'],
      additionalProperties: false,
    },
    guide: (kind) => `${QUERY_GUIDE[kind]} Use scope "focused" for a specific name or term, "broad" to gather surrounding context.`,
    toCall: (a, ctx) => ({
      tool: 'kb_search',
      query: String(a.query),
      k: ctx.k,
      expand: a.scope !== 'focused',
      max_chars: ctx.maxChars,
    }),
    observe: (out) => {
      const items = arr(obj(out).items).map((v) => {
        const n = node(v);
        return { ...hit(n, n.snippet || n.description || '', 'search'), distance: num(obj(v).distance), matched_by: str(obj(v).matched_by) || undefined };
      });
      return evidence(dedupeHits(items), 'passage(s)');
    },
  },
  {
    id: 'kb_read_lines',
    title: 'Read lines',
    option: 'Read a specific line range of a file already seen in the results',
    description: 'Reads up to 400 lines of one file in the knowledge base, e.g. to see more around a search hit.',
    kinds: ['document', 'code', 'mixed'],
    risk: 'read',
    command: 'ug get_code -f <file> -s <start> -e <end> --json',
    schema: {
      type: 'object',
      properties: {
        file: { type: 'string', minLength: 1, maxLength: 256, description: 'file path exactly as shown in the results' },
        start: { type: 'integer', minimum: 1, maximum: 1000000 },
        end: { type: 'integer', minimum: 1, maximum: 1000000 },
      },
      required: ['file', 'start', 'end'],
      additionalProperties: false,
    },
    guide: () => 'Pick `file` from the tool results so far (copy the path exactly) and a line window around the part that matters.',
    toCall: (a) => ({ tool: 'kb_get_code', symbol: null, file: String(a.file), start: Number(a.start), end: Number(a.end) }),
    observe: (out) => {
      const slices = arr(obj(out).slices).map(node);
      const hits = slices.filter((s) => s.code).map((s) => hit(s, s.code ?? '', 'lines'));
      return evidence(hits, 'excerpt(s)', errorsIn(slices));
    },
  },
  {
    id: 'kb_file_context',
    title: 'File outline',
    option: 'Outline one file: its sections or symbols, and what it connects to',
    description: 'For one file: its outline (headings or declared symbols), importers, imports, tests and neighbours.',
    kinds: ['document', 'code', 'mixed'],
    risk: 'read',
    command: 'ug file_context <file> --max-chars 4000 --json',
    schema: {
      type: 'object',
      properties: { file: { type: 'string', minLength: 1, maxLength: 256, description: 'file path or unique file name' } },
      required: ['file'],
      additionalProperties: false,
    },
    guide: () => 'Use a file path from the tool results or one the user named.',
    toCall: (a) => ({ tool: 'kb_file_context', file: String(a.file), max_chars: 4000 }),
    observe: (out) => {
      const files = arr(obj(out).files).map(obj);
      const hits = files.map((f) => {
        const items = arr(f.items).map(node);
        const lines = items.map((n) => `- ${n.role ? `[${n.role}] ` : ''}${n.node_type} ${n.name} (${n.file}:${n.start_line})${n.doc ? ` — ${n.doc}` : ''}`);
        return hit(
          { id: str(f.id), name: str(f.file), node_type: 'File', file: str(f.file), start_line: 1, end_line: num(obj(f.facts).lines) },
          lines.join('\n'),
          'file',
        );
      });
      return evidence(hits, 'file outline(s)', files.map((f) => str(f.error)).filter(Boolean));
    },
  },
  {
    id: 'kb_overview',
    title: 'Overview',
    option: 'Get an overview of what the knowledge base contains',
    description: 'Size, languages, node types, the biggest files and the most depended-upon symbols. Takes no arguments.',
    kinds: ['document', 'code', 'mixed'],
    risk: 'read',
    command: 'ug project_overview --json',
    schema: null,
    guide: () => '',
    toCall: () => ({ tool: 'kb_overview' }),
    observe: (out) => {
      const o = obj(out);
      const idx = obj(o.index);
      const names = (v: unknown) => arr(v).map((x) => `${str(obj(x).name)} (${num(obj(x).count)})`).join(', ');
      const hot = arr(o.hotspots).map(node).map((n) => `${n.name} @ ${n.file}:${n.start_line}`).join(', ');
      const text = [
        `Kind: ${str(o.kb_type) || 'unknown'}; ${num(idx.files)} files, ${num(idx.symbols)} symbols, ${num(idx.lines)} lines.`,
        `Languages: ${names(o.languages) || '—'}`,
        `Node types: ${names(o.node_types) || '—'}`,
        `Biggest files: ${names(o.biggest_files) || '—'}`,
        hot ? `Most depended-upon: ${hot}` : '',
      ]
        .filter(Boolean)
        .join('\n');
      const h = hit({ id: 'overview', name: 'Knowledge base overview', node_type: 'Overview', file: '(overview)' }, text, 'overview');
      return { hits: [h], summary: text.split('\n').slice(0, 2).join(' ') };
    },
  },
  {
    id: 'kb_find_symbols',
    title: 'Find symbols',
    option: 'Look up functions, classes or other code symbols by exact name or wildcard pattern',
    description: 'Exact or wildcard lookup of code symbols (no embeddings), e.g. `parse*` or `*Config`. Gives their files, lines and doc comments.',
    kinds: ['code', 'mixed'],
    risk: 'read',
    command: 'ug find_symbols <name…> [--node-type <type>] -k 20 --include-docs --json',
    schema: {
      type: 'object',
      properties: {
        names: { type: 'array', items: { type: 'string', maxLength: 128 }, minItems: 1, maxItems: 5, description: 'names or wildcard patterns' },
        node_type: { type: 'string', enum: NODE_TYPES },
      },
      required: ['names', 'node_type'],
      additionalProperties: false,
    },
    guide: () =>
      'Give 1–5 identifiers or wildcard patterns (`*` matches any run of characters, and must match the whole name: `auth*`, `*Handler`). Use node_type "any" unless the user asked for a specific kind.',
    toCall: (a) => ({
      tool: 'kb_find_symbols',
      names: (a.names as string[]) ?? [],
      node_type: a.node_type && a.node_type !== 'any' ? String(a.node_type) : null,
      file_prefix: null,
    }),
    observe: (out) => {
      const items = arr(obj(out).queries).flatMap((q) => arr(obj(q).items)).map(node);
      const hits = items.map((n) => hit(n, `${n.node_type} ${n.name}${n.doc ? `\n${n.doc}` : ''}`, 'symbol'));
      return evidence(hits, 'symbol(s)');
    },
  },
  {
    id: 'kb_symbol_context',
    title: 'Symbol context',
    option: 'Explain one code symbol in depth: its source, callers, tests and dependencies',
    description: 'One budgeted bundle for a symbol: its source and doc, direct callers with call sites, tests reaching it, and what it depends on.',
    kinds: ['code', 'mixed'],
    risk: 'read',
    command: 'ug context <symbol> --max-chars 4000 --json',
    schema: {
      type: 'object',
      properties: { symbol: { type: 'string', minLength: 1, maxLength: 128, description: 'exact symbol name or node id' } },
      required: ['symbol'],
      additionalProperties: false,
    },
    guide: () => 'Give the exact symbol name (from the request or the results so far).',
    toCall: (a) => ({ tool: 'kb_symbol_context', symbol: String(a.symbol), max_chars: 4000 }),
    observe: (out) => {
      const items = arr(obj(out).items).map(node);
      const hits = items.map((n) =>
        hit(n, `[${n.role || 'related'}${n.why ? `: ${n.why}` : ''}] ${n.node_type} ${n.name}${n.doc ? `\n${n.doc}` : ''}${n.code ? `\n${n.code}` : ''}`, 'context'),
      );
      return evidence(hits, 'related node(s)', [str(obj(out).error)].filter(Boolean));
    },
  },
  {
    id: 'kb_get_code',
    title: 'Read symbol source',
    option: 'Read the full source code of a named function, class or other symbol',
    description: 'The source of a symbol (or every symbol a wildcard matches, up to 25), with its doc comment.',
    kinds: ['code', 'mixed'],
    risk: 'read',
    command: 'ug get_code <symbol> --max-chars 8000 --json',
    schema: {
      type: 'object',
      properties: { symbol: { type: 'string', minLength: 1, maxLength: 128, description: 'exact symbol name, node id or wildcard' } },
      required: ['symbol'],
      additionalProperties: false,
    },
    guide: () => 'Give the exact symbol name (from the request or the results so far).',
    toCall: (a) => ({ tool: 'kb_get_code', symbol: String(a.symbol), file: null, start: null, end: null }),
    observe: (out) => {
      const slices = arr(obj(out).slices).map((v) => ({ ...node(v), name: str(obj(v).title) }));
      const hits = slices.filter((s) => s.code).map((s) => hit(s, s.code ?? '', 'code'));
      return evidence(hits, 'source slice(s)', errorsIn(slices));
    },
  },
  {
    id: 'kb_find_usages',
    title: 'Find usages',
    option: 'Find who calls, imports or references a code symbol',
    description: 'Direct users of a symbol: callers, importers, references, with the call-site lines.',
    kinds: ['code', 'mixed'],
    risk: 'read',
    command: 'ug find_usages <symbol> --json',
    schema: {
      type: 'object',
      properties: { symbol: { type: 'string', minLength: 1, maxLength: 128, description: 'exact symbol name or node id' } },
      required: ['symbol'],
      additionalProperties: false,
    },
    guide: () => 'Give the exact symbol name whose users are wanted.',
    toCall: (a) => ({ tool: 'kb_find_usages', symbol: String(a.symbol) }),
    observe: (out) => {
      const nodes = arr(obj(out).nodes).map(obj);
      const hits = nodes.flatMap((n) => {
        const subject = node(n.subject);
        return arr(n.users).map((u) => {
          const user = node(u);
          const sites = arr(obj(u).call_sites)
            .map((c) => `  ${num(obj(c).line)}: ${str(obj(c).text)}`)
            .join('\n');
          return hit(user, `${user.node_type} ${user.name} → ${str(obj(u).via_edge) || 'uses'} ${subject.name}${sites ? `\n${sites}` : ''}`, 'usage');
        });
      });
      return evidence(hits, 'usage(s)', nodes.map((n) => str(n.error)).filter(Boolean));
    },
  },
];
