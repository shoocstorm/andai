// Typed wrappers over the Rust `ug` bridge (src-tauri/src/ug.rs).
import { invoke, isTauri } from '@tauri-apps/api/core';
import type { SourceView } from './source';

export type Source = {
  file: string;
  original: string;
  kind: 'PDF' | 'MD' | 'TXT' | 'CSV' | 'CODE';
  bytes: number;
  approxTokens: number | null;
  addedAt: number;
  status: 'pending' | 'indexed' | 'failed';
};

/** What a KB holds; decides which agent tools apply (src-tauri/src/ug.rs `KbKind`). */
export type KbKind = 'document' | 'code' | 'mixed';

export type KbInfo = {
  slug: string;
  name: string;
  createdAt: number;
  sources: Source[];
  lastIndexedAt: number | null;
  lastError: string | null;
  /** The user's override; null when `kind` is derived from the sources. */
  kindOverride: KbKind | null;
  dir: string;
  status: 'empty' | 'pending' | 'indexing' | 'ready' | 'failed';
  kind: KbKind;
  nodes: number;
  edges: number;
  sizeBytes: number;
};

export type UgStatus = { found: boolean; path: string | null; version: string | null };

export type SearchHit = {
  id: string;
  name: string;
  node_type: string;
  file: string;
  start_line: number;
  end_line: number;
  description?: string | null;
  snippet?: string | null;
  distance?: number;
  hop?: number;
  matched_by?: string;
};

export const inTauri = isTauri();

function call<T>(cmd: string, args?: Record<string, unknown>): Promise<T> {
  if (!inTauri) return Promise.reject(new Error('Knowledge bases need the Andai desktop app (ug runs natively).'));
  return invoke<T>(cmd, args);
}

export const ugStatus = () =>
  inTauri ? invoke<UgStatus>('ug_status') : Promise.resolve<UgStatus>({ found: false, path: null, version: null });
/** Opens UltraGraph's site (install instructions) in the system browser; Rust holds the URL. */
export const openUgWebsite = () => call<void>('open_ug_website');
export const kbList = () => (inTauri ? invoke<KbInfo[]>('kb_list') : Promise.resolve<KbInfo[]>([]));
export const kbCreate = (name: string) => call<KbInfo>('kb_create', { name });
/** Rust opens the dialog and grants what the user picks (src-tauri/src/grants.rs). */
export const kbPickFiles = (title: string) => call<string[]>('kb_pick_files', { title });
/** Only paths the user dropped or picked are accepted; others come back as per-file errors. */
export const kbAddFiles = (slug: string, paths: string[]) =>
  call<[KbInfo, string[]]>('kb_add_files', { slug, paths });
export const kbRemoveSource = (slug: string, file: string) => call<KbInfo>('kb_remove_source', { slug, file });
export const kbDelete = (slug: string) => call<void>('kb_delete', { slug });
export const kbIndex = (slug: string) => call<KbInfo>('kb_index', { slug });

/** A source's metadata, stored text and ug's outline of it, for the source dialog; Rust reads only files `kb.json` lists. */
export const kbSource = (slug: string, file: string) => call<SourceView>('kb_source', { slug, file });

export const kbSetKind = (slug: string, kind: KbKind | null) => call<KbInfo>('kb_set_kind', { slug, kind });
/** Adds a bundled sample (kb/samples.ts) as a knowledge base, or returns the one already added. Index it next. */
export const kbAddSample = (sample: string) => call<KbInfo>('kb_add_sample', { sample });

/** One read-only ug tool call; mirrors `ToolCall` in src-tauri/src/tools.rs, which re-validates it. */
export type KbToolCall =
  | { tool: 'kb_search'; query: string; k: number; expand: boolean; max_chars: number }
  | { tool: 'kb_find_symbols'; names: string[]; node_type: string | null; file_prefix: string | null }
  | { tool: 'kb_symbol_context'; symbol: string; max_chars: number }
  | { tool: 'kb_get_code'; symbol: string | null; file: string | null; start: number | null; end: number | null }
  | { tool: 'kb_find_usages'; symbol: string }
  | { tool: 'kb_file_context'; file: string; max_chars: number }
  | { tool: 'kb_overview' };

export type KbToolOutput = {
  /** ug's JSON, or its clipped text when the output was cut at the cap. */
  output: unknown;
  truncated: boolean;
  bytes: number;
  ms: number;
  /** The ug arguments as run. */
  argv: string[];
};

export const kbTool = (slug: string, toolCall: KbToolCall) =>
  call<KbToolOutput>('kb_tool', { slug, call: toolCall });

export async function kbSearch(slug: string, query: string, k: number, maxChars: number): Promise<SearchHit[]> {
  const res = await call<{ items?: SearchHit[] }>('kb_search', { slug, query, k, maxChars });
  return dedupeHits(res.items ?? []);
}

/**
 * ug returns a document node and its sections side by side; keep the most
 * specific hit for any overlapping line range so the prompt isn't repeated.
 */
export function dedupeHits(items: SearchHit[]): SearchHit[] {
  const withText = items.filter((h) => (h.snippet ?? h.description ?? '').trim());
  const out: SearchHit[] = [];
  const sorted = [...withText].sort(
    (a, b) => a.end_line - a.start_line - (b.end_line - b.start_line),
  );
  for (const h of sorted) {
    const overlaps = out.some(
      (o) => o.file === h.file && h.start_line <= o.start_line && h.end_line >= o.end_line,
    );
    if (!overlaps) out.push(h);
  }
  // restore ug's relevance order
  return withText.filter((h) => out.includes(h));
}
