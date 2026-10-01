import { listen } from '@tauri-apps/api/event';
import { create } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';
import {
  inTauri,
  kbAddFiles,
  kbAddSample,
  kbCreate,
  kbDelete,
  kbIndex,
  kbList,
  kbRemoveSource,
  kbSource,
  ugStatus,
  type KbInfo,
  type KbKind,
  type UgStatus,
} from '../kb/api';
import type { SourceView } from '../kb/source';
import { logApp } from './activity';
import { toast } from './ui';

type KbState = {
  kbs: KbInfo[];
  loaded: boolean;
  ug: UgStatus | null;
  /** KB shown on the Knowledge screen. */
  selected: string | null;
  /** KB that grounds chat answers (null = none). */
  grounding: string | null;
  k: number;
  maxChars: number;
  /** What the user says a KB holds, by slug, over what Andai derives; decides which agent tools apply. */
  kindOverrides: Record<string, KbKind>;
  logs: Record<string, { at: number; line: string }[]>;
  lastSearch: { ms: number; hits: number } | null;
  hits24h: number;
};

export const useKb = create<KbState>()(
  persist(
    (): KbState => ({
      kbs: [],
      loaded: false,
      ug: null,
      selected: null,
      grounding: null,
      k: 8,
      maxChars: 6000,
      kindOverrides: {},
      logs: {},
      lastSearch: null,
      hits24h: 0,
    }),
    {
      name: 'andai.kb',
      storage: createJSONStorage(() => localStorage),
      partialize: (s) => ({ selected: s.selected, grounding: s.grounding, k: s.k, maxChars: s.maxChars, kindOverrides: s.kindOverrides }),
    },
  ),
);

const errText = (e: unknown) => (e instanceof Error ? e.message : String(e));
const nameOf = (slug: string) => useKb.getState().kbs.find((k) => k.slug === slug)?.name ?? slug;
/** The activity log's record of a knowledge base event (state/activity.ts; written only while the log is on). */
const logKb = (action: 'create' | 'index' | 'add' | 'sample' | 'remove' | 'delete' | 'kind', slug: string, extra: Record<string, unknown> = {}) =>
  logApp('kb', { action, kb: nameOf(slug), slug, ...extra });

/** Rust reports what a KB holds; the user's override, kept here, wins. */
function withKind(kb: KbInfo): KbInfo {
  const override = useKb.getState().kindOverrides[kb.slug] ?? null;
  return { ...kb, kindOverride: override, kind: override ?? kb.kind };
}

function upsert(raw: KbInfo) {
  const kb = withKind(raw);
  const kbs = useKb.getState().kbs;
  const i = kbs.findIndex((k) => k.slug === kb.slug);
  useKb.setState({ kbs: i === -1 ? [...kbs, kb] : kbs.map((k) => (k.slug === kb.slug ? kb : k)) });
}

function pushLog(slug: string, line: string) {
  const logs = useKb.getState().logs;
  const next = [...(logs[slug] ?? []), { at: Date.now(), line }].slice(-200);
  useKb.setState({ logs: { ...logs, [slug]: next } });
}

export async function refreshKbs() {
  const [raw, ug] = await Promise.all([kbList().catch(() => []), ugStatus()]);
  const kbs = raw.map(withKind);
  const s = useKb.getState();
  const exists = (slug: string | null) => !!slug && kbs.some((k) => k.slug === slug);
  useKb.setState({
    kbs,
    ug,
    loaded: true,
    selected: exists(s.selected) ? s.selected : (kbs[0]?.slug ?? null),
    grounding: exists(s.grounding) ? s.grounding : null,
  });
}

export async function createKb(name: string): Promise<KbInfo | null> {
  try {
    const kb = await kbCreate(name);
    upsert(kb);
    useKb.setState({ selected: kb.slug, grounding: useKb.getState().grounding ?? kb.slug });
    logKb('create', kb.slug);
    toast({ tone: 'ok', title: `Knowledge base “${kb.name}” created` });
    return kb;
  } catch (e) {
    logApp('kb', { action: 'create', kb: name, error: errText(e) });
    toast({ tone: 'error', title: 'Could not create knowledge base', body: errText(e) });
    return null;
  }
}

export async function indexKb(slug: string) {
  const kb = useKb.getState().kbs.find((k) => k.slug === slug);
  if (kb) upsert({ ...kb, status: 'indexing' });
  const started = performance.now();
  try {
    const next = await kbIndex(slug);
    upsert(next);
    const ms = Math.round(performance.now() - started);
    if (next.status === 'failed') logKb('index', slug, { ms, error: next.lastError ?? 'indexing failed' });
    else logKb('index', slug, { ms, sources: next.sources.length, nodes: next.nodes, edges: next.edges, kind: next.kind });
    if (next.status === 'failed') {
      toast({ tone: 'error', title: `Indexing “${next.name}” failed`, body: next.lastError ?? undefined });
    } else if (next.sources.length) {
      toast({
        tone: 'ok',
        title: `“${next.name}” indexed`,
        body: `${next.nodes.toLocaleString()} nodes · ${next.edges.toLocaleString()} edges`,
      });
    }
  } catch (e) {
    logKb('index', slug, { ms: Math.round(performance.now() - started), error: errText(e) });
    toast({ tone: 'error', title: 'Indexing failed', body: errText(e) });
    await refreshKbs();
  }
}

export async function addFiles(slug: string, paths: string[]) {
  if (!paths.length) return;
  try {
    const [kb, errors] = await kbAddFiles(slug, paths);
    upsert(kb);
    logKb('add', slug, { files: paths.length - errors.length, names: paths.map((p) => p.split(/[\\/]/).pop()), skipped: errors });
    for (const e of errors) toast({ tone: 'warn', title: 'Skipped a file', body: e });
    if (kb.sources.some((s) => s.status === 'pending')) await indexKb(slug);
  } catch (e) {
    logKb('add', slug, { error: errText(e) });
    toast({ tone: 'error', title: 'Could not add files', body: errText(e) });
  }
}

/**
 * Adds a sample knowledge base (kb/samples.ts), selects it, grounds chat in
 * it, and indexes it if it isn't yet. Adding one twice reuses the first.
 */
export async function addSample(id: string): Promise<KbInfo | null> {
  try {
    const kb = await kbAddSample(id);
    upsert(kb);
    logKb('sample', kb.slug, { sample: id });
    useKb.setState({ selected: kb.slug, grounding: kb.slug });
    if (kb.status === 'pending' || kb.status === 'failed') await indexKb(kb.slug);
    return useKb.getState().kbs.find((k) => k.slug === kb.slug) ?? kb;
  } catch (e) {
    logApp('kb', { action: 'sample', kb: id, error: errText(e) });
    toast({ tone: 'error', title: 'Could not add the sample', body: errText(e) });
    return null;
  }
}

export async function removeSource(slug: string, file: string) {
  try {
    upsert(await kbRemoveSource(slug, file));
    logKb('remove', slug, { file });
    await indexKb(slug);
  } catch (e) {
    logKb('remove', slug, { file, error: errText(e) });
    toast({ tone: 'error', title: 'Could not remove source', body: errText(e) });
  }
}

export async function deleteKb(slug: string) {
  const name = nameOf(slug);
  try {
    await kbDelete(slug);
    logApp('kb', { action: 'delete', kb: name, slug });
    const s = useKb.getState();
    const kbs = s.kbs.filter((k) => k.slug !== slug);
    useKb.setState({
      kbs,
      selected: s.selected === slug ? (kbs[0]?.slug ?? null) : s.selected,
      grounding: s.grounding === slug ? null : s.grounding,
    });
    toast({ tone: 'info', title: 'Knowledge base deleted' });
  } catch (e) {
    logApp('kb', { action: 'delete', kb: name, slug, error: errText(e) });
    toast({ tone: 'error', title: 'Could not delete', body: errText(e) });
  }
}

/** Overrides what the KB is taken to hold (null: what Andai derives); decides which agent tools apply. */
export function setKind(slug: string, kind: KbKind | null) {
  const { [slug]: _, ...rest } = useKb.getState().kindOverrides;
  useKb.setState({ kindOverrides: kind ? { ...rest, [slug]: kind } : rest });
  const kb = useKb.getState().kbs.find((k) => k.slug === slug);
  if (kb) upsert({ ...kb, kind: kb.kindOverride ? derivedKind(kb) : kb.kind });
  logKb('kind', slug, { kind });
}

/** What the KB holds without the user's override: ug's kind is lost once overridden, so derive it from the files. */
export function derivedKind(kb: Pick<KbInfo, 'sources'>): KbKind {
  const code = kb.sources.filter((s) => s.kind === 'CODE').length;
  return code === 0 ? 'document' : code === kb.sources.length ? 'code' : 'mixed';
}

/** Loads what the source dialog shows about one source; throws a message the dialog can show. */
export async function viewSource(slug: string, file: string): Promise<SourceView> {
  try {
    return await kbSource(slug, file);
  } catch (e) {
    throw new Error(errText(e));
  }
}

export function recordSearch(ms: number, hits: number) {
  useKb.setState((s) => ({ lastSearch: { ms, hits }, hits24h: s.hits24h + hits }));
}

let listening = false;
export async function startKbEvents() {
  if (!inTauri || listening) return;
  listening = true;
  await listen<{ slug: string; line: string }>('kb-progress', (e) => pushLog(e.payload.slug, e.payload.line));
}
