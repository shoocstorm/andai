import { listen } from '@tauri-apps/api/event';
import { create } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';
import {
  inTauri,
  kbAddFiles,
  kbCreate,
  kbDelete,
  kbIndex,
  kbList,
  kbRemoveSource,
  kbSetKind,
  ugStatus,
  type KbInfo,
  type KbKind,
  type UgStatus,
} from '../kb/api';
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
      logs: {},
      lastSearch: null,
      hits24h: 0,
    }),
    {
      name: 'andai.kb',
      storage: createJSONStorage(() => localStorage),
      partialize: (s) => ({ selected: s.selected, grounding: s.grounding, k: s.k, maxChars: s.maxChars }),
    },
  ),
);

const errText = (e: unknown) => (e instanceof Error ? e.message : String(e));

function upsert(kb: KbInfo) {
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
  const [kbs, ug] = await Promise.all([kbList().catch(() => []), ugStatus()]);
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
    toast({ tone: 'ok', title: `Knowledge base “${kb.name}” created` });
    return kb;
  } catch (e) {
    toast({ tone: 'error', title: 'Could not create knowledge base', body: errText(e) });
    return null;
  }
}

export async function indexKb(slug: string) {
  const kb = useKb.getState().kbs.find((k) => k.slug === slug);
  if (kb) upsert({ ...kb, status: 'indexing' });
  try {
    const next = await kbIndex(slug);
    upsert(next);
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
    toast({ tone: 'error', title: 'Indexing failed', body: errText(e) });
    await refreshKbs();
  }
}

export async function addFiles(slug: string, paths: string[]) {
  if (!paths.length) return;
  try {
    const [kb, errors] = await kbAddFiles(slug, paths);
    upsert(kb);
    for (const e of errors) toast({ tone: 'warn', title: 'Skipped a file', body: e });
    if (kb.sources.some((s) => s.status === 'pending')) await indexKb(slug);
  } catch (e) {
    toast({ tone: 'error', title: 'Could not add files', body: errText(e) });
  }
}

export async function removeSource(slug: string, file: string) {
  try {
    upsert(await kbRemoveSource(slug, file));
    await indexKb(slug);
  } catch (e) {
    toast({ tone: 'error', title: 'Could not remove source', body: errText(e) });
  }
}

export async function deleteKb(slug: string) {
  try {
    await kbDelete(slug);
    const s = useKb.getState();
    const kbs = s.kbs.filter((k) => k.slug !== slug);
    useKb.setState({
      kbs,
      selected: s.selected === slug ? (kbs[0]?.slug ?? null) : s.selected,
      grounding: s.grounding === slug ? null : s.grounding,
    });
    toast({ tone: 'info', title: 'Knowledge base deleted' });
  } catch (e) {
    toast({ tone: 'error', title: 'Could not delete', body: errText(e) });
  }
}

/** Overrides what the KB is taken to hold (null: derive it from the sources); decides which agent tools apply. */
export async function setKind(slug: string, kind: KbKind | null) {
  try {
    upsert(await kbSetKind(slug, kind));
  } catch (e) {
    toast({ tone: 'error', title: 'Could not change the knowledge base kind', body: errText(e) });
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
