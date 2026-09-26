// Knowledge-base store actions with the Rust bridge mocked.
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { KbInfo } from '../kb/api';

const bridge = vi.hoisted(() => ({
  kbs: [] as KbInfo[],
  indexCalls: [] as string[],
  addErrors: [] as string[],
  indexResult: null as Partial<KbInfo> | null,
  fail: {} as Record<string, Error>,
}));

const mk = (over: Partial<KbInfo> = {}): KbInfo => ({
  slug: 'docs',
  name: 'Docs',
  createdAt: 1,
  sources: [],
  lastIndexedAt: null,
  lastError: null,
  kindOverride: null,
  kind: 'document',
  dir: '/tmp/docs',
  status: 'empty',
  nodes: 0,
  edges: 0,
  sizeBytes: 0,
  ...over,
});

vi.mock('../kb/api', async (orig) => {
  const real = await orig<typeof import('../kb/api')>();
  const guard = (name: string) => {
    if (bridge.fail[name]) throw bridge.fail[name];
  };
  return {
    ...real,
    kbList: async () => bridge.kbs,
    ugStatus: async () => ({ found: true, path: '/bin/ug', version: 'ug version 0.1.21' }),
    kbCreate: async (name: string) => {
      guard('create');
      return mk({ slug: name.toLowerCase(), name });
    },
    kbAddFiles: async (slug: string, paths: string[]) => {
      guard('add');
      const sources = paths.map((p) => ({
        file: p.split('/').pop()!,
        original: p,
        kind: 'MD' as const,
        bytes: 4,
        approxTokens: 1,
        addedAt: 1,
        status: 'pending' as const,
      }));
      return [mk({ slug, sources, status: 'pending' }), bridge.addErrors];
    },
    kbIndex: async (slug: string) => {
      bridge.indexCalls.push(slug);
      guard('index');
      const sources = [{ file: 'a.md', original: '/x/a.md', kind: 'MD' as const, bytes: 4, approxTokens: 1, addedAt: 1, status: 'indexed' as const }];
      return mk({ slug, sources, status: 'ready', nodes: 9, edges: 8, ...bridge.indexResult });
    },
    kbRemoveSource: async (slug: string) => mk({ slug, sources: [], status: 'pending' }),
    kbDelete: async () => guard('delete'),
  };
});

const { addFiles, createKb, deleteKb, refreshKbs, useKb } = await import('./kb');
const { useUi } = await import('./ui');

beforeEach(() => {
  Object.assign(bridge, { kbs: [], indexCalls: [], addErrors: [], indexResult: null, fail: {} });
  useKb.setState({ kbs: [], selected: null, grounding: null, logs: {} });
  useUi.setState({ toasts: [] });
});
const lastToast = () => useUi.getState().toasts.at(-1);

describe('knowledge base actions', () => {
  it('createKb selects the new KB and grounds chat in it when nothing else does', async () => {
    const kb = await createKb('Specs');
    expect(kb?.slug).toBe('specs');
    expect(useKb.getState()).toMatchObject({ selected: 'specs', grounding: 'specs' });
    expect(lastToast()).toMatchObject({ tone: 'ok' });
  });

  it('createKb keeps an existing grounding choice', async () => {
    useKb.setState({ grounding: 'other' });
    await createKb('Specs');
    expect(useKb.getState().grounding).toBe('other');
  });

  it('addFiles indexes pending sources and reports the result', async () => {
    await createKb('Docs');
    await addFiles('docs', ['/x/a.md', '/x/b.md']);
    expect(bridge.indexCalls).toEqual(['docs']);
    expect(useKb.getState().kbs[0]).toMatchObject({ status: 'ready', nodes: 9 });
    expect(lastToast()).toMatchObject({ tone: 'ok', body: '9 nodes · 8 edges' });
  });

  it('addFiles warns about skipped files but still indexes the rest', async () => {
    bridge.addErrors = ['pic.png: unsupported type'];
    await addFiles('docs', ['/x/a.md', '/x/pic.png']);
    expect(useUi.getState().toasts.some((t) => t.tone === 'warn' && /unsupported/.test(t.body ?? ''))).toBe(true);
    expect(bridge.indexCalls).toEqual(['docs']);
  });

  it('a failed index surfaces ug’s error to the user', async () => {
    bridge.indexResult = { status: 'failed', lastError: 'ug gen failed: disk full' };
    await addFiles('docs', ['/x/a.md']);
    expect(lastToast()).toMatchObject({ tone: 'error', body: 'ug gen failed: disk full' });
  });

  it('a bridge exception becomes an error toast, not an unhandled rejection', async () => {
    bridge.fail.add = new Error('permission denied');
    await expect(addFiles('docs', ['/x/a.md'])).resolves.toBeUndefined();
    expect(lastToast()).toMatchObject({ tone: 'error', body: 'permission denied' });
  });

  it('deleteKb clears selections that pointed at it', async () => {
    useKb.setState({ kbs: [mk(), mk({ slug: 'b', name: 'B' })], selected: 'docs', grounding: 'docs' });
    await deleteKb('docs');
    expect(useKb.getState()).toMatchObject({ selected: 'b', grounding: null });
    expect(useKb.getState().kbs.map((k) => k.slug)).toEqual(['b']);
  });

  it('refreshKbs drops selections for knowledge bases that no longer exist', async () => {
    useKb.setState({ selected: 'gone', grounding: 'gone' });
    bridge.kbs = [mk({ slug: 'kept' })];
    await refreshKbs();
    expect(useKb.getState()).toMatchObject({ selected: 'kept', grounding: null, loaded: true });
    expect(useKb.getState().ug?.found).toBe(true);
  });

  it('persists only preferences, never the KB list or logs', async () => {
    useKb.setState({ kbs: [mk()], logs: { docs: [{ at: 1, line: 'x' }] }, k: 16, grounding: 'docs' });
    const saved = JSON.parse(localStorage.getItem('andai.kb')!).state;
    expect(Object.keys(saved).sort()).toEqual(['grounding', 'k', 'maxChars', 'selected']);
  });
});
