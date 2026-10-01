// Knowledge-base store actions with the Rust bridge mocked.
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { KbInfo } from '../kb/api';

const bridge = vi.hoisted(() => ({
  kbs: [] as KbInfo[],
  indexCalls: [] as string[],
  addErrors: [] as string[],
  indexResult: null as Partial<KbInfo> | null,
  fail: {} as Record<string, Error>,
  analyze: [] as { preset: string; target: string | null }[],
  analyzeFail: null as Error | null,
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
  managed: true,
  root: '/tmp/docs',
  sourceCount: 0,
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
    ugStatus: async () => ({ found: true, path: '/bin/ug', version: 'ug version 0.1.21', canInstall: true, installCommand: '' }),
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
      const sources = [{ file: 'a.md', kind: 'MD' as const, bytes: 4, approxTokens: 1, addedAt: 1, status: 'indexed' as const }];
      return mk({ slug, sources, status: 'ready', nodes: 9, edges: 8, ...bridge.indexResult });
    },
    kbRemoveSource: async (slug: string) => mk({ slug, sources: [], status: 'pending' }),
    kbDelete: async () => guard('delete'),
    kbAnalyze: async (_slug: string, preset: string, target: string | null = null) => {
      bridge.analyze.push({ preset, target });
      if (bridge.analyzeFail) throw bridge.analyzeFail;
      if (preset === 'impact') return { output: { title: preset, columns: ['file'], rows: [], targetNotIndexed: [target] } };
      return { output: { title: preset, columns: ['file', 'symbols'], rows: [['a.ts', 3]], rowsTotal: 1 } };
    },
  };
});

const { addFiles, askAbout, createKb, deleteKb, fileImpact, loadInsights, refreshKbs, setKind, useKb } = await import('./kb');
const { useUi } = await import('./ui');

beforeEach(() => {
  Object.assign(bridge, { kbs: [], indexCalls: [], addErrors: [], indexResult: null, fail: {}, analyze: [], analyzeFail: null });
  useKb.setState({ kbs: [], selected: null, grounding: null, logs: {}, kindOverrides: {}, insights: {} });
  useUi.setState({ toasts: [], prefill: null });
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

  it('keeps the user’s kind for a knowledge base in settings, over what Rust reports, until cleared', async () => {
    const code = { file: 'a.ts', kind: 'CODE' as const, bytes: 1, approxTokens: 1, addedAt: 1, status: 'indexed' as const };
    bridge.kbs = [mk({ kind: 'code', sources: [code] })];
    await refreshKbs();
    setKind('docs', 'mixed');
    expect(useKb.getState().kbs[0]).toMatchObject({ kind: 'mixed', kindOverride: 'mixed' });
    await refreshKbs();
    expect(useKb.getState().kbs[0]).toMatchObject({ kind: 'mixed', kindOverride: 'mixed' });
    expect(JSON.parse(localStorage.getItem('andai.kb')!).state.kindOverrides).toEqual({ docs: 'mixed' });
    setKind('docs', null);
    expect(useKb.getState().kbs[0]).toMatchObject({ kind: 'code', kindOverride: null });
  });

  it('persists only preferences, never the KB list or logs', async () => {
    useKb.setState({ kbs: [mk()], logs: { docs: [{ at: 1, line: 'x' }] }, k: 16, grounding: 'docs' });
    const saved = JSON.parse(localStorage.getItem('andai.kb')!).state;
    expect(Object.keys(saved).sort()).toEqual(['grounding', 'k', 'kindOverrides', 'maxChars', 'selected']);
  });
});

describe('code insights', () => {
  const code = (over: Partial<KbInfo> = {}) => mk({ slug: 'repo', kind: 'code', status: 'ready', lastIndexedAt: 5, nodes: 9, edges: 8, ...over });

  it('runs each preset once per index, and again after a re-index', async () => {
    useKb.setState({ kbs: [code()] });
    await loadInsights('repo', ['biggest_files', 'coupling_matrix']);
    expect(bridge.analyze.map((c) => c.preset).sort()).toEqual(['biggest_files', 'coupling_matrix']);
    expect(useKb.getState().insights.repo.results.biggest_files?.rows).toEqual([['a.ts', 3]]);
    expect(useKb.getState().insights.repo.loading).toBe(false);

    await loadInsights('repo', ['biggest_files']);
    expect(bridge.analyze).toHaveLength(2);

    useKb.setState({ kbs: [code({ lastIndexedAt: 6 })] });
    await loadInsights('repo', ['biggest_files']);
    expect(bridge.analyze).toHaveLength(3);
    expect(Object.keys(useKb.getState().insights.repo.results)).toEqual(['biggest_files']);
  });

  it('skips a KB that isn’t indexed, and keeps each preset’s error, an old ug’s as an update hint', async () => {
    useKb.setState({ kbs: [code({ status: 'pending' })] });
    await loadInsights('repo', ['biggest_files']);
    expect(bridge.analyze).toEqual([]);

    useKb.setState({ kbs: [code()] });
    bridge.analyzeFail = new Error('unknown command: analyze');
    await loadInsights('repo', ['biggest_files']);
    expect(useKb.getState().insights.repo.errors.biggest_files).toMatch(/ug upgrade/);
  });

  it('reads a file’s blast radius, with its caveats', async () => {
    const r = await fileImpact('repo', 'src/a.ts');
    expect(bridge.analyze.map((c) => c.target)).toEqual(['src/a.ts', 'src/a.ts', 'src/a.ts', 'src/a.ts']);
    expect(r.impact?.caveats.join(' ')).toMatch(/src\/a\.ts isn't in the index/);
    expect(r.error).toBeNull();
  });

  it('Ask grounds the chat in the KB and hands the question to the composer', () => {
    askAbout('repo', 'Who calls withRetry?');
    expect(useKb.getState().grounding).toBe('repo');
    expect(useUi.getState()).toMatchObject({ prefill: 'Who calls withRetry?', route: 'command' });
  });
});
