import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { SAMPLES } from '../kb/samples';
import { addSample, useKb } from '../state/kb';
import type { KbInfo } from '../kb/api';
import { Knowledge, Samples } from './Knowledge';

vi.mock('../state/kb', async (original) => ({
  ...(await original<typeof import('../state/kb')>()),
  addSample: vi.fn().mockResolvedValue({ slug: 'x' }),
}));

beforeEach(() => {
  useKb.setState({ kbs: [], grounding: null });
  vi.mocked(addSample).mockClear();
});

describe('sample knowledge bases', () => {
  it('offers each bundled sample and adds the one picked', async () => {
    const user = userEvent.setup();
    const onAdded = vi.fn();
    render(<Samples title="Or try a sample" onAdded={onAdded} />);
    const list = within(screen.getByRole('region', { name: 'Sample knowledge bases' }));
    for (const s of SAMPLES) expect(list.getByText(s.name)).toBeInTheDocument();
    await user.click(list.getByRole('button', { name: 'Add sample Tidewater Ferries · Code' }));
    expect(addSample).toHaveBeenCalledWith('tidewater-code');
    expect(onAdded).toHaveBeenCalled();
  });

  it('shows a sample that is already added as added', () => {
    useKb.setState({ kbs: [{ slug: 'andai-tidewater-ferries-documents', name: 'Tidewater Ferries · Documents' } as never] });
    render(<Samples title="Samples" />);
    const added = screen.getByRole('button', { name: 'Add sample Tidewater Ferries · Documents' });
    expect(added).toBeDisabled();
    expect(added).toHaveTextContent('Added');
    expect(screen.getByRole('button', { name: 'Add sample Tidewater Ferries · Code' })).toBeEnabled();
  });
});

describe('knowledge bases from ug', () => {
  const file = { file: 'src/fare.ts', kind: 'CODE' as const, bytes: 10, approxTokens: 2, addedAt: 1, status: 'indexed' as const };
  const project = (over: Partial<KbInfo> = {}): KbInfo => ({
    slug: 'tidewater',
    name: 'tidewater',
    managed: false,
    root: '/Users/me/code/tidewater',
    createdAt: 1,
    sources: [file],
    sourceCount: 1,
    lastIndexedAt: 2,
    lastError: null,
    kindOverride: null,
    status: 'ready',
    kind: 'code',
    nodes: 24,
    edges: 63,
    sizeBytes: 1,
    ...over,
  });

  it('shows a ug project as read-only: its folder, no adding or removing files', () => {
    useKb.setState({ kbs: [project()], selected: 'tidewater' });
    render(<Knowledge />);
    expect(screen.getByText('Indexed by ug')).toBeInTheDocument();
    expect(screen.getAllByText('/Users/me/code/tidewater').length).toBeGreaterThan(0);
    expect(screen.queryByRole('button', { name: 'Upload_Local' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'View src/fare.ts' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Remove src/fare.ts' })).not.toBeInTheDocument();
  });

  it('lets Andai’s own knowledge bases take and drop files', () => {
    useKb.setState({ kbs: [project({ slug: 'andai-notes', name: 'notes', managed: true })], selected: 'andai-notes' });
    render(<Knowledge />);
    expect(screen.getByRole('button', { name: 'Upload_Local' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Remove src/fare.ts' })).toBeInTheDocument();
  });

  it('says a knowledge base needs UltraGraph when ug is missing', () => {
    useKb.setState({ kbs: [project({ status: 'offline', nodes: 0 })], selected: 'tidewater' });
    render(<Knowledge />);
    expect(screen.getAllByText('Needs ug')).toHaveLength(2); // the knowledge base, and its file
    expect(screen.getByRole('button', { name: /Re-index/ })).toBeDisabled();
  });
});

describe('long lists page', () => {
  const kbOf = (i: number): KbInfo => ({
    slug: `p${i}`,
    name: `project-${i}`,
    managed: false,
    root: `/r/${i}`,
    createdAt: i,
    sources: [],
    sourceCount: 0,
    lastIndexedAt: null,
    lastError: null,
    kindOverride: null,
    status: 'empty',
    kind: 'document',
    nodes: 0,
    edges: 0,
    sizeBytes: 0,
  });

  it('pages the knowledge base tabs, opening on the selected one', async () => {
    const user = userEvent.setup();
    useKb.setState({ kbs: Array.from({ length: 11 }, (_, i) => kbOf(i)), selected: 'p9' });
    render(<Knowledge />);
    const tabs = within(screen.getByRole('navigation', { name: 'Knowledge bases' }));
    expect(tabs.getByText('9–11 of 11')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /project-9/ })).toHaveAttribute('aria-pressed', 'true');
    expect(screen.queryByRole('button', { name: /project-0/ })).not.toBeInTheDocument();
    await user.click(tabs.getByRole('button', { name: 'Previous page' }));
    expect(screen.getByRole('button', { name: /project-0/ })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /project-9/ })).not.toBeInTheDocument();
  });

  it('pages a large source list, 50 files at a time', async () => {
    const user = userEvent.setup();
    const sources = Array.from({ length: 120 }, (_, i) => ({
      file: `src/f${String(i).padStart(3, '0')}.ts`,
      kind: 'CODE' as const,
      bytes: 1,
      approxTokens: 1,
      addedAt: 1,
      status: 'indexed' as const,
    }));
    useKb.setState({ kbs: [{ ...kbOf(0), sources, sourceCount: 120, status: 'ready' }], selected: 'p0' });
    render(<Knowledge />);
    const pager = within(screen.getByRole('navigation', { name: 'Sources' }));
    expect(pager.getByText('1–50 of 120')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'View src/f049.ts' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'View src/f050.ts' })).not.toBeInTheDocument();
    await user.click(pager.getByRole('button', { name: 'Next page' }));
    expect(screen.getByRole('button', { name: 'View src/f050.ts' })).toBeInTheDocument();
    expect(pager.getByText('51–100 of 120')).toBeInTheDocument();
  });
});
