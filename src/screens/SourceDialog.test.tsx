import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { KbInfo, Source } from '../kb/api';
import type { SourceView } from '../kb/source';
import { viewSource } from '../state/kb';
import { Sources } from './Knowledge';

vi.mock('../state/kb', async (original) => ({
  ...(await original<typeof import('../state/kb')>()),
  viewSource: vi.fn(),
}));

const src = (file: string, kind: Source['kind'] = 'MD'): Source => ({
  file,
  kind,
  bytes: 686,
  approxTokens: 171,
  addedAt: 1_700_000_000,
  status: 'indexed',
});

const kb: KbInfo = {
  slug: 'andai-ferries',
  name: 'Ferries',
  createdAt: 1,
  sources: [src('operations.md'), src('refund-policy.md')],
  lastIndexedAt: 1_700_000_100,
  lastError: null,
  kindOverride: null,
  managed: true,
  root: '/tmp/docs',
  sourceCount: 0,
  status: 'ready',
  kind: 'document',
  nodes: 10,
  edges: 9,
  sizeBytes: 1,
};

const TEXT = '# Handbook\n\n## Fleet\n\nMV Kestrel carries 212.\n\n## Routes\n\nHarlow to Pellin.\n';
const view = (file: string, over: Partial<SourceView> = {}): SourceView => ({
  source: src(file),
  path: `/Users/me/Library/Application Support/dev.andai.agent/kb/ferries/docs/${file}`,
  text: TEXT,
  textTruncated: false,
  structure: {
    files: [{
      facts: { language: 'markdown', classification: 'documentation', lines: 9 },
      items: [
        { role: 'outline', id: 'h:Handbook', name: 'Handbook', node_type: 'Concept', start_line: 1, end_line: 9 },
        { role: 'outline', id: 'h:Fleet', name: 'Fleet', node_type: 'Concept', start_line: 3, end_line: 6, doc: 'MV Kestrel carries 212.' },
        { role: 'outline', id: 'h:Routes', name: 'Routes', node_type: 'Concept', start_line: 7, end_line: 9 },
        { role: 'sibling', name: 'refund-policy.md', file: 'refund-policy.md' },
        { role: 'sibling', name: 'elsewhere.md', file: 'elsewhere.md' },
      ],
    }],
  },
  structureError: null,
  ...over,
});

beforeEach(() => {
  vi.mocked(viewSource).mockReset().mockImplementation(async (_slug, file) => view(file));
});

const openDialog = async () => {
  const user = userEvent.setup();
  render(<Sources kb={kb} onDelete={() => {}} />);
  await user.click(screen.getByRole('button', { name: 'View operations.md' }));
  const dialog = await screen.findByRole('dialog', { name: 'Source operations.md' });
  await within(dialog).findByText(/kb\/ferries\/docs\/operations\.md$/);
  return { user, dialog: within(dialog) };
};

describe('source dialog', () => {
  it('opens from a source name with its metadata', async () => {
    const { dialog } = await openDialog();
    expect(viewSource).toHaveBeenCalledWith('andai-ferries', 'operations.md');
    expect(dialog.getByRole('tab', { name: 'Overview' })).toHaveAttribute('aria-selected', 'true');
    expect(dialog.getByText('andai-ferries')).toBeInTheDocument();
    expect(dialog.getByText('documentation')).toBeInTheDocument();
  });

  it('shows the content rendered or as numbered text', async () => {
    const { user, dialog } = await openDialog();
    await user.click(dialog.getByRole('tab', { name: 'Content' }));
    expect(dialog.getByRole('heading', { name: 'Fleet' })).toBeInTheDocument();
    await user.click(dialog.getByRole('button', { name: 'Text' }));
    const lines = dialog.getByRole('region', { name: 'File text with line numbers' });
    expect(lines.querySelectorAll('[data-line]')).toHaveLength(9);
    expect(lines).toHaveTextContent('MV Kestrel carries 212.');
  });

  it('jumps from an outline entry to its lines', async () => {
    const { user, dialog } = await openDialog();
    await user.click(dialog.getByRole('tab', { name: /Structure/ }));
    await user.click(dialog.getByRole('button', { name: /Fleet/ }));
    expect(dialog.getByRole('tab', { name: 'Content' })).toHaveAttribute('aria-selected', 'true');
    const hits = [...dialog.getByRole('region', { name: 'File text with line numbers' }).querySelectorAll('.hit')];
    expect(hits.map((h) => h.getAttribute('data-line'))).toEqual(['3', '4', '5', '6']);
  });

  it('opens a related file that is in the knowledge base, and only those', async () => {
    const { user, dialog } = await openDialog();
    expect(dialog.queryByRole('button', { name: 'elsewhere.md' })).toBeNull();
    await user.click(dialog.getByRole('button', { name: 'refund-policy.md' }));
    await waitFor(() => expect(viewSource).toHaveBeenLastCalledWith('andai-ferries', 'refund-policy.md'));
    expect(await screen.findByRole('dialog', { name: 'Source refund-policy.md' })).toBeInTheDocument();
  });

  it('says why a structure is missing, and shows a load error', async () => {
    vi.mocked(viewSource).mockResolvedValueOnce(view('operations.md', { structure: null, structureError: 'Not indexed yet.' }));
    const { user, dialog } = await openDialog();
    await user.click(dialog.getByRole('tab', { name: /Structure/ }));
    expect(dialog.getByText('Not indexed yet.')).toBeInTheDocument();
    await user.click(dialog.getByRole('button', { name: 'Close dialog' }));
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());

    vi.mocked(viewSource).mockRejectedValueOnce(new Error('“x.md” isn’t in this knowledge base.'));
    await user.click(screen.getByRole('button', { name: 'View refund-policy.md' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('isn’t in this knowledge base');
  });
});
