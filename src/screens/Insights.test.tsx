import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { readAnalysis } from '../kb/analysis';
import type { KbInfo } from '../kb/api';
import { askAbout, indexStamp, useKb } from '../state/kb';
import { Insights } from './Insights';

vi.mock('../state/kb', async (original) => ({
  ...(await original<typeof import('../state/kb')>()),
  loadInsights: vi.fn(async () => {}),
  askAbout: vi.fn(),
}));

const kb: KbInfo = {
  slug: 'repo',
  name: 'Repo',
  createdAt: 1,
  sources: [],
  lastIndexedAt: 5,
  lastError: null,
  kindOverride: null,
  managed: false,
  root: '/tmp/repo',
  sourceCount: 0,
  status: 'ready',
  kind: 'code',
  nodes: 9,
  edges: 8,
  sizeBytes: 1,
};

const results = {
  language_breakdown: readAnalysis({ columns: ['language', 'symbols', 'code_lines'], rows: [['typescript', 594, 14364]] }),
  dependency_fanin: readAnalysis({
    columns: ['id', 'depended_on_by', 'loc'],
    rows: [
      ['function_declaration:src/retry.ts:withRetry', 3, 12],
      ['class:src/big.ts:Monolith', 9, 400],
    ],
  }),
  risky_symbols: readAnalysis({ columns: ['id', 'depended_on_by', 'loc'], rows: [['class:src/big.ts:Monolith', 9, 400]] }),
  coupling_matrix: readAnalysis({
    columns: ['from_folder', 'to_folder', 'edges'],
    rows: [
      ['src/modules', 'src', 40],
      ['src', 'src/modules', 26],
    ],
  }),
  dead_code: readAnalysis({ columns: ['id', 'loc'], rows: [], unindexed: ['name_mentions'] }),
};

const seed = (over: Partial<{ results: object; errors: object; loading: boolean }> = {}) =>
  useKb.setState({ insights: { repo: { stamp: indexStamp(kb), results, errors: {}, loading: false, ...over } } });

beforeEach(() => vi.mocked(askAbout).mockClear());

describe('Insights', () => {
  it('shows each card from ug’s results, and flags a risky load-bearing symbol', () => {
    seed();
    render(<Insights kb={kb} />);
    expect(within(screen.getByRole('article', { name: 'Shape' })).getByText('typescript')).toBeInTheDocument();
    const load = screen.getByRole('article', { name: 'Load-bearing' });
    expect(within(load).getByRole('button', { name: 'Open withRetry in src/retry.ts' })).toBeInTheDocument();
    expect(within(load).getAllByText('risky')).toHaveLength(1);
  });

  it('draws the folder coupling as labelled cells', () => {
    seed();
    render(<Insights kb={kb} />);
    expect(screen.getByLabelText('src/modules → src: 40 edges')).toHaveTextContent('40');
    expect(screen.getByLabelText('src → src/modules: 26 edges')).toBeInTheDocument();
    expect(screen.getByLabelText('src: same folder')).toHaveTextContent('');
  });

  it('says when a result is incomplete instead of showing an empty list as “none”', () => {
    seed();
    render(<Insights kb={kb} />);
    const dead = screen.getByRole('article', { name: 'Dead code' });
    expect(within(dead).getByText(/no name_mentions yet/)).toBeInTheDocument();
    expect(within(dead).queryByText('None found.')).toBeNull();
  });

  it('Ask hands a question about the symbol to the chat', async () => {
    seed();
    render(<Insights kb={kb} />);
    const load = screen.getByRole('article', { name: 'Load-bearing' });
    await userEvent.click(within(load).getByRole('button', { name: 'Ask about withRetry' }));
    expect(askAbout).toHaveBeenCalledWith('repo', 'Who calls withRetry?');
  });

  it('shows one notice when every analysis failed', () => {
    seed({ results: {}, errors: { biggest_files: 'This version of ug has no `ug analyze`.', dead_code: 'This version of ug has no `ug analyze`.' } });
    render(<Insights kb={kb} />);
    expect(screen.getByText(/has no `ug analyze`/)).toBeInTheDocument();
    expect(screen.queryByRole('article')).toBeNull();
  });
});
