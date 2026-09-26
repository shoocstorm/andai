import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it } from 'vitest';
import { TOOLS } from '../agent/tools/registry';
import type { KbInfo } from '../kb/api';
import { useKb } from '../state/kb';
import { useTools } from '../state/tools';
import { Tools } from './Tools';

const kb = (kind: KbInfo['kind']): KbInfo => ({
  slug: 'docs',
  name: 'Docs',
  createdAt: 0,
  sources: [],
  lastIndexedAt: 1,
  lastError: null,
  kindOverride: null,
  kind,
  dir: '/tmp',
  status: 'ready',
  nodes: 5,
  edges: 4,
  sizeBytes: 1,
});

beforeEach(() => {
  useTools.setState({ agentMode: true, maxSteps: 4, minConfidence: 0.3, policies: {}, stats: {} });
  useKb.setState({ kbs: [kb('document')], grounding: 'docs' });
});

describe('Tools', () => {
  it('lists every registered tool with what it runs and its policy', () => {
    render(<Tools />);
    for (const t of TOOLS) {
      const card = within(screen.getByRole('article', { name: t.title }));
      expect(card.getByText(t.command)).toBeInTheDocument();
      expect(card.getByText('Read-only')).toBeInTheDocument();
      expect(card.getByRole('button', { name: 'Auto' })).toHaveAttribute('aria-pressed', 'true');
    }
  });

  it('says which tools a document knowledge base does not get', () => {
    render(<Tools />);
    const code = within(screen.getByRole('article', { name: 'Find usages' }));
    expect(code.getByText(/not offered for “Docs”/)).toBeInTheDocument();
    expect(within(screen.getByRole('article', { name: 'Knowledge search' })).queryByText(/not offered/)).toBeNull();
  });

  it('sets a tool’s policy and the agent settings', async () => {
    const user = userEvent.setup();
    render(<Tools />);
    const card = within(screen.getByRole('article', { name: 'Knowledge search' }));
    await user.click(card.getByRole('button', { name: 'Ask' }));
    expect(useTools.getState().policies.kb_search).toBe('ask');
    await user.click(screen.getByRole('switch', { name: 'Agent mode' }));
    expect(useTools.getState().agentMode).toBe(false);
  });

  it('shows usage stats and who makes the decisions', () => {
    useTools.setState({ stats: { kb_search: { calls: 4, errors: 1, totalMs: 200, lastAt: null } } });
    render(<Tools />);
    expect(screen.getByText('4 calls · 1 failed · avg 50 ms')).toBeInTheDocument();
    expect(screen.getByText('No model loaded')).toBeInTheDocument();
  });
});
