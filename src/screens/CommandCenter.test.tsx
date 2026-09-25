import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it } from 'vitest';
import { addMessage, clearChat } from '../state/chat';
import { useKb } from '../state/kb';
import { useLayout } from '../state/layout';
import { CommandCenter } from './CommandCenter';

beforeEach(() => {
  useLayout.setState({ traceOpen: true });
  clearChat();
  useKb.setState({ kbs: [], grounding: null });
});

describe('Command Center', () => {
  it('guides first-run setup when no model is loaded', () => {
    render(<CommandCenter />);
    expect(screen.getByText(/standing by/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /download/i })).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/load a model/i)).toBeInTheDocument();
    for (const b of screen.getAllByRole('button', { name: /brief me|draft a status|explain a concept|offline/i })) {
      expect(b).toBeDisabled();
    }
  });

  it('only enables send once there is text', async () => {
    const user = userEvent.setup();
    render(<CommandCenter />);
    const send = screen.getByTitle(/send/i);
    expect(send).toBeDisabled();
    await user.type(screen.getByRole('textbox'), 'hello');
    expect(send).toBeEnabled();
  });

  it('renders a completed turn: reasoning chips, answer, sources and trace', () => {
    addMessage({ id: 'u', role: 'user', content: 'What headers?', createdAt: 0 });
    addMessage({
      id: 'a',
      role: 'assistant',
      content: '<think>\n\n</think>\n\nUse **COOP** and COEP [1].',
      createdAt: 0,
      kbName: 'Docs',
      steps: [
        { kind: 'analyze', title: 'Analyze query', detail: 'Identifying ‘headers’ parameters…', status: 'done', ms: 120 },
        { kind: 'retrieve', title: 'Knowledge retrieval', detail: 'Retrieved 1 passage from “Docs” · 90 ms', status: 'done', ms: 90 },
        { kind: 'build', title: 'Assemble context', detail: '1 message · ~800 tokens', status: 'done' },
        { kind: 'generate', title: 'Generate · Qwen3 0.6B', detail: '9 tokens · 30.0 tok/s', status: 'done' },
      ],
      sources: [{ id: 's', name: 'Run it', node_type: 'Concept', file: 'README.md', start_line: 11, end_line: 33, snippet: 'x' }],
      stats: { tokens: 9, tokPerSec: 30, promptTokens: 800, nCtx: 4096, totalMs: 900, model: 'Qwen3 0.6B' },
    });
    render(<CommandCenter />);
    expect(screen.getAllByText('What headers?')).toHaveLength(2); // bubble + trace root task
    expect(screen.getByText(/Analyzing query/)).toBeInTheDocument();
    expect(screen.getByText('COOP').tagName).toBe('STRONG');
    expect(screen.queryByText(/thought process/i)).not.toBeInTheDocument(); // empty think pair is hidden
    expect(screen.getByText(/retrieval\.log — Docs/)).toBeInTheDocument();
    const trace = screen.getByText('Execution Trace').closest('aside')!;
    expect(within(trace).getAllByText(/Completed/i)).toHaveLength(4);
    expect(within(trace).getByText('30.0')).toBeInTheDocument();
    expect(within(trace).getByText('20%')).toBeInTheDocument(); // (800+9)/4096
  });

  it('shows a folded thought process when the model reasoned', () => {
    addMessage({ id: 'a', role: 'assistant', content: '<think>because</think>\n\nanswer', createdAt: 0, steps: [] });
    render(<CommandCenter />);
    expect(screen.getByText(/thought process/i)).toBeInTheDocument();
  });

  it('hides and shows the execution trace panel', async () => {
    const user = userEvent.setup();
    render(<CommandCenter />);
    expect(screen.getByRole('complementary', { name: /execution trace/i })).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Hide execution trace' }));
    expect(useLayout.getState().traceOpen).toBe(false);
    await waitFor(() => expect(screen.queryByRole('complementary', { name: /execution trace/i })).not.toBeInTheDocument());
    await user.click(screen.getByRole('button', { name: 'Show execution trace' }));
    expect(await screen.findByRole('complementary', { name: /execution trace/i })).toBeInTheDocument();
  });

  it('shows live step progress in the header while the trace is hidden', () => {
    useLayout.setState({ traceOpen: false });
    addMessage({
      id: 'a',
      role: 'assistant',
      content: '',
      createdAt: 0,
      streaming: true,
      steps: [
        { kind: 'analyze', title: 'Analyze query', detail: '', status: 'done' },
        { kind: 'retrieve', title: 'Knowledge retrieval', detail: 'Searching…', status: 'running' },
        { kind: 'build', title: 'Assemble context', detail: '', status: 'queued' },
        { kind: 'generate', title: 'Generate', detail: '', status: 'queued' },
      ],
    });
    render(<CommandCenter />);
    expect(screen.getByRole('status')).toHaveTextContent('Step 2/4 · Knowledge retrieval');
  });
});
