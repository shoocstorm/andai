import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it } from 'vitest';
import { addMessage, clearChat, type AgentStep, type ToolCallRecord } from '../state/chat';
import { requestApproval } from '../state/tools';
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
      stats: { tokens: 9, tokPerSec: 30, promptTokens: 800, nCtx: 4096, totalMs: 900, firstTokenMs: 400, model: 'Qwen3 0.6B' },
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

  it('folds the retrieval log until opened', async () => {
    const user = userEvent.setup();
    addMessage({
      id: 'a',
      role: 'assistant',
      content: 'answer [1]',
      createdAt: 0,
      kbName: 'Docs',
      steps: [],
      sources: [{ id: 's', name: 'Run it', node_type: 'Concept', file: 'README.md', start_line: 11, end_line: 33, snippet: 'x' }],
    });
    render(<CommandCenter />);
    const log = screen.getByText(/retrieval\.log — Docs/).closest('details')!;
    expect(log).not.toHaveAttribute('open');
    await user.click(screen.getByText(/retrieval\.log — Docs/));
    expect(log).toHaveAttribute('open');
  });

  it('copies the operator’s question from its bubble', async () => {
    const user = userEvent.setup();
    addMessage({ id: 'u', role: 'user', content: 'What headers?', createdAt: 0 });
    render(<CommandCenter />);
    await user.click(screen.getByRole('button', { name: 'Copy question' }));
    expect(await navigator.clipboard.readText()).toBe('What headers?');
  });

  it('lists only real tools under Choose tool, with no simulated ones', async () => {
    const user = userEvent.setup();
    render(<CommandCenter />);
    await user.click(screen.getByRole('button', { name: /choose tool/i }));
    expect(screen.getByRole('button', { name: /knowledge search/i })).toBeInTheDocument();
    expect(screen.queryByText(/python interpreter|web search|email dispatcher|simulated/i)).toBeNull();
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

describe('Command Center (agent mode)', () => {
  const decision = {
    question: 'q',
    options: [
      { id: 'answer_now', label: 'A', text: 'Answer now', probability: 0.1 },
      { id: 'kb_search', label: 'B', text: 'Search', probability: 0.85 },
      { id: 'ask_clarification', label: 'C', text: 'Ask', probability: 0.05 },
    ],
    chosen: 'kb_search',
    confidence: 0.85,
    model: 'Qwen3 1.7B',
    slot: 'decider' as const,
    ms: 140,
    seed: 7,
    promptTokens: 420,
  };
  const call = (over: Partial<ToolCallRecord> = {}): ToolCallRecord => ({
    tool: 'kb_search',
    title: 'Knowledge search',
    args: { query: 'wllama COOP headers', scope: 'broad' },
    argsRaw: null,
    argModel: 'Qwen3 0.6B',
    argAttempts: 1,
    policy: 'auto',
    argv: ['search', 'wllama COOP headers', '-k', '8', '--json'],
    startedAt: 0,
    ms: 42,
    status: 'done',
    output: '{"items":[{"file":"README.md"}]}',
    outputBytes: 33,
    observation: '1 passage(s): Run it @ README.md:11-33',
    hits: 1,
    ...over,
  });
  const turn = (agent: AgentStep[], streaming = false) => {
    addMessage({ id: 'u', role: 'user', content: 'What headers?', createdAt: 0 });
    addMessage({
      id: 'a',
      role: 'assistant',
      content: streaming ? '' : 'Use COOP [1].',
      createdAt: 0,
      kbName: 'Docs',
      streaming,
      steps: [
        { kind: 'plan', title: 'Plan & use tools', detail: '1 tool call · 1 passage from “Docs”', status: streaming ? 'running' : 'done' },
        { kind: 'build', title: 'Assemble context', detail: '', status: streaming ? 'queued' : 'done' },
        { kind: 'generate', title: 'Generate', detail: '', status: streaming ? 'queued' : 'done' },
      ],
      agent,
    });
  };

  it('shows each tool call as a chip, and every decision and call in the trace', async () => {
    const user = userEvent.setup();
    turn([
      { id: 's1', index: 0, at: 0, decision, action: 'kb_search', call: call() },
      { id: 's2', index: 1, at: 0, decision: { ...decision, chosen: 'answer_now', confidence: 0.7 }, action: 'answer_now' },
    ]);
    render(<CommandCenter />);
    expect(screen.getByText(/Knowledge search \(query: "wllama COOP headers"/)).toBeInTheDocument();
    const trace = within(screen.getByText('Execution Trace').closest('aside')!);
    expect(trace.getByText(/Step 1 ·/)).toBeInTheDocument();
    expect(trace.getByText('Answer now')).toBeInTheDocument();
    expect(trace.getByText(/Chosen with 85% of 3 options/)).toBeInTheDocument();

    await user.click(trace.getAllByRole('button', { name: 'Details' })[0]);
    const bars = trace.getByRole('list', { name: /probabilities/i });
    expect(within(bars).getAllByRole('listitem').map((li) => li.textContent)).toEqual([
      expect.stringMatching(/B Knowledge search.*85%/),
      expect.stringMatching(/A Answer now.*10%/),
      expect.stringMatching(/C Ask to clarify.*5\.0%/),
    ]);
    expect(trace.getByText(/Qwen3 1\.7B · decision model · 140 ms/)).toBeInTheDocument();
    // shown exactly as a shell would run it
    expect(trace.getByText("ug search 'wllama COOP headers' -k 8 --json")).toBeInTheDocument();
    expect(trace.getByText(/"file":"README.md"/)).toBeInTheDocument();
    expect(trace.getByRole('button', { name: /copy trace/i })).toBeInTheDocument();
  });

  it('asks for approval inline and resolves the waiting call', async () => {
    const user = userEvent.setup();
    const ctl = new AbortController();
    const approved = requestApproval('s1', ctl.signal);
    turn([{ id: 's1', index: 0, at: 0, decision, action: 'kb_search', call: call({ status: 'awaiting', approval: 'pending', policy: 'ask', output: undefined }) }], true);
    render(<CommandCenter />);
    const card = within(screen.getByRole('group', { name: /approve knowledge search/i }));
    expect(card.getByText(/"query": "wllama COOP headers"/)).toBeInTheDocument();
    await user.click(card.getByRole('button', { name: /approve/i }));
    await expect(approved).resolves.toBe(true);
  });

  it('copies a step, a call’s shell command and the turn’s debug report', async () => {
    const user = userEvent.setup();
    turn([{ id: 's1', index: 0, at: 0, decision, action: 'kb_search', call: call() }]);
    render(<CommandCenter />);
    const trace = within(screen.getByText('Execution Trace').closest('aside')!);

    await user.click(trace.getByRole('button', { name: 'Copy step 1 as JSON' }));
    expect(JSON.parse(await navigator.clipboard.readText())).toMatchObject({ action: 'kb_search', call: { args: { query: 'wllama COOP headers' } } });
    expect(trace.getByRole('button', { name: 'Copy step 1 as JSON' }).querySelector('svg')).toBeTruthy();

    await user.click(trace.getByRole('button', { name: 'Details' }));
    await user.click(trace.getByRole('button', { name: 'Copy command' }));
    expect(await navigator.clipboard.readText()).toBe("ug search 'wllama COOP headers' -k 8 --json");

    await user.click(screen.getAllByRole('button', { name: 'Copy debug report' })[0]);
    const report = await navigator.clipboard.readText();
    expect(report).toContain('### Question\nWhat headers?');
    expect(report).toContain('1. kb_search @ 85%');
  });

  it('shows how long each decision took, and the turn’s total, without opening details', () => {
    turn([
      { id: 's1', index: 0, at: 0, decision, action: 'kb_search', call: call() },
      { id: 's2', index: 1, at: 0, decision: { ...decision, chosen: 'answer_now', ms: 1480 }, action: 'answer_now' },
    ]);
    render(<CommandCenter />);
    const trace = within(screen.getByText('Execution Trace').closest('aside')!);
    expect(trace.getByText('decided in 140 ms')).toBeInTheDocument();
    expect(trace.getByText('decided in 1.48 s')).toBeInTheDocument();
    expect(trace.getByLabelText('Decision time')).toHaveTextContent('2 decisions · 1.62 s total · 810 ms each on average · Qwen3 1.7B');
    expect(screen.getByText(/decided 2× in 1\.62 s/)).toBeInTheDocument();
  });

  it('shows a decision call’s input and output in the details, and copies it', async () => {
    const user = userEvent.setup();
    const io = {
      request: {
        messages: [
          { role: 'system', content: 'Make the requested decision.' },
          { role: 'user', content: 'State:\nUser request:\nWhat headers?\n\nAllowed options:\nA. Answer now\nB. Search' },
        ],
        params: { max_tokens: 1, grammar: 'root ::= "A" | "B" | "C"' },
      },
      response: { sampled: 'B', topLogprobs: [{ token: 'B', logprob: Math.log(0.85) }, { token: 'A', logprob: Math.log(0.1) }] },
    };
    turn([{ id: 's1', index: 0, at: 0, decision: { ...decision, io }, action: 'kb_search', call: call() }]);
    render(<CommandCenter />);
    const trace = within(screen.getByText('Execution Trace').closest('aside')!);
    await user.click(trace.getByRole('button', { name: 'Details' }));
    expect(trace.getByText('Decision call')).toBeInTheDocument();
    // the prompt reads as text, newlines intact, one block per message
    expect(trace.getByText((_, el) => el?.tagName === 'PRE' && !!el.textContent?.startsWith('State:\nUser request:\nWhat headers?\n'))).toBeInTheDocument();
    expect(trace.getByText(/"grammar": "root ::= \\"A\\"/)).toBeInTheDocument();
    const table = trace.getByRole('table');
    expect(within(table).getAllByRole('row').map((r) => r.textContent)).toEqual([
      'TokenLogprobShare of listed',
      expect.stringMatching(/^"B" Knowledge search-0\.163\d+%$/),
      expect.stringMatching(/^"A" Answer now-2\.303\d+%$/),
    ]);
    await user.click(trace.getByRole('button', { name: 'Copy decision call' }));
    expect(JSON.parse(await navigator.clipboard.readText())).toEqual(io);
    await user.click(trace.getByRole('button', { name: 'Copy input: prompt' }));
    expect(await navigator.clipboard.readText()).toBe(`[system]\nMake the requested decision.\n\n[user]\n${io.request.messages[1].content}`);
  });

  it('shows a failed decision call with its time, error and request', async () => {
    const user = userEvent.setup();
    const io = { request: { messages: [{ role: 'user', content: 'State: failing' }], params: {} }, response: null };
    turn([
      {
        id: 's1',
        index: 0,
        at: 0,
        decision: null,
        action: 'kb_search',
        fallback: 'decision-failed',
        failedDecision: { error: 'context overflow', ms: 640, io },
        note: 'Decision failed (context overflow), so searching the knowledge base instead.',
        call: call(),
      },
    ]);
    render(<CommandCenter />);
    const trace = within(screen.getByText('Execution Trace').closest('aside')!);
    expect(trace.getByText('Decision failed after 640 ms')).toBeInTheDocument();
    await user.click(trace.getByRole('button', { name: 'Details' }));
    expect(trace.getByText('State: failing')).toBeInTheDocument();
    expect(trace.getByText(/No reply: the call failed/)).toBeInTheDocument();
  });

  it('shows a fallback note when the loop overrode the decision', () => {
    turn([{ id: 's1', index: 0, at: 0, decision: null, action: 'kb_search', note: 'Decision failed (boom), so searching the knowledge base instead.', call: call() }]);
    render(<CommandCenter />);
    expect(screen.getByText(/Decision failed \(boom\)/)).toBeInTheDocument();
  });
});
