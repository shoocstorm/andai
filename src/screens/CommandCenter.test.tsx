import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { addMessage, clearChat, type AgentStep, type Message, type ToolCallRecord } from '../state/chat';
import { requestApproval } from '../state/tools';
import { useKb } from '../state/kb';
import { useLayout } from '../state/layout';
import { CommandCenter, reasoningSummary } from './CommandCenter';

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

  it('suggests the sample’s own questions when chat is grounded in a sample knowledge base', () => {
    const kb = { slug: 'tidewater-ferries-code', name: 'Tidewater Ferries · Code', kind: 'code', status: 'ready', sources: [] };
    useKb.setState({ kbs: [kb as never], grounding: kb.slug });
    render(<CommandCenter />);
    expect(screen.getByText(/Try asking Tidewater Ferries · Code/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Which functions call computeFare?' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /brief me/i })).toBeNull();
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
    expect(screen.getByText(/^Passages found in “Docs”$/)).toBeInTheDocument();
    const trace = screen.getByText('Execution Trace').closest('aside')!;
    expect(within(trace).getAllByText(/Completed/i)).toHaveLength(4);
    expect(within(trace).getByText('30.0')).toBeInTheDocument();
    expect(within(trace).getByText('20%')).toBeInTheDocument(); // (800+9)/4096
  });

  it('folds the passages found until opened', async () => {
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
    const log = screen.getByText(/^Passages found in “Docs”$/).closest('details')!;
    expect(log).not.toHaveAttribute('open');
    await user.click(screen.getByText(/^Passages found in “Docs”$/));
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

  it('folds a finished turn’s reasoning to one line that opens to every step', async () => {
    addMessage({ id: 'u', role: 'user', content: 'q', createdAt: 0 });
    addMessage({
      id: 'a',
      role: 'assistant',
      content: 'Use COOP [1].',
      createdAt: 0,
      kbName: 'Docs',
      sources: [{ id: 'h1', name: 'n', node_type: 'Section', file: 'a.md', start_line: 1, end_line: 2, snippet: 's' }],
      stats: { tokens: 10, tokPerSec: 50, promptTokens: 300, nCtx: 4096, totalMs: 1260, firstTokenMs: 900, model: 'Qwen3' },
      steps: [
        { kind: 'retrieve', title: 'Knowledge retrieval', detail: 'Retrieved 1 passage', status: 'done' },
        { kind: 'generate', title: 'Generate', detail: '10 tokens', status: 'done' },
      ],
    });
    render(<CommandCenter />);
    const folded = document.querySelector('.cc-reasoning') as HTMLDetailsElement;
    expect(folded.open).toBe(false);
    expect(within(folded.querySelector('summary')!).getByText('Reasoned in 1.3 s · 1 passage')).toBeInTheDocument();
    await userEvent.click(folded.querySelector('summary')!);
    expect(folded.open).toBe(true);
    expect(within(folded).getByText('Searching knowledge base: Retrieved 1 passage')).toBeVisible();
  });

  it('numbers each step’s chip as the trace does, shows its icon on the trace card, and flashes that card on a click', async () => {
    const raf = vi.spyOn(window, 'requestAnimationFrame').mockImplementation((cb) => {
      cb(0);
      return 0;
    });
    useLayout.setState({ traceOpen: true });
    addMessage({ id: 'u', role: 'user', content: 'q', createdAt: 0 });
    addMessage({
      id: 'a',
      role: 'assistant',
      content: 'Use COOP.',
      createdAt: 0,
      steps: [
        { kind: 'analyze', title: 'Analyze query', detail: '', status: 'skipped' },
        { kind: 'retrieve', title: 'Knowledge retrieval', detail: 'Retrieved 1 passage', status: 'done' },
        { kind: 'generate', title: 'Generate', detail: '10 tokens', status: 'done' },
      ],
    });
    render(<CommandCenter />);
    // a skipped step keeps its number, as in the trace: retrieval is step 02
    const chip = screen.getByRole('button', { name: 'Show step 02 in the trace: Searching knowledge base: Retrieved 1 passage' });
    expect(within(chip).getByText('02')).toHaveClass('step-no');
    const card = document.getElementById('trace-a-retrieve')!;
    expect(card).toHaveTextContent('Step 02');
    expect(card.querySelector('.trace-step-icon')).not.toBeNull();
    await userEvent.click(chip);
    expect(card).toHaveClass('flash');
    raf.mockRestore();
  });

  it('shows a running turn as one live line: a dot per step and what the current one is doing', () => {
    addMessage({ id: 'u', role: 'user', content: 'q', createdAt: 0 });
    addMessage({
      id: 'a',
      role: 'assistant',
      content: '',
      createdAt: 0,
      streaming: true,
      steps: [
        { kind: 'retrieve', title: 'Knowledge retrieval', detail: 'Retrieved 1 passage', status: 'done' },
        { kind: 'generate', title: 'Generate', detail: '12 tokens · 40 tok/s', status: 'running' },
      ],
    });
    render(<CommandCenter />);
    const live = document.querySelector('.cc-live') as HTMLElement;
    expect(live).toHaveTextContent('Generating response: 12 tokens · 40 tok/s');
    expect(live.querySelectorAll('.cc-live-dots i')).toHaveLength(2);
    expect(document.querySelector('.cc-reasoning')).toBeNull();
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

describe('reasoningSummary', () => {
  const base = { id: 'a', role: 'assistant' as const, content: '', createdAt: 0 };
  it('says what a turn used, and flags a failed step before a claim to check', () => {
    expect(reasoningSummary({ ...base, stats: { tokens: 1, tokPerSec: 1, promptTokens: 1, nCtx: 1, totalMs: 12_400, firstTokenMs: 1, model: 'm' }, kbName: 'Docs' })).toEqual({ text: 'Reasoned in 12 s · no passages', warn: null });
    const flagged = { model: 'L', modelId: 'laya-en', ms: 1, modelMs: 1, flagBelow: 0.1, items: [{ sentence: 'A b.', n: 1, cites: [1], source: 's', score: 0.01, flagged: true, inputTokens: 1, truncated: false }] };
    expect(reasoningSummary({ ...base, support: flagged }).warn).toBe('1 claim to check');
    expect(reasoningSummary({ ...base, support: flagged, steps: [{ kind: 'verify', title: 'Claim check', detail: '', status: 'error' }] }).warn).toBe('1 step failed');
    expect(reasoningSummary({ ...base, stopped: true }).text).toBe('Stopped');
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

  const whyDialog = async (user: ReturnType<typeof userEvent.setup>, trace: ReturnType<typeof within>, n = 0) => {
    await user.click(trace.getAllByRole('button', { name: /why this step/i })[n]);
    return within(screen.getByRole('dialog', { name: `Why step ${n + 1}` }));
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

    // the tool call opens in a dialog, without the decision
    await user.click(trace.getByRole('button', { name: 'Tool call' }));
    const dialog = within(screen.getByRole('dialog', { name: 'Tool call, step 1' }));
    expect(dialog.getByText("ug search 'wllama COOP headers' -k 8 --json")).toBeInTheDocument();
    await user.click(dialog.getByText('Raw output from ug'));
    expect(dialog.getByText(/"file":"README.md"/)).toBeInTheDocument();
    expect(dialog.queryByRole('list', { name: /probabilities/i })).toBeNull();
    await user.click(dialog.getByRole('button', { name: /^close$/i }));
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
    // a step without a tool call has no tool-call button
    expect(trace.getAllByRole('button', { name: /tool call$/i })).toHaveLength(1);
    expect(trace.getByRole('button', { name: /copy trace/i })).toBeInTheDocument();
  });

  it('explains a step in a dialog: why, what the model saw, what it returned', async () => {
    const user = userEvent.setup();
    const io = {
      request: {
        state: 'User request:\nWhat headers?\n\nTool results so far: none.',
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
    const dialog = await whyDialog(user, within(screen.getByText('Execution Trace').closest('aside')!));
    expect(dialog.getByRole('heading', { name: 'Why: Knowledge search' })).toBeInTheDocument();
    expect(dialog.getByText(/Qwen3 1\.7B · decision model · decided in 140 ms · 420 input tokens/)).toBeInTheDocument();
    const why = within(dialog.getByRole('region', { name: 'Why this step' }));
    expect(why.getAllByRole('listitem').map((li) => li.textContent)).toEqual([
      'The agent asked Qwen3 1.7B “q” and it scored 3 options in 140 ms.',
      'Knowledge search scored highest at 85%, ahead of Answer now at 10%.',
      'So the agent ran Knowledge search with query: "wllama COOP headers", scope: "broad".',
    ]);
    const saw = within(dialog.getByRole('region', { name: 'What the model saw' }));
    // the state reads as text, newlines intact
    expect(saw.getByText((_, el) => el?.tagName === 'PRE' && el.textContent === io.request.state)).toBeInTheDocument();
    expect(saw.getAllByRole('listitem').map((li) => li.textContent)).toEqual(['AAnswer now Answer now', 'BKnowledge search Search', 'CAsk to clarify Ask']);
    const bars = within(dialog.getByRole('region', { name: 'What it returned' })).getByRole('list', { name: /probabilities/i });
    expect(within(bars).getAllByRole('listitem').map((li) => li.textContent)).toEqual([
      expect.stringMatching(/B Knowledge search.*85%/),
      expect.stringMatching(/A Answer now.*10%/),
      expect.stringMatching(/C Ask to clarify.*5\.0%/),
    ]);
    // the exact call is folded away, raw readout included
    await user.click(dialog.getByText(/Exact call/));
    expect(within(dialog.getByRole('table')).getAllByRole('row').map((r) => r.textContent)).toEqual([
      'TokenLogprobShare of listed',
      expect.stringMatching(/^"B" Knowledge search-0\.163\d+%$/),
      expect.stringMatching(/^"A" Answer now-2\.303\d+%$/),
    ]);
    await user.click(dialog.getByRole('button', { name: 'Copy input: prompt' }));
    expect(await navigator.clipboard.readText()).toBe(`[system]\nMake the requested decision.\n\n[user]\n${io.request.messages[1].content}`);
    await user.click(dialog.getByRole('button', { name: 'Copy decision call' }));
    expect(JSON.parse(await navigator.clipboard.readText())).toMatchObject({ step: 1, action: 'kb_search', decision: { io } });
    await user.click(dialog.getByRole('button', { name: /^close$/i }));
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull()); // after its exit animation
  });

  it('shows what the argument writer saw: each attempt’s prompt, parameters, grammar, raw reply and why it was rejected', async () => {
    const user = userEvent.setup();
    const params = { max_tokens: 200, temperature: 0, grammar: 'root ::= "{" ws query ws "}"', chat_template_kwargs: { enable_thinking: false } };
    const argIO = {
      schema: { type: 'object' as const, properties: { query: { type: 'string' as const } }, required: ['query'], additionalProperties: false as const },
      calls: [
        {
          model: 'Qwen3 1.7B · MLX',
          messages: [
            { role: 'system' as const, content: 'You fill in the arguments for one tool call.' },
            { role: 'user' as const, content: 'User request:\nWhat headers?\n\nTool: Knowledge search.' },
          ],
          params,
          reply: '{"query":"x"}',
          errors: ['missing scope'],
          ms: 84,
          promptTokens: 412,
        },
        { model: 'Qwen3 1.7B · MLX', messages: [{ role: 'user' as const, content: 'That was not valid: missing scope.' }], params, reply: '{"query":"wllama COOP headers","scope":"broad"}', errors: [], ms: 90, promptTokens: 450 },
      ],
    };
    turn([{ id: 's1', index: 0, at: 0, decision, action: 'kb_search', call: call({ argIO, argAttempts: 2 }) }]);
    render(<CommandCenter />);
    await user.click(within(screen.getByText('Execution Trace').closest('aside')!).getByRole('button', { name: 'Tool call' }));
    const dialog = within(screen.getByRole('dialog', { name: 'Tool call, step 1' }));
    // folded under the arguments, with who wrote them and a one-line summary
    expect(within(dialog.getByRole('region', { name: 'What it was given' })).getByText('Written by Qwen3 0.6B in 2 attempts')).toBeInTheDocument();
    const fold = dialog.getByText('How the arguments were written: Qwen3 1.7B · MLX · 2 attempts · 174 ms · accepted').closest('details')!;
    expect(fold).not.toHaveAttribute('open');
    await user.click(dialog.getByText(/How the arguments were written/));
    expect(fold).toHaveAttribute('open');
    const writer = within(dialog.getByRole('region', { name: 'What the argument writer saw' }));
    const first = within(writer.getByLabelText('Attempt 1'));
    expect(first.getByText(/Attempt 1/).parentElement).toHaveTextContent('Attempt 1 · Qwen3 1.7B · MLX · 84 ms · 412 prompt tokens · rejected');
    expect(first.getByText('You fill in the arguments for one tool call.')).toBeInTheDocument();
    expect(first.getByText((_, el) => el?.tagName === 'PRE' && el.textContent === 'User request:\nWhat headers?\n\nTool: Knowledge search.')).toBeInTheDocument();
    // parameters as sent, the grammar folded away
    expect(first.getByText((_, el) => el?.tagName === 'PRE' && /"temperature": 0/.test(el.textContent ?? '') && /"max_tokens": 200/.test(el.textContent ?? ''))).toBeInTheDocument();
    await user.click(first.getByText(/Grammar the reply was held to/));
    expect(first.getByText('root ::= "{" ws query ws "}"')).toBeInTheDocument();
    expect(first.getByText('{"query":"x"}')).toBeInTheDocument();
    expect(first.getByText('missing scope')).toBeInTheDocument();
    expect(within(writer.getByLabelText('Attempt 2')).getByText(/accepted/)).toBeInTheDocument();
    await user.click(dialog.getByRole('button', { name: 'Copy argument call' }));
    expect(JSON.parse(await navigator.clipboard.readText())).toEqual(argIO);
  });

  it('says when the agent set a tool’s arguments without the argument writer', async () => {
    const user = userEvent.setup();
    turn([{ id: 's1', index: 0, at: 0, decision, action: 'kb_read_lines', call: call({ argIO: { schema: null, calls: [], note: 'No argument writer: the agent set the range to the whole section the search had clipped.' } }) }]);
    render(<CommandCenter />);
    await user.click(within(screen.getByText('Execution Trace').closest('aside')!).getByRole('button', { name: 'Tool call' }));
    expect(screen.getByText('How the arguments were written: no request to the chat model')).toBeInTheDocument();
    const writer = within(screen.getByRole('region', { name: 'What the argument writer saw' }));
    expect(writer.getByText(/No argument writer: the agent set the range/)).toBeInTheDocument();
    expect(writer.queryByLabelText('Attempt 1')).toBeNull();
  });

  it('shows Laya’s scores for what a tool call found, the pieces of a long passage, and a failed scoring', async () => {
    const user = userEvent.setup();
    const scored = {
      model: 'Laya English',
      ms: 41,
      rows: 4,
      items: [
        { source: 'fleet.md:10-37', name: 'Dry-dock schedule', score: 0.82, chunks: 3, inputTokens: 480, truncated: false },
        { source: 'log.md:1-5', name: 'Incident log', score: 0.03, chunks: 1, inputTokens: 120, truncated: true },
      ],
    };
    turn([
      { id: 's1', index: 0, at: 0, decision, action: 'kb_search', call: call({ observation: '2 passage(s): Dry-dock schedule; Incident log', scored }) },
      { id: 's2', index: 1, at: 0, decision, action: 'kb_search', call: call({ observation: '1 passage(s)', scored: { model: 'Laya English', ms: 0, rows: 0, items: [], error: 'No Laya model is loaded.' } }) },
    ]);
    render(<CommandCenter />);
    const trace = within(screen.getByText('Execution Trace').closest('aside')!);
    await user.click(trace.getAllByRole('button', { name: 'Tool call' })[0]);
    let dialog = within(screen.getByRole('dialog', { name: 'Tool call, step 1' }));
    expect(dialog.getByText('Scored by Laya English')).toBeInTheDocument();
    expect(dialog.getByText('2 passages in 4 pieces · 41 ms · most useful: “Dry-dock schedule” (82% likely to help)')).toBeInTheDocument();
    expect(within(dialog.getByRole('list', { name: 'Passage scores' })).getAllByRole('listitem').map((li) => li.textContent)).toEqual([
      'fleet.md:10-37 · 82% · best of 3 pieces',
      'log.md:1-5 · 3% · cut to fit',
    ]);
    await user.click(dialog.getByRole('button', { name: /^close$/i }));
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
    await user.click(trace.getAllByRole('button', { name: 'Tool call' })[1]);
    dialog = within(screen.getByRole('dialog', { name: 'Tool call, step 2' }));
    expect(dialog.getByText(/Couldn’t score these passages \(No Laya model is loaded\.\)/)).toBeInTheDocument();
  });

  it('explains a tool call in a dialog: what happened, what it was given, what it ran and found', async () => {
    const user = userEvent.setup();
    const output = JSON.stringify({
      items: [
        { id: 'n1', name: 'Run it', node_type: 'Concept', file: 'README.md', start_line: 11, end_line: 33, snippet: 'Serve with COOP and COEP headers.' },
        { id: 'n2', name: 'vite.config.ts', node_type: 'File', file: 'vite.config.ts', snippet: 'headers: { COOP }' },
      ],
    });
    turn([{ id: 's1', index: 0, at: 0, decision, action: 'kb_search', call: call({ output, outputBytes: 2048, hits: 2 }) }]);
    render(<CommandCenter />);
    await user.click(within(screen.getByText('Execution Trace').closest('aside')!).getByRole('button', { name: 'Tool call' }));
    const dialog = within(screen.getByRole('dialog', { name: 'Tool call, step 1' }));
    expect(dialog.getByRole('heading', { name: /Knowledge search/ })).toHaveTextContent('Done');
    expect(within(dialog.getByRole('region', { name: 'What happened' })).getAllByRole('listitem').map((li) => li.textContent)).toEqual([
      'Qwen3 0.6B filled in the arguments: query: "wllama COOP headers", scope: "broad".',
      'It ran without asking: the tool is set to Auto, and it only reads the selected knowledge base.',
      'ug answered in 42 ms with 2,048 bytes of results.',
      '2 passages went into the answer’s context, new or with text the context didn’t have yet.',
    ]);
    // each argument by name, with what the tool's schema says it means
    const given = within(dialog.getByRole('region', { name: 'What it was given' }));
    expect(given.getByText('scope').closest('div')).toHaveTextContent(/scope.*focused returns direct matches only.*broad/);
    // the passages, read from the output the way the agent read them
    const found = within(dialog.getByRole('list', { name: 'Passages found' }));
    expect(found.getAllByRole('listitem').map((li) => li.textContent)).toEqual([
      expect.stringMatching(/^README\.md:11-33Run it · ConceptServe with COOP and COEP headers\.$/),
      expect.stringMatching(/^vite\.config\.tsFileheaders: \{ COOP \}$/),
    ]);
    await user.click(dialog.getByRole('button', { name: 'Copy tool call' }));
    expect(JSON.parse(await navigator.clipboard.readText())).toMatchObject({ step: 1, tool: 'kb_search', args: { scope: 'broad' } });
  });

  it('explains a denied call and a failed one', async () => {
    const user = userEvent.setup();
    turn([
      { id: 's1', index: 0, at: 0, decision, action: 'kb_search', call: call({ status: 'denied', policy: 'ask', approval: 'denied', argv: undefined, output: undefined, outputBytes: undefined, ms: undefined, hits: undefined, observation: 'The user declined this call; do not ask for it again.' }) },
      { id: 's2', index: 1, at: 0, decision, action: 'kb_search', call: call({ status: 'error', args: null, argsRaw: '{"query":', argAttempts: 2, argv: undefined, output: undefined, hits: undefined, error: 'Could not produce valid arguments: bad JSON' }) },
    ]);
    render(<CommandCenter />);
    const trace = within(screen.getByText('Execution Trace').closest('aside')!);
    await user.click(trace.getAllByRole('button', { name: 'Tool call' })[0]);
    let lines = within(within(screen.getByRole('dialog', { name: 'Tool call, step 1' })).getByRole('region', { name: 'What happened' })).getAllByRole('listitem');
    expect(lines.map((li) => li.textContent)[1]).toBe('The tool is set to Ask, and you denied this call, so it did not run.');
    await user.click(screen.getByRole('button', { name: /^close$/i }));
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
    await user.click(trace.getAllByRole('button', { name: 'Tool call' })[1]);
    const dialog = within(screen.getByRole('dialog', { name: 'Tool call, step 2' }));
    lines = within(dialog.getByRole('region', { name: 'What happened' })).getAllByRole('listitem');
    expect(lines.map((li) => li.textContent)).toEqual(['Qwen3 0.6B could not write valid arguments in 2 attempts, so the tool never ran.']);
    expect(dialog.getByText('Could not produce valid arguments: bad JSON')).toBeInTheDocument();
    expect(dialog.getByText('No command ran.')).toBeInTheDocument();
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

    await user.click(trace.getByRole('button', { name: 'Tool call' }));
    await user.click(within(screen.getByRole('dialog', { name: 'Tool call, step 1' })).getByRole('button', { name: 'Copy command' }));
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

  it('explains Laya’s stop question: on the step and in the dialog', async () => {
    const user = userEvent.setup();
    const stop = { statement: 'The tool results above already contain the information needed.', probability: 0.78 };
    turn([
      {
        id: 's1',
        index: 0,
        at: 0,
        decision: { ...decision, model: 'Laya Multilingual', chosen: 'kb_search', stop },
        action: 'answer_now',
        note: 'The tool results cover the request (78% likely), so answering.',
      },
    ]);
    render(<CommandCenter />);
    const trace = within(screen.getByText('Execution Trace').closest('aside')!);
    expect(trace.getByText(/Results suffice: 78% · Chosen with/)).toBeInTheDocument();
    const dialog = await whyDialog(user, trace);
    const lines = within(dialog.getByRole('region', { name: 'Why this step' })).getAllByRole('listitem').map((li) => li.textContent);
    expect(lines[1]).toBe('With tool results in hand, it also asked whether they already answer the request: 78% yes, so no more tools were needed.');
    expect(lines.at(-1)).toBe('So the agent stopped using tools and answered.');
    expect(lines.join(' ')).not.toMatch(/The tool results cover/); // said once, not twice
    expect(dialog.getByText(stop.statement)).toBeInTheDocument();
    expect(dialog.getByText(/Results suffice\?/)).toHaveTextContent('Results suffice? 78% yes');
  });

  it('explains a failed decision with its time, error and request', async () => {
    const user = userEvent.setup();
    const io = { request: { state: 'State: failing', messages: [{ role: 'user', content: 'State: failing' }], params: {} }, response: null };
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
    const dialog = await whyDialog(user, trace);
    expect(dialog.getByText('Decision failed after 640 ms')).toBeInTheDocument();
    expect(dialog.getByText(/call failed: context overflow/)).toBeInTheDocument();
    expect(dialog.getByText('State: failing', { selector: 'pre.dd-state' })).toBeInTheDocument();
    await user.click(dialog.getByText(/Exact call/));
    expect(dialog.getByText(/No reply: the call failed/)).toBeInTheDocument();
  });

  it('shows the relevance check: each passage’s score, what was kept and dropped', () => {
    const relevance = {
      model: 'Laya Multilingual',
      modelId: 'laya-multilingual',
      request: 'What headers?',
      ms: 41,
      modelMs: 38,
      keepTop: 2,
      dropBelow: 0.1,
      tokensSaved: 180,
      items: [
        { file: 'README.md', start_line: 1, end_line: 9, name: 'a', score: 0.92, kept: true, reason: 'top' as const, chars: 400, text: 'Set COOP and COEP.', inputTokens: 90, truncated: false },
        { file: 'notes.md', start_line: 3, end_line: 8, name: 'b', score: 0.03, kept: true, reason: 'top' as const, chars: 300, text: 'Notes.', inputTokens: 40, truncated: false },
        { file: 'old.md', start_line: 1, end_line: 5, name: 'c', score: 0.04, kept: false, reason: 'low' as const, chars: 700, text: 'Old release notes about fonts.', inputTokens: 60, truncated: false },
        { file: 'api.md', start_line: 2, end_line: 6, name: 'd', score: 0.61, kept: true, reason: 'score' as const, chars: 200, text: 'The API needs COOP.', inputTokens: 55, truncated: true },
      ],
    };
    addMessage({ id: 'u', role: 'user', content: 'What headers?', createdAt: 0 });
    addMessage({
      id: 'a',
      role: 'assistant',
      content: 'Use COOP [1].',
      createdAt: 0,
      kbName: 'Docs',
      relevance,
      steps: [
        { kind: 'retrieve', title: 'Knowledge retrieval', detail: 'Retrieved 4 passages', status: 'done' },
        { kind: 'filter', title: 'Relevance check', detail: 'Kept 3 of 4 passages · ~180 tokens less to read · 41 ms', status: 'done' },
        { kind: 'build', title: 'Assemble context', detail: '', status: 'done' },
        { kind: 'generate', title: 'Generate', detail: '', status: 'done' },
      ],
    });
    render(<CommandCenter />);
    expect(screen.getByText(/Relevance check: Kept 3 of 4 passages/)).toBeInTheDocument();
    const list = within(screen.getByRole('region', { name: 'Relevance check' }));
    expect(list.getAllByRole('listitem').map((li) => li.getAttribute('aria-label'))).toEqual([
      'README.md: 92%, kept',
      'notes.md: 3.0%, kept',
      'old.md: 4.0%, dropped',
      'api.md: 61%, kept',
    ]);
    // kept passages carry the number the answer cites them by
    expect(list.getByText('[3]', { exact: false }).closest('li')).toHaveAttribute('aria-label', 'api.md: 61%, kept');
    expect(list.getByText(/1 dropped · ~180 fewer prompt tokens/)).toBeInTheDocument();
    // folded to one line until opened
    expect(list.getByText('4 passages scored · 1 dropped').closest('details')).not.toHaveAttribute('open');
  });

  it('opens a passage’s relevance check from its name: the request, the passage, Laya’s answer and why it was dropped', async () => {
    const user = userEvent.setup();
    addMessage({ id: 'u', role: 'user', content: 'What headers?', createdAt: 0 });
    addMessage({
      id: 'a',
      role: 'assistant',
      content: 'Use COOP [1].',
      createdAt: 0,
      relevance: {
        model: 'Laya Multilingual',
        modelId: 'laya-multilingual',
        request: 'What headers?',
        ms: 41,
        modelMs: 38,
        keepTop: 2,
        dropBelow: 0.1,
        tokensSaved: 180,
        items: [
          { file: 'README.md', start_line: 1, end_line: 9, name: 'a', score: 0.92, kept: true, reason: 'top', chars: 400, text: 'Set COOP.', inputTokens: 90, truncated: false },
          { file: 'old.md', start_line: 1, end_line: 5, name: 'c', score: 0.04, kept: false, reason: 'low', chars: 700, text: 'Old release notes about fonts.', inputTokens: 60, truncated: true },
        ],
      },
      steps: [{ kind: 'filter', title: 'Relevance check', detail: 'Kept 1 of 2 passages', status: 'done' }],
    });
    render(<CommandCenter />);
    await user.click(screen.getByRole('button', { name: 'Relevance check: old.md:1-5' }));
    const dialog = within(screen.getByRole('dialog', { name: 'Relevance check for old.md:1-5' }));
    expect(dialog.getByText('Dropped: unlikely to help')).toBeInTheDocument();
    expect(dialog.getAllByText('Old release notes about fonts.').length).toBeGreaterThan(0);
    const why = within(dialog.getByRole('region', { name: 'Why this verdict' }));
    expect(why.getByText(/answered 4\.0% yes, below the 10% cut, so the passage was left out/)).toBeInTheDocument();
    expect(why.getByText(/cut to fit the model’s input/)).toBeInTheDocument();
    expect(why.getByText(/76% of the time/)).toBeInTheDocument();
  });

  it('fills in a relevance record saved before the request and passage text were kept, and says what can’t be recovered (reported)', async () => {
    const user = userEvent.setup();
    // The shape saved by earlier builds: no modelId, request, text, inputTokens or truncated.
    const old = {
      model: 'Laya Multilingual',
      ms: 29,
      modelMs: 28,
      keepTop: 2,
      dropBelow: 0.1,
      tokensSaved: 20,
      items: [
        { file: 'release-notes.md', start_line: 13, end_line: 18, name: 'a', score: 0.22, kept: true, reason: 'score', chars: 120 },
        { file: 'operations.md', start_line: 3, end_line: 9, name: 'b', score: 0.02, kept: false, reason: 'low', chars: 90 },
      ],
    } as unknown as NonNullable<Message['relevance']>;
    addMessage({ id: 'u', role: 'user', content: 'Can I get a refund in cash if my sailing is canceled?', createdAt: 0 });
    addMessage({
      id: 'a',
      role: 'assistant',
      content: 'No [1].',
      createdAt: 0,
      sources: [{ id: 'h1', name: '2.4.0', node_type: 'Section', file: 'release-notes.md', start_line: 13, end_line: 18, snippet: 'Group bookings: parties of 10 or more get 15% off.' }],
      relevance: old,
      steps: [{ kind: 'filter', title: 'Relevance check', detail: 'All passages look relevant', status: 'done' }],
    });
    render(<CommandCenter />);
    await user.click(screen.getByRole('button', { name: 'Relevance check: release-notes.md:13-18' }));
    let dialog = within(screen.getByRole('dialog', { name: 'Relevance check for release-notes.md:13-18' }));
    expect(dialog.getByText('Can I get a refund in cash if my sailing is canceled?')).toBeInTheDocument();
    expect(dialog.getAllByText(/Group bookings: parties of 10 or more get 15% off\./).length).toBe(2); // passage, and the exact input
    expect(dialog.queryByText('(empty)')).toBeNull();
    expect(dialog.getByText(/76% of the time/)).toBeInTheDocument(); // the model id, recovered from the name
    await user.click(dialog.getAllByRole('button', { name: 'Close dialog' })[0]);
    await user.click(screen.getByRole('button', { name: 'Relevance check: operations.md:3-9' }));
    dialog = within(await screen.findByRole('dialog', { name: 'Relevance check for operations.md:3-9' }));
    expect(dialog.getAllByText(/Not recorded: this check ran before Andai kept the text of dropped passages\./).length).toBe(2);
  });

  it('opens a claim saved before cites and source were kept', async () => {
    const user = userEvent.setup();
    const old = { model: 'Laya English', ms: 30, modelMs: 25, flagBelow: 0.1, items: [{ sentence: 'Cars pay 18.5.', n: 1, score: 0.04, flagged: true }] } as unknown as NonNullable<Message['support']>;
    addMessage({ id: 'u', role: 'user', content: 'q', createdAt: 0 });
    addMessage({
      id: 'a',
      role: 'assistant',
      content: 'Cars pay 18.5 [1].',
      createdAt: 0,
      sources: [{ id: 'h1', name: 'fares', node_type: 'Code', file: 'fares.ts', start_line: 12, end_line: 12, snippet: 'export const VEHICLE_SURCHARGE = 18.5;' }],
      support: old,
    });
    render(<CommandCenter />);
    await user.click(within(document.querySelector('.cc-support') as HTMLElement).getByRole('button', { name: 'Claim check: [1] Cars pay 18.5.' }));
    const dialog = within(screen.getByRole('dialog', { name: 'Claim check for source 1' }));
    expect(dialog.getByText(/whether passage \[1\] \(fares\.ts:12-12\) supports/)).toBeInTheDocument();
    expect(dialog.getByText(/82% of the time/)).toBeInTheDocument();
  });

  it('shows in the passages found how ug found each passage, and nothing for one no search returned', () => {
    addMessage({ id: 'u', role: 'user', content: 'q', createdAt: 0 });
    addMessage({
      id: 'a',
      role: 'assistant',
      content: 'A [1].',
      createdAt: 0,
      kbName: 'Docs',
      sources: [
        { id: 'h1', name: 'Processing', node_type: 'Concept', file: 'refund-policy.md', start_line: 17, end_line: 20, snippet: 'cash', distance: -0.08, matched_by: 'semantic' },
        { id: 'h2', name: 'Handbook', node_type: 'Concept', file: 'operations.md', start_line: 1, end_line: 25, snippet: 'ops', distance: -0.04, hop: 1, matched_by: 'graph' },
        { id: 'h3', name: 'booking.ts', node_type: 'File', file: 'booking.ts', start_line: 15, end_line: 60, snippet: 'lines read' },
      ],
    });
    render(<CommandCenter />);
    const log = within(document.querySelector('.cc-log') as HTMLElement);
    expect(log.getByLabelText('semantic, 100% of the best match')).toBeInTheDocument();
    expect(log.getByLabelText('graph · 1 hop, 50% of the best match')).toHaveAttribute('title', expect.stringMatching(/following links.*ug rank score -0\.0400/));
    expect(within(log.getByRole('button', { name: /^Source 3/ })).queryByText(/semantic|keyword|graph/)).toBeNull();
  });

  it('makes each citation in the answer and each passage in the list open that source, with what the checks said', async () => {
    const user = userEvent.setup();
    addMessage({ id: 'u', role: 'user', content: 'q', createdAt: 0 });
    addMessage({
      id: 'a',
      role: 'assistant',
      content: 'Cars pay 18.5 [2]. See `arr[1]` in code, and [9] is not a source.',
      createdAt: 0,
      kbName: 'Docs',
      sources: [
        { id: 'h1', name: 'README', node_type: 'Section', file: 'README.md', start_line: 1, end_line: 9, snippet: 'Set COOP.' },
        { id: 'h2', name: 'VEHICLE_SURCHARGE', node_type: 'Constant', file: 'fares.ts', start_line: 12, end_line: 12, snippet: 'export const VEHICLE_SURCHARGE = 18.5;' },
      ],
      support: {
        model: 'Laya English',
        modelId: 'laya-en',
        ms: 30,
        modelMs: 25,
        flagBelow: 0.1,
        items: [{ sentence: 'Cars pay 18.5.', n: 2, cites: [2], source: 'fares.ts:12-12', score: 0.9, flagged: false, inputTokens: 30, truncated: false }],
      },
    });
    render(<CommandCenter />);
    // only real citations are links: not inside code, not past the source list
    expect(screen.getAllByRole('button', { name: /^Source \d+$/ }).map((b) => b.getAttribute('aria-label'))).toEqual(['Source 2']);
    await user.click(screen.getByRole('button', { name: 'Source 2' }));
    let dialog = within(screen.getByRole('dialog', { name: 'Source 2' }));
    expect(dialog.getByText('VEHICLE_SURCHARGE')).toBeInTheDocument();
    expect(dialog.getByText('export const VEHICLE_SURCHARGE = 18.5;')).toBeInTheDocument();
    expect(dialog.getByText(/Cars pay 18\.5\./)).toBeInTheDocument();
    await user.click(dialog.getAllByRole('button', { name: 'Close dialog' })[0]);
    await user.click(screen.getByRole('button', { name: 'Source 1: README.md:1-9' }));
    dialog = within(await screen.findByRole('dialog', { name: 'Source 1' }));
    expect(dialog.getByText('Set COOP.')).toBeInTheDocument();
  });

  it('doesn’t list the passages again under the plan step: the tool calls and the relevance check do', () => {
    addMessage({ id: 'u', role: 'user', content: 'q', createdAt: 0 });
    addMessage({
      id: 'a',
      role: 'assistant',
      content: 'A [1].',
      createdAt: 0,
      sources: [{ id: 'h9', name: 'x', node_type: 'Section', file: 'zzz-unique.md', start_line: 1, end_line: 2 }],
      steps: [{ kind: 'plan', title: 'Plan & use tools', detail: '1 tool call', status: 'done' }],
      agent: [{ id: 's1', index: 0, at: 0, decision: null, action: 'kb_search', call: call() }],
    });
    render(<CommandCenter />);
    // the per-passage rows the retrieve step shows (trace-sub), not the passages listed under the answer
    expect([...document.querySelectorAll('.trace-sub')].some((el) => el.textContent?.includes('zzz-unique.md'))).toBe(false);
  });

  it('notes under the answer which cited sentences may not be supported, and lists every claim in the trace', () => {
    const support = {
      model: 'Laya English',
      modelId: 'laya-en',
      ms: 60,
      modelMs: 52,
      flagBelow: 0.1,
      items: [
        { sentence: 'Use COOP.', n: 1, cites: [1], source: 'README.md:1-9', score: 0.93, flagged: false, inputTokens: 60, truncated: false },
        { sentence: 'Cars pay 18.5.', n: 2, cites: [1, 2], source: 'fares.ts:12-12', score: 0.04, flagged: true, inputTokens: 44, truncated: false },
      ],
    };
    addMessage({ id: 'u', role: 'user', content: 'What headers?', createdAt: 0 });
    addMessage({
      id: 'a',
      role: 'assistant',
      content: 'Use COOP [1]. Cars pay 18.5 [1][2].',
      createdAt: 0,
      kbName: 'Docs',
      sources: [
        { id: 'h1', name: 'README', node_type: 'Section', file: 'README.md', start_line: 1, end_line: 9, snippet: 'Set COOP and COEP.' },
        { id: 'h2', name: 'fares', node_type: 'Code', file: 'fares.ts', start_line: 12, end_line: 12, snippet: 'export const VEHICLE_SURCHARGE = 18.5;' },
      ],
      support,
      steps: [
        { kind: 'generate', title: 'Generate', detail: '', status: 'done' },
        { kind: 'verify', title: 'Claim check', detail: '1 of 2 cited sentences may not be supported by their source · 60 ms', status: 'done' },
      ],
    });
    render(<CommandCenter />);
    expect(screen.getByText('1 of 2 cited sentences may not be supported by the source it cites')).toBeInTheDocument();
    const list = within(screen.getByRole('region', { name: 'Claim check' }));
    expect(list.getAllByRole('listitem').map((li) => li.getAttribute('aria-label'))).toEqual([
      '[1] Use COOP.: 93%, supported',
      '[2] Cars pay 18.5.: 4.0%, may not be supported',
    ]);
  });

  it('builds a claim’s dialog only when it is opened, so a closed one never costs or breaks the chat', () => {
    // A claim check saved by an earlier build (before `cites`, `source` and
    // `inputTokens`) blanked the whole app on launch: every closed dialog ran
    // its explanation on render.
    const support = {
      model: 'Laya English',
      ms: 60,
      modelMs: 52,
      flagBelow: 0.1,
      items: [{ sentence: 'Cars pay 18.5.', n: 1, score: 0.04, flagged: true }],
    } as unknown as NonNullable<Message['support']>;
    addMessage({ id: 'u', role: 'user', content: 'q', createdAt: 0 });
    addMessage({
      id: 'a',
      role: 'assistant',
      content: 'Cars pay 18.5 [1].',
      createdAt: 0,
      sources: [{ id: 'h1', name: 'fares', node_type: 'Code', file: 'fares.ts', start_line: 12, end_line: 12, snippet: 'export const VEHICLE_SURCHARGE = 18.5;' }],
      support,
      steps: [{ kind: 'verify', title: 'Claim check', detail: '', status: 'done' }],
    });
    render(<CommandCenter />);
    expect(screen.getByRole('region', { name: 'Claim check' })).toBeInTheDocument();
    expect(screen.queryByRole('dialog')).toBeNull();
  });

  it('opens a claim’s check from the note: the passage it was checked against, Laya’s answer and why', async () => {
    const support = {
      model: 'Laya English',
      modelId: 'laya-en',
      ms: 60,
      modelMs: 52,
      flagBelow: 0.1,
      items: [{ sentence: 'Cars pay 18.5.', n: 2, cites: [1, 2], source: 'fares.ts:12-12', score: 0.04, flagged: true, inputTokens: 44, truncated: true }],
    };
    addMessage({ id: 'u', role: 'user', content: 'q', createdAt: 0 });
    addMessage({
      id: 'a',
      role: 'assistant',
      content: 'Cars pay 18.5 [1][2].',
      createdAt: 0,
      sources: [
        { id: 'h1', name: 'README', node_type: 'Section', file: 'README.md', start_line: 1, end_line: 9, snippet: 'Set COOP.' },
        { id: 'h2', name: 'fares', node_type: 'Code', file: 'fares.ts', start_line: 12, end_line: 12, snippet: 'export const VEHICLE_SURCHARGE = 18.5;' },
      ],
      support,
    });
    render(<CommandCenter />);
    await userEvent.click(within(document.querySelector('.cc-support') as HTMLElement).getByRole('button', { name: 'Claim check: [2] Cars pay 18.5.' }));
    expect(within(document.querySelector('.cc-support') as HTMLElement).getByText(/^A rough automatic check: it can miss real problems and flag correct ones\./)).toBeInTheDocument();
    const dialog = within(screen.getByRole('dialog', { name: 'Claim check for source 2' }));
    expect(dialog.getByText('May not be supported by its source')).toBeInTheDocument();
    expect(dialog.getAllByText('export const VEHICLE_SURCHARGE = 18.5;').length).toBeGreaterThan(0);
    const why = within(dialog.getByRole('region', { name: 'Why this verdict' }));
    expect(why.getByText(/answered 4\.0% yes, below the 10% cut/)).toBeInTheDocument();
    expect(why.getByText(/also cites \[1\], and each source is checked on its own/)).toBeInTheDocument();
    expect(why.getByText(/cut to fit the model’s input/)).toBeInTheDocument();
    expect(why.getByText(/82% of the time.*a hint to read the source, not a verdict/)).toBeInTheDocument();
  });

  it('says nothing under the answer when every cited sentence looks supported', () => {
    addMessage({ id: 'u', role: 'user', content: 'q', createdAt: 0 });
    addMessage({
      id: 'a',
      role: 'assistant',
      content: 'Use COOP [1].',
      createdAt: 0,
      support: {
        model: 'Laya English',
        modelId: 'laya-en',
        ms: 30,
        modelMs: 25,
        flagBelow: 0.1,
        items: [{ sentence: 'Use COOP.', n: 1, cites: [1], source: 'README.md:1-9', score: 0.9, flagged: false, inputTokens: 30, truncated: false }],
      },
    });
    render(<CommandCenter />);
    expect(screen.queryByText(/may not be supported/)).toBeNull();
  });

  it('numbers a tool call’s chip under the plan step, and points it at its card in the trace', async () => {
    const raf = vi.spyOn(window, 'requestAnimationFrame').mockImplementation((cb) => {
      cb(0);
      return 0;
    });
    turn([{ id: 's1', index: 0, at: 0, decision: null, action: 'kb_search', call: call() }]);
    render(<CommandCenter />);
    const chip = screen.getByRole('button', { name: /^Show step 01\.1 in the trace: Knowledge search/ });
    await userEvent.click(chip);
    expect(document.getElementById('trace-a-call-s1')).toHaveClass('flash');
    raf.mockRestore();
  });

  it('sums up a knowledge search by how its passages were found, instead of listing them', () => {
    const long = '10 passage(s): 2.5.1 (2026-05-14) @ release-notes.md:3-6; 2.5.0 (2026-04-20) @ release-notes.md:7-12; +8 more';
    turn([{ id: 's1', index: 0, at: 0, decision: null, action: 'kb_search', call: call({ observation: long, found: { total: 10, by: { semantic: 7, graph: 2, keyword: 1 } } }) }]);
    render(<CommandCenter />);
    expect(screen.getAllByText('10 passages · 7 semantic · 1 keyword · 2 graph').length).toBeGreaterThan(0);
    expect(screen.queryByText(long)).toBeNull();
    expect(screen.getByRole('button', { name: 'Show step 01.1 in the trace: Knowledge search' })).toHaveTextContent(/— 10 passages · 7 semantic · 1 keyword · 2 graph$/);
  });

  it('counts a knowledge search recorded before the counts were kept from its stored output', () => {
    const output = JSON.stringify({ items: [{ id: 'a', name: 'A', node_type: 'Concept', file: 'a.md', start_line: 1, end_line: 3, snippet: 'x', matched_by: 'semantic' }, { id: 'b', name: 'B', node_type: 'Concept', file: 'b.md', start_line: 1, end_line: 3, snippet: 'y', matched_by: 'graph', hop: 1 }] });
    turn([{ id: 's1', index: 0, at: 0, decision: null, action: 'kb_search', call: call({ output, observation: '2 passage(s): A @ a.md:1-3; B @ b.md:1-3' }) }]);
    render(<CommandCenter />);
    expect(screen.getAllByText('2 passages · 1 semantic · 1 graph').length).toBeGreaterThan(0);
  });

  it('lists the tool calls right under the plan step, before the steps that came after', () => {
    turn([{ id: 's1', index: 0, at: 0, decision: null, action: 'kb_search', call: call() }]);
    render(<CommandCenter />);
    const order = [...document.querySelectorAll('.cc-reasoning button.cc-chip')].map((b) => /^Show step ([\d.]+) /.exec(b.getAttribute('aria-label')!)![1]);
    expect(order).toEqual(['01', '01.1', '02', '03']);
  });

  it('opens what the chat model was sent from the Assemble context card', async () => {
    const user = userEvent.setup();
    addMessage({ id: 'u0', role: 'user', content: 'Earlier question', createdAt: 0 });
    addMessage({ id: 'u', role: 'user', content: 'What headers?', createdAt: 0 });
    addMessage({
      id: 'a',
      role: 'assistant',
      content: 'Use COOP [1].',
      createdAt: 0,
      stats: { tokens: 5, tokPerSec: 50, promptTokens: 400, nCtx: 4096, totalMs: 900, firstTokenMs: 500, model: 'Qwen3 8B' },
      context: {
        nCtx: 4096,
        replyTokens: 512,
        budget: { context: 6389, history: 2662 },
        passages: [
          { n: 1, chars: 900, used: 900, status: 'in', source: 'README.md:1-9' },
          { n: 2, chars: 6000, used: 5489, status: 'clipped', source: 'big.md:1-200' },
          { n: 3, chars: 400, used: 0, status: 'left out', source: 'late.md:1-9' },
        ],
        history: { sent: 1, of: 3, chars: 16 },
        system: 'You are Andai.\n\n<passage>\n[1] README.md (lines 1-9)\nSet COOP.\n</passage>',
        messages: [
          { role: 'system', chars: 6500 },
          { role: 'user', chars: 16 },
          { role: 'user', chars: 13 },
        ],
        tokens: 2040,
      },
      steps: [{ kind: 'build', title: 'Assemble context', detail: '2 messages · ~2,040 tokens', status: 'done' }],
    });
    render(<CommandCenter />);
    await user.click(screen.getByRole('button', { name: 'What went in?' }));
    const dialog = within(screen.getByRole('dialog', { name: 'Assemble context' }));
    const did = within(dialog.getByRole('region', { name: 'What it did' }));
    expect(did.getByText(/Qwen3 8B reads at most 4,096 tokens.*kept 512 of them for the reply/)).toBeInTheDocument();
    expect(did.getByText(/2 of 3 passages went in \(1 cut short to fit\); 1 was left out/)).toBeInTheDocument();
    expect(did.getByText(/The last 1 of 3 earlier messages went in/)).toBeInTheDocument();
    expect(did.getByText(/about 2,040 tokens, 50% of the window/)).toBeInTheDocument();
    expect(dialog.getByText('left out')).toBeInTheDocument();
    expect(dialog.getByText('user (the question)')).toBeInTheDocument();
  });

  it('shows a fallback note when the loop overrode the decision', () => {
    turn([{ id: 's1', index: 0, at: 0, decision: null, action: 'kb_search', note: 'Decision failed (boom), so searching the knowledge base instead.', call: call() }]);
    render(<CommandCenter />);
    expect(screen.getByText(/Decision failed \(boom\)/)).toBeInTheDocument();
  });
});
