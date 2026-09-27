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
      ms: 41,
      modelMs: 38,
      keepTop: 2,
      dropBelow: 0.1,
      tokensSaved: 180,
      items: [
        { file: 'README.md', start_line: 1, end_line: 9, name: 'a', score: 0.92, kept: true, reason: 'top' as const, chars: 400 },
        { file: 'notes.md', start_line: 3, end_line: 8, name: 'b', score: 0.03, kept: true, reason: 'top' as const, chars: 300 },
        { file: 'old.md', start_line: 1, end_line: 5, name: 'c', score: 0.04, kept: false, reason: 'low' as const, chars: 700 },
        { file: 'api.md', start_line: 2, end_line: 6, name: 'd', score: 0.61, kept: true, reason: 'score' as const, chars: 200 },
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
  });

  it('notes under the answer which cited sentences may not be supported, and lists every claim in the trace', () => {
    const support = {
      model: 'Laya English',
      ms: 60,
      modelMs: 52,
      flagBelow: 0.1,
      items: [
        { sentence: 'Use COOP.', n: 1, score: 0.93, flagged: false },
        { sentence: 'Cars pay 18.5.', n: 2, score: 0.04, flagged: true },
      ],
    };
    addMessage({ id: 'u', role: 'user', content: 'What headers?', createdAt: 0 });
    addMessage({
      id: 'a',
      role: 'assistant',
      content: 'Use COOP [1]. Cars pay 18.5 [2].',
      createdAt: 0,
      kbName: 'Docs',
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

  it('says nothing under the answer when every cited sentence looks supported', () => {
    addMessage({ id: 'u', role: 'user', content: 'q', createdAt: 0 });
    addMessage({
      id: 'a',
      role: 'assistant',
      content: 'Use COOP [1].',
      createdAt: 0,
      support: { model: 'Laya English', ms: 30, modelMs: 25, flagBelow: 0.1, items: [{ sentence: 'Use COOP.', n: 1, score: 0.9, flagged: false }] },
    });
    render(<CommandCenter />);
    expect(screen.queryByText(/may not be supported/)).toBeNull();
  });

  it('shows a fallback note when the loop overrode the decision', () => {
    turn([{ id: 's1', index: 0, at: 0, decision: null, action: 'kb_search', note: 'Decision failed (boom), so searching the knowledge base instead.', call: call() }]);
    render(<CommandCenter />);
    expect(screen.getByText(/Decision failed \(boom\)/)).toBeInTheDocument();
  });
});
