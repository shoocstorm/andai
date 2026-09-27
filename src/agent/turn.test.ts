// The agent turn orchestrator with the LLM engine and ug mocked out: verifies
// the step sequence the Execution Trace shows, and every failure path, for
// both the fixed pipeline and agent mode (a scripted decider and tools).
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { KbInfo, SearchHit } from '../kb/api';
import { MODELS } from '../llm/models';
import { clearChat, useChat } from '../state/chat';
import { useKb } from '../state/kb';
import { usePersona } from '../state/persona';

const engine = vi.hoisted(() => ({
  loaded: true,
  deltas: ['Hello', ' world'] as string[],
  fail: null as Error | null,
  seen: [] as { messages: { role: string; content: string }[]; opts: Record<string, unknown> }[],
}));
const search = vi.hoisted(() => ({ hits: [] as unknown[], fail: null as Error | null, calls: 0 }));
type Scripted = { chosen: string; confidence?: number; stop?: number; scope?: string } | Error;
const agent = vi.hoisted(() => ({
  decisions: [] as Scripted[],
  laya: false,
  seenStops: [] as (string | null)[],
  seenStates: [] as string[],
  seenOptions: [] as string[][],
  fills: {} as Record<string, unknown>,
  fillFail: null as string | null,
  fillFiles: [] as (string[] | undefined)[],
  fillSymbols: [] as (string[] | undefined)[],
  fillRanges: [] as (string[] | undefined)[],
  failTools: [] as string[],
  tool: [] as { slug: string; call: Record<string, unknown> }[],
  output: {} as Record<string, unknown>,
  toolFail: null as Error | null,
  toolDelayMs: 0,
  seenChoices: [] as string[][],
  fillFixed: [] as (Record<string, string> | undefined)[],
  // layaChoices: the scope it picks on its own pass, or an error
  ownPick: 'focused' as string | Error,
  ownPasses: 0,
}));

vi.mock('../llm/decide', async (orig) => ({
  ...(await orig<typeof import('../llm/decide')>()),
  decidesWithLaya: () => agent.laya,
  layaChoices: async (_state: string, choices: { id: string; tool: string; arg: string }[]) => {
    agent.ownPasses++;
    if (agent.ownPick instanceof Error) throw agent.ownPick;
    return { picks: choices.map((c) => ({ ...c, value: agent.ownPick as string, probability: 0.7, scores: [] })), ms: 8, model: 'Laya Multilingual' };
  },
  decide: async (
    state: string,
    _question: string,
    options: { id: string; text: string }[],
    _signal?: AbortSignal,
    extra: { stop?: string; choices?: { id: string; tool: string; arg: string }[] } = {},
  ) => {
    agent.seenChoices.push((extra.choices ?? []).map((c) => c.id));
    agent.seenStates.push(state);
    agent.seenOptions.push(options.map((o) => o.id));
    agent.seenStops.push(extra.stop ?? null);
    const next = agent.decisions.shift() ?? { chosen: 'answer_now' };
    if (next instanceof Error) throw next;
    const confidence = next.confidence ?? 0.9;
    return {
      options: options.map((o, i) => ({
        ...o,
        label: 'ABCDEFGHIJKLMNOP'[i],
        probability: o.id === next.chosen ? confidence : (1 - confidence) / (options.length - 1),
        logprob: 0,
      })),
      chosen: next.chosen,
      confidence,
      bounded: [],
      model: 'Qwen3 1.7B',
      slot: 'decider',
      ms: 12,
      promptTokens: 300,
      ...(extra.stop && next.stop != null ? { stop: { statement: extra.stop, probability: next.stop } } : {}),
      ...(next.scope && extra.choices?.length
        ? { picks: extra.choices.map((c) => ({ ...c, value: next.scope!, probability: 0.9, scores: [] })) }
        : {}),
    };
  },
}));

vi.mock('./tools/argfill', () => ({
  fillArgs: async (tool: { id: string; schema: unknown }, ctx: { known?: { files?: string[]; symbols?: string[]; ranges?: string[] }; fixed?: Record<string, string> }) => {
    agent.fillFixed.push(ctx.fixed);
    agent.fillFiles.push(ctx.known?.files);
    agent.fillSymbols.push(ctx.known?.symbols);
    agent.fillRanges.push(ctx.known?.ranges);
    if (!tool.schema) return { ok: true, args: {}, raw: '{}', attempts: 0, model: null };
    if (agent.fillFail) return { ok: false, errors: [agent.fillFail], raw: 'nope', attempts: 2, model: 'Qwen3 0.6B' };
    // a list is consumed one fill per call
    const f = agent.fills[tool.id];
    const args = (Array.isArray(f) ? f.shift() : f) as Record<string, unknown>;
    return { ok: true, args, raw: JSON.stringify(args), attempts: 1, model: 'Qwen3 0.6B' };
  },
}));

const rel = vi.hoisted(() => ({
  scores: null as number[] | null,
  fail: null as Error | null,
  support: null as number[] | null,
  supportFail: null as Error | null,
  claims: [] as unknown[][],
}));
vi.mock('../llm/laya', () => ({
  layaRelevance: async (_request: string, passages: unknown[]) => {
    if (rel.fail) throw rel.fail;
    const scores = rel.scores ?? passages.map(() => 0.9);
    return { scores, inputTokens: scores.map(() => 100), truncated: scores.map(() => false), ms: 20, model: 'laya-multilingual' };
  },
  layaSupport: async (claims: unknown[]) => {
    rel.claims.push(claims);
    if (rel.supportFail) throw rel.supportFail;
    const scores = rel.support ?? claims.map(() => 0.9);
    return { scores, inputTokens: scores.map(() => 50), truncated: scores.map(() => false), ms: 9, model: 'laya-multilingual' };
  },
}));

vi.mock('../llm/engine', async () => {
  const { create } = await import('zustand');
  const { MODELS } = await import('../llm/models');
  const useEngine = create(() => ({ tokPerSec: 42 as number | null }));
  return {
    useEngine,
    loadedModel: () => (engine.loaded ? MODELS[0] : undefined),
    isAbort: (e: unknown) => e instanceof Error && e.name === 'AbortError',
    chat: async function* (messages: never, opts: { signal: AbortSignal } & Record<string, unknown>) {
      engine.seen.push({ messages, opts });
      if (engine.fail) throw engine.fail;
      let n = 0;
      for (const d of engine.deltas) {
        if (opts.signal.aborted) throw Object.assign(new Error('aborted'), { name: 'AbortError' });
        n++;
        await new Promise((r) => setTimeout(r, 1));
        yield { type: 'delta', text: d, tokens: n, tokPerSec: 42 };
      }
      yield { type: 'done', promptTokens: 120, completionTokens: n };
    },
  };
});

vi.mock('../kb/api', async (orig) => ({
  ...(await orig<typeof import('../kb/api')>()),
  kbSearch: async () => {
    search.calls++;
    if (search.fail) throw search.fail;
    return search.hits;
  },
  kbTool: async (slug: string, call: Record<string, unknown>) => {
    agent.tool.push({ slug, call });
    if (agent.toolDelayMs) await new Promise((r) => setTimeout(r, agent.toolDelayMs));
    if (agent.toolFail) throw agent.toolFail;
    if (agent.failTools.includes(call.tool as string)) throw new Error(`No indexed file matches '${call.file}'.`);
    return { output: agent.output[call.tool as string] ?? {}, truncated: false, bytes: 10, ms: 3, argv: ['search', 'x'] };
  },
}));

const { runTurn, stopTurn } = await import('./turn');
const { STOP } = await import('./loop');
const { resolveApproval, setAgent, setPolicy, useTools } = await import('../state/tools');

const kb = (over: Partial<KbInfo> = {}): KbInfo => ({
  slug: 'docs',
  name: 'Docs',
  createdAt: 0,
  sources: [],
  lastIndexedAt: 1,
  lastError: null,
  kindOverride: null,
  kind: 'document',
  dir: '/tmp/docs',
  status: 'ready',
  nodes: 12,
  edges: 11,
  sizeBytes: 1,
  ...over,
});
const hit: SearchHit = {
  id: 'h1',
  name: 'Run it',
  node_type: 'Concept',
  file: 'README.md',
  start_line: 11,
  end_line: 33,
  snippet: 'serve.json adds the COOP/COEP headers wllama needs',
};

const assistant = () => useChat.getState().messages.find((m) => m.role === 'assistant')!;
const statuses = () => Object.fromEntries(assistant().steps!.map((s) => [s.kind, s.status]));

beforeEach(() => {
  clearChat();
  engine.loaded = true;
  engine.deltas = ['Hello', ' world'];
  engine.fail = null;
  engine.seen = [];
  search.hits = [hit];
  search.fail = null;
  search.calls = 0;
  useKb.setState({ kbs: [kb()], grounding: 'docs', k: 8, maxChars: 6000 });
  usePersona.getState().reset();
  // Most tests script the first decision; the searchFirst tests turn it on.
  useTools.setState({ agentMode: false, maxSteps: 4, minConfidence: 0.3, searchFirst: false, policies: {}, stats: {} });
  Object.assign(agent, {
    decisions: [],
    laya: false,
    seenStops: [],
    seenStates: [],
    seenOptions: [],
    fills: { kb_search: { query: 'wllama COOP COEP headers', scope: 'broad' }, kb_find_usages: { symbol: 'add' } },
    fillFail: null,
    fillFiles: [],
    fillSymbols: [],
    fillRanges: [],
    failTools: [],
    tool: [],
    output: { kb_search: { items: [hit] } },
    toolFail: null,
    toolDelayMs: 0,
    seenChoices: [],
    fillFixed: [],
    ownPick: 'focused',
    ownPasses: 0,
  });
});

describe('claim check (with a Laya decision model)', () => {
  beforeEach(() => {
    rel.support = null;
    rel.supportFail = null;
    rel.claims = [];
  });

  it('checks the cited sentences against their sources after the answer, and records what may not be supported', async () => {
    agent.laya = true;
    engine.deltas = ['Use COOP [1]. ', 'It needs COEP too [1].'];
    rel.support = [0.8, 0.03];
    await runTurn('What headers does wllama need?');
    expect(rel.claims).toEqual([
      [
        { statement: 'Use COOP.', source: expect.stringContaining(':'), text: expect.any(String) },
        { statement: 'It needs COEP too.', source: expect.stringContaining(':'), text: expect.any(String) },
      ],
    ]);
    expect(assistant().support!.items.map((x) => x.flagged)).toEqual([false, true]);
    expect(statuses().verify).toBe('done');
    expect(assistant().steps!.find((s) => s.kind === 'verify')!.detail).toMatch(/^1 of 2 cited sentences may not be supported by its source · \d+ ms$/);
    expect(assistant().content).toBe('Use COOP [1]. It needs COEP too [1].');
  });

  it('leaves the answer as it is when the check fails', async () => {
    agent.laya = true;
    engine.deltas = ['Use COOP [1].'];
    rel.supportFail = new Error('No Laya model is loaded.');
    await runTurn('What headers does wllama need?');
    expect(statuses().verify).toBe('error');
    expect(assistant()).toMatchObject({ content: 'Use COOP [1].', streaming: false });
    expect(assistant().stopped).toBeUndefined();
    expect(assistant().support).toBeUndefined();
  });

  it('says so when the answer cites, but nothing it cites can be checked', async () => {
    agent.laya = true;
    engine.deltas = ['Use COOP [9].'];
    await runTurn('What headers does wllama need?');
    expect(statuses().verify).toBe('skipped');
    expect(assistant().steps!.find((s) => s.kind === 'verify')!.detail).toMatch(/^The answer’s citations name no listed source/);
    expect(rel.claims).toEqual([]);
  });

  it('checks a sentence whose citation comes after its full stop', async () => {
    agent.laya = true;
    engine.deltas = ['Use COOP and COEP. [1]'];
    await runTurn('What headers does wllama need?');
    expect(statuses().verify).toBe('done');
    expect(rel.claims[0]).toEqual([expect.objectContaining({ statement: 'Use COOP and COEP.' })]);
  });

  it('skips the check when the answer cites nothing, and never runs it without Laya', async () => {
    agent.laya = true;
    await runTurn('What headers does wllama need?');
    expect(statuses().verify).toBe('skipped');
    clearChat();
    agent.laya = false;
    engine.deltas = ['Use COOP [1].'];
    await runTurn('What headers does wllama need?');
    expect(assistant().steps!.some((s) => s.kind === 'verify')).toBe(false);
    expect(rel.claims).toEqual([]);
  });
});

describe('relevance check (with a Laya decision model)', () => {
  const hits = [1, 2, 3, 4].map((i) => ({ ...hit, id: `h${i}`, file: `f${i}.md`, snippet: `passage ${i}` }));
  beforeEach(() => {
    rel.scores = null;
    rel.fail = null;
  });

  it('drops a clearly unhelpful passage before the prompt, and numbers sources as the prompt cites them', async () => {
    agent.laya = true;
    search.hits = hits;
    rel.scores = [0.9, 0.05, 0.01, 0.8];
    await runTurn('What headers does wllama need?');
    expect(statuses().filter).toBe('done');
    expect(assistant().sources!.map((h) => h.id)).toEqual(['h1', 'h2', 'h4']);
    expect(assistant().relevance!.items.map((x) => x.kept)).toEqual([true, true, false, true]);
    expect(assistant().steps!.find((s) => s.kind === 'filter')!.detail).toMatch(/^Kept 3 of 4 passages · ~\d+ tokens less to read/);
    // the dropped passage never reaches the chat model
    const system = engine.seen.at(-1)!.messages[0].content;
    expect(system).toContain('[3] f4.md');
    expect(system).not.toContain('f3.md');
  });

  it('keeps every passage when the check fails', async () => {
    agent.laya = true;
    search.hits = hits;
    rel.fail = new Error('No Laya model is loaded.');
    await runTurn('What headers does wllama need?');
    expect(statuses().filter).toBe('error');
    expect(assistant().steps!.find((s) => s.kind === 'filter')!.detail).toMatch(/kept every passage/);
    expect(assistant().sources).toHaveLength(4);
    expect(statuses().generate).toBe('done');
  });

  it('has nothing to check with only the top results, and no step without Laya', async () => {
    agent.laya = true;
    search.hits = hits.slice(0, 2);
    await runTurn('What headers does wllama need?');
    expect(statuses().filter).toBe('skipped');
    clearChat();
    agent.laya = false;
    search.hits = hits;
    await runTurn('What headers does wllama need?');
    expect(statuses().filter).toBeUndefined();
    expect(assistant().sources).toHaveLength(4);
  });
});

describe('runTurn (fixed pipeline)', () => {
  it('runs analyze → retrieve → build → generate and records the answer, sources and stats', async () => {
    await runTurn('What headers does wllama need?');
    const m = assistant();
    expect(statuses()).toEqual({ analyze: 'done', retrieve: 'done', build: 'done', generate: 'done' });
    expect(m.content).toBe('Hello world');
    expect(m.streaming).toBe(false);
    expect(m.sources).toEqual([hit]);
    expect(m.kbName).toBe('Docs');
    expect(m.stats).toMatchObject({ tokens: 2, promptTokens: 120, nCtx: MODELS[0].n_ctx, model: MODELS[0].name });
    expect(m.steps!.find((s) => s.kind === 'retrieve')!.detail).toMatch(/Retrieved 1 passage from “Docs”/);
  });

  it('records time to first token, measured from the start of the turn (docs/performance.md)', async () => {
    await runTurn('What headers does wllama need?');
    const { firstTokenMs, totalMs } = assistant().stats!;
    // analyze alone waits 120 ms before retrieval starts
    expect(firstTokenMs).toBeGreaterThanOrEqual(120);
    expect(firstTokenMs).toBeLessThanOrEqual(totalMs);
  });

  it('leaves time to first token empty when the model produced no tokens', async () => {
    engine.deltas = [];
    await runTurn('hi');
    expect(assistant().stats!.firstTokenMs).toBeNull();
  });

  it('grounds the system prompt in retrieved passages and passes persona settings to the model', async () => {
    usePersona.getState().set({ temperature: 0.2, maxTokens: 300, verbose: true });
    await runTurn('headers?');
    const call = engine.seen[0];
    expect(call.messages[0].role).toBe('system');
    expect(call.messages[0].content).toContain('[1] README.md (lines 11-33)');
    expect(call.messages.at(-1)).toEqual({ role: 'user', content: 'headers?' });
    expect(call.opts).toMatchObject({ temperature: 0.2, maxTokens: 300, thinking: true });
  });

  it('skips retrieval when no knowledge base grounds the chat', async () => {
    useKb.setState({ grounding: null });
    await runTurn('hi');
    expect(search.calls).toBe(0);
    expect(statuses().retrieve).toBe('skipped');
    expect(engine.seen[0].messages[0].content).not.toContain('Knowledge base');
  });

  it('skips retrieval for a knowledge base that is not indexed yet', async () => {
    useKb.setState({ kbs: [kb({ status: 'empty', nodes: 0 })] });
    await runTurn('hi');
    expect(search.calls).toBe(0);
    expect(assistant().steps!.find((s) => s.kind === 'retrieve')!.detail).toMatch(/not indexed/);
  });

  it('still answers when ug search fails, and marks retrieval failed', async () => {
    search.fail = new Error('ug exploded');
    await runTurn('hi');
    expect(statuses()).toMatchObject({ retrieve: 'error', generate: 'done' });
    expect(assistant().content).toBe('Hello world');
  });

  it('refuses to run without a model and says how to fix it', async () => {
    engine.loaded = false;
    await runTurn('hi');
    const msgs = useChat.getState().messages;
    expect(msgs.map((m) => m.role)).toEqual(['user', 'error']);
    expect(msgs[1].content).toMatch(/Settings → Models/);
    expect(engine.seen).toHaveLength(0);
  });

  it('surfaces generation errors on the trace and as an error message', async () => {
    engine.fail = new Error('kv cache full');
    await runTurn('hi');
    expect(statuses().generate).toBe('error');
    expect(useChat.getState().messages.at(-1)).toMatchObject({ role: 'error', content: 'kv cache full' });
  });

  it('stops cleanly when the operator aborts mid-stream', async () => {
    engine.deltas = Array.from({ length: 50 }, (_, i) => `t${i} `);
    const p = runTurn('long answer please');
    await new Promise((r) => setTimeout(r, 8));
    stopTurn();
    await p;
    const m = assistant();
    expect(m.stopped).toBe(true);
    expect(m.streaming).toBe(false);
    expect(m.steps!.every((s) => s.status !== 'running' && s.status !== 'queued')).toBe(true);
    expect(useChat.getState().messages.some((x) => x.role === 'error')).toBe(false);
  });

  it('ignores blank input and a second turn while one is running', async () => {
    await runTurn('   ');
    expect(useChat.getState().messages).toHaveLength(0);
    const first = runTurn('one');
    await runTurn('two');
    await first;
    expect(useChat.getState().messages.filter((m) => m.role === 'user').map((m) => m.content)).toEqual(['one']);
  });

  it('includes prior turns as history, without reasoning', async () => {
    engine.deltas = ['<think>hidden</think>\n\nfirst answer'];
    await runTurn('first');
    engine.deltas = ['second answer'];
    await runTurn('second');
    const msgs = engine.seen[1].messages.slice(1);
    expect(msgs).toEqual([
      { role: 'user', content: 'first' },
      { role: 'assistant', content: 'first answer' },
      { role: 'user', content: 'second' },
    ]);
  });
});

describe('runTurn (agent mode)', () => {
  beforeEach(() => setAgent({ agentMode: true }));
  const steps = () => assistant().agent!;

  it('answers without tools when the decision is to answer, and says the KB was not consulted', async () => {
    agent.decisions = [{ chosen: 'answer_now' }];
    await runTurn('hi there');
    expect(agent.tool).toHaveLength(0);
    expect(search.calls).toBe(0);
    expect(Object.fromEntries(assistant().steps!.map((s) => [s.kind, s.status]))).toEqual({ plan: 'done', build: 'done', generate: 'done' });
    expect(steps()).toHaveLength(1);
    expect(steps()[0]).toMatchObject({ action: 'answer_now', decision: { chosen: 'answer_now', model: 'Qwen3 1.7B', slot: 'decider' } });
    expect(engine.seen[0].messages[0].content).toContain('“Docs” was not consulted');
  });

  it('searches with the rewritten query, then answers grounded in the results', async () => {
    agent.decisions = [{ chosen: 'kb_search' }, { chosen: 'answer_now' }];
    await runTurn('What headers does wllama need?');
    expect(agent.tool).toEqual([
      { slug: 'docs', call: { tool: 'kb_search', query: 'wllama COOP COEP headers', k: 8, expand: true, max_chars: 6000 } },
    ]);
    const call = steps()[0].call!;
    expect(call).toMatchObject({ tool: 'kb_search', status: 'done', args: { query: 'wllama COOP COEP headers' }, argv: ['search', 'x'], hits: 1 });
    expect(call.output).toContain('serve.json adds the COOP/COEP headers');
    expect(call.observation).toMatch(/^1 passage/);
    // counted by how ug found them, for the trace's one-line summary
    expect(call.found).toEqual({ total: 1, by: {} });
    expect(assistant().sources).toHaveLength(1);
    expect(engine.seen[0].messages[0].content).toContain('[1] README.md (lines 11-33)');
    // the trace's "Assemble context" dialog shows exactly what was sent
    const ctx = assistant().context!;
    expect(ctx.system).toBe(engine.seen[0].messages[0].content);
    expect(ctx.messages.map((x) => x.role)).toEqual(engine.seen[0].messages.map((x) => x.role));
    expect(ctx.passages).toEqual([expect.objectContaining({ n: 1, status: 'in', source: 'README.md:11-33' })]);
    expect(ctx.history).toEqual({ sent: 0, of: 0, chars: 0 });
    // the second decision saw the first result
    expect(agent.seenStates[1]).toContain('kb_search {"query":"wllama COOP COEP headers","scope":"broad"} → 1 passage');
    expect(useTools.getState().stats.kb_search.calls).toBe(1);
  });

  it('offers only the tools that fit the knowledge base, answer first and clarify last', async () => {
    await runTurn('q');
    const docOptions = agent.seenOptions[0];
    expect(docOptions[0]).toBe('answer_now');
    expect(docOptions.at(-1)).toBe('ask_clarification');
    expect(docOptions).not.toContain('kb_find_usages');
    useKb.setState({ kbs: [kb({ kind: 'code' })] });
    await runTurn('who calls add?');
    expect(agent.seenOptions[1]).toContain('kb_find_usages');
  });

  it('looks symbols up first when a symbol tool is chosen before any symbol was seen, then holds `symbol` to them', async () => {
    // Free-text symbol names drew "No symbol named …" from ug (docs/agentic-rag-improvements.md, item 2).
    useKb.setState({ kbs: [kb({ kind: 'code' })] });
    agent.output.kb_find_symbols = { queries: [{ items: [{ id: 'f:add', name: 'add', node_type: 'Function', file: 'math.ts', start_line: 1, end_line: 3 }] }] };
    agent.fills.kb_find_symbols = { names: ['add*'], node_type: 'any' };
    agent.decisions = [{ chosen: 'kb_find_usages' }, { chosen: 'kb_find_usages' }, { chosen: 'answer_now' }];
    await runTurn('who calls add?');
    // The option list doesn't change: a shorter one moved Qwen3 0.6B's choices.
    expect(agent.seenOptions[0]).toContain('kb_find_usages');
    expect(steps()[0]).toMatchObject({ action: 'kb_find_symbols', fallback: 'needs-symbol' });
    expect(steps()[0].decision!.chosen).toBe('kb_find_usages');
    expect(steps()[0].note).toMatch(/Find usages needs a symbol name, .* looking symbols up first/);
    expect(agent.tool.map((t) => t.call.tool)).toEqual(['kb_find_symbols', 'kb_find_usages']);
    expect(agent.fillSymbols).toEqual([[], ['add']]);
  });

  it('reads lines only around what was found: the hit padded by 20 lines', async () => {
    // A probe asked for lines 100–200 of a file it hadn't seen (docs/agentic-rag-improvements.md, item 3).
    agent.decisions = [{ chosen: 'kb_search' }, { chosen: 'kb_read_lines' }, { chosen: 'answer_now' }];
    agent.fills.kb_read_lines = { range: 'README.md:1-53' };
    await runTurn('what does the readme say about headers?');
    expect(agent.fillRanges[1]).toEqual(['README.md:1-53']);
    expect(agent.tool[1].call).toMatchObject({ tool: 'kb_get_code', file: 'README.md', start: 1, end: 53 });
  });

  it('searches first when Read lines is chosen before any line range turned up', async () => {
    agent.decisions = [{ chosen: 'kb_read_lines' }, { chosen: 'answer_now' }];
    await runTurn('show me the headers section');
    expect(agent.seenOptions[0]).toContain('kb_read_lines');
    expect(steps()[0]).toMatchObject({ action: 'kb_search', fallback: 'needs-range' });
    expect(steps()[0].note).toMatch(/Read lines needs a line range .* searching first/);
  });

  it('keeps what a later call adds to a passage it already has, and answers from the merged evidence', async () => {
    // docs/agentic-rag-improvements.md, item 5: the call site Find usages found was dropped as a duplicate.
    useKb.setState({ kbs: [kb({ kind: 'code' })] });
    const cancel = { id: 'f:cancel', name: 'cancelBooking', node_type: 'Function', file: 'booking.ts', start_line: 35, end_line: 40, snippet: 'function cancelBooking() {…}' };
    agent.output.kb_search = { items: [cancel] };
    agent.output.kb_find_usages = {
      nodes: [{ subject: { name: 'refundFraction' }, users: [{ ...cancel, call_sites: [{ line: 37, text: 'const amount = refundFraction(hours)' }] }] }],
    };
    agent.fills.kb_find_usages = { symbol: 'cancelBooking' };
    agent.decisions = [{ chosen: 'kb_search' }, { chosen: 'kb_find_usages' }, { chosen: 'answer_now' }];
    await runTurn('who calls refundFraction?');
    expect(steps()[1].call).toMatchObject({ status: 'done', hits: 1 });
    expect(assistant().sources).toHaveLength(1);
    const system = engine.seen[0].messages[0].content;
    expect(system).toContain('function cancelBooking() {…}\nFunction cancelBooking → uses refundFraction\n  37: const amount = refundFraction(hours)');
  });

    it('searches first without a decision when the request is not small talk', async () => {
    // docs/agentic-rag-improvements.md, item 7: step 1 chose search 23 of 24 times.
    useTools.setState({ searchFirst: true });
    agent.decisions = [{ chosen: 'answer_now' }];
    await runTurn('What headers does wllama need?');
    expect(steps()[0]).toMatchObject({ action: 'kb_search', decision: null });
    expect(steps()[0].note).toMatch(/Searched first, without a decision/);
    expect(agent.tool.map((t) => t.call.tool)).toEqual(['kb_search']);
    expect(agent.seenOptions).toHaveLength(1); // only the second step was decided
  });

  it('searches first in a code or mixed knowledge base too, even for a question that names a symbol (item 14)', async () => {
    for (const kind of ['code', 'mixed'] as const) {
      clearChat();
      agent.tool = [];
      agent.seenOptions = [];
      useKb.setState({ kbs: [kb({ kind })] });
      useTools.setState({ searchFirst: true });
      agent.decisions = [{ chosen: 'answer_now' }];
      await runTurn('Which functions call computeFare?');
      expect(steps()[0]).toMatchObject({ action: 'kb_search', decision: null });
      expect(steps()[0].note).toBe('Searched first, without a decision: a question about the knowledge base’s content starts with a search, which reads the matching text.');
    }
  });

  it('still lets the model decide on small talk, and after the first step', async () => {
    useTools.setState({ searchFirst: true });
    agent.decisions = [{ chosen: 'answer_now' }];
    await runTurn('Thanks, that is all!');
    expect(steps()[0]).toMatchObject({ action: 'answer_now' });
    expect(steps()[0].decision).not.toBeNull();
    expect(agent.tool).toHaveLength(0);
  });

  it('lets the model decide on a question to the assistant, instead of searching for it', async () => {
    useTools.setState({ searchFirst: true });
    agent.decisions = [{ chosen: 'answer_now' }];
    await runTurn('who are u?');
    expect(steps()[0]).toMatchObject({ action: 'answer_now' });
    expect(agent.tool).toHaveLength(0);
  });

  it('does not search first when search is switched off', async () => {
    useTools.setState({ searchFirst: true });
    setPolicy('kb_search', 'off');
    agent.decisions = [{ chosen: 'answer_now' }];
    await runTurn('What headers does wllama need?');
    expect(steps()[0].decision).not.toBeNull();
  });

  it('does not count document sections as symbols', async () => {
    useKb.setState({ kbs: [kb({ kind: 'mixed' })] });
    agent.decisions = [{ chosen: 'kb_search' }, { chosen: 'kb_get_code' }, { chosen: 'answer_now' }];
    agent.fills.kb_find_symbols = { names: ['readme'], node_type: 'any' };
    await runTurn('what does the readme say?');
    expect(steps()[1]).toMatchObject({ action: 'kb_find_symbols', fallback: 'needs-symbol' });
  });

  it('counts code hits from a search as symbols', async () => {
    useKb.setState({ kbs: [kb({ kind: 'mixed' })] });
    agent.output.kb_search = { items: [hit, { ...hit, id: 'c:VAT', name: 'VAT_RATE', node_type: 'Constant', file: 'tax.ts' }] };
    agent.decisions = [{ chosen: 'kb_search' }, { chosen: 'kb_get_code' }, { chosen: 'answer_now' }];
    agent.fills.kb_get_code = { symbol: 'VAT_RATE' };
    await runTurn('what is the VAT rate?');
    expect(steps()[1]).toMatchObject({ action: 'kb_get_code' });
    expect(steps()[1].fallback).toBeUndefined();
    expect(agent.fillSymbols[1]).toEqual(['VAT_RATE']);
  });

  it('never offers a tool the user switched off', async () => {
    setPolicy('kb_search', 'off');
    await runTurn('q');
    expect(agent.seenOptions[0]).not.toContain('kb_search');
  });

  it('asks for approval under the Ask policy, and runs only once approved', async () => {
    setPolicy('kb_search', 'ask');
    agent.decisions = [{ chosen: 'kb_search' }, { chosen: 'answer_now' }];
    const p = runTurn('headers?');
    await vi.waitFor(() => expect(steps()[0].call?.status).toBe('awaiting'));
    expect(agent.tool).toHaveLength(0);
    expect(steps()[0].call!.approval).toBe('pending');
    resolveApproval(steps()[0].id, true);
    await p;
    expect(agent.tool).toHaveLength(1);
    expect(steps()[0].call).toMatchObject({ status: 'done', approval: 'approved' });
  });

  it('records a denial, tells the decider, and does not offer that tool again', async () => {
    setPolicy('kb_search', 'ask');
    agent.decisions = [{ chosen: 'kb_search' }, { chosen: 'answer_now' }];
    const p = runTurn('headers?');
    await vi.waitFor(() => expect(steps()[0].call?.status).toBe('awaiting'));
    resolveApproval(steps()[0].id, false);
    await p;
    expect(agent.tool).toHaveLength(0);
    expect(steps()[0].call).toMatchObject({ status: 'denied', approval: 'denied' });
    expect(agent.seenStates[1]).toContain('The user declined this call');
    expect(agent.seenOptions[1]).not.toContain('kb_search');
  });

  it('does not offer a tool again once it returned results, and switches the answer option to use them', async () => {
    agent.decisions = [{ chosen: 'kb_search' }, { chosen: 'answer_now' }];
    await runTurn('headers?');
    expect(agent.seenOptions[0]).toContain('kb_search');
    expect(agent.seenOptions[1]).not.toContain('kb_search');
  });

  it('lets an empty search be retried once, then stops offering it', async () => {
    agent.output = { kb_search: { items: [] } };
    agent.fills.kb_search = [
      { query: 'first', scope: 'broad' },
      { query: 'second', scope: 'broad' },
    ];
    agent.decisions = [{ chosen: 'kb_search' }, { chosen: 'kb_search' }, { chosen: 'answer_now' }];
    await runTurn('headers?');
    expect(agent.tool.map((t) => t.call.query)).toEqual(['first', 'second']);
    expect(agent.seenOptions[1]).toContain('kb_search');
    expect(agent.seenOptions[2]).not.toContain('kb_search');
  });

  it('uses the given seed for the option shuffle, so a run can be repeated (the agent eval)', async () => {
    agent.decisions = [{ chosen: 'kb_search' }, { chosen: 'answer_now' }];
    await runTurn('headers?', { seed: 5 });
    expect(steps().map((s) => s.decision?.seed)).toEqual([5, 6]);
  });

  it('treats a choice that was not offered as a failed decision', async () => {
    setPolicy('kb_overview', 'off');
    agent.decisions = [{ chosen: 'kb_overview' }, { chosen: 'answer_now' }];
    await runTurn('headers?');
    expect(steps()[0].note).toMatch(/wasn't offered/);
    expect(agent.tool.map((t) => t.call.tool)).toEqual(['kb_search']);
  });

  it('does not run the same call twice in a turn', async () => {
    agent.output = { kb_search: { items: [] } };
    agent.decisions = [{ chosen: 'kb_search' }, { chosen: 'kb_search' }, { chosen: 'answer_now' }];
    await runTurn('headers?');
    expect(agent.tool).toHaveLength(1);
    expect(steps()[1].call).toMatchObject({ status: 'skipped' });
    expect(steps()[1].call!.error).toMatch(/already ran at step 1/);
  });

  it('stops when the model keeps repeating a failed call, instead of looping to the step limit', async () => {
    // Reported: "show an overview of kb1" → overview, then File outline with
    // {"file":"kb1"} failed and was re-chosen, skipped, over and over.
    setAgent({ maxSteps: 8 });
    agent.failTools = ['kb_file_context'];
    agent.fills.kb_file_context = { file: 'kb1' };
    agent.decisions = [{ chosen: 'kb_overview' }, ...Array.from({ length: 7 }, () => ({ chosen: 'kb_file_context' }))];
    await runTurn('show an overview of kb1');
    expect(steps().map((s) => s.call?.status ?? s.action)).toEqual(['done', 'error', 'skipped', 'answer_now']);
    expect(steps().at(-1)!.note).toMatch(/kept repeating/);
    expect(agent.tool.map((t) => t.call.tool)).toEqual(['kb_overview', 'kb_file_context']);
  });

  it('tells the argument writer which files exist: indexed sources, then files in the results', async () => {
    const src = (file: string, status: 'indexed' | 'failed') =>
      ({ file, original: file, kind: 'MD', bytes: 1, approxTokens: 1, addedAt: 0, status }) as const;
    useKb.setState({ kbs: [kb({ sources: [src('notes.md', 'indexed'), src('broken.md', 'failed')] })] });
    agent.decisions = [{ chosen: 'kb_search' }, { chosen: 'kb_file_context' }, { chosen: 'answer_now' }];
    agent.fills.kb_file_context = { file: 'README.md' };
    await runTurn('outline the readme');
    expect(agent.fillFiles).toEqual([['notes.md'], ['notes.md', 'README.md']]);
  });

  it('stops at the step limit and answers with what it has', async () => {
    setAgent({ maxSteps: 2 });
    agent.decisions = [{ chosen: 'kb_search' }, { chosen: 'kb_overview' }, { chosen: 'kb_file_context' }];
    agent.fills.kb_file_context = { file: 'README.md' };
    await runTurn('everything');
    expect(agent.tool.map((t) => t.call.tool)).toEqual(['kb_search', 'kb_overview']);
    expect(steps().at(-1)).toMatchObject({ action: 'answer_now', decision: null });
    expect(steps().at(-1)!.note).toMatch(/limit of 2 tool calls/);
    expect(statusesOf(assistant()).generate).toBe('done');
  });

  it('falls back to one search when the decision is not confident enough', async () => {
    agent.decisions = [{ chosen: 'kb_overview', confidence: 0.2 }, { chosen: 'answer_now' }];
    await runTurn('headers?');
    expect(agent.tool.map((t) => t.call.tool)).toEqual(['kb_search']);
    expect(steps()[0].note).toMatch(/Low confidence \(20% < 30%\) in “kb_overview”/);
    expect(steps()[0].fallback).toBe('low-confidence');
  });

  it('falls back to the fixed search when the decision model fails, with the reason on the trace', async () => {
    agent.decisions = [new Error('no scores for every option'), { chosen: 'answer_now' }];
    await runTurn('headers?');
    expect(agent.tool.map((t) => t.call.tool)).toEqual(['kb_search']);
    expect(steps()[0].note).toMatch(/Decision failed \(no scores for every option\)/);
    expect(steps()[0].fallback).toBe('decision-failed');
  });

  it('with Laya, asks whether the results suffice once there are some, and answers on yes', async () => {
    agent.laya = true;
    agent.decisions = [{ chosen: 'kb_search' }, { chosen: 'kb_overview', stop: 0.8 }];
    await runTurn('What headers does wllama need?');
    expect(agent.tool.map((t) => t.call.tool)).toEqual(['kb_search']);
    // no stop question before any results; after, answer_now leaves the choice
    expect(agent.seenStops).toEqual([null, STOP]);
    expect(agent.seenOptions[1]).not.toContain('answer_now');
    expect(steps().at(-1)).toMatchObject({ action: 'answer_now', decision: { stop: { statement: STOP, probability: 0.8 } } });
    expect(steps().at(-1)!.note).toMatch(/results cover the request \(80% likely\)/);
  });

  it('with Laya, keeps using tools while it says the results fall short', async () => {
    agent.laya = true;
    agent.decisions = [{ chosen: 'kb_search' }, { chosen: 'kb_overview', stop: 0.2 }, { chosen: 'kb_overview', stop: 0.9 }];
    await runTurn('What headers does wllama need?');
    expect(agent.tool.map((t) => t.call.tool)).toEqual(['kb_search', 'kb_overview']);
    expect(steps().at(-1)!.action).toBe('answer_now');
  });

  it('with Laya, asks the search scope in the decision pass and pins it for the argument fill', async () => {
    agent.laya = true;
    agent.decisions = [{ chosen: 'kb_search', scope: 'focused' }, { chosen: 'kb_overview', stop: 0.9 }];
    await runTurn('Which function implements the group discount?');
    expect(agent.seenChoices[0]).toContain('kb_search_scope');
    expect(agent.ownPasses).toBe(0);
    expect(agent.fillFixed[0]).toEqual({ scope: 'focused' });
    expect(steps()[0].call!.argChoices).toEqual([{ arg: 'scope', value: 'focused', probability: 0.9, model: 'Qwen3 1.7B' }]);
  });

  it('with Laya, picks the scope in a pass of its own when the search came without a decision', async () => {
    agent.laya = true;
    useTools.setState({ searchFirst: true });
    agent.decisions = [{ chosen: 'answer_now', stop: 0.9 }];
    await runTurn('What headers does wllama need?');
    expect(agent.ownPasses).toBe(1);
    expect(agent.fillFixed[0]).toEqual({ scope: 'focused' });
    expect(steps()[0].call!.argChoices?.[0]).toMatchObject({ arg: 'scope', value: 'focused', model: 'Laya Multilingual' });
  });

  it('lets the chat model write the scope when Laya can’t pick it', async () => {
    agent.laya = true;
    useTools.setState({ searchFirst: true });
    agent.ownPick = new Error('No Laya model is loaded.');
    agent.decisions = [{ chosen: 'answer_now', stop: 0.9 }];
    await runTurn('What headers does wllama need?');
    expect(agent.tool.map((t) => t.call.tool)).toEqual(['kb_search']);
    expect(agent.fillFixed[0]).toEqual({});
    expect(steps()[0].call!.argChoices).toBeUndefined();
  });

  it('never asks the stop question without Laya', async () => {
    agent.decisions = [{ chosen: 'kb_search' }, { chosen: 'answer_now' }];
    await runTurn('What headers does wllama need?');
    expect(agent.seenStops.every((s) => s === null)).toBe(true);
    expect(agent.seenOptions.at(-1)).toContain('answer_now');
    // nor argument choices: the chat model writes the scope
    expect(agent.seenChoices.every((c) => c.length === 0)).toBe(true);
    expect(agent.ownPasses).toBe(0);
  });

  it('records a failed decision call on the trace, with its time and what was sent', async () => {
    const { DecisionError } = await import('../llm/decide');
    const io = { request: { messages: [{ role: 'user', content: 'State: …' }], params: { max_tokens: 1 } }, response: { sampled: 'x', topLogprobs: [{ token: 'x', logprob: -1 }] } };
    agent.decisions = [new DecisionError('The decision model scored none of the options (A, B).', io, 640), { chosen: 'answer_now' }];
    await runTurn('headers?');
    expect(steps()[0].failedDecision).toEqual({ error: 'The decision model scored none of the options (A, B).', ms: 640, io });
    expect(steps()[1].failedDecision).toBeUndefined();
  });

  it('searches with the question as written when the arguments are invalid', async () => {
    agent.fillFail = 'query is too short';
    agent.decisions = [{ chosen: 'kb_search' }, { chosen: 'answer_now' }];
    await runTurn('What headers does wllama need?');
    expect(agent.tool[0].call).toMatchObject({ query: 'What headers does wllama need?' });
    expect(steps()[0].call!.error).toMatch(/searched with the question as written/);
  });

  it('keeps going after a tool error, then stops after repeated errors', async () => {
    agent.toolFail = new Error('No symbol named add');
    useKb.setState({ kbs: [kb({ kind: 'code' })] });
    agent.fills.kb_find_symbols = { names: ['add'], node_type: 'any' };
    agent.decisions = [{ chosen: 'kb_find_symbols' }, { chosen: 'kb_search' }, { chosen: 'kb_overview' }];
    await runTurn('who calls add?');
    expect(steps().map((s) => s.call?.status ?? s.action)).toEqual(['error', 'error', 'answer_now']);
    expect(steps()[0].call!.error).toBe('No symbol named add');
    expect(agent.seenStates[1]).toContain('kb_find_symbols {"names":["add"],"node_type":"any"} → Failed: No symbol named add');
    expect(steps().at(-1)!.note).toMatch(/repeated tool errors/);
    expect(statusesOf(assistant()).generate).toBe('done');
  });

  it('asks a clarifying question when that is the decision', async () => {
    agent.decisions = [{ chosen: 'ask_clarification' }];
    await runTurn('do the thing');
    expect(engine.seen[0].messages[0].content).toContain('Ask the user one short clarifying question');
  });

  it('stops mid-tool when the operator aborts, leaving nothing marked running', async () => {
    agent.toolDelayMs = 50;
    agent.decisions = [{ chosen: 'kb_search' }];
    const p = runTurn('headers?');
    await vi.waitFor(() => expect(steps()[0]?.call?.status).toBe('running'));
    stopTurn();
    await p;
    expect(assistant().stopped).toBe(true);
    expect(steps()[0].call!.status).toBe('skipped');
    expect(engine.seen).toHaveLength(0);
  });

  it('stops while waiting for approval', async () => {
    setPolicy('kb_search', 'ask');
    agent.decisions = [{ chosen: 'kb_search' }];
    const p = runTurn('headers?');
    await vi.waitFor(() => expect(steps()[0]?.call?.status).toBe('awaiting'));
    stopTurn();
    await p;
    expect(steps()[0].call).toMatchObject({ status: 'skipped', approval: undefined });
    expect(agent.tool).toHaveLength(0);
  });

  it('uses the fixed pipeline when no knowledge base grounds the chat', async () => {
    useKb.setState({ grounding: null });
    await runTurn('hi');
    expect(agent.seenStates).toHaveLength(0);
    expect(assistant().agent).toBeUndefined();
  });
});

function statusesOf(m: ReturnType<typeof assistant>) {
  return Object.fromEntries(m.steps!.map((s) => [s.kind, s.status]));
}
