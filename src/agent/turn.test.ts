// The agent turn orchestrator with the LLM engine and ug mocked out: verifies
// the step sequence the Execution Trace shows, and every failure path.
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
}));

const { runTurn, stopTurn } = await import('./turn');

const kb = (over: Partial<KbInfo> = {}): KbInfo => ({
  slug: 'docs',
  name: 'Docs',
  createdAt: 0,
  sources: [],
  lastIndexedAt: 1,
  lastError: null,
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
});

describe('runTurn', () => {
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
