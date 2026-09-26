// Agent eval harness (docs/agentic-rag-improvements.md, item 1): runs each
// question through `runTurn` in the real app, against knowledge bases built
// from tests/fixtures/eval/, and reports what the agent did. The runner
// (scripts/eval-agent.mjs) passes the cases in VITE_EVAL, grants the fixture
// files, and scores the CASE lines this prints:
//   bun run eval:agent
// Loaded only when VITE_SMOKE=eval (main.tsx), so none of it ships in the app.
import { invoke } from '@tauri-apps/api/core';
import { runTurn } from './agent/turn';
import { loadModel, useEngine } from './llm/engine';
import { clearChat, uid, useChat, type Message } from './state/chat';
import { addFiles, createKb, deleteKb, useKb } from './state/kb';
import { usePersona } from './state/persona';
import { TOOL_DEFAULTS, useTools } from './state/tools';

export type EvalCase = {
  id: string;
  /** Key into `kbs`: which knowledge base the question runs against. */
  kb: string;
  prompt: string;
  /** Earlier turns, for follow-up questions. */
  history?: { role: 'user' | 'assistant'; content: string }[];
};

export type EvalInput = {
  model: string;
  /** Fixes the agent's option shuffle, so two runs see the same prompts. */
  seed: number;
  /** Knowledge base key → absolute paths of its (granted) files. */
  kbs: Record<string, string[]>;
  cases: EvalCase[];
};

const log = (line: string) => invoke('dev_log', { line }).catch(() => console.log(line));

/** What the runner scores: every step of the turn, the sources and the answer. */
function report(c: EvalCase, msg: Message | undefined, error: string | null, ms: number) {
  return {
    id: c.id,
    ms: Math.round(ms),
    error,
    steps: (msg?.agent ?? []).map((a) => ({
      action: a.action,
      fallback: a.fallback ?? null,
      note: a.note ?? null,
      decision: a.decision && {
        chosen: a.decision.chosen,
        confidence: a.decision.confidence,
        ms: a.decision.ms,
        promptTokens: a.decision.promptTokens,
        model: a.decision.model,
      },
      call: a.call && {
        tool: a.call.tool,
        status: a.call.status,
        args: a.call.args,
        argAttempts: a.call.argAttempts,
        // A done call that carries an error ran with fallback arguments (loop.ts: kb_search with the question).
        argsFallback: a.call.status === 'done' && !!a.call.error,
        argv: a.call.argv ?? null,
        error: a.call.error ?? null,
        observation: a.call.observation ?? null,
        hits: a.call.hits ?? 0,
        ms: a.call.ms ?? null,
      },
    })),
    sources: (msg?.sources ?? []).map((h) => `${h.file}:${h.start_line}-${h.end_line}`),
    answer: msg?.content ?? '',
    stats: msg?.stats ? { promptTokens: msg.stats.promptTokens, tokens: msg.stats.tokens, firstTokenMs: msg.stats.firstTokenMs } : null,
  };
}

export async function runEval(input: EvalInput) {
  // Shares webview storage with the real app (like runE2E): snapshot what it touches.
  const savedChat = { messages: useChat.getState().messages, session: useChat.getState().session };
  const { grounding, selected, k, maxChars } = useKb.getState();
  const savedKb = { grounding, selected, k, maxChars };
  const { agentMode, maxSteps, minConfidence, policies, stats } = useTools.getState();
  const savedTools = { agentMode, maxSteps, minConfidence, policies, stats };
  const { set: _set, reset: _reset, ...savedPersona } = usePersona.getState();
  const slugs: string[] = [];
  const cleanup = async () => {
    if (!import.meta.env.VITE_SMOKE_KEEP) for (const slug of slugs) await deleteKb(slug);
    useChat.setState(savedChat);
    useKb.setState(savedKb);
    useTools.setState(savedTools);
    usePersona.setState(savedPersona);
  };
  try {
    await new Promise((r) => setTimeout(r, 1500)); // let App mount + refresh KBs
    const slugOf: Record<string, string> = {};
    for (const [key, files] of Object.entries(input.kbs)) {
      const name = `eval-${key}`;
      const stale = useKb.getState().kbs.find((x) => x.name === name);
      if (stale) await deleteKb(stale.slug);
      const kb = await createKb(name);
      if (!kb) throw new Error(`createKb ${name} failed`);
      slugs.push(kb.slug);
      await addFiles(kb.slug, files);
      const after = useKb.getState().kbs.find((x) => x.slug === kb.slug)!;
      if (after.status !== 'ready' || after.sources.some((s) => s.status !== 'indexed')) {
        throw new Error(`${name} did not index: ${after.status} ${after.lastError ?? ''} ${JSON.stringify(after.sources.map((s) => [s.file, s.status]))}`);
      }
      slugOf[key] = kb.slug;
      await log(`KB ${JSON.stringify({ key, kind: after.kind, nodes: after.nodes, edges: after.edges, files: after.sources.length })}`);
    }

    await loadModel(input.model);
    while (useEngine.getState().status === 'loading') await new Promise((r) => setTimeout(r, 200));
    if (useEngine.getState().loadedId !== input.model) await loadModel(input.model);
    if (useEngine.getState().status !== 'ready') throw new Error(`load failed: ${useEngine.getState().error}`);

    // The shipped defaults, with a greedy answer so a run is repeatable.
    useTools.setState({ ...TOOL_DEFAULTS, agentMode: true, policies: {}, stats: {} });
    useKb.setState({ k: 8, maxChars: 6000 });
    usePersona.getState().reset();
    usePersona.setState({ temperature: 0 });
    await log(`START ${JSON.stringify({ model: input.model, seed: input.seed, cases: input.cases.length, engine: useEngine.getState().info })}`);

    for (const c of input.cases) {
      const slug = slugOf[c.kb];
      if (!slug) throw new Error(`${c.id}: unknown kb “${c.kb}”`);
      useKb.setState({ grounding: slug });
      clearChat();
      const history: Message[] = (c.history ?? []).map((h) => ({ id: uid(), role: h.role, content: h.content, createdAt: Date.now() }));
      useChat.setState({ messages: history });
      const started = performance.now();
      let error: string | null = null;
      try {
        await runTurn(c.prompt, { seed: input.seed });
      } catch (e) {
        error = e instanceof Error ? e.message : String(e);
      }
      const ms = performance.now() - started;
      const after = useChat.getState().messages.slice(history.length);
      error ??= after.find((m) => m.role === 'error')?.content ?? null;
      await log(`CASE ${JSON.stringify(report(c, after.find((m) => m.role === 'assistant'), error, ms))}`);
    }
    await cleanup();
    await log('OK');
    await invoke('dev_exit', { code: 0 });
  } catch (e) {
    await cleanup().catch(() => {});
    await log(`FAIL ${e instanceof Error ? `${e.message}\n${e.stack}` : String(e)}`);
    await invoke('dev_exit', { code: 1 });
  }
}
