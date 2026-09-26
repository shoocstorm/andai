// Headless-ish end-to-end check of the engine inside the real Tauri webview:
//   VITE_SMOKE=tiny bun run tauri dev     (1.2 MB model)
//   VITE_SMOKE=qwen3-0.6b bun run tauri dev
// Reports capabilities, load time and tok/s to the terminal, then exits.
import { invoke } from '@tauri-apps/api/core';
import { decisionOptions, QUESTION } from './agent/loop';
import { agentState } from './agent/prompt';
import { fillArgs } from './agent/tools/argfill';
import { available, toolById } from './agent/tools/registry';
import { runTurn } from './agent/turn';
import { decide } from './llm/decide';
import { chat, loadModel, useEngine } from './llm/engine';
import { clearChat, useChat } from './state/chat';
import { kbAddFiles, kbTool } from './kb/api';
import { modelById } from './llm/models';
import { addFiles, createKb, deleteKb, useKb } from './state/kb';
import { TOOL_DEFAULTS, useTools } from './state/tools';

const log = (line: string) => invoke('dev_log', { line }).catch(() => console.log(line));

export async function runSmoke(which: string) {
  const id = which === 'tiny' ? 'stories-260k' : which;
  try {
    await log(`caps ${JSON.stringify(useEngine.getState().caps)} ua=${navigator.userAgent}`);
    const t = performance.now();
    await loadModel(id);
    const s = useEngine.getState();
    if (s.status !== 'ready') throw new Error(`load failed: ${s.error}`);
    await log(`loaded ${id} in ${(performance.now() - t).toFixed(0)}ms info=${JSON.stringify(s.info)}`);
    let text = '';
    let rate = 0;
    const thinking = false;
    for await (const ev of chat(
      [{ role: 'user', content: id === 'stories-260k' ? 'Once upon a time' : 'In one sentence: what is WebAssembly?' }],
      { temperature: 0.7, maxTokens: 64, thinking, signal: new AbortController().signal },
    )) {
      if (ev.type === 'delta') {
        text += ev.text;
        rate = ev.tokPerSec;
      } else await log(`done prompt=${ev.promptTokens} completion=${ev.completionTokens}`);
    }
    await log(`OK ${rate.toFixed(1)} tok/s :: ${JSON.stringify(text.slice(0, 300))}`);
    await invoke('dev_exit', { code: 0 });
  } catch (e) {
    await log(`FAIL ${e instanceof Error ? `${e.message}\n${e.stack}` : String(e)}`);
    await invoke('dev_exit', { code: 1 });
  }
}

/**
 * AGENTS.md §9 in the real webview, where the unit tests can't reach: Rust
 * refuses paths the user never granted, the window can't leave the app
 * origin, and the CSP blocks egress to hosts outside the allowlist.
 */
async function securityChecks(slug: string) {
  const [, errors] = await kbAddFiles(slug, ['/etc/hosts']);
  if (!errors[0]?.includes('not added by you')) throw new Error(`ungranted path was not refused: ${JSON.stringify(errors)}`);

  const origin = location.origin;
  location.href = 'https://example.com/';
  await new Promise((r) => setTimeout(r, 800));
  if (location.origin !== origin) throw new Error('navigation left the app origin');
  if (window.open('https://example.com/')) throw new Error('window.open was allowed');

  // A cached document can keep a stale (or no) CSP (see vite.config.ts), so
  // first prove the policy is enforced at all: it has no 'unsafe-eval'.
  let evalRan = true;
  try {
    new Function('return 1')();
  } catch {
    evalRan = false;
  }
  if (evalRan) throw new Error('CSP is not enforced: eval ran');

  // The runner's canary answers any origin, so only the CSP can stop these;
  // it also asserts that no request arrived. (WKWebView rejects with a bare
  // "Load failed" and doesn't fire securitypolicyviolation for connect-src.)
  const canary = import.meta.env.VITE_SMOKE_CANARY as string;
  const leaked = await fetch(`${canary}?via=fetch&q=secret`).then(
    () => true,
    () => false,
  );
  if (leaked) throw new Error('fetch to a host outside connect-src succeeded');
  await new Promise<void>((resolve) => {
    const img = new Image();
    img.onload = img.onerror = () => resolve();
    img.src = `${canary}?via=img&q=secret`;
  });
  await log('security ok: ungranted path refused, navigation blocked, egress attempted (runner checks the canary)');
}

/**
 * Agent mode against the real model and ug (AGENTS.md §1.6: the decision
 * readout's logit bias and post-sampling logprobs are llama.cpp behavior that
 * only a real run can confirm). Each part is probed on its own so a failure
 * says which one broke, then one full agent turn is traced.
 */
async function agentProbes(slug: string, question: string) {
  const kb = useKb.getState().kbs.find((k) => k.slug === slug)!;
  const tools = available(kb.kind, {});
  const state = agentState({
    prompt: question,
    history: [],
    kb: { name: kb.name, kind: kb.kind, nodes: kb.nodes, files: kb.sources.length },
    observations: [],
    step: 0,
    maxSteps: TOOL_DEFAULTS.maxSteps,
  });
  const out: Record<string, unknown> = { kind: kb.kind, tools: tools.map((t) => t.id) };
  try {
    const d = await decide(state, QUESTION, decisionOptions(tools, 1));
    out.decision = { chosen: d.chosen, confidence: d.confidence, ms: d.ms, model: d.model, slot: d.slot, options: d.options.map((o) => [o.id, o.probability]) };
  } catch (e) {
    out.decision = { error: e instanceof Error ? e.message : String(e) };
  }
  await log(`agent decision ${JSON.stringify(out.decision)}`);
  const t = performance.now();
  const fill = await fillArgs(toolById('kb_search')!, { state, kind: kb.kind }).catch((e) => ({ ok: false, errors: [String(e)], raw: '' }));
  out.fill = { ...fill, ms: performance.now() - t };
  await log(`agent fill ${JSON.stringify(out.fill)}`);
  out.overview = await kbTool(slug, { tool: 'kb_overview' })
    .then((r) => ({ ok: true, argv: r.argv, ms: r.ms, kbType: (r.output as { kb_type?: string })?.kb_type }))
    .catch((e) => ({ ok: false, error: String(e) }));
  await log(`agent overview ${JSON.stringify(out.overview)}`);

  useTools.setState({ agentMode: true });
  clearChat();
  const started = performance.now();
  await runTurn(question);
  const m = useChat.getState().messages.find((x) => x.role === 'assistant');
  out.turn = {
    ms: performance.now() - started,
    steps: m?.steps?.map((x) => ({ kind: x.kind, status: x.status, detail: x.detail })),
    agent: m?.agent?.map((a) => ({
      action: a.action,
      note: a.note,
      confidence: a.decision?.confidence,
      decisionMs: a.decision?.ms,
      call: a.call && { status: a.call.status, args: a.call.args, argv: a.call.argv, error: a.call.error, observation: a.call.observation, ms: a.call.ms },
    })),
    sources: m?.sources?.map((h) => h.file),
    answer: m?.content,
  };
  await log(`agent turn ${JSON.stringify(out.turn).slice(0, 4000)}`);
  useTools.setState({ agentMode: false });
  return out;
}

/**
 * Full pipeline inside the real app: create KB → ingest via ug → load model →
 * one RAG turn through the same orchestrator the UI uses. The App is rendered
 * alongside, so UI code runs in WKWebView too.
 *   VITE_SMOKE=e2e VITE_SMOKE_FILES=/abs/a.md,/abs/b.pdf bun run tauri dev
 */
export async function runE2E(files: string[], model = 'qwen3-0.6b') {
  // The harness shares webview storage with the real app on this machine, so
  // snapshot what it touches and put it back afterwards.
  const savedChat = { messages: useChat.getState().messages, session: useChat.getState().session };
  const savedKb = { grounding: useKb.getState().grounding, selected: useKb.getState().selected };
  const { agentMode, maxSteps, minConfidence, searchFirst, policies, stats } = useTools.getState();
  const savedTools = { agentMode, maxSteps, minConfidence, searchFirst, policies, stats };
  const restore = () => {
    useChat.setState(savedChat);
    useKb.setState(savedKb);
    useTools.setState(savedTools);
  };
  try {
    await new Promise((r) => setTimeout(r, 1500)); // let App mount + refresh KBs
    const stale = useKb.getState().kbs.find((k) => k.name === 'e2e-docs');
    if (stale) await deleteKb(stale.slug);
    const kb = await createKb('e2e-docs');
    if (!kb) throw new Error('createKb failed');
    await log(`kb created ${kb.slug}`);
    await securityChecks(kb.slug);
    const t = performance.now();
    await addFiles(kb.slug, files);
    const ingestMs = performance.now() - t;
    const after = useKb.getState().kbs.find((k) => k.slug === kb.slug)!;
    await log(
      `indexed in ${ingestMs.toFixed(0)}ms status=${after.status} nodes=${after.nodes} edges=${after.edges} sources=${JSON.stringify(after.sources.map((s) => [s.file, s.kind, s.status]))} err=${after.lastError}`,
    );
    const logs = useKb.getState().logs[kb.slug] ?? [];
    await log(`ug log lines=${logs.length} last=${JSON.stringify(logs.slice(-2).map((l) => l.line))}`);
    useKb.setState({ grounding: kb.slug });

    await loadModel(model);
    while (useEngine.getState().status === 'loading') await new Promise((r) => setTimeout(r, 200));
    if (useEngine.getState().loadedId !== model) await loadModel(model);
    if (useEngine.getState().status !== 'ready') throw new Error(`load failed: ${useEngine.getState().error}`);
    // The fixed pipeline first: its grounding checks and perf baselines
    // (docs/performance.md) are about one search + one answer.
    useTools.setState({ ...TOOL_DEFAULTS, agentMode: false, policies: {}, stats: {} });
    clearChat();
    const question = 'What HTTP headers does wllama need for multi-threading, and why?';
    await runTurn(question);
    const msg = useChat.getState().messages.find((m) => m.role === 'assistant');
    // Read before the agent turn, which records searches of its own.
    const searchMs = useKb.getState().lastSearch?.ms ?? null;
    const agent = await agentProbes(kb.slug, question);
    const info = useEngine.getState().info;
    const result = {
      caps: useEngine.getState().caps,
      engine: info,
      verifyMs: useEngine.getState().lastVerifyMs,
      kb: { status: after.status, nodes: after.nodes, edges: after.edges, error: after.lastError, sources: after.sources.map((x) => ({ file: x.file, kind: x.kind, status: x.status })) },
      ugLogLines: logs.length,
      steps: msg?.steps?.map((x) => ({ kind: x.kind, status: x.status, detail: x.detail })),
      sources: msg?.sources?.map((h) => h.file),
      stats: msg?.stats,
      answer: msg?.content,
      agent,
      // Compared with perf/baseline.json by the runner (docs/performance.md).
      perf: {
        ingestMs,
        searchMs,
        loadMs: useEngine.getState().lastLoadMs,
        verifyMs: useEngine.getState().lastVerifyMs,
        modelBytes: modelById(model)?.bytes ?? null,
        firstTokenMs: msg?.stats?.firstTokenMs ?? null,
        tokPerSec: msg?.stats?.tokPerSec ?? null,
        promptTokPerSec: msg?.stats?.promptTokPerSec ?? null,
        promptTokens: msg?.stats?.promptTokens ?? null,
      },
    };
    await log(`RESULT ${JSON.stringify(result)}`);
    // clean up BEFORE reporting OK: the runner may kill the app as soon as it sees it
    if (!import.meta.env.VITE_SMOKE_KEEP) {
      await deleteKb(kb.slug);
      restore();
    }
    await log(`OK answer=${JSON.stringify(msg?.content.slice(0, 300))}`);
    await invoke('dev_exit', { code: 0 });
  } catch (e) {
    restore();
    await log(`FAIL ${e instanceof Error ? `${e.message}\n${e.stack}` : String(e)}`);
    await invoke('dev_exit', { code: 1 });
  }
}
