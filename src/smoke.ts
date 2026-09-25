// Headless-ish end-to-end check of the engine inside the real Tauri webview:
//   VITE_SMOKE=tiny npm run tauri dev     (1.2 MB model)
//   VITE_SMOKE=qwen3-0.6b npm run tauri dev
// Reports capabilities, load time and tok/s to the terminal, then exits.
import { invoke } from '@tauri-apps/api/core';
import { runTurn } from './agent/turn';
import { chat, loadModel, useEngine } from './llm/engine';
import { clearChat, useChat } from './state/chat';
import { addFiles, createKb, deleteKb, useKb } from './state/kb';

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
 * Full pipeline inside the real app: create KB → ingest via ug → load model →
 * one RAG turn through the same orchestrator the UI uses. The App is rendered
 * alongside, so UI code runs in WKWebView too.
 *   VITE_SMOKE=e2e VITE_SMOKE_FILES=/abs/a.md,/abs/b.pdf npm run tauri dev
 */
export async function runE2E(files: string[], model = 'qwen3-0.6b') {
  // The harness shares webview storage with the real app on this machine, so
  // snapshot what it touches and put it back afterwards.
  const savedChat = { messages: useChat.getState().messages, session: useChat.getState().session };
  const savedKb = { grounding: useKb.getState().grounding, selected: useKb.getState().selected };
  const restore = () => {
    useChat.setState(savedChat);
    useKb.setState(savedKb);
  };
  try {
    await new Promise((r) => setTimeout(r, 1500)); // let App mount + refresh KBs
    const stale = useKb.getState().kbs.find((k) => k.name === 'e2e-docs');
    if (stale) await deleteKb(stale.slug);
    const kb = await createKb('e2e-docs');
    if (!kb) throw new Error('createKb failed');
    await log(`kb created ${kb.slug}`);
    const t = performance.now();
    await addFiles(kb.slug, files);
    const after = useKb.getState().kbs.find((k) => k.slug === kb.slug)!;
    await log(
      `indexed in ${(performance.now() - t).toFixed(0)}ms status=${after.status} nodes=${after.nodes} edges=${after.edges} sources=${JSON.stringify(after.sources.map((s) => [s.file, s.kind, s.status]))} err=${after.lastError}`,
    );
    const logs = useKb.getState().logs[kb.slug] ?? [];
    await log(`ug log lines=${logs.length} last=${JSON.stringify(logs.slice(-2).map((l) => l.line))}`);
    useKb.setState({ grounding: kb.slug });

    await loadModel(model);
    while (useEngine.getState().status === 'loading') await new Promise((r) => setTimeout(r, 200));
    if (useEngine.getState().loadedId !== model) await loadModel(model);
    if (useEngine.getState().status !== 'ready') throw new Error(`load failed: ${useEngine.getState().error}`);
    clearChat();
    await runTurn('What HTTP headers does wllama need for multi-threading, and why?');
    const msg = useChat.getState().messages.find((m) => m.role === 'assistant');
    const info = useEngine.getState().info;
    const result = {
      caps: useEngine.getState().caps,
      engine: info,
      kb: { status: after.status, nodes: after.nodes, edges: after.edges, error: after.lastError, sources: after.sources.map((x) => ({ file: x.file, kind: x.kind, status: x.status })) },
      ugLogLines: logs.length,
      steps: msg?.steps?.map((x) => ({ kind: x.kind, status: x.status, detail: x.detail })),
      sources: msg?.sources?.map((h) => h.file),
      stats: msg?.stats,
      answer: msg?.content,
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
