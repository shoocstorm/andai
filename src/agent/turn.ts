// One agent turn: analyze → retrieve (ug GraphRAG) → build prompt → generate.
// Every step is written to the assistant message's trace, which drives the
// "Processing reasoning" chips and the Execution Trace panel.

import { kbSearch, type SearchHit } from '../kb/api';
import { budgets, buildHistory, buildSystem, CHARS_PER_TOKEN, keywords } from './prompt';
import { chat, isAbort, loadedModel, useEngine, type ChatMessage } from '../llm/engine';
import {
  addMessage,
  patchMessage,
  patchStep,
  uid,
  useChat,
  type Message,
  type TraceStep,
} from '../state/chat';
import { recordSearch, useKb } from '../state/kb';
import { usePersona } from '../state/persona';

let controller: AbortController | null = null;

export const isBusy = () => controller !== null;
export const stopTurn = () => controller?.abort();

export async function runTurn(text: string): Promise<void> {
  const prompt = text.trim();
  if (!prompt || controller) return;
  const model = loadedModel();
  const persona = usePersona.getState();
  const kbState = useKb.getState();
  const kb = kbState.kbs.find((k) => k.slug === kbState.grounding) ?? null;
  const searchable = kb && kb.status !== 'empty' && kb.nodes > 0;

  addMessage({ id: uid(), role: 'user', content: prompt, createdAt: Date.now() });

  if (!model) {
    addMessage({
      id: uid(),
      role: 'error',
      content: 'No model is loaded. Open **Settings → Models** and load one — Qwen3 0.6B is a good start.',
      createdAt: Date.now(),
    });
    return;
  }

  const kw = keywords(prompt);
  const steps: TraceStep[] = [
    {
      kind: 'analyze',
      title: 'Analyze query',
      detail: kw.length ? `Identifying ${kw.map((k) => `‘${k}’`).join(' and ')} parameters…` : 'Parsing intent…',
      status: 'queued',
    },
    {
      kind: 'retrieve',
      title: 'Knowledge retrieval',
      detail: searchable ? `Searching knowledge base “${kb.name}”…` : kb ? `“${kb.name}” is not indexed yet` : 'No knowledge base selected',
      status: searchable ? 'queued' : 'skipped',
    },
    { kind: 'build', title: 'Assemble context', detail: 'Persona + retrieved passages + history', status: 'queued' },
    { kind: 'generate', title: `Generate · ${model.name}`, detail: 'Waiting for first token…', status: 'queued' },
  ];
  const id = uid();
  const msg: Message = {
    id,
    role: 'assistant',
    content: '',
    createdAt: Date.now(),
    steps,
    sources: [],
    kbName: kb?.name ?? null,
    streaming: true,
  };
  addMessage(msg);
  controller = new AbortController();
  const signal = controller.signal;
  const started = performance.now();

  try {
    patchStep(id, 'analyze', { status: 'running' });
    await new Promise((r) => setTimeout(r, 120)); // let the chip render before the search blocks
    patchStep(id, 'analyze', { status: 'done' });

    let hits: SearchHit[] = [];
    if (searchable) {
      patchStep(id, 'retrieve', { status: 'running' });
      const t = performance.now();
      try {
        hits = await kbSearch(kb.slug, prompt, kbState.k, kbState.maxChars);
        const ms = Math.round(performance.now() - t);
        recordSearch(ms, hits.length);
        patchMessage(id, { sources: hits });
        patchStep(id, 'retrieve', {
          status: 'done',
          detail: hits.length
            ? `Retrieved ${hits.length} passage${hits.length === 1 ? '' : 's'} from “${kb.name}” · ${ms} ms`
            : `No matching passages in “${kb.name}”`,
        });
      } catch (e) {
        patchStep(id, 'retrieve', { status: 'error', detail: `Search failed: ${e instanceof Error ? e.message : e}` });
      }
    }
    if (signal.aborted) throw new DOMException('aborted', 'AbortError');

    patchStep(id, 'build', { status: 'running' });
    const nCtx = model.n_ctx;
    const budget = budgets(nCtx, persona.maxTokens);
    const system = buildSystem(persona, hits, budget.context, kb?.name ?? null);
    const messages: ChatMessage[] = [
      { role: 'system', content: system },
      ...buildHistory(useChat.getState().messages, budget.history, id),
    ];
    patchStep(id, 'build', {
      status: 'done',
      detail: `${messages.length - 1} message${messages.length === 2 ? '' : 's'} · ~${Math.round(
        messages.reduce((n, m) => n + m.content.length, 0) / CHARS_PER_TOKEN,
      ).toLocaleString()} tokens`,
    });

    patchStep(id, 'generate', { status: 'running' });
    let reply = '';
    let lastPaint = 0;
    for await (const ev of chat(messages, {
      temperature: persona.temperature,
      maxTokens: persona.maxTokens,
      thinking: persona.verbose,
      signal,
    })) {
      if (ev.type === 'delta') {
        reply += ev.text;
        const now = performance.now();
        if (now - lastPaint > 33) {
          lastPaint = now;
          patchMessage(id, { content: reply });
          patchStep(id, 'generate', { detail: `${ev.tokens} tokens · ${ev.tokPerSec.toFixed(1)} tok/s` });
        }
      } else {
        const secs = (performance.now() - started) / 1000;
        patchMessage(id, {
          content: reply,
          stats: {
            tokens: ev.completionTokens,
            tokPerSec: useEngine.getState().tokPerSec ?? 0,
            promptTokens: ev.promptTokens,
            nCtx,
            totalMs: secs * 1000,
            model: model.name,
          },
        });
        patchStep(id, 'generate', {
          status: 'done',
          detail: `${ev.completionTokens} tokens · ${(useEngine.getState().tokPerSec ?? 0).toFixed(1)} tok/s`,
        });
      }
    }
    patchMessage(id, { streaming: false });
  } catch (e) {
    if (isAbort(e) || signal.aborted) {
      patchMessage(id, (m) => ({
        streaming: false,
        stopped: true,
        steps: m.steps?.map((s) => (s.status === 'running' || s.status === 'queued' ? { ...s, status: 'skipped', detail: 'Stopped by operator' } : s)),
      }));
    } else {
      const message = e instanceof Error ? e.message : String(e);
      patchMessage(id, (m) => ({
        streaming: false,
        steps: m.steps?.map((s) => (s.status === 'running' ? { ...s, status: 'error', detail: message } : s)),
      }));
      addMessage({ id: uid(), role: 'error', content: message, createdAt: Date.now() });
    }
  } finally {
    controller = null;
  }
}
