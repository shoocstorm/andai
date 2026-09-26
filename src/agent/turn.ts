// One agent turn. In agent mode: plan (loop.ts: decide → tools → observe) →
// build prompt → generate. Otherwise, or with no knowledge base to use, the
// fixed pipeline: analyze → retrieve (one ug search on the question) → build
// → generate. Every step is written to the assistant message's trace, which
// drives the "Processing reasoning" chips and the Execution Trace panel.

import { kbSearch, type SearchHit } from '../kb/api';
import { budgets, buildHistory, buildSystem, CHARS_PER_TOKEN, keywords } from './prompt';
import { chat, isAbort, loadedModel, type ChatMessage } from '../llm/engine';
import {
  addMessage,
  patchMessage,
  patchStep,
  settleCall,
  uid,
  useChat,
  type Message,
  type TraceStep,
} from '../state/chat';
import { recordSearch, useKb } from '../state/kb';
import { usePersona } from '../state/persona';
import { useTools } from '../state/tools';
import { runAgent } from './loop';
import { checkRelevance, KEEP_TOP } from './relevance';
import { decidesWithLaya } from '../llm/decide';

let controller: AbortController | null = null;

export const isBusy = () => controller !== null;
export const stopTurn = () => controller?.abort();

/** `seed` fixes the agent's option shuffle (the agent eval, src/eval.ts). */
export async function runTurn(text: string, opts: { seed?: number } = {}): Promise<void> {
  const prompt = text.trim();
  if (!prompt || controller) return;
  const model = loadedModel();
  const persona = usePersona.getState();
  const kbState = useKb.getState();
  const kb = kbState.kbs.find((k) => k.slug === kbState.grounding) ?? null;
  const searchable = kb && kb.status !== 'empty' && kb.nodes > 0;

  const history = useChat.getState().messages;
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

  // The agent's tools all work on the grounding knowledge base; without one
  // there is nothing to decide, so the turn is a plain answer.
  const agent = useTools.getState().agentMode && !!searchable;
  const kw = keywords(prompt);
  const steps: TraceStep[] = agent
    ? [{ kind: 'plan', title: 'Plan & use tools', detail: `Deciding how to use “${kb.name}”…`, status: 'queued' }]
    : [
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
      ];
  // With a Laya decision model, retrieved passages are checked for relevance
  // before they go into the prompt (agent/relevance.ts).
  const relevance = !!searchable && decidesWithLaya();
  if (relevance) steps.push({ kind: 'filter', title: 'Relevance check', detail: 'Waiting for passages…', status: 'queued' });
  steps.push(
    { kind: 'build', title: 'Assemble context', detail: 'Persona + retrieved passages + history', status: 'queued' },
    { kind: 'generate', title: `Generate · ${model.name}`, detail: 'Waiting for first token…', status: 'queued' },
  );
  const id = uid();
  const msg: Message = {
    id,
    role: 'assistant',
    content: '',
    createdAt: Date.now(),
    steps,
    sources: [],
    ...(agent ? { agent: [] } : {}),
    kbName: kb?.name ?? null,
    streaming: true,
  };
  addMessage(msg);
  controller = new AbortController();
  const signal = controller.signal;
  const started = performance.now();

  try {
    let hits: SearchHit[] = [];
    let searched = true;
    let clarify = false;
    if (agent) {
      patchStep(id, 'plan', { status: 'running' });
      const res = await runAgent({ msgId: id, prompt, history, kb, k: kbState.k, maxChars: kbState.maxChars, signal, seed: opts.seed });
      hits = res.hits;
      searched = res.calls > 0;
      clarify = res.clarify;
      patchMessage(id, { sources: hits });
      patchStep(id, 'plan', {
        status: 'done',
        detail: res.calls
          ? `${res.calls} tool call${res.calls === 1 ? '' : 's'} · ${hits.length} passage${hits.length === 1 ? '' : 's'} from “${kb.name}”`
          : clarify
            ? 'The request needs clarifying'
            : 'No tools needed',
      });
    } else {
      patchStep(id, 'analyze', { status: 'running' });
      await new Promise((r) => setTimeout(r, 120)); // let the chip render before the search blocks
      patchStep(id, 'analyze', { status: 'done' });
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
    }
    if (signal.aborted) throw new DOMException('aborted', 'AbortError');

    if (relevance) {
      if (hits.length <= KEEP_TOP) {
        patchStep(id, 'filter', { status: 'skipped', detail: hits.length ? `${hits.length} passage${hits.length === 1 ? '' : 's'}, all kept (the top ${KEEP_TOP} always are)` : 'No passages to check' });
      } else {
        patchStep(id, 'filter', { status: 'running', detail: `Scoring ${hits.length} passages…` });
        try {
          const r = await checkRelevance(prompt, hits);
          if (r) {
            hits = r.hits;
            // Sources are numbered as the prompt cites them, so they're set after the check.
            patchMessage(id, { sources: hits, relevance: r.record });
            const dropped = r.record.items.length - hits.length;
            patchStep(id, 'filter', {
              status: 'done',
              detail: dropped
                ? `Kept ${hits.length} of ${r.record.items.length} passages · ~${r.record.tokensSaved.toLocaleString()} tokens less to read · ${r.record.ms} ms`
                : `All ${hits.length} passages look relevant · ${r.record.ms} ms`,
            });
          }
        } catch (e) {
          if (signal.aborted) throw e;
          // A failed check never costs the answer its context.
          patchStep(id, 'filter', { status: 'error', detail: `Check failed (${e instanceof Error ? e.message : e}); kept every passage` });
        }
      }
    }
    if (signal.aborted) throw new DOMException('aborted', 'AbortError');

    patchStep(id, 'build', { status: 'running' });
    const nCtx = model.n_ctx;
    const budget = budgets(nCtx, persona.maxTokens);
    const system = buildSystem(persona, hits, budget.context, kb?.name ?? null, { searched, clarify });
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
    let firstTokenMs: number | null = null;
    for await (const ev of chat(messages, {
      temperature: persona.temperature,
      maxTokens: persona.maxTokens,
      thinking: persona.verbose,
      signal,
    })) {
      if (ev.type === 'delta') {
        reply += ev.text;
        const now = performance.now();
        firstTokenMs ??= now - started;
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
            tokPerSec: ev.tokPerSec ?? 0,
            promptTokPerSec: ev.promptTokPerSec,
            promptTokens: ev.promptTokens,
            nCtx,
            totalMs: secs * 1000,
            firstTokenMs,
            model: model.name,
          },
        });
        patchStep(id, 'generate', {
          status: 'done',
          detail: `${ev.completionTokens} tokens · ${(ev.tokPerSec ?? 0).toFixed(1)} tok/s${ev.promptTokPerSec ? ` · prompt ${Math.round(ev.promptTokPerSec)} tok/s` : ''}`,
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
        agent: m.agent?.map(settleCall),
      }));
    } else {
      const message = e instanceof Error ? e.message : String(e);
      patchMessage(id, (m) => ({
        streaming: false,
        steps: m.steps?.map((s) => (s.status === 'running' ? { ...s, status: 'error', detail: message } : s)),
        agent: m.agent?.map(settleCall),
      }));
      addMessage({ id: uid(), role: 'error', content: message, createdAt: Date.now() });
    }
  } finally {
    controller = null;
  }
}
