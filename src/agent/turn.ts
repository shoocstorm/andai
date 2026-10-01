// One agent turn. In agent mode: plan (loop.ts: decide → tools → observe) →
// build prompt → generate. Otherwise, or with no knowledge base to use, the
// fixed pipeline: analyze → retrieve (one ug search on the question) → build
// → generate. Every step is written to the assistant message's trace, which
// drives the "Processing reasoning" chips and the Execution Trace panel.

import { kbSearch, type SearchHit } from '../kb/api';
import { budgets, buildHistory, buildSystem, CHARS_PER_TOKEN, keywords, planPassages, type ContextRecord } from './prompt';
import { chat, deciderLaya, isAbort, loadedModel, type ChatMessage } from '../llm/engine';
import {
  addMessage,
  patchMessage,
  patchStep,
  settleCall,
  splitThink,
  uid,
  useChat,
  type Message,
  type TraceStep,
} from '../state/chat';
import { recordSearch, useKb } from '../state/kb';
import { usePersona } from '../state/persona';
import { useTools } from '../state/tools';
import { runAgent } from './loop';
import { checkRelevance, KEEP_TOP, scoringRequest, type PassageScorer } from './relevance';
import { checkClaims, supportSummary } from './claims';
import { decidesWithLaya } from '../llm/decide';
import { flushActivity, logActivity } from '../state/activity';

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
  const searchable = kb && kb.status !== 'empty' && kb.status !== 'offline' && kb.nodes > 0;

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
          detail: searchable
            ? `Searching knowledge base “${kb.name}”…`
            : kb?.status === 'offline'
              ? `UltraGraph (ug) isn't installed, so “${kb.name}” can't be searched; answering without it`
              : kb
                ? `“${kb.name}” is not indexed yet`
                : 'No knowledge base selected',
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
  // And, after the answer, its cited sentences are checked against their passages (agent/claims.ts).
  if (relevance) steps.push({ kind: 'verify', title: 'Claim check', detail: 'Waiting for the answer…', status: 'queued' });
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
  // The activity log (state/activity.ts), when it's on: the turn, then each step as it happens.
  const log = (kind: Parameters<typeof logActivity>[1], data: unknown) => logActivity(id, kind, data);
  const tools = useTools.getState();
  log('turn', {
    question: prompt,
    kb: kb ? { name: kb.name, kind: kb.kind, nodes: kb.nodes } : null,
    mode: agent ? 'agent' : 'fixed',
    model: model.name,
    decider: deciderLaya()?.name ?? null,
    seed: opts.seed ?? null,
    settings: { maxSteps: tools.maxSteps, minConfidence: tools.minConfidence, searchFirst: tools.searchFirst, plan: tools.plan, searchAgain: tools.searchAgain },
    history: history.length,
  });
  controller = new AbortController();
  const signal = controller.signal;
  const started = performance.now();
  let outcome: 'answered' | 'stopped' | 'failed' = 'answered';

  try {
    let hits: SearchHit[] = [];
    let searched = true;
    let clarify = false;
    // The agent loop's passage scores, reused by the relevance check.
    let scorer: PassageScorer | null = null;
    if (agent) {
      patchStep(id, 'plan', { status: 'running' });
      const res = await runAgent({ msgId: id, prompt, history, kb, k: kbState.k, maxChars: kbState.maxChars, signal, seed: opts.seed });
      hits = res.hits;
      searched = res.calls > 0;
      clarify = res.clarify;
      scorer = res.scorer;
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
          log('retrieve', { kb: kb.name, query: prompt, k: kbState.k, maxChars: kbState.maxChars, hits: hits.length, ms, sources: hits.map((h) => `${h.file}:${h.start_line}-${h.end_line}`) });
          patchMessage(id, { sources: hits });
          patchStep(id, 'retrieve', {
            status: 'done',
            detail: hits.length
              ? `Retrieved ${hits.length} passage${hits.length === 1 ? '' : 's'} from “${kb.name}” · ${ms} ms`
              : `No matching passages in “${kb.name}”`,
          });
        } catch (e) {
          log('retrieve', { kb: kb.name, query: prompt, error: e instanceof Error ? e.message : String(e) });
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
          const r = await checkRelevance(scoringRequest(prompt, history), hits, scorer, deciderLaya()?.input.tokens);
          if (r) {
            hits = r.hits;
            // Sources are numbered as the prompt cites them, so they're set after the check.
            patchMessage(id, { sources: hits, relevance: r.record });
            log('relevance', r.record);
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
    const sentHistory = buildHistory(useChat.getState().messages, budget.history, id);
    const messages: ChatMessage[] = [{ role: 'system', content: system }, ...sentHistory];
    const chars = messages.reduce((n, m) => n + m.content.length, 0);
    // What the chat model is sent and why, for the trace's "Assemble context" dialog.
    const earlier = useChat.getState().messages.filter((m) => m.id !== id && m.role !== 'error' && (m.role === 'assistant' ? splitThink(m.content).answer : m.content).trim());
    const plan = searched && !clarify && kb ? planPassages(hits, budget.context) : [];
    const context: ContextRecord = {
      nCtx,
      replyTokens: persona.maxTokens,
      budget,
      passages: plan.map((p, i) => ({ ...p, source: `${hits[i].file}:${hits[i].start_line}-${hits[i].end_line}` })),
      // The question itself is the last message; it isn't "earlier".
      history: { sent: Math.max(0, sentHistory.length - 1), of: Math.max(0, earlier.length - 1), chars: sentHistory.slice(0, -1).reduce((n, m) => n + m.content.length, 0) },
      system,
      messages: messages.map((m) => ({ role: m.role, chars: m.content.length })),
      tokens: Math.round(chars / CHARS_PER_TOKEN),
    };
    patchMessage(id, { context });
    log('context', { ...context, history: sentHistory.slice(0, -1) });
    patchStep(id, 'build', {
      status: 'done',
      detail: `${messages.length - 1} message${messages.length === 2 ? '' : 's'} · ~${context.tokens.toLocaleString()} tokens`,
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
    log('answer', { text: reply, stats: useChat.getState().messages.find((m) => m.id === id)?.stats ?? null, sources: hits.map((h) => `${h.file}:${h.start_line}-${h.end_line}`) });

    if (relevance) {
      const sources = useChat.getState().messages.find((m) => m.id === id)?.sources ?? [];
      patchStep(id, 'verify', { status: 'running', detail: 'Checking cited sentences against their passages…' });
      try {
        const answer = splitThink(reply).answer;
        const r = await checkClaims(answer, sources);
        if (!r) {
          patchStep(id, 'verify', {
            status: 'skipped',
            detail: /\[\d+(?:\s*,\s*\d+)*\]/.test(answer)
              ? `The answer’s citations name no listed source, or no sentence to check against it (${sources.length} source${sources.length === 1 ? '' : 's'})`
              : 'The answer cites no passages',
          });
        } else {
          patchMessage(id, { support: r });
          log('claims', r);
          patchStep(id, 'verify', { status: 'done', detail: `${supportSummary(r.items)} · ${r.ms} ms` });
        }
      } catch (e) {
        // A failed or stopped check never touches the answer, which is complete.
        patchStep(
          id,
          'verify',
          signal.aborted ? { status: 'skipped', detail: 'Stopped by operator' } : { status: 'error', detail: `Check failed (${e instanceof Error ? e.message : e})` },
        );
      }
    }
  } catch (e) {
    outcome = isAbort(e) || signal.aborted ? 'stopped' : 'failed';
    log('error', { stopped: outcome === 'stopped', message: e instanceof Error ? e.message : String(e) });
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
    const m = useChat.getState().messages.find((x) => x.id === id);
    log('done', { outcome, ms: Math.round(performance.now() - started), toolCalls: m?.agent?.filter((st) => st.call).length ?? 0, passages: m?.sources?.length ?? 0 });
    void flushActivity();
  }
}
