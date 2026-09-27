// Agent mode: decide → fill arguments → gate → run → observe, until the
// decision is to answer (or ask), or the step budget runs out.
//
// Every iteration is written to `Message.agent` as it happens, which drives
// the chips and the Execution Trace: the decision with every option's
// probability, the arguments, the approval, the ug command line, the output
// and timings. Nothing a tool returns is trusted: it reaches the model only
// as fenced passages (prompt.ts) and a clipped summary line.

import { kbTool, type KbInfo, type SearchHit } from '../kb/api';
import { decide, decidesWithLaya, DecisionError, layaChoices, seededShuffle, type ArgChoice, type ArgPick, type Decision, type DecisionOption } from '../llm/decide';
import {
  addAgentStep,
  OUTPUT_KEEP,
  patchCall,
  patchStep,
  uid,
  type AgentStep,
  type Message,
  type DecisionRecord,
  type ToolCallRecord,
} from '../state/chat';
import { recordSearch } from '../state/kb';
import { recordToolRun, requestApproval, useTools } from '../state/tools';
import { addEvidence, mergeEvidence, type Found } from './evidence';
import { agentState, needsLookup, type Observation } from './prompt';
import { fillArgs, parseObject } from './tools/argfill';
import { available, policyOf } from './tools/registry';
import { needsSymbol, SYMBOL_TYPES } from './tools/ug';
import type { ToolDef } from './tools/types';

export const ANSWER = 'answer_now';
export const CLARIFY = 'ask_clarification';

// The wording is measured (Qwen3 0.6B, 5 requests × 3 option orders): "needs no
// lookup (… general knowledge …)" drew *answer* for 9/9 knowledge questions;
// this narrower text scored 15/15 with answer first, 12/15 last, 11/15 shuffled.
const PSEUDO: Record<string, string> = {
  [ANSWER]:
    'Answer without using the knowledge base: only for greetings, thanks or small talk, or when the tool results above already answer the request',
  // Once a tool has returned something (measured with the rule in `offered`: 9/9 answered).
  answerWithResults: 'Answer now from the tool results above: they already cover the request',
  [CLARIFY]: 'Ask the user a clarifying question: the request is too ambiguous to act on',
};

export const QUESTION = 'What should the assistant do next to fulfil the user’s request?';

/**
 * Laya picked a tool over *answer* after nearly every search that found
 * passages (agent eval, 2026-09-26: 4.6 decisions per question), so with
 * Laya, once there are results, stopping is its own yes/no question, asked in
 * the same pass as the tool choice, and *answer* leaves the choice.
 */
export const STOP = 'The tool results above already contain the information needed to answer the user’s request.';
/** Answer when Laya says the results suffice at least this likely. */
export const STOP_AT = 0.5;

/** A tool that failed or found nothing may be retried (rephrased) up to this many calls per turn. */
const MAX_CALLS_PER_TOOL = 2;

/**
 * Tools still worth offering this turn. A small model can't judge whether
 * results suffice: with kb_search still offered after a search that found
 * passages, Qwen3 0.6B searched again 9/9 times (64–100%); without it, it
 * answered 9/9. So a tool that returned something isn't offered again, and
 * one that failed or came back empty gets one more try.
 */
export function offered(tools: ToolDef[], usage: Map<string, { calls: number; found: number }>, denied: Set<string>): ToolDef[] {
  return tools.filter((t) => {
    const u = usage.get(t.id);
    return !denied.has(t.id) && !(u && (u.found > 0 || u.calls >= MAX_CALLS_PER_TOOL));
  });
}

/** Lines of context Read lines gets on each side of a passage. */
const RANGE_PAD = 20;

/** `file:start-end` around each passage that has lines, padded, first seen first. */
export const rangesIn = (hits: SearchHit[]) => [
  ...new Set(
    hits
      .filter((h) => h.start_line > 0 && h.end_line >= h.start_line && h.file && !h.file.startsWith('('))
      .map((h) => `${h.file}:${Math.max(1, h.start_line - RANGE_PAD)}-${h.end_line + RANGE_PAD}`),
  ),
];

/** Names of the code symbols among the hits, first seen first. */
export const symbolsIn = (hits: SearchHit[]) => [...new Set(hits.filter((h) => SYMBOL_TYPES.has(h.node_type) && h.name).map((h) => h.name))];

/** JS-side cap on one tool run; Rust stops ug at 20 s (src-tauri/src/tools.rs). */
const CALL_TIMEOUT_MS = 30_000;
const MAX_CONSECUTIVE_ERRORS = 2;

export type AgentResult = { hits: SearchHit[]; clarify: boolean; calls: number };

type LoopInput = {
  msgId: string;
  prompt: string;
  /** The conversation before this turn. */
  history: Message[];
  kb: KbInfo;
  k: number;
  maxChars: number;
  signal: AbortSignal;
  /** Seed for the option shuffle; random when unset. The agent eval fixes it so runs are comparable. */
  seed?: number;
};

const aborted = () => new DOMException('aborted', 'AbortError');

/** Resolves with `p`, or rejects on timeout or abort. The ug process itself is bounded by Rust. */
function bounded<T>(p: Promise<T>, ms: number, signal: AbortSignal): Promise<T> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) return reject(aborted());
    const timer = setTimeout(() => reject(new Error(`No result after ${ms / 1000} s.`)), ms);
    const onAbort = () => reject(aborted());
    signal.addEventListener('abort', onAbort, { once: true });
    p.then(resolve, reject).finally(() => {
      clearTimeout(timer);
      signal.removeEventListener('abort', onAbort);
    });
  });
}

/** answer_now is always A and ask_clarification last; the tools between them are shuffled by `seed`. */
export function decisionOptions(tools: ToolDef[], seed: number, haveResults = false): DecisionOption[] {
  return [
    { id: ANSWER, text: haveResults ? PSEUDO.answerWithResults : PSEUDO[ANSWER] },
    ...seededShuffle(
      tools.map((t) => ({ id: t.id, text: t.option })),
      seed,
    ),
    { id: CLARIFY, text: PSEUDO[CLARIFY] },
  ];
}

const record = (d: Decision, seed: number): DecisionRecord => ({
  question: QUESTION,
  options: d.options.map(({ id, label, text, probability }) => ({ id, label, text, probability })),
  chosen: d.chosen,
  confidence: d.confidence,
  bounded: d.bounded,
  model: d.model,
  slot: d.slot,
  ms: Math.round(d.ms),
  seed,
  promptTokens: d.promptTokens,
  io: d.io,
  ...(d.stop ? { stop: d.stop } : {}),
  ...(d.truncated ? { truncated: true } : {}),
});

/** The enum arguments of `tools` that Laya picks as typed choices (`ToolDef.choices`), as Laya questions. */
export const argChoices = (tools: ToolDef[]): ArgChoice[] =>
  tools.flatMap((t) => (t.choices ?? []).map((c) => ({ id: `${t.id}_${c.arg}`, tool: t.id, arg: c.arg, question: c.question, options: c.options })));

const argKey = (tool: string, args: Record<string, unknown>) =>
  `${tool} ${JSON.stringify(Object.keys(args).sort().map((k) => [k, args[k]]))}`;

/** Whether the model's reply parses to exactly `args`, i.e. nothing was trimmed or repaired. */
function sameJson(raw: string, args: Record<string, unknown>) {
  try {
    return JSON.stringify(parseObject(raw)) === JSON.stringify(args);
  } catch {
    return false;
  }
}

const errText = (e: unknown) => (e instanceof Error ? e.message : String(e));

export async function runAgent(input: LoopInput): Promise<AgentResult> {
  const { msgId, prompt, kb, signal } = input;
  const settings = useTools.getState();
  const kind = kb.kind;
  const history = input.history;
  const seed = input.seed ?? Math.floor(Math.random() * 2 ** 31);
  const observations: Observation[] = [];
  // Every passage this turn, with the call that found it; merged for the answer (evidence.ts).
  const found: Found[] = [];
  const hits = () => found.map((f) => f.hit);
  const done = new Map<string, number>(); // argKey → step index
  const denied = new Set<string>();
  const usage = new Map<string, { calls: number; found: number }>();
  const used = (id: string, found: number) => {
    const u = usage.get(id) ?? { calls: 0, found: 0 };
    usage.set(id, { calls: u.calls + 1, found: u.found + found });
  };
  let consecutiveErrors = 0;
  let calls = 0;
  let index = 0;

  const state = () =>
    agentState({
      prompt,
      history,
      kb: { name: kb.name, kind, nodes: kb.nodes, files: kb.sources.length },
      observations,
      step: calls,
      maxSteps: settings.maxSteps,
    });
  // What a `file` argument may name: indexed sources, then files the tools have shown.
  const knownFiles = () => [
    ...new Set([
      ...kb.sources.filter((src) => src.status === 'indexed').map((src) => src.file),
      ...hits().map((h) => h.file).filter((f) => f && !f.startsWith('(')),
    ]),
  ];
  const progress = (detail: string) => patchStep(msgId, 'plan', { detail });
  const step = (s: Omit<AgentStep, 'id' | 'index' | 'at'>): AgentStep => {
    const full = { id: uid(), index: index++, at: Date.now(), ...s };
    addAgentStep(msgId, full);
    return full;
  };

  while (true) {
    if (signal.aborted) throw aborted();
    const tools = offered(available(kind, settings.policies), usage, denied);
    if (calls >= settings.maxSteps) {
      step({ decision: null, action: ANSWER, note: `Reached the limit of ${settings.maxSteps} tool calls for one turn.` });
      break;
    }
    if (!tools.length) {
      step({ decision: null, action: ANSWER, note: 'No tools left to offer: switched off, denied, or already used this turn.' });
      break;
    }

    // ── decide ──
    progress(`Step ${index + 1} · choosing the next action…`);
    let decision: DecisionRecord | null = null;
    let action: string;
    let note: string | undefined;
    let fallback: AgentStep['fallback'];
    let failedDecision: AgentStep['failedDecision'];
    let picks: ArgPick[] = [];
    const searchTool = tools.find((t) => t.id === 'kb_search');
    // What to do when the decision can't be trusted: search once if nothing
    // was looked up yet (the fixed pipeline's behavior), else answer.
    const fallbackAction = () => (calls === 0 && searchTool ? searchTool.id : ANSWER);
    // With a knowledge base selected, the first decision picked search on 23 of
    // 24 lookup questions, and the miss was a follow-up that needed one, so
    // it's skipped (about 0.7 s) when the request plainly asks about content.
    // Anything else, small talk or a question to the assistant, still goes to
    // the model (docs/agentic-rag-improvements.md, item 7).
    if (settings.searchFirst && index === 0 && searchTool && needsLookup(prompt)) {
      action = searchTool.id;
      note = 'Searched first, without a decision: with a knowledge base selected, a question almost always needs a search.';
    } else {
      try {
        const options = decisionOptions(tools, seed + index, found.length > 0);
        const withLaya = decidesWithLaya();
        const stopCheck = found.length > 0 && withLaya;
        const d = await decide(state(), QUESTION, stopCheck ? options.filter((o) => o.id !== ANSWER) : options, signal, {
          ...(stopCheck ? { stop: STOP } : {}),
          // Enum arguments ride in the same pass, a few ms, in case their tool is chosen.
          ...(withLaya ? { choices: argChoices(tools) } : {}),
        });
        decision = record(d, seed + index);
        picks = d.picks ?? [];
        action = d.chosen;
        if (d.stop && d.stop.probability >= STOP_AT) {
          action = ANSWER;
          note = `The tool results cover the request (${Math.round(d.stop.probability * 100)}% likely), so answering.`;
        }
        if (action !== ANSWER && action !== CLARIFY && !tools.some((t) => t.id === action)) {
          throw new Error(`chose “${action}”, which wasn't offered`);
        }
        if (action !== ANSWER && action !== CLARIFY && d.confidence < settings.minConfidence) {
          const next = fallbackAction();
          fallback = 'low-confidence';
          note = `Low confidence (${Math.round(d.confidence * 100)}% < ${Math.round(settings.minConfidence * 100)}%) in “${action}”, so ${next === ANSWER ? 'answering' : 'searching the knowledge base'} instead.`;
          action = next;
        }
      } catch (e) {
        if (signal.aborted) throw aborted();
        action = fallbackAction();
        fallback = 'decision-failed';
        failedDecision = e instanceof DecisionError ? { error: e.message, ms: Math.round(e.ms), io: e.io } : { error: errText(e), ms: 0 };
        note = `Decision failed (${errText(e)}), so ${action === ANSWER ? 'answering' : 'searching the knowledge base'} instead.`;
      }
    }

    // A symbol tool needs a name ug knows, and none has been seen yet: look
    // symbols up first. Not offering these tools until then was measured
    // instead, and it cost more than it saved: with the shorter option list,
    // Qwen3 0.6B answered without searching on 12 of 16 code and mixed
    // questions (docs/agentic-rag-improvements.md, item 2).
    // The same for Read lines, which reads around a passage already found.
    const lookup = tools.find((t) => t.id === 'kb_find_symbols') ?? searchTool;
    const chosenTool = tools.find((t) => t.id === action);
    if (chosenTool && needsSymbol(chosenTool) && !symbolsIn(hits()).length && lookup) {
      fallback = 'needs-symbol';
      note = `${chosenTool.title} needs a symbol name, and none has been seen yet, so ${lookup.id === 'kb_find_symbols' ? 'looking symbols up' : 'searching'} first.`;
      action = lookup.id;
    } else if (chosenTool?.id === 'kb_read_lines' && !rangesIn(hits()).length && searchTool) {
      fallback = 'needs-range';
      note = `${chosenTool.title} needs a line range from an earlier result, and none has turned up yet, so searching first.`;
      action = searchTool.id;
    }

    if (action === ANSWER || action === CLARIFY) {
      step({ decision, action, note, fallback, failedDecision });
      progress(action === CLARIFY ? 'Asking a clarifying question' : `Answering · ${calls} tool call${calls === 1 ? '' : 's'}`);
      return { hits: mergeEvidence(found), clarify: action === CLARIFY, calls };
    }

    const tool = tools.find((t) => t.id === action)!;
    const policy = policyOf(tool, settings.policies);
    // Laya picks the tool's enum arguments: from the decision's pass, or, when
    // the tool came without one (search first, a fallback), in a pass of their own.
    let mine = picks.filter((p) => p.tool === tool.id);
    let pickModel = decision?.model ?? '';
    const asked = argChoices([tool]);
    if (asked.length && mine.length < asked.length && decidesWithLaya()) {
      try {
        const r = await layaChoices(state(), asked);
        mine = r.picks;
        pickModel = decision?.model ?? r.model;
      } catch {
        // The chat model writes them instead, as without Laya.
        mine = [];
      }
      if (signal.aborted) throw aborted();
    }
    const fixed = Object.fromEntries(mine.map((p) => [p.arg, p.value]));
    const call: ToolCallRecord = {
      tool: tool.id,
      title: tool.title,
      args: null,
      argsRaw: null,
      argModel: null,
      argAttempts: 0,
      policy,
      startedAt: Date.now(),
      status: 'filling',
      ...(mine.length ? { argChoices: mine.map((p) => ({ arg: p.arg, value: p.value, probability: p.probability, model: pickModel })) } : {}),
    };
    const s = step({ decision, action, note, fallback, failedDecision, call });
    const patch = (p: Partial<ToolCallRecord>) => patchCall(msgId, s.id, p);
    calls++;

    // ── arguments ──
    progress(`Step ${s.index + 1} · ${tool.title}: writing arguments…`);
    let args: Record<string, unknown>;
    try {
      const fill = await fillArgs(tool, { state: state(), kind, known: { files: knownFiles(), symbols: symbolsIn(hits()), ranges: rangesIn(hits()) }, fixed, signal });
      if (fill.ok) {
        args = fill.args;
        patch({ args, argsRaw: sameJson(fill.raw, args) ? null : fill.raw, argModel: fill.model, argAttempts: fill.attempts });
      } else if (tool.id === 'kb_search') {
        // The fixed pipeline's query: the question itself.
        args = { query: prompt.slice(0, 300), scope: fixed.scope ?? 'broad' };
        patch({
          args,
          argsRaw: fill.raw,
          argModel: fill.model,
          argAttempts: fill.attempts,
          error: `Arguments were invalid (${fill.errors.join('; ')}); searched with the question as written.`,
        });
      } else {
        throw new Error(`Could not produce valid arguments: ${fill.errors.join('; ')}`);
      }
    } catch (e) {
      if (signal.aborted) throw aborted();
      const error = errText(e);
      patch({ status: 'error', error, endedAt: Date.now(), ms: Date.now() - call.startedAt });
      observations.push({ tool: tool.id, args: null, summary: `Failed: ${error}` });
      if (++consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
        step({ decision: null, action: ANSWER, note: 'Stopped after repeated tool errors.' });
        break;
      }
      continue;
    }

    // ── repeat guard ──
    const key = argKey(tool.id, args);
    if (done.has(key)) {
      const summary = `Skipped: the same call already ran at step ${done.get(key)! + 1}; its result is above.`;
      patch({ status: 'skipped', error: summary, observation: summary, endedAt: Date.now(), ms: 0 });
      observations.push({ tool: tool.id, args, summary });
      // A repeat is a wasted step: it counts toward the tool's retries and the
      // error streak, or a model that keeps picking it loops until the step limit.
      used(tool.id, 0);
      if (++consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
        step({ decision: null, action: ANSWER, note: 'Stopped: the same calls kept repeating.' });
        break;
      }
      continue;
    }
    done.set(key, s.index);

    // ── gate ──
    if (policy === 'ask') {
      patch({ status: 'awaiting', approval: 'pending' });
      progress(`Step ${s.index + 1} · waiting for your approval to run ${tool.title}`);
      const ok = await requestApproval(s.id, signal);
      if (!ok) {
        const summary = 'The user declined this call; do not ask for it again.';
        patch({ status: 'denied', approval: 'denied', observation: summary, endedAt: Date.now() });
        observations.push({ tool: tool.id, args, summary });
        denied.add(tool.id);
        continue;
      }
      patch({ approval: 'approved' });
    }

    // ── run ──
    patch({ status: 'running', startedAt: Date.now() });
    progress(`Step ${s.index + 1} · running ${tool.title}…`);
    const started = Date.now();
    try {
      const out = await bounded(
        kbTool(kb.slug, tool.toCall(args, { kind, k: input.k, maxChars: input.maxChars })),
        CALL_TIMEOUT_MS,
        signal,
      );
      const ev = tool.observe(out.output);
      const added = addEvidence(found, tool.id, ev.hits);
      used(tool.id, ev.hits.length);
      if (tool.id === 'kb_search') recordSearch(out.ms, ev.hits.length);
      const text = typeof out.output === 'string' ? out.output : JSON.stringify(out.output, null, 2);
      patch({
        status: 'done',
        argv: out.argv,
        output: text.length > OUTPUT_KEEP ? `${text.slice(0, OUTPUT_KEEP)}\n… (${text.length - OUTPUT_KEEP} more characters not kept)` : text,
        outputBytes: out.bytes,
        truncated: out.truncated,
        observation: ev.summary,
        hits: added,
        endedAt: Date.now(),
        ms: out.ms,
      });
      recordToolRun(tool.id, Date.now() - started, true);
      observations.push({ tool: tool.id, args, summary: ev.summary });
      consecutiveErrors = 0;
    } catch (e) {
      if (signal.aborted) throw aborted();
      const error = errText(e);
      patch({ status: 'error', error, observation: `Failed: ${error}`, endedAt: Date.now(), ms: Date.now() - started });
      recordToolRun(tool.id, Date.now() - started, false);
      used(tool.id, 0);
      observations.push({ tool: tool.id, args, summary: `Failed: ${error}` });
      if (++consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
        step({ decision: null, action: ANSWER, note: 'Stopped after repeated tool errors.' });
        break;
      }
    }
  }
  progress(`Answering · ${calls} tool call${calls === 1 ? '' : 's'}`);
  return { hits: mergeEvidence(found), clarify: false, calls };
}
