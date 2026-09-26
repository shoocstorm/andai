// Choice-based decisions, after SemIf (../SemIf, webgpu-demo/worker.js): the
// model never writes an answer. It gets a state, a question and lettered
// options, and one forward pass is read out as next-token log-probabilities of
// the option letters; a softmax over just those letters gives a probability
// per option. That is fast (one token), always yields a valid choice, and
// shows the user how sure the model was.
//
// Measured in the app (wllama 3.6.1, Qwen3 0.6B; AGENTS.md §2): the
// logprobs are the raw next-token distribution, before sampling, so every
// letter shows up in `top_logprobs` when the prompt asks for a letter. A GBNF
// grammar keeps the sampled token a letter. `post_sampling_probs` makes wllama
// return no `top_logprobs` at all, and a logit bias doesn't change the
// reported values, so neither is used. A letter missing from the top-N list
// scored at most the lowest listed logprob, so it is bounded at that value and
// flagged (`bounded`); with no letter scored at all, the decision fails.

import { complete, deciderLaya, slotFor, type Completion, type Slot } from './engine';
import { layaDecide } from './laya';

export const LETTERS = 'ABCDEFGHIJKLMNOP';
export const MAX_OPTIONS = LETTERS.length;

export type DecisionOption = { id: string; text: string };

export type Decision = {
  options: { id: string; label: string; text: string; probability: number; logprob: number }[];
  /** Argmax option id. */
  chosen: string;
  /** Probability of `chosen`. Uncalibrated (SemIf docs/CALIBRATION.md): a ranking signal, not a confidence. */
  confidence: number;
  /** Option ids that weren't in the readout and were scored at its lowest logprob (an upper bound). */
  bounded: string[];
  model: string;
  slot: Slot;
  ms: number;
  promptTokens: number | null;
  /** The call as sent and the raw readout, for the Execution Trace. */
  io: DecisionIO;
  /** Laya only: an option, the question or the state was cut to fit its input. */
  truncated?: boolean;
  /** Laya only, when asked: the probability that the stop statement holds (the results suffice). */
  stop?: { statement: string; probability: number };
};

/**
 * One decision call as the model saw it and as it answered: the messages and
 * sampling parameters sent, and the first token's top candidates. Shown in the
 * Execution Trace so an odd decision can be inspected and copied.
 */
export type DecisionIO = {
  /** `state` is the situation the model judged, apart from the prompt around it. */
  request: { state?: string; messages: { role: string; content: string }[]; params: Record<string, unknown> };
  /** Null when the call itself failed (no reply to read). */
  response: {
    sampled: string | null;
    topLogprobs: { token: string; logprob: number }[];
    /** Laya: calibrated score per option id, its input size, whether input was cut, and its own time (Rust, ms). */
    laya?: { scores: { id: string; probability: number }[]; inputTokens: number; truncated: boolean; ms: number; stop?: number };
  } | null;
};

/** A decision that was sent but couldn't be read out; carries the call for the trace. */
export class DecisionError extends Error {
  constructor(
    message: string,
    readonly io: DecisionIO,
    readonly ms: number,
  ) {
    super(message);
    this.name = 'DecisionError';
  }
}

export const labelsFor = (n: number) => LETTERS.slice(0, n).split('');

export function decisionMessages(state: string, question: string, options: DecisionOption[]) {
  const labels = labelsFor(options.length);
  const block = options.map((o, i) => `${labels[i]}. ${o.text}`).join('\n');
  return [
    { role: 'system' as const, content: 'Make the requested decision from the supplied state. Follow the output format exactly.' },
    {
      role: 'user' as const,
      content: `State:\n${state}\n\nQuestion:\n${question}\n\nAllowed options:\n${block}\n\nReply with exactly one option letter from: ${labels.join(', ')}.`,
    },
  ];
}

export function softmax(values: number[]): number[] {
  const max = Math.max(...values);
  const exps = values.map((v) => Math.exp(v - max));
  const total = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / total);
}

type TopLogprob = { token: string; logprob: number; bytes?: number[] | null };

/**
 * The log-probability of each label among the first generated token's top
 * candidates. A label that isn't listed scored at most the lowest listed
 * value, so it gets that value and is reported in `bounded`. Throws when no
 * label is listed: that's no decision, and the agent loop falls back.
 */
export function optionLogprobs(response: Pick<Completion, 'choices'>, labels: string[]): { values: number[]; bounded: number[] } {
  const entries: TopLogprob[] = response.choices?.[0]?.logprobs?.content?.[0]?.top_logprobs ?? [];
  const found = labels.map((label) => {
    const code = label.charCodeAt(0);
    const e = entries.find((t) => t.token === label || (t.bytes?.length === 1 && t.bytes[0] === code));
    return Number(e?.logprob);
  });
  if (!found.some(Number.isFinite)) {
    throw new Error(`The decision model scored none of the options (${labels.join(', ')}).`);
  }
  const floor = Math.min(...entries.map((t) => t.logprob).filter(Number.isFinite));
  const bounded = found.flatMap((v, i) => (Number.isFinite(v) ? [] : [i]));
  return { values: found.map((v) => (Number.isFinite(v) ? v : floor)), bounded };
}

/**
 * Deterministic shuffle (mulberry32). Option order moves small models'
 * choices (SemIf: reversing it flipped 10 of 36), so tool options are
 * shuffled per turn and the seed goes in the trace to make a run replayable.
 */
export function seededShuffle<T>(items: T[], seed: number): T[] {
  let a = seed >>> 0;
  const rand = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const out = [...items];
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

/** Scores `options` for `question` given `state`, on the decision model (or the chat model as fallback). */
/** Whether decisions go to a Laya checkpoint, which can also answer a yes/no `stop` question. */
export const decidesWithLaya = () => deciderLaya() != null;

/**
 * Scores `options` for `question` given `state`. With a Laya checkpoint, a
 * `stop` statement is asked in the same pass as a yes/no question and its
 * probability comes back as `Decision.stop`; other deciders ignore it.
 */
export async function decide(
  state: string,
  question: string,
  options: DecisionOption[],
  signal?: AbortSignal,
  extra: { stop?: string } = {},
): Promise<Decision> {
  if (options.length < 2 || options.length > MAX_OPTIONS) {
    throw new Error(`A decision needs 2–${MAX_OPTIONS} options, got ${options.length}.`);
  }
  const started = performance.now();
  const labels = labelsFor(options.length);
  const laya = deciderLaya();
  if (laya) return decideWithLaya(laya.name, state, question, options, labels, started, extra.stop);
  const target = slotFor('decider');
  if (!target) throw new Error('No model loaded — open Settings → Models to load one.');
  if (!target.def.decider) throw new Error(`${target.def.name} can't make decisions; load a decision model in Settings → Models.`);
  const messages = decisionMessages(state, question, options);
  const params = {
    max_tokens: 1,
    // Neutral sampling, so the logprobs are a plain readout of the logits.
    temperature: 1,
    top_k: 0,
    top_p: 1,
    logprobs: true,
    // Room for every letter even when a few other tokens rank among them.
    top_logprobs: 20,
    grammar: `root ::= ${labels.map((l) => `"${l}"`).join(' | ')}`,
    // A separate decision model only ever sees decisions, and consecutive
    // ones share most of their prompt (instructions, request, knowledge base,
    // earlier results), so its KV cache is reused. The chat model alternates
    // decisions with argument writing and answers, which would evict it.
    cache_prompt: target.slot === 'decider',
    chat_template_kwargs: { enable_thinking: false },
  };
  const io: DecisionIO = { request: { state, messages, params }, response: null };
  let result: Awaited<ReturnType<typeof complete>>;
  try {
    result = await complete(target.slot, { messages, ...params, abortSignal: signal });
  } catch (e) {
    if (signal?.aborted) throw e;
    throw new DecisionError(e instanceof Error ? e.message : String(e), io, performance.now() - started);
  }
  const { response, slot, def } = result;
  const first = response.choices?.[0]?.logprobs?.content?.[0];
  io.response = {
    sampled: first?.token ?? response.choices?.[0]?.message?.content ?? null,
    topLogprobs: (first?.top_logprobs ?? []).map((t: TopLogprob) => ({ token: t.token, logprob: t.logprob })),
  };
  let readout: ReturnType<typeof optionLogprobs>;
  try {
    readout = optionLogprobs(response, labels);
  } catch (e) {
    throw new DecisionError(e instanceof Error ? e.message : String(e), io, performance.now() - started);
  }
  const { values: logprobs, bounded } = readout;
  const probs = softmax(logprobs);
  const scored = options.map((o, i) => ({ id: o.id, label: labels[i], text: o.text, probability: probs[i], logprob: logprobs[i] }));
  const best = scored.reduce((a, b) => (b.probability > a.probability ? b : a));
  return {
    options: scored,
    chosen: best.id,
    confidence: best.probability,
    bounded: bounded.map((i) => options[i].id),
    model: def.name,
    slot,
    ms: performance.now() - started,
    promptTokens: response.usage?.prompt_tokens ?? null,
    io,
  };
}

/**
 * The same decision on a Laya checkpoint (Rust, MLX): the options go in as
 * `id: text` criteria and come back as calibrated probabilities, in ~10–20 ms.
 * A `stop` statement rides along as a yes/no question in the same batch. No
 * letters or logprobs; the trace shows the state, questions and options as
 * the input and the scores as the output.
 */
async function decideWithLaya(
  name: string,
  state: string,
  question: string,
  options: DecisionOption[],
  labels: string[],
  started: number,
  stop?: string,
): Promise<Decision> {
  const io: DecisionIO = {
    request: {
      state,
      messages: [
        { role: 'state', content: state },
        { role: 'question', content: question },
        { role: 'options', content: options.map((o) => `${o.id}: ${o.text}`).join('\n') },
        ...(stop ? [{ role: 'yes/no', content: stop }] : []),
      ],
      params: { model: name, type: stop ? 'choice + noul' : 'choice' },
    },
    response: null,
  };
  let r: Awaited<ReturnType<typeof layaDecide>>;
  try {
    r = await layaDecide(state, [
      { id: 'next', kind: 'choice', question, options: options.map(({ id, text }) => ({ id, text })) },
      ...(stop ? [{ id: 'stop', kind: 'noul' as const, question: stop }] : []),
    ]);
  } catch (e) {
    throw new DecisionError(e instanceof Error ? e.message : String(e), io, performance.now() - started);
  }
  const choice = r.answers.find((a) => a.id === 'next');
  const p = choice?.probabilities ?? [];
  if (p.length !== options.length || !p.every(Number.isFinite)) {
    throw new DecisionError(`${name} returned ${p.length} scores for ${options.length} options.`, io, performance.now() - started);
  }
  const stopP = stop ? r.answers.find((a) => a.id === 'stop')?.probabilities[1] : undefined;
  if (stop && !Number.isFinite(stopP)) throw new DecisionError(`${name} didn't answer the yes/no question.`, io, performance.now() - started);
  const truncated = r.answers.some((a) => a.truncated);
  io.response = {
    sampled: null,
    topLogprobs: [],
    laya: {
      scores: options.map((o, i) => ({ id: o.id, probability: p[i] })),
      inputTokens: choice!.inputTokens,
      truncated,
      ms: r.ms,
      ...(stopP != null ? { stop: stopP } : {}),
    },
  };
  const scored = options.map((o, i) => ({ id: o.id, label: labels[i], text: o.text, probability: p[i], logprob: Math.log(p[i]) }));
  const best = scored.reduce((a, b) => (b.probability > a.probability ? b : a));
  return {
    options: scored,
    chosen: best.id,
    confidence: best.probability,
    bounded: [],
    model: name,
    slot: 'decider',
    ms: performance.now() - started,
    promptTokens: choice!.inputTokens,
    io,
    truncated,
    ...(stop && stopP != null ? { stop: { statement: stop, probability: stopP } } : {}),
  };
}
