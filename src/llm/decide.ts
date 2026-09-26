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

import { complete, slotFor, type Completion, type Slot } from './engine';

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
};

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
export async function decide(
  state: string,
  question: string,
  options: DecisionOption[],
  signal?: AbortSignal,
): Promise<Decision> {
  if (options.length < 2 || options.length > MAX_OPTIONS) {
    throw new Error(`A decision needs 2–${MAX_OPTIONS} options, got ${options.length}.`);
  }
  const started = performance.now();
  const labels = labelsFor(options.length);
  const target = slotFor('decider');
  if (!target) throw new Error('No model loaded — open Settings → Models to load one.');
  if (!target.def.decider) throw new Error(`${target.def.name} can't make decisions; load a decision model in Settings → Models.`);
  const { response, slot, def } = await complete(target.slot, {
    messages: decisionMessages(state, question, options),
    max_tokens: 1,
    // Neutral sampling, so the logprobs are a plain readout of the logits.
    temperature: 1,
    top_k: 0,
    top_p: 1,
    logprobs: true,
    // Room for every letter even when a few other tokens rank among them.
    top_logprobs: 20,
    grammar: `root ::= ${labels.map((l) => `"${l}"`).join(' | ')}`,
    cache_prompt: false,
    chat_template_kwargs: { enable_thinking: false },
    abortSignal: signal,
  });
  const { values: logprobs, bounded } = optionLogprobs(response, labels);
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
  };
}
