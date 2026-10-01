import { create } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';
import type { Policy } from '../agent/tools/types';
import type { SearchHit } from '../kb/api';
import type { RelevanceRecord } from '../agent/relevance';
import type { SupportRecord } from '../agent/claims';
import type { ContextRecord } from '../agent/prompt';
import type { Found } from '../kb/match';
import type { DecisionIO } from '../llm/decide';
import type { FillIO } from '../agent/tools/argfill';

/**
 * `retrieve` belongs to the fixed pipeline; `plan` to agent mode, whose tool
 * calls are in `Message.agent`. `filter` is the relevance check (agent/relevance.ts,
 * with a Laya decision model only), recorded in `Message.relevance`; `verify`
 * the claim check after the answer (agent/claims.ts, Laya only), in `Message.support`.
 */
export type StepKind = 'analyze' | 'retrieve' | 'plan' | 'filter' | 'build' | 'generate' | 'verify';
export type StepStatus = 'queued' | 'running' | 'done' | 'skipped' | 'error';

export type TraceStep = {
  kind: StepKind;
  title: string;
  detail: string;
  status: StepStatus;
  startedAt?: number;
  ms?: number;
};

/** One SemIf-style readout (llm/decide.ts), kept whole so the user sees every option's probability. */
export type DecisionRecord = {
  question: string;
  options: { id: string; label: string; text: string; probability: number }[];
  chosen: string;
  confidence: number;
  /** Options missing from the readout, scored at its lowest value (an upper bound). */
  bounded?: string[];
  model: string;
  slot: 'chat' | 'decider';
  ms: number;
  /** Seed of the option shuffle, so the order can be reproduced. */
  seed: number;
  promptTokens: number | null;
  /** What the decision model was sent and its raw readout (llm/decide.ts). */
  io?: DecisionIO;
  /** Laya: its input was cut to fit (an option, the question or the state). */
  truncated?: boolean;
  /** Laya: how likely the tool results already suffice (asked with the choice once there are results). */
  stop?: { statement: string; probability: number };
};

export type CallStatus = 'filling' | 'awaiting' | 'running' | 'done' | 'error' | 'denied' | 'skipped';

export type ToolCallRecord = {
  tool: string;
  title: string;
  /** Validated arguments, as passed on; null until filled or when filling failed. */
  args: Record<string, unknown> | null;
  /** The model's raw argument text, when it differs from `args` or failed to parse. */
  argsRaw: string | null;
  /** Model that wrote the arguments, and how many tries it took. */
  argModel: string | null;
  argAttempts: number;
  /**
   * What the argument writer (the chat model) was sent and replied, every
   * attempt; `note` says where the arguments came from when not (only) from it:
   * set by the agent, or the question as written after a failed fill.
   */
  argIO?: FillIO & { note?: string };
  /** Arguments Laya picked as typed choices (the chat model wrote the rest). */
  argChoices?: { arg: string; value: string; probability: number; model: string }[];
  policy: Policy;
  approval?: 'pending' | 'approved' | 'denied';
  /** ug arguments as run (from Rust). */
  argv?: string[];
  startedAt: number;
  endedAt?: number;
  ms?: number;
  status: CallStatus;
  /** Tool output as JSON text, clipped to OUTPUT_KEEP for storage. */
  output?: string;
  outputBytes?: number;
  truncated?: boolean;
  error?: string;
  /** The line the next decision saw. */
  observation?: string;
  /** A search's passages counted by how ug found them (semantic, keyword, graph), for a short summary. */
  found?: Found;
  /** Passages this call added to the answer's context, or added text to (agent/evidence.ts). */
  hits?: number;
  /**
   * Laya's scores for the passages this call returned, read by the next
   * decision (agent/relevance.ts): each passage in pieces that fit a row
   * (`chunks`), scored by its best one. `error` when scoring failed.
   */
  scored?: {
    model: string;
    ms: number;
    rows: number;
    items: { source: string; name: string; score: number; chunks: number; inputTokens: number; truncated: boolean }[];
    error?: string;
  };
};

/** One loop iteration: a decision, and the tool call it led to (none for answer/clarify). */
export type AgentStep = {
  id: string;
  index: number;
  at: number;
  /** Null when no decision was needed or it failed (see `note`). */
  decision: DecisionRecord | null;
  /** Tool id, or `answer_now` / `ask_clarification`. */
  action: string;
  /** Why the loop did something other than the plain argmax (fallback, low confidence, guard). */
  note?: string;
  /**
   * Set when the loop didn't take the decision as chosen: it wasn't trusted
   * (search once, or answer), or it picked a tool whose argument nothing has
   * shown yet: a symbol (look symbols up first) or a line range (search first);
   * or it was about to answer from a passage the search had cut short (read it whole first),
   * or without the code of a symbol the request names (fetch it first).
   */
  fallback?: 'low-confidence' | 'decision-failed' | 'needs-symbol' | 'needs-range' | 'read-whole' | 'named-symbol' | 'whole-repo';
  /** A rule chose this step (agent/plan.ts: the evidence plan or search-again), not the decision model. */
  planned?: boolean;
  /** A decision that was sent but couldn't be read: the error, its time and the call, when it got that far. */
  failedDecision?: { error: string; ms: number; model?: string; io?: DecisionIO };
  call?: ToolCallRecord;
};

/** Stored tool output per call; the full output only lives for the turn. */
export const OUTPUT_KEEP = 16_000;

export type TurnStats = {
  tokens: number;
  /** Generation speed, from the first token on (engine.ts `StreamEvent`). */
  tokPerSec: number;
  /** How fast the prompt was read, before the first token; null when unknown. */
  promptTokPerSec?: number | null;
  promptTokens: number | null;
  nCtx: number;
  totalMs: number;
  /** From the start of the turn to the first streamed token: the wait the user sees. Null when no token came. */
  firstTokenMs: number | null;
  model: string;
};

export type Message = {
  id: string;
  role: 'user' | 'assistant' | 'error';
  content: string;
  createdAt: number;
  /** assistant turns only */
  steps?: TraceStep[];
  sources?: SearchHit[];
  /** Agent mode: every decision and tool call, in order. */
  agent?: AgentStep[];
  /** The relevance check: every retrieved passage's score and whether it went into the prompt. */
  relevance?: RelevanceRecord;
  /** What the chat model was sent: budgets, passages in / cut / left out, history kept, the system prompt. */
  context?: ContextRecord;
  /** The claim check: each cited sentence's support score, and which may not be supported. */
  support?: SupportRecord;
  kbName?: string | null;
  stats?: TurnStats;
  streaming?: boolean;
  stopped?: boolean;
};

type ChatState = {
  session: string;
  messages: Message[];
};

const newSession = () => `X-${Math.floor(100 + Math.random() * 900)}`;

export const useChat = create<ChatState>()(
  persist(() => ({ session: newSession(), messages: [] as Message[] }), {
    name: 'andai.chat',
    storage: createJSONStorage(() => localStorage),
    // a turn interrupted by quitting the app is shown as stopped, not stuck
    merge: (persisted, current) => {
      const p = persisted as Partial<ChatState>;
      return {
        ...current,
        ...p,
        messages: (p.messages ?? []).map((m) =>
          m.streaming
            ? { ...m, streaming: false, stopped: true, steps: m.steps?.map(settle), agent: m.agent?.map(settleCall) }
            : m,
        ),
      };
    },
  }),
);

const settle = (s: TraceStep): TraceStep =>
  s.status === 'running' || s.status === 'queued' ? { ...s, status: 'skipped' } : s;

const LIVE: CallStatus[] = ['filling', 'awaiting', 'running'];
/** A call that was still in flight when its turn ended is marked skipped. */
export const settleCall = (a: AgentStep): AgentStep =>
  a.call && LIVE.includes(a.call.status)
    ? { ...a, call: { ...a.call, status: 'skipped', approval: a.call.approval === 'pending' ? undefined : a.call.approval } }
    : a;

export const uid = () => Math.random().toString(36).slice(2, 10);

export function addMessage(m: Message) {
  useChat.setState((s) => ({ messages: [...s.messages, m] }));
}

export function patchMessage(id: string, patch: Partial<Message> | ((m: Message) => Partial<Message>)) {
  useChat.setState((s) => ({
    messages: s.messages.map((m) => (m.id === id ? { ...m, ...(typeof patch === 'function' ? patch(m) : patch) } : m)),
  }));
}

export function patchStep(id: string, kind: StepKind, patch: Partial<TraceStep>) {
  patchMessage(id, (m) => ({
    steps: m.steps?.map((s) => {
      if (s.kind !== kind) return s;
      const next = { ...s, ...patch };
      if (patch.status === 'running') next.startedAt = Date.now();
      if ((patch.status === 'done' || patch.status === 'error') && s.startedAt) next.ms = Date.now() - s.startedAt;
      return next;
    }),
  }));
}

export function addAgentStep(id: string, step: AgentStep) {
  patchMessage(id, (m) => ({ agent: [...(m.agent ?? []), step] }));
}

export function patchAgentStep(id: string, stepId: string, patch: Partial<AgentStep>) {
  patchMessage(id, (m) => ({ agent: m.agent?.map((a) => (a.id === stepId ? { ...a, ...patch } : a)) }));
}

export function patchCall(id: string, stepId: string, patch: Partial<ToolCallRecord>) {
  patchMessage(id, (m) => ({
    agent: m.agent?.map((a) => (a.id === stepId && a.call ? { ...a, call: { ...a.call, ...patch } } : a)),
  }));
}

export function clearChat() {
  useChat.setState({ messages: [], session: newSession() });
}

/** Split Qwen3's `<think>…</think>` from the answer (ported from wllama-chat). */
export function splitThink(raw: string): { thinking: string | null; answer: string; open: boolean } {
  const start = raw.indexOf('<think>');
  if (start === -1) return { thinking: null, answer: raw, open: false };
  const end = raw.indexOf('</think>');
  const thinking = raw.slice(start + 7, end === -1 ? raw.length : end).trim();
  const answer = end === -1 ? '' : raw.slice(end + 8).replace(/^\n+/, '');
  // With enable_thinking: false Qwen3 still emits an empty pair.
  return { thinking: thinking || null, answer, open: end === -1 };
}
