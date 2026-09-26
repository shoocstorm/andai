import { create } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';
import type { SearchHit } from '../kb/api';

export type StepKind = 'analyze' | 'retrieve' | 'build' | 'generate';
export type StepStatus = 'queued' | 'running' | 'done' | 'skipped' | 'error';

export type TraceStep = {
  kind: StepKind;
  title: string;
  detail: string;
  status: StepStatus;
  startedAt?: number;
  ms?: number;
};

export type TurnStats = {
  tokens: number;
  tokPerSec: number;
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
          m.streaming ? { ...m, streaming: false, stopped: true, steps: m.steps?.map(settle) } : m,
        ),
      };
    },
  }),
);

const settle = (s: TraceStep): TraceStep =>
  s.status === 'running' || s.status === 'queued' ? { ...s, status: 'skipped' } : s;

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
