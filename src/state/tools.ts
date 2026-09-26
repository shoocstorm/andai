import { create } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';
import type { Policy } from '../agent/tools/types';

export type ToolStats = { calls: number; errors: number; totalMs: number; lastAt: number | null };

type ToolsState = {
  /** Off → the fixed pipeline: one search with the raw question, then answer. */
  agentMode: boolean;
  /** Tool calls allowed per turn. */
  maxSteps: number;
  /** Below this, a tool choice isn't trusted: the agent falls back (loop.ts). */
  minConfidence: number;
  /** Per-tool user setting; a tool that isn't listed uses its default (registry.defaultPolicy). */
  policies: Record<string, Policy>;
  stats: Record<string, ToolStats>;
};

export const TOOL_DEFAULTS = { agentMode: true, maxSteps: 20, minConfidence: 0.3 };

export const useTools = create<ToolsState>()(
  persist((): ToolsState => ({ ...TOOL_DEFAULTS, policies: {}, stats: {} }), {
    name: 'andai.tools',
    storage: createJSONStorage(() => localStorage),
  }),
);

export const setPolicy = (id: string, policy: Policy) => useTools.setState((s) => ({ policies: { ...s.policies, [id]: policy } }));
export const setAgent = (patch: Partial<Pick<ToolsState, 'agentMode' | 'maxSteps' | 'minConfidence'>>) => useTools.setState(patch);

export function recordToolRun(id: string, ms: number, ok: boolean) {
  useTools.setState((s) => {
    const prev = s.stats[id] ?? { calls: 0, errors: 0, totalMs: 0, lastAt: null };
    return {
      stats: { ...s.stats, [id]: { calls: prev.calls + 1, errors: prev.errors + (ok ? 0 : 1), totalMs: prev.totalMs + ms, lastAt: Date.now() } },
    };
  });
}

// ── approvals ────────────────────────────────────────────────────────────
// A call under the Ask policy waits here until the user answers in the chat
// (CommandCenter's approval card). Not persisted: a pending approval never
// survives a reload, the turn is marked stopped instead (state/chat.ts).

const pending = new Map<string, (approved: boolean) => void>();

export function requestApproval(callId: string, signal: AbortSignal): Promise<boolean> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) return reject(new DOMException('aborted', 'AbortError'));
    const onAbort = () => {
      pending.delete(callId);
      reject(new DOMException('aborted', 'AbortError'));
    };
    signal.addEventListener('abort', onAbort, { once: true });
    pending.set(callId, (ok) => {
      signal.removeEventListener('abort', onAbort);
      pending.delete(callId);
      resolve(ok);
    });
  });
}

export function resolveApproval(callId: string, approved: boolean) {
  pending.get(callId)?.(approved);
}

export const isAwaitingApproval = (callId: string) => pending.has(callId);
