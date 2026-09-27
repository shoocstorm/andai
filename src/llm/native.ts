// Native chat models, run by Rust on MLX (src-tauri/src/llm/, Apple Silicon
// only): typed wrappers over its commands, and a completion in the shape
// wllama returns, so decide.ts and argfill.ts read either engine the same way.
// Downloads go through the checkpoint download shared with Laya (laya.ts).

import { Channel, invoke, isTauri } from '@tauri-apps/api/core';
import type { ChatMessage } from './engine';
import type { LayaCheckpoint } from './laya';

export type NativeSlot = 'chat' | 'decider';

export type NativeStatus = {
  supported: boolean;
  /** Checkpoint id loaded per slot. */
  chat: string | null;
  decider: string | null;
  checkpoints: LayaCheckpoint[];
};

export type NativeParams = {
  maxTokens: number;
  temperature: number;
  topK?: number;
  topP?: number;
  seed?: number;
  /** Qwen3's reasoning block (`enable_thinking`). */
  thinking?: boolean;
  topLogprobs?: number;
  /** GBNF without recursion: a choice between literals, or argument filling's JSON shape (llm/grammar.rs). */
  grammar?: string;
  /** Reuse the KV cache for the prompt prefix shared with the previous request (default on). */
  cachePrompt?: boolean;
};

export type TopLogprob = { token: string; bytes: number[]; logprob: number };

export type Generated = {
  text: string;
  promptTokens: number;
  /** Prompt tokens whose keys and values came from the cache. */
  cachedTokens: number;
  completionTokens: number;
  /** From the call to the first token: reading the prompt and the first step. */
  promptMs: number;
  /** From the first token to the last. */
  genMs: number;
  finish: 'stop' | 'length' | 'cancelled';
  first: { token: string; bytes: number[]; logprob: number; top_logprobs: TopLogprob[] } | null;
};

const UNSUPPORTED: NativeStatus = { supported: false, chat: null, decider: null, checkpoints: [] };

/** In a plain browser (no Tauri) there's no MLX: same answer as a Windows or Intel build. */
export const nativeStatus = (): Promise<NativeStatus> => (isTauri() ? invoke<NativeStatus>('llm_status') : Promise.resolve(UNSUPPORTED));
export type NativeLoaded = { ms: number; layers: number; bits: number; nCtx: number };
export const nativeLoad = (slot: NativeSlot, checkpoint: string, nCtx: number) => invoke<NativeLoaded>('llm_load', { slot, checkpoint, nCtx });
export const nativeUnload = (slot: NativeSlot) => invoke<void>('llm_unload', { slot });
export const nativeRemove = (checkpoint: string) => invoke<void>('llm_remove', { checkpoint });

/** Generation speed, timed from the first token (engine.ts `StreamEvent`). */
export const genTokPerSec = (g: Generated) => (g.completionTokens > 1 && g.genMs > 0 ? (g.completionTokens - 1) / (g.genMs / 1000) : null);
/** Prompt reading speed over the tokens actually read (not those reused from the cache). */
export const promptTokPerSec = (g: Generated) => {
  const read = g.promptTokens - g.cachedTokens;
  return read > 0 && g.promptMs > 0 ? read / (g.promptMs / 1000) : null;
};

/** A cancelled native generation, recognized by `isAbort` like wllama's. */
export class NativeAbortError extends Error {
  constructor() {
    super('Generation stopped.');
    this.name = 'AbortError';
  }
}

/**
 * Generates on the model in `slot`. Text arrives through `onText` as it's
 * written; aborting `signal` stops it after the current token and throws
 * `NativeAbortError`.
 */
export async function nativeGenerate(
  slot: NativeSlot,
  messages: ChatMessage[],
  params: NativeParams,
  onText: (piece: string) => void,
  signal?: AbortSignal,
): Promise<Generated> {
  if (signal?.aborted) throw new NativeAbortError();
  const request = crypto.getRandomValues(new Uint32Array(1))[0];
  const channel = new Channel<string>();
  channel.onmessage = onText;
  const cancel = () => void invoke('llm_cancel', { request }).catch(() => {});
  signal?.addEventListener('abort', cancel, { once: true });
  try {
    const out = await invoke<Generated>('llm_generate', { slot, request, messages, params, onText: channel });
    if (out.finish === 'cancelled' || signal?.aborted) throw new NativeAbortError();
    return out;
  } finally {
    signal?.removeEventListener('abort', cancel);
  }
}

/**
 * `out` as wllama's non-streaming chat completion: the text as the message,
 * and the first token's log-probabilities where decide.ts reads them.
 */
export function asCompletion(out: Generated) {
  return {
    choices: [
      {
        index: 0,
        message: { role: 'assistant' as const, content: out.text },
        finish_reason: out.finish === 'length' ? 'length' : 'stop',
        logprobs: out.first
          ? {
              content: [
                {
                  token: out.first.token,
                  logprob: out.first.logprob,
                  bytes: out.first.bytes,
                  top_logprobs: out.first.top_logprobs.map((t) => ({ token: t.token, logprob: t.logprob, bytes: t.bytes })),
                },
              ],
            }
          : null,
      },
    ],
    usage: { prompt_tokens: out.promptTokens, completion_tokens: out.completionTokens, total_tokens: out.promptTokens + out.completionTokens },
  };
}
