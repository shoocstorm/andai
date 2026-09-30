// An MLX model as the chat model or decider: loading downloads it first when
// needed, `chat` streams what Rust sends (and recovers text that arrives
// after the result), `complete` answers in wllama's shape so decide.ts reads
// it the same way, and an abort is an abort.
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { Generated } from './native';

const native = vi.hoisted(() => ({
  downloaded: false,
  log: [] as string[],
  /** What the next generation streams, and what it returns. */
  pieces: ['Hel', 'lo'] as string[],
  late: [] as string[],
  result: null as Partial<Generated> | null,
  lastParams: null as unknown,
  aborted: false,
}));

vi.mock('./native', async (importOriginal) => {
  const actual = await importOriginal<typeof import('./native')>();
  return {
    ...actual,
    nativeStatus: async () => ({
      supported: true,
      chat: null,
      decider: null,
      checkpoints: ['qwen3-1.7b-mlx', 'qwen3-0.6b-mlx'].map((id) => ({ id, repo: 'r/x', commit: 'c', bytes: 10, files: [], downloaded: native.downloaded })),
    }),
    nativeLoad: async (slot: string, id: string, nCtx: number) => {
      native.log.push(`load ${slot} ${id} ${nCtx}`);
      return { ms: 5, layers: 28, bits: 4, nCtx };
    },
    nativeUnload: async (slot: string) => void native.log.push(`unload ${slot}`),
    nativeRemove: async (id: string) => {
      native.log.push(`remove ${id}`);
      native.downloaded = false;
    },
    nativeGenerate: async (slot: string, _messages: unknown, params: unknown, onText: (p: string) => void, signal?: AbortSignal) => {
      native.log.push(`generate ${slot}`);
      native.lastParams = params;
      if (native.aborted || signal?.aborted) throw new actual.NativeAbortError();
      for (const p of native.pieces) onText(p);
      const text = [...native.pieces, ...native.late].join('');
      const out: Generated = {
        text,
        promptTokens: 500,
        cachedTokens: 100,
        completionTokens: 11,
        promptMs: 40,
        genMs: 25,
        finish: 'stop',
        first: null,
        ...native.result,
      };
      // Channel messages may land after the command resolves; these never do.
      return out;
    },
  };
});

// jsdom has no OPFS, so wllama's real cache throws; MLX models never use it.
vi.mock('@wllama/wllama', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@wllama/wllama')>()),
  ModelManager: class {
    async getModels() {
      return [];
    }
  },
}));

vi.mock('./laya', async (importOriginal) => ({
  ...(await importOriginal<typeof import('./laya')>()),
  downloadCheckpoint: async (kind: string, c: { id: string }, onProgress: (p: unknown) => void) => {
    native.log.push(`download ${kind} ${c.id}`);
    onProgress({ loaded: 5, total: 10, speed: 1, phase: 'Downloading…' });
    native.downloaded = true;
  },
}));

const { chat, complete, evictModel, isAbort, loadDecider, loadModel, unloadDecider, unloadModel, useEngine } = await import('./engine');
const { optionLogprobs } = await import('./decide');

beforeEach(async () => {
  await unloadModel();
  await unloadDecider();
  Object.assign(native, { downloaded: false, log: [], pieces: ['Hel', 'lo'], late: [], result: null, lastParams: null, aborted: false });
});

const drain = async (gen: AsyncGenerator<unknown>) => {
  const out: unknown[] = [];
  for await (const ev of gen) out.push(ev);
  return out;
};
const opts = () => ({ temperature: 0.7, maxTokens: 64, thinking: false, signal: new AbortController().signal });

describe('an MLX chat model', () => {
  it('downloads a missing model into Rust, loads it and reports the native engine', async () => {
    await loadModel('qwen3-1.7b-mlx');
    expect(native.log).toEqual(['download llm qwen3-1.7b-mlx', 'load chat qwen3-1.7b-mlx 4096']);
    const s = useEngine.getState();
    expect(s).toMatchObject({ status: 'ready', loadedId: 'qwen3-1.7b-mlx', error: null });
    expect(s.info).toMatchObject({ backend: 'MLX · Metal (native)', layers: 28, arch: 'qwen3 · 4-bit' });
    expect(s.cached['https://huggingface.co/mlx-community/Qwen3-1.7B-4bit/tree/3b1b1768f8f8cf8351c712464f906e86c2b8269e']).toBeGreaterThan(0);
  });

  it('skips the download when Rust already has it, and unloads its slot', async () => {
    native.downloaded = true;
    await loadModel('qwen3-0.6b-mlx');
    await unloadModel();
    expect(native.log).toEqual(['load chat qwen3-0.6b-mlx 4096', 'unload chat']);
  });

  it('streams the pieces, then reports Rust’s own timings', async () => {
    native.downloaded = true;
    await loadModel('qwen3-1.7b-mlx');
    const events = await drain(chat([{ role: 'user', content: 'hi' }], opts()));
    expect(events.filter((e) => (e as { type: string }).type === 'delta').map((e) => (e as { text: string }).text)).toEqual(['Hel', 'lo']);
    // 10 tokens after the first in 25 ms; 400 prompt tokens read (100 cached) in 40 ms.
    expect(events.at(-1)).toEqual({ type: 'done', promptTokens: 500, completionTokens: 11, tokPerSec: 400, promptTokPerSec: 10_000 });
    expect(native.lastParams).toMatchObject({ maxTokens: 64, temperature: 0.7, thinking: false });
  });

  it('takes text that hadn’t streamed yet from the result', async () => {
    native.downloaded = true;
    await loadModel('qwen3-1.7b-mlx');
    native.late = [' world'];
    const events = await drain(chat([{ role: 'user', content: 'hi' }], opts()));
    const text = events.flatMap((e) => ((e as { type: string }).type === 'delta' ? [(e as { text: string }).text] : [])).join('');
    expect(text).toBe('Hello world');
  });

  it('turns a stop into an abort the turn recognizes', async () => {
    native.downloaded = true;
    await loadModel('qwen3-1.7b-mlx');
    native.aborted = true;
    const err = await drain(chat([{ role: 'user', content: 'hi' }], opts())).catch((e: unknown) => e);
    expect(isAbort(err)).toBe(true);
    expect(useEngine.getState().generating).toBe(false);
  });

  it('answers complete() in wllama’s shape, so decisions read the letters', async () => {
    native.downloaded = true;
    await loadModel('qwen3-1.7b-mlx');
    native.result = {
      text: 'B',
      completionTokens: 1,
      first: {
        token: 'B',
        bytes: [66],
        logprob: -0.25,
        top_logprobs: [
          { token: 'B', bytes: [66], logprob: -0.25 },
          { token: 'C', bytes: [67], logprob: -1.5 },
          { token: 'The', bytes: [84, 104, 101], logprob: -6 },
        ],
      },
    };
    const { response, slot, def } = await complete('decider', {
      messages: [{ role: 'user', content: 'pick' }],
      max_tokens: 1,
      temperature: 1,
      top_k: 0,
      top_p: 1,
      logprobs: true,
      top_logprobs: 20,
      grammar: 'root ::= "A" | "B" | "C"',
      cache_prompt: false,
    } as never);
    expect([slot, def.id]).toEqual(['chat', 'qwen3-1.7b-mlx']);
    expect(response.choices[0].message?.content).toBe('B');
    expect(native.lastParams).toMatchObject({ maxTokens: 1, topK: 0, topLogprobs: 20, grammar: 'root ::= "A" | "B" | "C"', cachePrompt: false, thinking: true });
    // C is listed; A isn't, so it's bounded at the lowest listed value.
    expect(optionLogprobs(response, ['A', 'B', 'C'])).toEqual({ values: [-6, -0.25, -1.5], bounded: [0] });
  });

  it('runs decisions on an MLX decider slot', async () => {
    native.downloaded = true;
    await loadModel('qwen3-1.7b-mlx');
    await loadDecider('qwen3-0.6b-mlx');
    await complete('decider', { messages: [{ role: 'user', content: 'x' }], max_tokens: 1 } as never);
    expect(native.log).toEqual(['load chat qwen3-1.7b-mlx 4096', 'load decider qwen3-0.6b-mlx 4096', 'generate decider']);
    await unloadDecider();
    expect(native.log.at(-1)).toBe('unload decider');
  });

  it('evicting removes Rust’s copy, unloading it first', async () => {
    native.downloaded = true;
    await loadModel('qwen3-1.7b-mlx');
    await evictModel('qwen3-1.7b-mlx');
    expect(native.log.slice(-2)).toEqual(['unload chat', 'remove qwen3-1.7b-mlx']);
    expect(useEngine.getState().loadedId).toBeNull();
  });
});
