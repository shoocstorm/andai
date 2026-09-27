// Engine benchmark (VITE_SMOKE=bench, via scripts/bench-engine.mjs): loads one
// catalog model under several wllama settings inside the real webview and
// reports what llama.cpp says about the GPU and threads, plus prompt and
// generation speed. An MLX model runs once, natively in Rust (llm/native.ts),
// on the same prompt. The runner passes the settings in VITE_BENCH.
// Loaded only when VITE_SMOKE=bench (main.tsx), so none of it ships in the app.
import { invoke } from '@tauri-apps/api/core';
import { ModelManager, Wllama } from '@wllama/wllama';
import { downloadCheckpoint } from './llm/laya';
import { isMlx, modelById, type MlxDef } from './llm/models';
import { genTokPerSec, nativeGenerate, nativeLoad, nativeStatus, nativeUnload, promptTokPerSec } from './llm/native';

export type BenchConfig = { name: string; n_threads?: number; n_gpu_layers?: number; n_batch?: number; n_ubatch?: number };
/** `contextWords`: pad the prompt with a passage this long, like a grounded answer's context. */
export type BenchInput = { model: string; configs: BenchConfig[]; maxTokens: number; contextWords?: number };

const log = (line: string) => invoke('dev_log', { line }).catch(() => console.log(line));
const asset = (path: string) => new URL(path, window.location.href).href;
// llama.cpp lines that say where the work runs.
const INTERESTING = /webgpu|gpu|offload|metal|adapter|device|backend|n_threads|threads|CPU_REPACK|flash|buffer size|failed|error/i;

const PASSAGE =
  'Tidewater Ferries runs two vessels between Harlow, Pellin Island and Dunmere. Tickets are priced per passenger, with a surcharge for vehicles, a peak multiplier on Fridays and Sundays, and a discount for groups of ten or more. ';
const prompt = (words: number) => [
  { role: 'system' as const, content: `You are a concise assistant.${words ? `\n\nContext:\n${PASSAGE.repeat(Math.ceil(words / 40)).split(' ').slice(0, words).join(' ')}` : ''}` },
  { role: 'user' as const, content: 'Explain in about 150 words how a ferry company might price tickets for cars and foot passengers.' },
];

/** The MLX path: download into Rust if needed, load, warm up, then one measured answer. */
async function benchNative(def: MlxDef, input: BenchInput) {
  const c = (await nativeStatus()).checkpoints.find((x) => x.id === def.native);
  if (!c) throw new Error(`${def.name} needs an Apple Silicon Mac`);
  if (!c.downloaded) {
    let last = 0;
    await downloadCheckpoint('llm', c, (p) => {
      if (p.loaded - last > 100e6 || p.phase.startsWith('Verifying')) void log(`download ${p.phase} ${Math.round(p.loaded / 1e6)} / ${Math.round(p.total / 1e6)} MB`);
      last = p.loaded;
    });
  }
  const loaded = await nativeLoad('chat', def.native, 4096);
  const params = (max: number) => ({ maxTokens: max, temperature: 0, thinking: false, cachePrompt: false });
  await nativeGenerate('chat', prompt(input.contextWords ?? 0), params(8), () => {}); // warm-up: kernels, first allocations
  const t1 = performance.now();
  const r = await nativeGenerate('chat', prompt(input.contextWords ?? 0), params(input.maxTokens), () => {});
  const wall = performance.now() - t1;
  await log(
    `BENCH ${JSON.stringify({
      name: 'MLX · Metal (native)',
      params: {},
      threads: 0,
      compat: false,
      loadMs: Math.round(loaded.ms),
      promptTokens: r.promptTokens,
      promptTokPerSec: promptTokPerSec(r),
      genTokPerSec: genTokPerSec(r),
      genTokens: r.completionTokens,
      firstTokenMs: Math.round(r.promptMs),
      wallMs: Math.round(wall),
      log: [`${loaded.layers} layers, ${loaded.bits}-bit weights, MLX in Rust`],
    })}`,
  );
  await nativeUnload('chat');
}

export async function runBench(input: BenchInput) {
  try {
    const def = modelById(input.model);
    if (!def) throw new Error(`unknown model ${input.model}`);
    if (isMlx(def)) {
      await benchNative(def, input);
      await log('OK');
      await invoke('dev_exit', { code: 0 });
      return;
    }
    await log(`CAPS ${JSON.stringify({ cores: navigator.hardwareConcurrency, gpu: 'gpu' in navigator, isolated: crossOriginIsolated, ua: navigator.userAgent })}`);
    const adapter = await (navigator as unknown as { gpu?: { requestAdapter: () => Promise<{ info?: Record<string, string>; features?: Set<string> } | null> } }).gpu
      ?.requestAdapter()
      .catch(() => null);
    await log(`ADAPTER ${JSON.stringify(adapter ? { info: adapter.info && { vendor: adapter.info.vendor, architecture: adapter.info.architecture, device: adapter.info.device, description: adapter.info.description }, features: [...(adapter.features ?? [])].slice(0, 12) } : null)}`);
    const mm = new ModelManager();
    const cached = (await mm.getModels()).find((m) => m.url === def.url && m.size > 0);
    if (!cached) throw new Error(`${def.name} is not downloaded in this webview; load it in the app once first`);

    for (const cfg of input.configs) {
      const lines: string[] = [];
      const keep = (...a: unknown[]) => {
        const s = a.map(String).join(' ');
        if (INTERESTING.test(s) && lines.length < 60) lines.push(s.slice(0, 240));
      };
      const w = new Wllama({ default: asset('/wllama/default/wllama.wasm') }, { logger: { debug: keep, log: keep, warn: keep, error: keep } });
      w.setCompat({ worker: asset('/wllama/compat/wllama.js'), wasm: asset('/wllama/compat/wllama.wasm') });
      const { name, ...params } = cfg;
      const t0 = performance.now();
      try {
        await w.loadModel(cached, { n_ctx: 4096, reasoning_format: 'none', ...params } as Parameters<Wllama['loadModel']>[1]);
        const loadMs = performance.now() - t0;
        const ask = (max: number) =>
          w.createChatCompletion({ messages: prompt(input.contextWords ?? 0), max_tokens: max, temperature: 0, cache_prompt: false, chat_template_kwargs: { enable_thinking: false } } as never) as Promise<{
            timings?: { prompt_n: number; prompt_per_second: number; predicted_n: number; predicted_per_second: number };
          }>;
        await ask(8); // warm-up: shader compilation, first allocations
        const t1 = performance.now();
        const r = await ask(input.maxTokens);
        const wall = performance.now() - t1;
        await log(
          `BENCH ${JSON.stringify({
            name,
            params,
            threads: w.isMultithread() ? w.getNumThreads() : 1,
            compat: w.getWorkerResources().compat,
            loadMs: Math.round(loadMs),
            promptTokens: r.timings?.prompt_n,
            promptTokPerSec: r.timings?.prompt_per_second,
            genTokPerSec: r.timings?.predicted_per_second,
            genTokens: r.timings?.predicted_n,
            wallMs: Math.round(wall),
            log: lines,
          })}`,
        );
      } catch (e) {
        await log(`BENCH ${JSON.stringify({ name, params, error: e instanceof Error ? e.message : String(e), log: lines })}`);
      } finally {
        await w.exit().catch(() => {});
      }
    }
    await log('OK');
    await invoke('dev_exit', { code: 0 });
  } catch (e) {
    await log(`FAIL ${e instanceof Error ? e.message : String(e)}`);
    await invoke('dev_exit', { code: 1 });
  }
}
