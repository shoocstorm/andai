#!/usr/bin/env bun
// Engine benchmark: launches Andai with VITE_SMOKE=bench, which loads one
// model under several wllama settings in the real webview (src/bench.ts) and
// prints what llama.cpp reports (GPU adapter, layers offloaded, threads) and
// the prompt and generation speed of each.
//
//   bun run bench:engine                      # qwen3-1.7b, the default settings list
//   BENCH_MODEL=qwen3-0.6b bun run bench:engine
//   BENCH_CONTEXT_WORDS=0 …                  # prompt size: words of context (default 400, about a grounded answer)
//   BENCH_CONFIGS='[{"name":"cpu8","n_gpu_layers":0,"n_threads":8}]' bun run bench:engine
//
// The model must already be downloaded in the dev webview (load it once in
// `bun run tauri dev`). Uses a throwaway data dir; touches no knowledge base.
import { spawn } from 'node:child_process';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

const root = resolve(import.meta.dirname, '..');
const model = process.env.BENCH_MODEL ?? 'qwen3-1.7b';
const configs = JSON.parse(
  process.env.BENCH_CONFIGS ??
    JSON.stringify([
      { name: 'app default (GPU, wllama threads)' },
      { name: 'CPU only, wllama threads', n_gpu_layers: 0 },
      { name: 'GPU, 8 threads', n_threads: 8 },
      { name: 'GPU, 2 threads', n_threads: 2 },
      { name: 'GPU, n_batch 2048 / n_ubatch 512', n_batch: 2048, n_ubatch: 512 },
      { name: 'GPU, n_batch 1024 / n_ubatch 1024', n_batch: 1024, n_ubatch: 1024 },
    ]),
);
const dataDir = mkdtempSync(join(tmpdir(), 'andai-bench-'));
const env = {
  ...process.env,
  VITE_SMOKE: 'bench',
  VITE_BENCH: JSON.stringify({ model, configs, maxTokens: Number(process.env.BENCH_TOKENS ?? 192), contextWords: Number(process.env.BENCH_CONTEXT_WORDS ?? 400) }),
  ANDAI_DATA_DIR: dataDir,
  ANDAI_SMOKE: '1',
};
const child = spawn('bun', ['run', 'tauri', 'dev'], { cwd: root, env, detached: true });
const results = [];
await new Promise((done) => {
  const onData = (buf) => {
    for (const line of buf.toString().split('\n')) {
      if (!line.includes('[webview]')) continue;
      const text = line.slice(line.indexOf('[webview]') + 10);
      if (text.startsWith('BENCH ')) {
        const r = JSON.parse(text.slice(6));
        results.push(r);
        console.log(`\n[bench] ${r.name}${r.error ? `: ERROR ${r.error}` : `: ${r.genTokPerSec?.toFixed(1)} tok/s generation, ${r.promptTokPerSec?.toFixed(0)} tok/s prompt (${r.promptTokens} tokens), ${r.threads} threads, load ${r.loadMs} ms`}`);
        for (const l of r.log ?? []) console.log(`    ${l}`);
      } else console.log(`[bench] ${text.slice(0, 400)}`);
      if (text.startsWith('OK') || text.startsWith('FAIL')) done();
    }
  };
  child.stdout.on('data', onData);
  child.stderr.on('data', onData);
  child.on('exit', done);
  setTimeout(done, Number(process.env.BENCH_TIMEOUT_MS ?? 900_000));
});
try {
  process.kill(-child.pid, 'SIGTERM');
} catch {}
rmSync(dataDir, { recursive: true, force: true });
console.log(`\n── engine bench · ${model} ──`);
for (const r of results) console.log(`${(r.genTokPerSec ?? 0).toFixed(1).padStart(7)} tok/s gen  ${String(Math.round(r.promptTokPerSec ?? 0)).padStart(6)} tok/s prompt  ${r.name}${r.error ? `  (${r.error})` : ''}`);
process.exit(results.length === configs.length ? 0 : 1);
