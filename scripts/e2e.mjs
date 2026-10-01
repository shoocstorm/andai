#!/usr/bin/env bun
// End-to-end test of the real app: launches Andai with VITE_SMOKE=e2e, which
// (inside the actual WKWebView) creates a knowledge base, ingests the fixtures
// through ug, loads the model, runs one grounded agent turn and prints a
// RESULT line. This script asserts on it.
//
//   bun run test:e2e              # dev build (tauri dev)
//   bun run test:e2e --release    # release binary: http://localhost origin + ACL
//   E2E_MODEL=stories-260k ...    # engine plumbing only (answer not checked)
//   bun run test:e2e --update-perf  # record this run as the perf baseline
//   E2E_MLX=none ...              # skip the native MLX chat model probe (Apple Silicon)
//   E2E_LAYA=none ...             # skip the Laya decision probe (Apple Silicon; default laya-multilingual)
//
// Needs: ug on PATH (or ~/.local/bin), network on first run for the model.
import { spawn, spawnSync } from 'node:child_process';
import { mkdtempSync, rmSync } from 'node:fs';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { devOnPort } from './dev-port.mjs';
import { keepCheckpoint, seedCheckpoint } from './checkpoint-cache.mjs';
import { byLine, compare } from './perf-lib.mjs';

const root = resolve(import.meta.dirname, '..');
const release = process.argv.includes('--release');
const updatePerf = process.argv.includes('--update-perf');
const model = process.env.E2E_MODEL ?? 'qwen3-0.6b';
const timeoutMs = Number(process.env.E2E_TIMEOUT_MS ?? 900_000);
const fixtures = ['tests/fixtures/wllama-notes.md', 'tests/fixtures/hello.pdf'].map((f) => resolve(root, f));
// Egress canary (AGENTS.md §9): a local server the CSP doesn't allow. It
// answers with permissive CORS/CORP so only the CSP can stop a request, and
// counts every one that arrives. The webview tries to reach it; zero hits
// proves the block without any real outbound traffic.
const canaryHits = [];
const canary = createServer((req, res) => {
  canaryHits.push(req.url);
  res.writeHead(200, { 'Access-Control-Allow-Origin': '*', 'Cross-Origin-Resource-Policy': 'cross-origin' });
  res.end('leaked');
});
await new Promise((r) => canary.listen(0, '127.0.0.1', r));
const canaryUrl = `http://127.0.0.1:${canary.address().port}/exfil`;

// Knowledge-base files go to a throwaway dir, never the user's real ones.
const dataDir = mkdtempSync(join(tmpdir(), 'andai-e2e-'));
// One real Laya decision on Apple Silicon (laya/, AGENTS.md §2). The checkpoint
// downloads once into a test cache (scripts/checkpoint-cache.mjs), never user data.
const appleSilicon = process.platform === 'darwin' && process.arch === 'arm64';
const laya = appleSilicon && process.env.E2E_LAYA !== 'none' ? (process.env.E2E_LAYA ?? 'laya-multilingual') : '';
// And one answer on a native MLX chat model (llm/, AGENTS.md §2), cached the same way.
const mlx = appleSilicon && process.env.E2E_MLX !== 'none' ? (process.env.E2E_MLX ?? 'qwen3-0.6b-mlx') : '';
const env = {
  ...process.env,
  VITE_SMOKE: 'e2e',
  VITE_SMOKE_LAYA: laya,
  VITE_SMOKE_MLX: mlx,
  VITE_SMOKE_FILES: fixtures.join(','),
  VITE_SMOKE_MODEL: model,
  VITE_SMOKE_CANARY: canaryUrl,
  ANDAI_DATA_DIR: dataDir,
  // ug's projects too: Andai lists every ug project, so the run gets its own
  // and never sees or touches the user's (~/.ug). ug's embedder cache stays shared.
  UG_HOME: join(dataDir, 'ug'),
  // Harness-only switches, read by Rust at startup (AGENTS.md §9): enable
  // dev_log/dev_exit, and grant the fixtures the way a drop would.
  ANDAI_SMOKE: '1',
  ANDAI_E2E_FILES: fixtures.join(','),
};

for (const id of [laya, mlx]) if (id) seedCheckpoint(dataDir, id);

function launch() {
  if (!release) {
    const dev = devOnPort();
    return spawn('bun', ['run', 'tauri', 'dev', ...dev.args], { cwd: root, env: { ...env, ...dev.env }, detached: true });
  }
  console.log('[e2e] building release binary with the e2e harness…');
  const b = spawnSync('bun', ['run', 'tauri', 'build', '--no-bundle'], { cwd: root, env, stdio: 'inherit' });
  if (b.status !== 0) process.exit(b.status ?? 1);
  // Minimal environment, like a Finder launch: proves ug is found without the shell PATH.
  return spawn(resolve(root, 'src-tauri/target/release/andai'), [], {
    cwd: root,
    env: {
      HOME: process.env.HOME,
      PATH: '/usr/bin:/bin',
      ANDAI_DATA_DIR: dataDir,
      UG_HOME: env.UG_HOME,
      ANDAI_SMOKE: env.ANDAI_SMOKE,
      ANDAI_E2E_FILES: env.ANDAI_E2E_FILES,
    },
    detached: true,
  });
}

const child = launch();
const lines = [];
let result = null;
let failure = null;

const done = new Promise((resolveDone) => {
  const onData = (lines) => {
    for (const line of lines) {
      if (!line.includes('[webview]')) continue;
      const text = line.slice(line.indexOf('[webview]') + 10);
      lines.push(text);
      console.log(`[e2e] ${text.slice(0, 220)}`);
      if (text.startsWith('RESULT ')) result = JSON.parse(text.slice(7));
      if (text.startsWith('FAIL')) failure = text;
      if (text.startsWith('OK') || text.startsWith('FAIL')) resolveDone();
    }
  };
  child.stdout.on('data', byLine(onData));
  child.stderr.on('data', byLine(onData));
  child.on('exit', () => resolveDone());
  setTimeout(() => {
    failure ??= `timed out after ${timeoutMs / 1000}s`;
    resolveDone();
  }, timeoutMs);
});

await done;
try {
  process.kill(-child.pid, 'SIGTERM');
} catch {}
for (const id of [laya, mlx]) if (id) keepCheckpoint(dataDir, id);
rmSync(dataDir, { recursive: true, force: true });
canary.close();

// ── assertions ─────────────────────────────────────────────────────────────
const checks = [];
const check = (name, ok, detail = '') => checks.push({ name, ok: !!ok, detail });

if (failure || !result) {
  check('harness completed', false, failure ?? 'no RESULT line');
} else {
  const { caps, engine, kb, steps, sources, stats, answer } = result;
  check('webview is cross-origin isolated', caps.isolated && caps.sharedArrayBuffer);
  check('CSP blocked egress (the canary got no requests)', canaryHits.length === 0, JSON.stringify(canaryHits));
  check('wllama runs multi-threaded', /multi/.test(engine?.threads ?? ''), engine?.threads);
  // llama.cpp's own load log, not just navigator.gpu (engine.ts gpuFromLog).
  const layers = /WebGPU · (\d+)\/(\d+) layers/.exec(engine?.backend ?? '');
  check('every layer runs on the GPU (WebGPU)', layers && layers[1] === layers[2], engine?.backend);
  check('knowledge base indexed by ug', kb.status === 'ready' && kb.nodes > 0 && !kb.error, JSON.stringify(kb));
  check('every source indexed', kb.sources.length === fixtures.length && kb.sources.every((s) => s.status === 'indexed'));
  check('ug progress streamed to the UI', result.ugLogLines > 5, `${result.ugLogLines} lines`);
  // With a Laya decision model the turn also has the relevance and claim checks, which skip
  // when there's nothing to check (no more than two passages, an answer that cites nothing).
  const LAYA_STEPS = ['filter', 'verify'];
  const settled = (list) => list?.every((s) => s.status === 'done' || (LAYA_STEPS.includes(s.kind) && s.status === 'skipped'));
  const core = (list) => (list ?? []).filter((s) => !LAYA_STEPS.includes(s.kind));
  check('all four agent steps completed', core(steps).length === 4 && core(steps).every((s) => s.status === 'done') && settled(steps), JSON.stringify(steps));
  check('retrieval returned the notes', sources?.includes('wllama-notes.md'), JSON.stringify(sources));
  console.log(`[e2e] model sha256 check: ${result.verifyMs == null ? 'reused a verified copy' : `${Math.round(result.verifyMs)} ms`}`);
  check('generation produced tokens', stats?.tokens > 0 && stats?.tokPerSec > 1, JSON.stringify(stats));
  if (model !== 'stories-260k') {
    check('answer is grounded in the retrieved passage', /cross-origin|coop|coep/i.test(answer ?? ''), answer?.slice(0, 200));
  }
  // Agent mode. The model's choices aren't asserted (a small model picks
  // loosely); the machinery is: every option scored, arguments valid,
  // ug run through the Rust boundary, and every step traced.
  const a = result.agent ?? {};
  if (model !== 'stories-260k') {
    const probs = (a.decision?.options ?? []).map(([, p]) => p);
    check(
      'decision readout scores every option',
      !a.decision?.error && probs.length === (a.tools?.length ?? 0) + 2 && Math.abs(probs.reduce((x, y) => x + y, 0) - 1) < 1e-6,
      JSON.stringify(a.decision),
    );
    check('tool arguments come back schema-valid', a.fill?.ok === true, JSON.stringify(a.fill));
  }
  check('kb_tool runs ug through the Rust boundary', a.overview?.ok && a.overview.argv?.[0] === 'project_overview', JSON.stringify(a.overview));
  check(
    'agent turn completes with every step traced',
    settled(a.turn?.steps) && a.turn?.agent?.length > 0 &&
      a.turn.agent.every((s) => !s.call || ['done', 'error', 'skipped'].includes(s.call.status)),
    JSON.stringify(a.turn?.agent),
  );
  if (laya) {
    const l = result.laya ?? {};
    check('Laya decides in Rust on the GPU, under 100 ms', !l.error && l.slot === 'decider' && l.sum > 0.999 && l.sum < 1.001 && l.ms < 100, JSON.stringify(l));
    console.log(`[e2e] laya: ${l.model} chose ${l.chosen} @${Math.round((l.confidence ?? 0) * 100)}% in ${l.ms?.toFixed(1)} ms (model ${l.modelMs?.toFixed(1)} ms, ${l.inputTokens} tokens${l.truncated ? ', cut to fit' : ''})`);
  }
  if (mlx) {
    const n = result.mlx ?? {};
    check('MLX model answers natively, grounded, over 100 tok/s', !n.error && /MLX/.test(n.backend ?? '') && n.stats?.tokPerSec > 100 && /cross-origin|coop|coep/i.test(n.answer ?? ''), JSON.stringify(n).slice(0, 600));
    check('MLX argument fill is held to its grammar', n.fill?.ok === true && /^\{/.test(n.fill.raw ?? ''), JSON.stringify(n.fill));
    console.log(`[e2e] mlx: ${n.backend} ${n.stats?.tokPerSec?.toFixed(0)} tok/s, first token ${n.stats?.firstTokenMs?.toFixed(0)} ms, load ${n.loadMs?.toFixed(0)} ms, fill ${n.fill?.ms?.toFixed(0)} ms`);
  }
  const hub = result.hub ?? {};
  check('Hugging Face search and inspect work from the app (CSP, CORS), pinned to a commit', !hub.error && hub.results > 0 && hub.gguf?.ok && /^[0-9a-f]{40}$/.test(hub.gguf.commit ?? '') && hub.gguf.recommended, JSON.stringify(hub).slice(0, 600));
  if (mlx) {
    check('Rust accepts a runnable MLX model from Hugging Face, and forgets it', hub.mlx?.ok && hub.mlx.layers === 28 && hub.mlx.downloaded === false && /^hf-/.test(hub.mlx.id ?? ''), JSON.stringify(hub.mlx));
  }
  const sm = result.sample ?? {};
  check('a sample knowledge base arrives with its bundled files', !sm.error && sm.kind === 'mixed' && sm.sources?.length === 7 && sm.sources.every(([, , st]) => st === 'pending'), JSON.stringify(sm));
  if (a.turn?.agent) {
    console.log(`[e2e] agent actions: ${a.turn.agent.map((s) => `${s.action}${s.confidence != null ? `@${Math.round(s.confidence * 100)}%` : ''}${s.note ? ' (fallback)' : ''}`).join(' → ')}`);
  }
}

// ── performance (docs/performance.md) ─────────────────────────────────────
// Only the default model's numbers mean anything; a failed run has none.
if (result?.perf && model === 'qwen3-0.6b' && !failure) {
  const p = result.perf;
  const lower = (value, unit, slack) => ({ value, unit, better: 'lower', tolerance: 1.5, slack });
  const metrics = {
    'ingest-ms': lower(p.ingestMs, 'ms', 1000),
    'search-ms': lower(p.searchMs, 'ms', 150),
    'first-token-ms': lower(p.firstTokenMs, 'ms', 300),
    // Timed from the first token; reading the prompt is its own metric.
    'generation-tok-per-sec': { value: p.tokPerSec, unit: 'tok/s', better: 'higher', tolerance: 1.5 },
    'prompt-tok-per-sec': { value: p.promptTokPerSec, unit: 'tok/s', better: 'higher', tolerance: 1.5 },
    // Deterministic for a fixed KB and question: growth means the prompt got bigger.
    'prompt-tokens': { value: p.promptTokens, unit: 'tokens', better: 'lower', tolerance: 1.25 },
  };
  // A load that downloaded or re-verified the model says nothing about warm load
  // time, and a verified-copy load has no hash throughput; each run measures one.
  if (p.verifyMs == null) metrics['warm-load-ms'] = lower(p.loadMs, 'ms', 1000);
  else metrics['verify-mb-per-sec'] = { value: p.modelBytes / 2 ** 20 / (p.verifyMs / 1000), unit: 'MB/s', better: 'higher', tolerance: 1.5 };
  for (const [k, m] of Object.entries(metrics)) if (m.value == null || Number.isNaN(m.value)) delete metrics[k];

  const section = release ? 'e2e-release' : 'e2e-dev';
  // Keep the metric this run couldn't measure (warm load vs. verify) in the baseline.
  const optional = p.verifyMs == null ? 'verify-mb-per-sec' : 'warm-load-ms';
  const failures = compare(section, metrics, { update: updatePerf, label: section, optional: [optional] });
  check('performance within the baseline (docs/performance.md)', failures === 0, `${failures} metric(s) regressed`);
}

console.log('\n── e2e results ' + (release ? '(release)' : '(dev)') + ' ──');
for (const c of checks) console.log(`${c.ok ? '✓' : '✗'} ${c.name}${c.ok || !c.detail ? '' : `\n    ${c.detail}`}`);
const failed = checks.filter((c) => !c.ok).length;
console.log(failed ? `\n${failed} check(s) failed` : `\nall ${checks.length} checks passed`);
process.exit(failed ? 1 : 0);
