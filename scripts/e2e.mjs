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
//
// Needs: ug on PATH (or ~/.local/bin), network on first run for the model.
import { spawn, spawnSync } from 'node:child_process';
import { mkdtempSync, rmSync } from 'node:fs';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { compare } from './perf-lib.mjs';

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
const env = {
  ...process.env,
  VITE_SMOKE: 'e2e',
  VITE_SMOKE_FILES: fixtures.join(','),
  VITE_SMOKE_MODEL: model,
  VITE_SMOKE_CANARY: canaryUrl,
  ANDAI_DATA_DIR: dataDir,
  // Harness-only switches, read by Rust at startup (AGENTS.md §9): enable
  // dev_log/dev_exit, and grant the fixtures the way a drop would.
  ANDAI_SMOKE: '1',
  ANDAI_E2E_FILES: fixtures.join(','),
};

function launch() {
  if (!release) return spawn('bun', ['run', 'tauri', 'dev'], { cwd: root, env, detached: true });
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
  const onData = (buf) => {
    for (const line of buf.toString().split('\n')) {
      if (!line.includes('[webview]')) continue;
      const text = line.slice(line.indexOf('[webview]') + 10);
      lines.push(text);
      console.log(`[e2e] ${text.slice(0, 220)}`);
      if (text.startsWith('RESULT ')) result = JSON.parse(text.slice(7));
      if (text.startsWith('FAIL')) failure = text;
      if (text.startsWith('OK') || text.startsWith('FAIL')) resolveDone();
    }
  };
  child.stdout.on('data', onData);
  child.stderr.on('data', onData);
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
  check('knowledge base indexed by ug', kb.status === 'ready' && kb.nodes > 0 && !kb.error, JSON.stringify(kb));
  check('every source indexed', kb.sources.length === fixtures.length && kb.sources.every((s) => s.status === 'indexed'));
  check('ug progress streamed to the UI', result.ugLogLines > 5, `${result.ugLogLines} lines`);
  check('all four agent steps completed', steps?.length === 4 && steps.every((s) => s.status === 'done'), JSON.stringify(steps));
  check('retrieval returned the notes', sources?.includes('wllama-notes.md'), JSON.stringify(sources));
  console.log(`[e2e] model sha256 check: ${result.verifyMs == null ? 'reused a verified copy' : `${Math.round(result.verifyMs)} ms`}`);
  check('generation produced tokens', stats?.tokens > 0 && stats?.tokPerSec > 1, JSON.stringify(stats));
  if (model !== 'stories-260k') {
    check('answer is grounded in the retrieved passage', /cross-origin|coop|coep/i.test(answer ?? ''), answer?.slice(0, 200));
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
    'generation-tok-per-sec': { value: p.tokPerSec, unit: 'tok/s', better: 'higher', tolerance: 1.5 },
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
