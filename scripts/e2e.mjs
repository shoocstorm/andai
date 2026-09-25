#!/usr/bin/env node
// End-to-end test of the real app: launches Andai with VITE_SMOKE=e2e, which
// (inside the actual WKWebView) creates a knowledge base, ingests the fixtures
// through ug, loads the model, runs one grounded agent turn and prints a
// RESULT line. This script asserts on it.
//
//   npm run test:e2e              # dev build (tauri dev)
//   npm run test:e2e -- --release # release binary: http://localhost origin + ACL
//   E2E_MODEL=stories-260k ...    # engine plumbing only (answer not checked)
//
// Needs: ug on PATH (or ~/.local/bin), network on first run for the model.
import { spawn, spawnSync } from 'node:child_process';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

const root = resolve(import.meta.dirname, '..');
const release = process.argv.includes('--release');
const model = process.env.E2E_MODEL ?? 'qwen3-0.6b';
const timeoutMs = Number(process.env.E2E_TIMEOUT_MS ?? 900_000);
const fixtures = ['tests/fixtures/wllama-notes.md', 'tests/fixtures/hello.pdf'].map((f) => resolve(root, f));
// Knowledge-base files go to a throwaway dir, never the user's real ones.
const dataDir = mkdtempSync(join(tmpdir(), 'andai-e2e-'));
const env = {
  ...process.env,
  VITE_SMOKE: 'e2e',
  VITE_SMOKE_FILES: fixtures.join(','),
  VITE_SMOKE_MODEL: model,
  ANDAI_DATA_DIR: dataDir,
};

function launch() {
  if (!release) return spawn('npx', ['tauri', 'dev'], { cwd: root, env, detached: true });
  console.log('[e2e] building release binary with the e2e harness…');
  const b = spawnSync('npx', ['tauri', 'build', '--no-bundle'], { cwd: root, env, stdio: 'inherit' });
  if (b.status !== 0) process.exit(b.status ?? 1);
  // Minimal environment, like a Finder launch: proves ug is found without the shell PATH.
  return spawn(resolve(root, 'src-tauri/target/release/andai'), [], {
    cwd: root,
    env: { HOME: process.env.HOME, PATH: '/usr/bin:/bin', ANDAI_DATA_DIR: dataDir },
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

// ── assertions ─────────────────────────────────────────────────────────────
const checks = [];
const check = (name, ok, detail = '') => checks.push({ name, ok: !!ok, detail });

if (failure || !result) {
  check('harness completed', false, failure ?? 'no RESULT line');
} else {
  const { caps, engine, kb, steps, sources, stats, answer } = result;
  check('webview is cross-origin isolated', caps.isolated && caps.sharedArrayBuffer);
  check('wllama runs multi-threaded', /multi/.test(engine?.threads ?? ''), engine?.threads);
  check('knowledge base indexed by ug', kb.status === 'ready' && kb.nodes > 0 && !kb.error, JSON.stringify(kb));
  check('every source indexed', kb.sources.length === fixtures.length && kb.sources.every((s) => s.status === 'indexed'));
  check('ug progress streamed to the UI', result.ugLogLines > 5, `${result.ugLogLines} lines`);
  check('all four agent steps completed', steps?.length === 4 && steps.every((s) => s.status === 'done'), JSON.stringify(steps));
  check('retrieval returned the notes', sources?.includes('wllama-notes.md'), JSON.stringify(sources));
  check('generation produced tokens', stats?.tokens > 0 && stats?.tokPerSec > 1, JSON.stringify(stats));
  if (model !== 'stories-260k') {
    check('answer is grounded in the retrieved passage', /cross-origin|coop|coep/i.test(answer ?? ''), answer?.slice(0, 200));
  }
}

console.log('\n── e2e results ' + (release ? '(release)' : '(dev)') + ' ──');
for (const c of checks) console.log(`${c.ok ? '✓' : '✗'} ${c.name}${c.ok || !c.detail ? '' : `\n    ${c.detail}`}`);
const failed = checks.filter((c) => !c.ok).length;
console.log(failed ? `\n${failed} check(s) failed` : `\nall ${checks.length} checks passed`);
process.exit(failed ? 1 : 0);
