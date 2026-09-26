#!/usr/bin/env bun
// Performance baselines (docs/performance.md). Measures, then compares with
// perf/baseline.json and fails on a regression past a metric's tolerance.
//
//   bun run perf                  # bundle + micro
//   bun run perf bundle           # build output sizes only (what CI runs)
//   bun run perf micro            # hot-path benchmarks only
//   bun run perf --update         # record the measurement as the new baseline
//   bun run perf bundle --no-build   # reuse an existing dist/
//
// The e2e baselines (model load, retrieval, first token, tok/s) are checked
// by `bun run test:e2e`, which is the only place the real engine runs.
import { spawnSync } from 'node:child_process';
import { mkdtempSync, readdirSync, readFileSync, rmSync, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { extname, join, resolve } from 'node:path';
import { compare } from './perf-lib.mjs';

const root = resolve(import.meta.dirname, '..');
const args = process.argv.slice(2);
const update = args.includes('--update');
const suites = args.filter((a) => !a.startsWith('--'));
const run = (s) => suites.length === 0 || suites.includes(s);

let failed = 0;

if (run('bundle')) {
  if (!args.includes('--no-build')) {
    const b = spawnSync('bun', ['run', 'build'], { cwd: root, stdio: 'inherit' });
    if (b.status !== 0) process.exit(b.status ?? 1);
  }
  failed += compare('bundle', bundleMetrics(join(root, 'dist')), { update, machineBound: false, label: 'bundle (dist/)' });
}

if (run('micro')) {
  const out = mkdtempSync(join(tmpdir(), 'andai-perf-'));
  const v = spawnSync(join(root, 'node_modules/.bin/vitest'), ['run', '--config', 'vitest.perf.config.ts'], {
    cwd: root,
    stdio: 'inherit',
    env: { ...process.env, PERF_OUT: out },
  });
  if (v.status !== 0) process.exit(v.status ?? 1);
  const metrics = {};
  for (const f of readdirSync(out)) Object.assign(metrics, JSON.parse(readFileSync(join(out, f), 'utf8')));
  rmSync(out, { recursive: true, force: true });
  failed += compare('micro', metrics, { update, label: 'micro-benchmarks (tests/perf/)' });
}

console.log(update ? '\nbaseline updated — review the diff of perf/baseline.json before committing' : failed ? `\n${failed} perf regression(s)` : '\nperf ok');
process.exit(failed ? 1 : 0);

/**
 * Raw bytes, not gzip: the release UI is served from a loopback origin, so
 * what costs time is parsing and compiling, which scales with raw size.
 */
function bundleMetrics(dist) {
  const files = walk(dist);
  const sum = (pred) => files.filter(pred).reduce((n, f) => n + f.size, 0);
  const inAssets = (f) => f.rel.startsWith('assets/');
  const size = (value) => ({ value, unit: 'bytes', better: 'lower', tolerance: 1.1 });
  return {
    'app-js': size(sum((f) => inAssets(f) && extname(f.rel) === '.js')),
    'app-css': size(sum((f) => inAssets(f) && extname(f.rel) === '.css')),
    fonts: size(sum((f) => inAssets(f) && /\.woff2?$/.test(f.rel))),
    'wllama-runtime': size(sum((f) => f.rel.startsWith('wllama/'))),
    'dist-total': size(sum(() => true)),
  };
}

function walk(dir, base = dir) {
  return readdirSync(dir).flatMap((name) => {
    const p = join(dir, name);
    const st = statSync(p);
    return st.isDirectory() ? walk(p, base) : [{ rel: p.slice(base.length + 1), size: st.size }];
  });
}
