#!/usr/bin/env bun
// Agent eval (docs/agentic-rag-improvements.md, item 1): launches Andai with
// VITE_SMOKE=eval, which (in the real webview) builds knowledge bases from
// tests/fixtures/eval/, runs every question in cases.json through the agent,
// and prints a CASE line per question. This script scores them, writes a JSON
// report and prints the scorecard.
//
//   bun run eval:agent                          # all questions, compared with perf/baseline.json
//   bun run eval:agent --update                 # record this run as the "agent-eval" baseline
//   bun run eval:agent --against .eval/a.json   # also list the questions whose outcome changed
//   bun run eval:agent --only doc-wind,code-peak
//   EVAL_MODEL=qwen3-1.7b EVAL_SEED=7 …
//
// Needs ug and the model (downloaded once), like the e2e run. Machine- and
// model-bound, and not part of `bun run check`.
import { spawn } from 'node:child_process';
import { copyFileSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { diffCases, loadCases, scoreCase, scorecard } from './eval-lib.mjs';
import { compare } from './perf-lib.mjs';

const root = resolve(import.meta.dirname, '..');
const argv = process.argv.slice(2);
const flag = (name) => argv.includes(name);
const value = (name) => {
  const i = argv.indexOf(name);
  return i === -1 ? undefined : argv[i + 1];
};
const model = process.env.EVAL_MODEL ?? 'qwen3-0.6b';
const seed = Number(process.env.EVAL_SEED ?? 7);
const timeoutMs = Number(process.env.EVAL_TIMEOUT_MS ?? 1_800_000);
const only = value('--only')?.split(',');

const fixtures = resolve(root, 'tests/fixtures/eval');
const { notFound, cases: allCases } = loadCases(join(fixtures, 'cases.json'));
const cases = only ? allCases.filter((c) => only.includes(c.id)) : allCases;
if (!cases.length) {
  console.error(`no cases match --only ${only}`);
  process.exit(2);
}

// A file grant is consumed when it's ingested (src-tauri/src/grants.rs), so
// every knowledge base gets its own copies. They live in a throwaway dir,
// with the app's data dir, never the user's.
const dataDir = mkdtempSync(join(tmpdir(), 'andai-eval-'));
const copies = (key, dirs) => {
  const out = join(dataDir, 'fixtures', key);
  mkdirSync(out, { recursive: true });
  return dirs.flatMap((d) =>
    readdirSync(join(fixtures, d)).map((f) => {
      copyFileSync(join(fixtures, d, f), join(out, f));
      return join(out, f);
    }),
  );
};
const kbs = { docs: copies('docs', ['docs']), code: copies('code', ['code']), mixed: copies('mixed', ['docs', 'code']) };
const used = new Set(cases.map((c) => c.kb));
for (const key of Object.keys(kbs)) if (!used.has(key)) delete kbs[key];
const files = Object.values(kbs).flat();

const env = {
  ...process.env,
  VITE_SMOKE: 'eval',
  VITE_EVAL: JSON.stringify({ model, seed, kbs, cases: cases.map(({ id, kb, prompt, history }) => ({ id, kb, prompt, history })) }),
  ANDAI_DATA_DIR: join(dataDir, 'app'),
  ANDAI_SMOKE: '1',
  ANDAI_E2E_FILES: files.join(','),
};

console.log(`[eval] ${cases.length} question(s), model ${model}, seed ${seed}`);
const child = spawn('bun', ['run', 'tauri', 'dev'], { cwd: root, env, detached: true });
const records = [];
let failure = null;
let start = null;
const byId = new Map(cases.map((c) => [c.id, c]));

await new Promise((resolveDone) => {
  const onData = (buf) => {
    for (const line of buf.toString().split('\n')) {
      if (!line.includes('[webview]')) continue;
      const text = line.slice(line.indexOf('[webview]') + 10);
      if (text.startsWith('CASE ')) {
        const r = JSON.parse(text.slice(5));
        records.push(r);
        const s = scoreCase(byId.get(r.id), r, notFound);
        const mark = (ok) => (ok === null ? '·' : ok ? '✓' : '✗');
        console.log(
          `[eval] ${String(records.length).padStart(2)}/${cases.length} ${mark(s.firstOk)} first ${mark(s.factsOk)} facts  ${s.id.padEnd(26)} ${s.actions.join(' → ')}  ${s.seconds.toFixed(1)} s${s.error ? `  ERROR ${s.error.slice(0, 120)}` : ''}`,
        );
      } else {
        console.log(`[eval] ${text.slice(0, 220)}`);
        if (text.startsWith('START ')) start = JSON.parse(text.slice(6));
        if (text.startsWith('FAIL')) failure = text;
      }
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
try {
  process.kill(-child.pid, 'SIGTERM');
} catch {}
rmSync(dataDir, { recursive: true, force: true });

if (failure || records.length !== cases.length) {
  console.error(`\n[eval] harness failed: ${failure ?? `${records.length} of ${cases.length} questions reported`}`);
  process.exit(1);
}

// ── report ─────────────────────────────────────────────────────────────────
const scored = records.map((r) => scoreCase(byId.get(r.id), r, notFound));
const card = scorecard(scored);
const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
const out = value('--out') ?? join(root, '.eval', `agent-eval-${stamp}.json`);
mkdirSync(resolve(out, '..'), { recursive: true });
writeFileSync(out, `${JSON.stringify({ at: new Date().toISOString(), model, seed, engine: start?.engine ?? null, only: only ?? null, scorecard: card, scored, records }, null, 2)}\n`);

const pct = (v) => (v == null ? '—' : `${(v * 100).toFixed(1)}%`);
const num = (v, d = 2) => (v == null ? '—' : v.toFixed(d));
console.log(`\n── agent eval · ${model} · ${card.questions} questions ──`);
console.log(`first-action accuracy   ${pct(card.firstActionAccuracy)}`);
console.log(`answer-fact hit rate    ${pct(card.factHitRate)}`);
console.log(`grounded answers        ${pct(card.groundedRate)}  (cites at least one source, and only listed ones)`);
console.log(`tool calls              ${card.calls}  wasted ${card.wastedCalls} (${num(card.wastedPerQuestion)}/question: errors, empty, skipped)`);
console.log(`invalid arguments       ${card.argsInvalid}   "No symbol named" ${card.noSymbolErrors}   fallbacks ${card.fallbacks}   turn errors ${card.errors}`);
console.log(`decisions / question    ${num(card.decisionsPerQuestion)}   ${num(card.msPerDecision, 0)} ms and ${num(card.promptTokensPerDecision, 0)} prompt tokens each`);
console.log(`seconds / question      ${num(card.secondsPerQuestion)}`);
const misses = scored.filter((s) => !s.firstOk || s.factsOk === false);
if (misses.length) {
  console.log('\nmissed:');
  for (const s of misses) {
    console.log(`  ${s.id}: ${!s.firstOk ? `first ${s.first}, expected ${byId.get(s.id).first.join('|')}` : ''}${!s.firstOk && s.factsOk === false ? '; ' : ''}${s.factsOk === false ? `missing ${s.missing.join(', ')}` : ''}`);
  }
}
console.log(`\nreport: ${out}`);

const against = value('--against');
if (against) {
  const changed = diffCases(JSON.parse(readFileSync(against, 'utf8')).scored, scored);
  console.log(`\nchanged against ${against}: ${changed.length ? '' : 'none'}`);
  for (const c of changed) console.log(`  ${c.id}: ${c.change}`);
}

// ── baseline (docs/performance.md) ────────────────────────────────────────
// Only a full run with the default model is comparable to the baseline.
if (only || model !== 'qwen3-0.6b') {
  console.log('\n(baseline not compared: a subset or another model)');
  process.exit(0);
}
const higher = (v, unit = '%') => ({ value: v == null ? null : v * 100, unit, better: 'higher', tolerance: 1.1 });
const lower = (v, unit, tolerance, slack) => ({ value: v, unit, better: 'lower', tolerance, slack });
const metrics = {
  'first-action-accuracy': higher(card.firstActionAccuracy),
  'fact-hit-rate': higher(card.factHitRate),
  'grounded-rate': higher(card.groundedRate),
  'wasted-calls-per-question': lower(card.wastedPerQuestion, 'calls', 1.5, 0.2),
  'decisions-per-question': lower(card.decisionsPerQuestion, 'decisions', 1.25, 0.1),
  'seconds-per-question': lower(card.secondsPerQuestion, 's', 1.5, 1),
  'ms-per-decision': lower(card.msPerDecision, 'ms', 1.5, 100),
  'prompt-tokens-per-decision': lower(card.promptTokensPerDecision, 'tokens', 1.25),
};
for (const [k, m] of Object.entries(metrics)) if (m.value == null || Number.isNaN(m.value)) delete metrics[k];
const failed = compare('agent-eval', metrics, { update: flag('--update'), label: `agent eval (${model})` });
process.exit(failed ? 1 : 0);
