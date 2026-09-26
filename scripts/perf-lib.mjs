// Performance baselines (docs/performance.md). One JSON file, perf/baseline.json,
// holds a section per suite: `bundle` (build output sizes, the same on every
// machine), `micro` (hot-path benchmarks) and `e2e-dev` / `e2e-release` (the
// real app). Each metric stores its measured value and how far it may drift
// before the suite fails. Timings only mean something on the machine that
// recorded them, so machine-bound sections are enforced only there.
import { readFileSync, writeFileSync } from 'node:fs';
import { cpus, totalmem } from 'node:os';
import { resolve } from 'node:path';

export const BASELINE = resolve(import.meta.dirname, '..', 'perf', 'baseline.json');

/** e.g. "Apple M5 Max · 18 cores · 128 GB". */
export const machine = () => `${cpus()[0]?.model ?? 'unknown CPU'} · ${cpus().length} cores · ${Math.round(totalmem() / 2 ** 30)} GB`;

export function readBaseline() {
  try {
    return JSON.parse(readFileSync(BASELINE, 'utf8'));
  } catch {
    return {};
  }
}

/**
 * The limit a metric must stay within: `value × tolerance + slack` when lower
 * is better, `value ÷ tolerance` when higher is better. Slack keeps tiny
 * timings (a 20 ms search) from failing on scheduler jitter.
 */
export function limit(m) {
  return m.better === 'higher' ? m.value / m.tolerance : m.value * m.tolerance + (m.slack ?? 0);
}

const within = (value, m) => (m.better === 'higher' ? value >= limit(m) : value <= limit(m));

const fmt = (v, unit) => {
  if (unit === 'bytes') return v >= 2 ** 20 ? `${(v / 2 ** 20).toFixed(2)} MB` : `${(v / 1024).toFixed(1)} KB`;
  const digits = v >= 100 ? 0 : v >= 10 ? 1 : 2;
  return `${v.toFixed(digits)} ${unit}`;
};

/**
 * Compares `measured` ({ name: { value, unit, better, tolerance, slack? } })
 * with the baseline section and prints a table. With `update`, records the
 * measurement as the new baseline instead (existing tolerances are kept, so
 * loosening one is a deliberate edit of baseline.json).
 * `optional` names metrics this run can't measure; they are neither
 * reported missing nor dropped from the baseline on update.
 * Returns the number of failures (always 0 when updating or not enforced).
 */
export function compare(section, measured, { update = false, machineBound = true, label = section, optional = [] } = {}) {
  const all = readBaseline();
  const base = all[section];
  const here = machine();
  const enforced = !update && !!base && (!machineBound || base.machine === here || process.env.PERF_ENFORCE === '1');

  console.log(`\n── perf: ${label} ──`);
  if (!base && !update) console.log(`no baseline yet for "${section}"; record one with --update (see docs/performance.md)`);
  else if (base && machineBound && base.machine !== here && !update) {
    console.log(`baseline was recorded on ${base.machine}; this is ${here}.`);
    console.log('Timings are shown but not enforced. Set PERF_ENFORCE=1 to enforce anyway.');
  }

  let failed = 0;
  const names = new Set([...Object.keys(measured), ...Object.keys(base?.metrics ?? {})]);
  for (const name of names) {
    const m = measured[name];
    const b = base?.metrics?.[name];
    if (!m && optional.includes(name)) {
      console.log(`· ${name}: not measured by this run`);
      continue;
    }
    if (!m) {
      // A metric that silently disappears would hide its regressions.
      console.log(`✗ ${name}: in the baseline but not measured`);
      if (enforced) failed++;
      continue;
    }
    if (!b) {
      console.log(`· ${name}: ${fmt(m.value, m.unit)} (new, not in the baseline)`);
      continue;
    }
    const ok = within(m.value, b);
    const delta = ((m.value - b.value) / b.value) * 100;
    const mark = !enforced ? '·' : ok ? '✓' : '✗';
    console.log(
      `${mark} ${name}: ${fmt(m.value, m.unit)}  baseline ${fmt(b.value, b.unit)} (${delta >= 0 ? '+' : ''}${delta.toFixed(1)}%)  limit ${b.better === 'higher' ? '≥' : '≤'} ${fmt(limit(b), b.unit)}`,
    );
    if (enforced && !ok) failed++;
  }

  if (update) {
    const metrics = {};
    for (const name of optional) if (!measured[name] && base?.metrics?.[name]) metrics[name] = base.metrics[name];
    for (const [name, m] of Object.entries(measured)) {
      const old = base?.metrics?.[name];
      metrics[name] = { ...m, value: round(m.value), tolerance: old?.tolerance ?? m.tolerance, ...(old?.slack != null ? { slack: old.slack } : {}) };
    }
    all[section] = {
      ...(machineBound ? { machine: here } : {}),
      recorded: new Date().toISOString().slice(0, 10),
      metrics,
    };
    writeFileSync(BASELINE, `${JSON.stringify(all, null, 2)}\n`);
    console.log(`recorded ${Object.keys(metrics).length} metrics as the "${section}" baseline in perf/baseline.json`);
  }
  return failed;
}

const round = (v) => (Math.abs(v) >= 100 ? Math.round(v) : Math.round(v * 1000) / 1000);

/**
 * A stream 'data' handler that passes on complete lines only. A chunk can end
 * mid-line, and a long harness line (a CASE or RESULT record) split across two
 * chunks was lost; the unfinished tail waits for the next chunk. Use one per
 * stream, since stdout and stderr interleave.
 */
export function byLine(onLines) {
  let rest = '';
  return (buf) => {
    const parts = (rest + buf.toString()).split('\n');
    rest = parts.pop();
    onLines(parts);
  };
}

