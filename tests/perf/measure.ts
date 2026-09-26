// Micro-benchmark helpers for tests/perf/*.perf.ts (docs/performance.md).
// These files measure; scripts/perf.mjs compares the numbers with
// perf/baseline.json. Run them with `bun run perf`, not `bun run test`.
import { mkdirSync, writeFileSync } from 'node:fs';
import { basename, join } from 'node:path';
import { afterAll } from 'vitest';

export type Metric = { value: number; unit: string; better: 'lower' | 'higher'; tolerance: number; slack?: number };

/** Micro timings in jsdom vary by ~10–20% run to run; 1.5× catches real regressions, not noise. */
export const MICRO_TOLERANCE = 1.5;

const results: Record<string, Metric> = {};

/**
 * Median milliseconds per call of `fn`. Warms up first (JIT), then sizes each
 * sample to at least `sampleMs` so timer resolution doesn't matter, and takes
 * the median of `samples` so one GC pause doesn't either.
 */
export function msPerOp(fn: () => unknown, { samples = 9, sampleMs = 40 } = {}): number {
  for (let i = 0; i < 3; i++) fn();
  let iters = 1;
  for (;;) {
    const t = performance.now();
    for (let i = 0; i < iters; i++) fn();
    const ms = performance.now() - t;
    if (ms >= sampleMs) break;
    iters = Math.max(iters * 2, Math.ceil((iters * sampleMs) / Math.max(ms, 0.01)));
  }
  const times: number[] = [];
  for (let s = 0; s < samples; s++) {
    const t = performance.now();
    for (let i = 0; i < iters; i++) fn();
    times.push((performance.now() - t) / iters);
  }
  times.sort((a, b) => a - b);
  return times[Math.floor(times.length / 2)];
}

export function record(name: string, metric: Omit<Metric, 'tolerance'> & { tolerance?: number }) {
  results[name] = { tolerance: MICRO_TOLERANCE, ...metric };
}

/** Call once per perf file: writes its metrics where scripts/perf.mjs collects them. */
export function reportTo(file: string) {
  afterAll(() => {
    const dir = process.env.PERF_OUT;
    if (!dir) return;
    mkdirSync(dir, { recursive: true });
    writeFileSync(join(dir, `${basename(file)}.json`), JSON.stringify(results));
  });
}
