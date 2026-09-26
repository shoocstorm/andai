// Micro-benchmarks (tests/perf/*.perf.ts), run by `bun run perf`, never by
// `bun run test`: timings belong to the machine, not the test suite.
import { defineConfig, mergeConfig } from 'vitest/config';
import viteConfig from './vite.config.ts';

export default mergeConfig(
  viteConfig,
  defineConfig({
    test: {
      environment: 'jsdom',
      setupFiles: ['./tests/setup.ts'],
      include: ['tests/perf/**/*.perf.{ts,tsx}'],
      // one file at a time, so benchmarks don't compete for cores
      fileParallelism: false,
      testTimeout: 120_000,
    },
  }),
);
