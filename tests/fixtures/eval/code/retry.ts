// Retries for calls to the card gateway, which times out under load.

export const DEFAULT_ATTEMPTS = 3;
export const BASE_DELAY_MS = 250;

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Runs `fn` up to `attempts` times, waiting BASE_DELAY_MS, then twice as long
 * after each failure (exponential backoff). Rethrows the last error.
 */
export async function withRetry<T>(fn: () => Promise<T>, attempts = DEFAULT_ATTEMPTS, baseDelayMs = BASE_DELAY_MS): Promise<T> {
  let last: unknown;
  for (let i = 0; i < attempts; i++) {
    try {
      return await fn();
    } catch (e) {
      last = e;
      if (i < attempts - 1) await sleep(baseDelayMs * 2 ** i);
    }
  }
  throw last;
}
