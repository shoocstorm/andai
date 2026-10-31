import { describe, expect, it } from 'vitest';
import { crashMessage, describeLoadError, detectDevice, loadPlan, MAX_LEVEL, planText, startLevel } from './device';

const IPHONE = 'Mozilla/5.0 (iPhone; CPU iPhone OS 18_7 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.5 Mobile/15E148 Safari/604.1';
const MAC = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.0 Safari/605.1.15';
const ANDROID = 'Mozilla/5.0 (Linux; Android 15; Pixel 9) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 Mobile Safari/537.36';
const phone = { mobile: true, ios: true };
const desktop = { mobile: false, ios: false };

describe('detectDevice', () => {
  it('tells phones and tablets from computers', () => {
    expect(detectDevice({ userAgent: IPHONE })).toEqual({ mobile: true, ios: true });
    expect(detectDevice({ userAgent: ANDROID })).toEqual({ mobile: true, ios: false });
    expect(detectDevice({ userAgent: MAC, maxTouchPoints: 0 })).toEqual(desktop);
    // iPadOS reports a Mac user agent; its touch screen gives it away
    expect(detectDevice({ userAgent: MAC, maxTouchPoints: 5 })).toEqual({ mobile: true, ios: true });
    expect(detectDevice(undefined)).toEqual(desktop);
  });
});

describe('load plans', () => {
  it('asks for less at each level, never more than the catalog', () => {
    expect(loadPlan(4096, 0)).toEqual({ level: 0, n_ctx: 4096, gpu: true });
    expect(loadPlan(4096, 1)).toEqual({ level: 1, n_ctx: 2048, gpu: false });
    expect(loadPlan(4096, 2)).toEqual({ level: MAX_LEVEL, n_ctx: 1024, gpu: false });
    expect(loadPlan(4096, 9)).toEqual(loadPlan(4096, MAX_LEVEL));
    expect(loadPlan(1024, 1).n_ctx).toBe(1024);
  });

  it('starts phones lighter, and below any level that failed here', () => {
    expect(startLevel(desktop, undefined)).toBe(0);
    expect(startLevel(phone, undefined)).toBe(1);
    expect(startLevel(desktop, 2)).toBe(2);
    expect(startLevel(phone, 0)).toBe(1);
    expect(startLevel(phone, 7)).toBe(MAX_LEVEL);
    expect(planText(loadPlan(4096, 1))).toBe('CPU only, 2,048-token context');
  });
});

describe('describeLoadError', () => {
  const plan = loadPlan(4096, 1);
  it('says what ran out and what to do', () => {
    for (const raw of ['RangeError: Out of memory', 'std::bad_alloc', 'failed to allocate CPU buffer of size 469762048', 'Aborted(OOM)']) {
      const f = describeLoadError(raw, 'Qwen3 0.6B', plan, phone);
      expect(f.retryLighter).toBe(true);
      expect(f.message).toBe('Qwen3 0.6B needs more memory than the browser gives this page on this iPhone (tried CPU only, 2,048-token context). Close other tabs and apps and try again, or pick a smaller model.');
    }
  });

  it('tells storage, network and integrity failures apart, and does not retry them lighter', () => {
    expect(describeLoadError('QuotaExceededError: The quota has been exceeded.', 'M', plan, phone)).toMatchObject({ retryLighter: false, message: expect.stringMatching(/storage space.*this iPhone/) });
    expect(describeLoadError('TypeError: Load failed', 'M', plan, phone)).toMatchObject({ retryLighter: false, message: expect.stringMatching(/couldn't be downloaded/) });
    const integrity = 'The downloaded M failed its integrity check (sha256 mismatch), so it was removed and not loaded.';
    expect(describeLoadError(integrity, 'M', null, desktop)).toEqual({ message: integrity, retryLighter: false });
  });

  it('retries a GPU failure on the CPU, and keeps an unknown error readable', () => {
    expect(describeLoadError('GPUDevice lost', 'M', loadPlan(4096, 0), desktop).retryLighter).toBe(true);
    expect(describeLoadError('timed out warming up', 'M', plan, desktop).message).toMatch(/^M didn't finish warming up in 3 minutes on this computer/);
    expect(describeLoadError('something odd', 'M', plan, desktop)).toEqual({ message: "M couldn't be loaded: something odd", retryLighter: true });
  });
});

describe('crashMessage', () => {
  it('names the cause and the next, lighter try', () => {
    expect(crashMessage('Qwen3 0.6B', loadPlan(4096, 1), loadPlan(4096, 2), phone)).toBe(
      'iOS closed the page while Qwen3 0.6B was warming up (CPU only, 2,048-token context), most likely because it ran out of memory. The next try uses less memory (CPU only, 1,024-token context). Close other tabs and apps first.',
    );
    expect(crashMessage('M', loadPlan(4096, 2), null, desktop)).toMatch(/^The page was closed .* smallest setting .* can't run M in the browser/);
  });
});
