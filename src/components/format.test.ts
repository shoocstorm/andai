import { afterEach, describe, expect, it, vi } from 'vitest';
import { fmtAgo, fmtBytes } from './ui';

describe('fmtBytes', () => {
  it.each([
    [512, '512 B'],
    [2048, '2.0 KB'],
    [639_446_688, '609.8 MB'],
    [2 * 1024 ** 3, '2.00 GB'],
  ])('%d → %s', (n, s) => expect(fmtBytes(n)).toBe(s));
});

describe('fmtAgo', () => {
  afterEach(() => vi.useRealTimers());
  it('renders relative times from unix seconds', () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-09-25T12:00:00Z'));
    const now = Date.now() / 1000;
    expect(fmtAgo(null)).toBe('—');
    expect(fmtAgo(now - 10)).toBe('Just now');
    expect(fmtAgo(now - 12 * 60)).toBe('12m ago');
    expect(fmtAgo(now - 3 * 3600)).toBe('3h ago');
    expect(fmtAgo(now - 2 * 86400)).toBe('2d ago');
  });
});
