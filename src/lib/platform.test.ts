import { act, renderHook } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { detectMac, PHONE_QUERY, shortcut, usePhone } from './platform';

afterEach(() => vi.unstubAllGlobals());

describe('platform', () => {
  it('detects macOS from navigator.platform or the user agent', () => {
    expect(detectMac({ platform: 'MacIntel', userAgent: '' })).toBe(true);
    expect(detectMac({ platform: '', userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit' })).toBe(true);
    expect(detectMac({ platform: 'Win32', userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Edg/140' })).toBe(false);
    expect(detectMac(undefined)).toBe(false);
  });

  it('follows the phone media query as the window resizes or rotates', () => {
    let matches = true;
    let onChange = () => {};
    const matchMedia = vi.fn((q: string) => ({
      matches: q === PHONE_QUERY && matches,
      addEventListener: (_: string, cb: () => void) => (onChange = cb),
      removeEventListener: () => {},
    }));
    vi.stubGlobal('matchMedia', matchMedia);
    const { result } = renderHook(() => usePhone());
    expect(result.current).toBe(true);
    act(() => {
      matches = false;
      onChange();
    });
    expect(result.current).toBe(false);
    expect(matchMedia).toHaveBeenCalledWith(PHONE_QUERY);
  });

  it('labels shortcuts with the platform modifier', () => {
    expect(shortcut('B', true)).toBe('⌘B');
    expect(shortcut('B', false)).toBe('Ctrl+B');
  });
});
