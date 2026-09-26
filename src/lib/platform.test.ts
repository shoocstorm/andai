import { describe, expect, it } from 'vitest';
import { detectMac, shortcut } from './platform';

describe('platform', () => {
  it('detects macOS from navigator.platform or the user agent', () => {
    expect(detectMac({ platform: 'MacIntel', userAgent: '' })).toBe(true);
    expect(detectMac({ platform: '', userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit' })).toBe(true);
    expect(detectMac({ platform: 'Win32', userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Edg/140' })).toBe(false);
    expect(detectMac(undefined)).toBe(false);
  });

  it('labels shortcuts with the platform modifier', () => {
    expect(shortcut('B', true)).toBe('⌘B');
    expect(shortcut('B', false)).toBe('Ctrl+B');
  });
});
