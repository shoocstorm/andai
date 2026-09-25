import { describe, expect, it } from 'vitest';
import { initTheme, nextMode, resolveTheme, useTheme } from './theme';

describe('resolveTheme', () => {
  it('follows the OS in system mode', () => {
    expect(resolveTheme('system', true)).toBe('dark');
    expect(resolveTheme('system', false)).toBe('light');
  });
  it('explicit modes ignore the OS', () => {
    expect(resolveTheme('light', true)).toBe('light');
    expect(resolveTheme('dark', false)).toBe('dark');
  });
});

describe('nextMode', () => {
  it('cycles system → light → dark → system', () => {
    expect(nextMode('system')).toBe('light');
    expect(nextMode('light')).toBe('dark');
    expect(nextMode('dark')).toBe('system');
  });
});

describe('applying the theme', () => {
  it('writes data-theme on <html> and follows mode changes', () => {
    initTheme();
    useTheme.getState().setMode('light');
    expect(document.documentElement.dataset.theme).toBe('light');
    useTheme.getState().setMode('dark');
    expect(document.documentElement.dataset.theme).toBe('dark');
  });
  it('persists only the mode, not the resolved value', () => {
    useTheme.getState().setMode('light');
    expect(JSON.parse(localStorage.getItem('andai.theme')!).state).toEqual({ mode: 'light' });
  });
});
