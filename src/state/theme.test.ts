import { afterEach, describe, expect, it, vi } from 'vitest';
import { initTheme, otherTheme, useTheme } from './theme';

/** A fresh copy of the store module, as on app start, with the OS reporting `dark`. */
async function freshStore(dark: boolean) {
  vi.resetModules();
  vi.stubGlobal('matchMedia', (q: string) => ({ matches: dark && q.includes('dark'), addEventListener() {} }));
  return (await import('./theme')).useTheme;
}

afterEach(() => {
  vi.unstubAllGlobals();
  localStorage.removeItem('andai.theme');
});

describe('otherTheme', () => {
  it('toggles between the two themes', () => {
    expect(otherTheme('light')).toBe('dark');
    expect(otherTheme('dark')).toBe('light');
  });
});

describe('the first-run theme', () => {
  it("is the OS's when nothing was chosen yet", async () => {
    expect((await freshStore(true)).getState().theme).toBe('dark');
    expect((await freshStore(false)).getState().theme).toBe('light');
  });

  it('is the chosen one afterwards, whatever the OS says', async () => {
    localStorage.setItem('andai.theme', JSON.stringify({ state: { theme: 'light' }, version: 0 }));
    expect((await freshStore(true)).getState().theme).toBe('light');
  });

  it('ignores a stored value that is not a theme (the old system mode)', async () => {
    localStorage.setItem('andai.theme', JSON.stringify({ state: { mode: 'system' }, version: 0 }));
    expect((await freshStore(false)).getState().theme).toBe('light');
  });
});

describe('applying the theme', () => {
  it('writes data-theme on <html> and follows changes', () => {
    initTheme();
    useTheme.getState().setTheme('light');
    expect(document.documentElement.dataset.theme).toBe('light');
    useTheme.getState().setTheme('dark');
    expect(document.documentElement.dataset.theme).toBe('dark');
  });

  // Regression: a "system" mode read prefers-color-scheme, which in the app
  // reports the window's forced theme, so it stayed on the last one picked.
  it('never reads prefers-color-scheme after startup', () => {
    const matchMedia = vi.fn(() => ({ matches: true, addEventListener() {} }));
    vi.stubGlobal('matchMedia', matchMedia);
    initTheme();
    useTheme.getState().setTheme('dark');
    useTheme.getState().setTheme('light');
    expect(matchMedia).not.toHaveBeenCalled();
    expect(document.documentElement.dataset.theme).toBe('light');
  });

  it('persists the theme', () => {
    useTheme.getState().setTheme('light');
    expect(JSON.parse(localStorage.getItem('andai.theme')!).state).toEqual({ theme: 'light' });
  });
});
