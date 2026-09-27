// Appearance: light or dark. The theme is written to <html data-theme>
// (tokens.css keys off it) and mirrored to the native window so the native
// title bar matches.
//
// There is no "follow the system" mode: setting the window's theme overrides
// its appearance, and WKWebView's prefers-color-scheme then reports the
// window's forced value, not the OS's, so a system mode stuck on whatever was
// picked last. The OS preference only picks the first-run default, read
// before anything sets the window's theme.

import { isTauri } from '@tauri-apps/api/core';
import { create } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';

export type Theme = 'light' | 'dark';

export const isTheme = (v: unknown): v is Theme => v === 'light' || v === 'dark';

export const otherTheme = (theme: Theme): Theme => (theme === 'light' ? 'dark' : 'light');

type ThemeState = {
  theme: Theme;
  setTheme: (theme: Theme) => void;
};

const osPrefersDark = () =>
  typeof window !== 'undefined' && window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)').matches : true;

export const useTheme = create<ThemeState>()(
  persist(
    (set) => ({
      theme: osPrefersDark() ? 'dark' : 'light',
      setTheme: (theme) => set({ theme }),
    }),
    {
      name: 'andai.theme',
      storage: createJSONStorage(() => localStorage),
      partialize: (s) => ({ theme: s.theme }),
      // Anything but a valid theme (an older format, a hand edit) keeps the default.
      merge: (persisted, current) => {
        const theme = (persisted as { theme?: unknown } | undefined)?.theme;
        return isTheme(theme) ? { ...current, theme } : current;
      },
    },
  ),
);

function apply(theme: Theme) {
  document.documentElement.dataset.theme = theme;
  if (isTauri()) {
    void import('@tauri-apps/api/window')
      .then(({ getCurrentWindow }) => getCurrentWindow().setTheme(theme))
      .catch(() => {});
  }
}

let started = false;

/** Apply now (before first paint) and keep following the store. */
export function initTheme() {
  apply(useTheme.getState().theme);
  if (started) return;
  started = true;
  useTheme.subscribe((s, prev) => {
    if (s.theme !== prev.theme) apply(s.theme);
  });
}
