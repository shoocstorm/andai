// Appearance: system / light / dark. The resolved theme is written to
// <html data-theme> (tokens.css keys off it) and mirrored to the native
// window so the macOS title bar and traffic lights match.

import { isTauri } from '@tauri-apps/api/core';
import { create } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';

export type ThemeMode = 'system' | 'light' | 'dark';
export type Theme = 'light' | 'dark';

export const THEME_MODES: ThemeMode[] = ['system', 'light', 'dark'];

export function resolveTheme(mode: ThemeMode, prefersDark: boolean): Theme {
  if (mode === 'system') return prefersDark ? 'dark' : 'light';
  return mode;
}

/** Order the top-bar button steps through. */
export function nextMode(mode: ThemeMode): ThemeMode {
  return THEME_MODES[(THEME_MODES.indexOf(mode) + 1) % THEME_MODES.length];
}

type ThemeState = {
  mode: ThemeMode;
  /** What is actually on screen. */
  resolved: Theme;
  setMode: (mode: ThemeMode) => void;
};

const media = () =>
  typeof window !== 'undefined' && window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null;
const prefersDark = () => media()?.matches ?? true;

export const useTheme = create<ThemeState>()(
  persist(
    (set) => ({
      mode: 'system',
      resolved: resolveTheme('system', prefersDark()),
      setMode: (mode) => set({ mode, resolved: resolveTheme(mode, prefersDark()) }),
    }),
    {
      name: 'andai.theme',
      storage: createJSONStorage(() => localStorage),
      partialize: (s) => ({ mode: s.mode }),
      onRehydrateStorage: () => (state) => {
        if (state) state.resolved = resolveTheme(state.mode, prefersDark());
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

/** Apply now (before first paint) and keep following the store and the OS. */
export function initTheme() {
  apply(useTheme.getState().resolved);
  if (started) return;
  started = true;
  useTheme.subscribe((s, prev) => {
    if (s.resolved !== prev.resolved) apply(s.resolved);
  });
  media()?.addEventListener('change', (e) => {
    const { mode } = useTheme.getState();
    if (mode === 'system') useTheme.setState({ resolved: e.matches ? 'dark' : 'light' });
  });
}
