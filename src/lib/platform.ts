// Andai ships for macOS and Windows, and its web app also runs on phones. The
// UI differences: the shortcut modifier (⌘ vs Ctrl), room for the macOS
// traffic lights in the desktop app's top bar, and the phone layout (a bottom
// tab bar instead of the sidebar). Shortcuts accept either key everywhere (App.tsx).

import { useSyncExternalStore } from 'react';

type Nav = Pick<Navigator, 'platform' | 'userAgent'>;

export function detectMac(nav: Nav | undefined): boolean {
  if (!nav) return false;
  return /Mac|iPhone|iPad/.test(nav.platform || nav.userAgent);
}

export const isMac = detectMac(typeof navigator === 'undefined' ? undefined : navigator);

/** A shortcut label: `⌘B` on macOS, `Ctrl+B` elsewhere. */
export function shortcut(key: string, mac = isMac): string {
  return mac ? `⌘${key}` : `Ctrl+${key}`;
}

/** Below this width the shell switches to the phone layout (app.css mirrors it). */
export const PHONE_QUERY = '(max-width: 640px)';

const phoneMedia = () => (typeof window !== 'undefined' && window.matchMedia ? window.matchMedia(PHONE_QUERY) : null);

/** Whether the window is phone-sized, updated as it resizes or rotates. */
export function usePhone(): boolean {
  return useSyncExternalStore(
    (cb) => {
      const m = phoneMedia();
      m?.addEventListener('change', cb);
      return () => m?.removeEventListener('change', cb);
    },
    () => phoneMedia()?.matches ?? false,
    () => false,
  );
}
