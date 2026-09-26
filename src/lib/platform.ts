// Andai ships for macOS and Windows. The only UI differences are the shortcut
// modifier (⌘ vs Ctrl) and room for the macOS traffic lights in the top bar.
// Shortcuts themselves accept either key on every platform (App.tsx).

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
