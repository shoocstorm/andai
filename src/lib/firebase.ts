// Google Analytics for the web deployment (Firebase Hosting, site
// `andai-agent`). Product decision, 2026-09-27, recorded in AGENTS.md §1.4:
// it runs ONLY outside the desktop app, which keeps its own promise that
// nothing leaves the machine — its CSP (tauri.conf.json) carries no Google
// hosts, and `tests/unit/firebase-hosting.test.ts` holds that line.
//
// The import from main.tsx is dynamic, so the desktop bundle never downloads
// or parses any of this. `bun run dev` served in a plain browser runs under
// the desktop CSP, which blocks gtag: the console warnings there are expected.
import { isTauri } from '@tauri-apps/api/core';

export async function initWebAnalytics(): Promise<void> {
  if (isTauri()) return;
  const { initializeApp } = await import('firebase/app');
  const { getAnalytics } = await import('firebase/analytics');

  // Firebase web configs are public client identifiers by design.
  const app = initializeApp({
    apiKey: 'AIzaSyD12U75gkRIfvFh2gHcCYJAlCqqyu8UHsI',
    authDomain: 'aldrick-ai.firebaseapp.com',
    projectId: 'aldrick-ai',
    storageBucket: 'aldrick-ai.firebasestorage.app',
    messagingSenderId: '914835400876',
    appId: '1:914835400876:web:c3953ce6d5613ead969709',
    measurementId: 'G-839EYLZRP9',
  });
  getAnalytics(app);
}
