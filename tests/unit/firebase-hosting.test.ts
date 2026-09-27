// Guards the web deployment (Firebase Hosting, site `andai-agent`, AGENTS.md
// §1.4/§9). The web build is the same `dist/` the release UI is served from,
// but its CSP is NOT the desktop one: it drops the Tauri IPC sources and adds
// only the hosts Google Analytics needs (product decision, 2026-09-27). The
// desktop app must never gain those hosts, or "nothing leaves the machine"
// silently stops holding for the shipped app.
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';

const ROOT = join(__dirname, '../..');
const read = (p: string) => readFileSync(join(ROOT, p), 'utf8');
const conf = JSON.parse(read('src-tauri/tauri.conf.json'));
const hosting = JSON.parse(read('firebase.json')).hosting;

const directives = (policy: string) =>
  Object.fromEntries(
    policy
      .split(';')
      .map((d) => d.trim().split(/\s+/))
      .map(([k, ...v]) => [k, v]),
  );
const web = directives(
  hosting.headers
    .flatMap((h: any) => h.headers)
    .find((h: any) => h.key === 'Content-Security-Policy').value,
);
const header = (key: string) =>
  hosting.headers
    .flatMap((h: any) => h.headers)
    .filter((h: any) => h.key === key)
    .map((h: any) => h.value);

/** The only hosts the web app may talk to: models (Hugging Face) + Analytics. */
const WEB_CONNECT = [
  "'self'",
  'https://huggingface.co',
  'https://*.hf.co',
  'https://www.googletagmanager.com',
  'https://www.google-analytics.com',
  'https://*.google-analytics.com',
  'https://*.analytics.google.com',
];

describe('firebase hosting config', () => {
  it('serves the Vite build for the andai-agent site', () => {
    expect(hosting.site).toBe('andai-agent');
    expect(hosting.public).toBe('dist');
  });

  it('keeps cross-origin isolation, so wllama gets SharedArrayBuffer outside WebKit', () => {
    const desktop = conf.app.security.headers;
    for (const [key, value] of Object.entries(desktop)) expect(header(key), key).toEqual([value]);
  });

  it('keeps the hardening directives of the desktop CSP', () => {
    expect(web['default-src']).toEqual(["'self'"]);
    for (const d of ['object-src', 'frame-src', 'base-uri', 'form-action']) expect(web[d], d).toEqual(["'none'"]);
    expect(web['style-src']).toEqual(["'self'", "'unsafe-inline'"]);
    expect(web['worker-src']).toEqual(["'self'", 'blob:']);
    expect(web['img-src']).toEqual(["'self'", 'data:', 'blob:']);
  });

  it('connects only to the web allowlist (no Tauri IPC, no other hosts)', () => {
    expect(web['connect-src'].filter((s: string) => !WEB_CONNECT.includes(s))).toEqual([]);
    expect(web['script-src'].filter((s: string) => !["'self'", "'wasm-unsafe-eval'", 'https://www.googletagmanager.com'].includes(s))).toEqual([]);
  });
});

describe('analytics stays web-only', () => {
  it('the desktop CSP names no Google host', () => {
    const sources = Object.values(conf.app.security.csp ?? {}).join(' ');
    expect(sources).not.toContain('google');
  });

  it('initWebAnalytics refuses to run inside the desktop app', () => {
    const src = read('src/lib/firebase.ts');
    expect(src).toMatch(/if\s*\(\s*isTauri\(\)\s*\)\s*return/);
    // The firebase packages must stay lazy chunks: the desktop app never
    // downloads them, and `bun run perf` keeps them out of the main bundle.
    expect(src).toContain("await import('firebase/app')");
    expect(src).not.toMatch(/from 'firebase/);
  });
});

describe('caching keeps the CSP current', () => {
  // Firebase Hosting must not serve a cached document with stale headers
  // (AGENTS.md §2: WebKit keeps cached headers on a 304).
  it('index.html is revalidated, hashed assets are immutable', () => {
    const forSource = (source: string) =>
      hosting.headers.find((h: any) => h.source === source)?.headers ?? [];
    expect(forSource('/index.html')).toContainEqual(
      expect.objectContaining({ key: 'Cache-Control', value: expect.stringContaining('no-cache') }),
    );
    expect(forSource('/assets/**')).toContainEqual(
      expect.objectContaining({ key: 'Cache-Control', value: expect.stringContaining('immutable') }),
    );
  });
});
