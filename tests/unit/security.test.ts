// Guards the security invariants in AGENTS.md §9. Each one is a line that,
// once crossed, silently breaks "nothing leaves the machine" (§1.4) or lets
// the untrusted webview reach further than it should. Loosening an allowlist
// here needs the same product decision as the change it allows.
import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';

const ROOT = join(__dirname, '../..');
const read = (p: string) => readFileSync(join(ROOT, p), 'utf8');
const conf = JSON.parse(read('src-tauri/tauri.conf.json'));
const cap = JSON.parse(read('src-tauri/capabilities/default.json'));
const csp: Record<string, string> = conf.app.security.csp ?? {};
const sources = (d: string) => new Set((csp[d] ?? '').split(/\s+/).filter(Boolean));

/** Every host the webview may talk to. Model downloads only: Hugging Face and its CDN (§2). */
const CONNECT_ALLOWLIST = ["'self'", 'ipc:', 'http://ipc.localhost', 'https://huggingface.co', 'https://*.hf.co'];
/** Plugin permissions the webview holds besides app commands (`allow-*`). */
const PERMISSION_ALLOWLIST = ['core:default', 'core:window:allow-set-theme', 'core:window:allow-start-dragging'];
/** The only files outside the harness that may name a remote URL. */
const URL_ALLOWLIST = ['llm/models.ts', 'lib/firebase.ts'];
/**
 * Files that may call a network API. GGUF models download through wllama's
 * ModelManager; Laya and MLX chat checkpoints can't, so llm/laya.ts fetches
 * them itself, only from pinned Hugging Face commit URLs (models.ts
 * `pinnedFileUrl`), and Rust verifies every file's sha256 before keeping it
 * (laya/store.rs).
 */
const NETWORK_ALLOWLIST = ['llm/laya.ts'];
/** The e2e harness names example.com to prove it gets blocked. */
const HARNESS = ['smoke.ts'];

const appFiles = readdirSync(join(ROOT, 'src'), { recursive: true })
  .map(String)
  .filter((f) => /\.tsx?$/.test(f) && !f.includes('.test.') && !HARNESS.includes(f));

describe('content security policy', () => {
  it('is set, and defaults to same-origin only', () => {
    expect(Object.keys(csp).length).toBeGreaterThan(0);
    expect(csp['default-src']).toBe("'self'");
    for (const d of ['object-src', 'frame-src', 'base-uri', 'form-action']) expect(csp[d], d).toBe("'none'");
  });

  it('never allows eval or inline scripts (wasm compile is the one exception)', () => {
    for (const [d, v] of Object.entries(csp)) expect(v, d).not.toContain("'unsafe-eval'");
    expect([...sources('script-src')].sort()).toEqual(["'self'", "'wasm-unsafe-eval'"]);
  });

  it('only connects to the allowlisted hosts', () => {
    expect([...sources('connect-src')].filter((s) => !CONNECT_ALLOWLIST.includes(s))).toEqual([]);
  });

  it('loads images and fonts only from the app itself, so model output cannot beacon', () => {
    for (const d of ['img-src', 'font-src', 'worker-src', 'style-src'])
      expect([...sources(d)].filter((s) => /^https?:|^\*$/.test(s)), d).toEqual([]);
  });

  it('lets Tauri add nonces everywhere except style-src (inline style attributes need it)', () => {
    expect(conf.app.security.dangerousDisableAssetCspModification).toEqual(['style-src']);
  });
});

describe('installers', () => {
  it('never download anything: the Windows installer does not fetch WebView2 (§1.4)', () => {
    // Tauri's default runs Microsoft's WebView2 bootstrapper, an outbound request.
    expect(conf.bundle.windows?.webviewInstallMode?.type).toBe('skip');
  });
});

describe('webview privileges', () => {
  it('holds no plugin permissions beyond the allowlist (no fs, shell, http, opener or JS dialog)', () => {
    const plugin = (cap.permissions as string[]).filter((p) => !p.startsWith('allow-'));
    expect(plugin.filter((p) => !PERMISSION_ALLOWLIST.includes(p))).toEqual([]);
  });

  it('grants the IPC bridge to exactly one remote origin, the release UI', () => {
    expect([...cap.remote.urls].sort()).toEqual(['http://localhost:14230', 'http://localhost:14230/*']);
  });

  it('gates the test-harness commands behind the launch environment', () => {
    const lib = read('src-tauri/src/lib.rs');
    for (const cmd of ['dev_log', 'dev_exit']) {
      const body = lib.match(new RegExp(`fn ${cmd}\\([^)]*\\)[^{]*\\{([\\s\\S]*?)\\n\\}`))?.[1] ?? '';
      expect(body, cmd).toContain('smoke_enabled()?');
    }
  });

  it('keeps the webview on the app origin', () => {
    const lib = read('src-tauri/src/lib.rs');
    expect(lib).toContain('.on_navigation(');
    expect(lib).toContain('NewWindowResponse::Deny');
  });
});

describe('egress (AGENTS.md §1.4)', () => {
  it('app code has no network API besides the model download in llm/', () => {
    const hits = appFiles.filter((f) => !NETWORK_ALLOWLIST.includes(f) && /\bfetch\(|XMLHttpRequest|WebSocket|sendBeacon|EventSource/.test(read(`src/${f}`)));
    expect(hits).toEqual([]);
  });

  it('checkpoint downloads fetch only pinned catalog URLs', () => {
    const src = read('src/llm/laya.ts');
    const calls = [...src.matchAll(/\bfetch\(([^,)]*)/g)].map((m) => m[1].trim());
    expect(calls).toEqual(['pinnedFileUrl(c.repo']);
    expect(src).not.toMatch(/XMLHttpRequest|WebSocket|sendBeacon|EventSource/);
  });

  it('only the model catalog names a remote URL', () => {
    const hits = appFiles.filter((f) => !URL_ALLOWLIST.includes(f) && /https?:\/\/(?!localhost[:/])/.test(read(`src/${f}`)));
    expect(hits).toEqual([]);
  });

  it('never renders raw HTML', () => {
    expect(appFiles.filter((f) => /dangerouslySetInnerHTML|rehype-raw/.test(read(`src/${f}`)))).toEqual([]);
    expect(read('package.json')).not.toContain('rehype-raw');
  });

  it('the Rust side has no HTTP client', () => {
    const deps = read('src-tauri/Cargo.toml');
    for (const crate of ['reqwest', 'ureq', 'hyper', 'isahc', 'surf', 'attohttpc', 'tauri-plugin-http', 'tauri-plugin-upload'])
      expect(deps, crate).not.toMatch(new RegExp(`^${crate}\\s*=`, 'm'));
  });

  it('the Laya tokenizer brings no HTTP client (tokenizers without its `http` feature)', () => {
    // tokenizers' `http` feature downloads from the Hub through hf-hub + ureq.
    // (reqwest and hyper are in the lockfile only as an optional Tauri feature
    // outside the build graph: `cargo tree -e normal -i reqwest` prints nothing.)
    const lock = read('src-tauri/Cargo.lock');
    for (const crate of ['hf-hub', 'ureq']) expect(lock, crate).not.toMatch(new RegExp(`^name = "${crate}"$`, 'm'));
    expect(read('src-tauri/Cargo.toml')).toMatch(/^tokenizers = \{[^}]*default-features = false/m);
  });
});
