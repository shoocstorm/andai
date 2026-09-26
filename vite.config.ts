import { readFileSync } from 'node:fs';
import { defineConfig, type Plugin } from 'vite';
import react from '@vitejs/plugin-react';

// Same headers as wllama-chat/serve.json: cross-origin isolation unlocks
// SharedArrayBuffer, which wllama needs for multi-threaded inference.
const isolation = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
  'Cross-Origin-Resource-Policy': 'cross-origin',
};

// The CSP lives in tauri.conf.json (AGENTS.md §9); dev and preview send the
// same policy so a violation shows up before release. Dev alone adds what
// Vite needs: the React Refresh inline preamble and the HMR socket.
const csp: Record<string, string> = JSON.parse(readFileSync('src-tauri/tauri.conf.json', 'utf8')).app.security.csp ?? {};
const policy = (extra: Record<string, string> = {}) =>
  Object.entries(csp)
    .map(([k, v]) => `${k} ${v}${extra[k] ? ` ${extra[k]}` : ''}`)
    .join('; ');
const devCsp = policy({ 'script-src': "'unsafe-inline'", 'connect-src': 'ws://localhost:1420' });

// Vite answers a revalidated index.html with a bare 304, and WebKit then keeps
// the cached copy's headers: a webview that cached the page before a header
// changed never sees the new CSP (measured: `new Function` still ran). Always
// send documents in full so their security headers are current.
const freshDocuments: Plugin = {
  name: 'andai-fresh-documents',
  configureServer(server) {
    server.middlewares.use((req, _res, next) => {
      if (req.headers.accept?.includes('text/html')) delete req.headers['if-none-match'];
      next();
    });
  },
};

export default defineConfig({
  plugins: [react(), freshDocuments],
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
    headers: { ...isolation, 'Content-Security-Policy': devCsp },
    watch: { ignored: ['**/src-tauri/**'] },
  },
  preview: { headers: { ...isolation, 'Content-Security-Policy': policy() } },
  optimizeDeps: { exclude: ['@wllama/wllama'] },
  build: { target: 'safari16', chunkSizeWarningLimit: 2000 },
});
