// Copy wllama's wasm builds into public/ so they are served from the app's own
// origin (COEP blocks cross-origin loads, and the app must work offline).
//   public/wllama/default/wllama.wasm  — Memory64 + JSPI build (Chromium)
//   public/wllama/compat/wllama.{js,wasm} — asyncify build (WKWebView / Safari)
import { cpSync, mkdirSync } from 'node:fs';

const copies = [
  ['node_modules/@wllama/wllama/esm/wasm/wllama.wasm', 'public/wllama/default/wllama.wasm'],
  ['node_modules/@wllama/wllama-compat/wasm/wllama.js', 'public/wllama/compat/wllama.js'],
  ['node_modules/@wllama/wllama-compat/wasm/wllama.wasm', 'public/wllama/compat/wllama.wasm'],
];

for (const [from, to] of copies) {
  mkdirSync(to.slice(0, to.lastIndexOf('/')), { recursive: true });
  cpSync(from, to);
}
console.log('[copy-wllama] wasm builds copied to public/wllama/');
