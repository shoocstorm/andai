import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Same headers as wllama-chat/serve.json: cross-origin isolation unlocks
// SharedArrayBuffer, which wllama needs for multi-threaded inference.
const isolation = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
  'Cross-Origin-Resource-Policy': 'cross-origin',
};

export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  server: { port: 1420, strictPort: true, headers: isolation, watch: { ignored: ['**/src-tauri/**'] } },
  preview: { headers: isolation },
  optimizeDeps: { exclude: ['@wllama/wllama'] },
  build: { target: 'safari16', chunkSizeWarningLimit: 2000 },
});
