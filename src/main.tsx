import '@fontsource/inter/400.css';
import '@fontsource/inter/500.css';
import '@fontsource/inter/600.css';
import '@fontsource/inter/700.css';
import '@fontsource/space-grotesk/500.css';
import '@fontsource/space-grotesk/600.css';
import '@fontsource/space-grotesk/700.css';
import '@fontsource/jetbrains-mono/400.css';
import '@fontsource/jetbrains-mono/500.css';
import '@fontsource/jetbrains-mono/600.css';
import '@fontsource/jetbrains-mono/700.css';
import './theme/app.css';
import './theme/screens.css';

import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { App } from './App';
import { runE2E, runSmoke } from './smoke';
import { isMac } from './lib/platform';
import { initTheme } from './state/theme';

// app.css keeps the top bar clear of the macOS traffic lights.
document.documentElement.dataset.platform = isMac ? 'mac' : 'other';
initTheme();

const smoke = import.meta.env.VITE_SMOKE as string | undefined;
if (smoke === 'bench') {
  // Engine benchmark (src/bench.ts): no UI, one model under several settings.
  void import('./bench').then((m) => m.runBench(JSON.parse(String(import.meta.env.VITE_BENCH))));
} else if (smoke && smoke !== 'e2e' && smoke !== 'eval') {
  void runSmoke(smoke);
} else {
  // The agent eval (src/eval.ts) is its own chunk, loaded only here.
  if (smoke === 'eval') void import('./eval').then((m) => m.runEval(JSON.parse(String(import.meta.env.VITE_EVAL))));
  if (smoke === 'e2e')
    void runE2E(
      String(import.meta.env.VITE_SMOKE_FILES ?? '').split(',').filter(Boolean),
      (import.meta.env.VITE_SMOKE_MODEL as string | undefined) ?? undefined,
    );
  createRoot(document.getElementById('root')!).render(
    <StrictMode>
      <App />
    </StrictMode>,
  );
}
