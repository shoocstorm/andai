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
import { initTheme } from './state/theme';

initTheme();

const smoke = import.meta.env.VITE_SMOKE as string | undefined;
if (smoke && smoke !== 'e2e') {
  void runSmoke(smoke);
} else {
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
