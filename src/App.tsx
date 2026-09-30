import { getCurrentWebview } from '@tauri-apps/api/webview';
import { CloudUpload } from 'lucide-react';
import { AnimatePresence, motion } from 'motion/react';
import { useEffect } from 'react';
import { stopTurn } from './agent/turn';
import { inTauri } from './kb/api';
import { initCustomModels } from './llm/custom';
import { autoload } from './llm/engine';
import { CommandCenter, composerRef } from './screens/CommandCenter';
import { Knowledge } from './screens/Knowledge';
import { Logs } from './screens/Logs';
import { Persona } from './screens/Persona';
import { Settings } from './screens/Settings';
import { Tools } from './screens/Tools';
import { WorkflowDetail } from './screens/WorkflowDetail';
import { Workflows } from './screens/Workflows';
import { ScreenBoundary } from './components/ui';
import { AboutModal, Sidebar, StatusBar, Toasts, TopBar } from './shell/Shell';
import { useLayout } from './state/layout';
import { addFiles, createKb, refreshKbs, startKbEvents, useKb } from './state/kb';
import { useUi, type Route } from './state/ui';

const SCREENS: Record<Route, () => React.ReactElement> = {
  command: CommandCenter,
  workflows: Workflows,
  'workflow-detail': WorkflowDetail,
  knowledge: Knowledge,
  tools: Tools,
  persona: Persona,
  settings: Settings,
  logs: Logs,
};

/** Files dropped from Finder go to the KB on screen (Knowledge) or the one grounding chat. */
async function ingestDrop(paths: string[]) {
  const { route } = useUi.getState();
  const { kbs, selected, grounding } = useKb.getState();
  let slug = route === 'knowledge' ? selected : (grounding ?? selected);
  if (!slug) {
    const kb = kbs[0] ?? (await createKb('My Knowledge'));
    if (!kb) return;
    slug = kb.slug;
    useKb.setState({ selected: slug, grounding: useKb.getState().grounding ?? slug });
  }
  await addFiles(slug, paths);
}

export function App() {
  const route = useUi((s) => s.route);
  const dragging = useUi((s) => s.dragging);
  const target = useKb((s) => {
    const slug = route === 'knowledge' ? s.selected : (s.grounding ?? s.selected);
    return s.kbs.find((k) => k.slug === slug)?.name ?? 'a new knowledge base';
  });
  const Screen = SCREENS[route];

  useEffect(() => {
    void refreshKbs();
    void startKbEvents();
    initCustomModels(); // before autoload: the last model may be one the user added
    void autoload();
  }, []);

  useEffect(() => {
    const go = useUi.getState().go;
    const onKey = (e: KeyboardEvent) => {
      if (!e.metaKey && !e.ctrlKey) {
        if (e.key === 'Escape') stopTurn();
        return;
      }
      const map: Record<string, Route> = { '1': 'command', '2': 'workflows', '3': 'knowledge', '4': 'persona', '5': 'tools', ',': 'settings', l: 'logs', L: 'logs' };
      if (map[e.key]) {
        e.preventDefault();
        go(map[e.key]);
      } else if (e.key.toLowerCase() === 'b') {
        e.preventDefault();
        useLayout.getState().toggleNav();
      } else if (e.key.toLowerCase() === 'j') {
        e.preventDefault();
        useLayout.getState().toggleTrace();
      } else if (e.key.toLowerCase() === 'k') {
        e.preventDefault();
        go('command');
        setTimeout(() => composerRef.current?.focus(), 30);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  useEffect(() => {
    if (!inTauri) return;
    let unlisten: (() => void) | undefined;
    void getCurrentWebview()
      .onDragDropEvent((e) => {
        const p = e.payload;
        if (p.type === 'enter' || p.type === 'over') useUi.setState({ dragging: true });
        else if (p.type === 'leave') useUi.setState({ dragging: false });
        else if (p.type === 'drop') {
          useUi.setState({ dragging: false });
          if (p.paths.length) void ingestDrop(p.paths);
        }
      })
      .then((u) => (unlisten = u));
    return () => unlisten?.();
  }, []);

  return (
    <div className="app">
      <TopBar />
      <Sidebar />
      <main className="main">
        <AnimatePresence mode="wait" initial={false}>
          <motion.div
            key={route}
            style={{ position: 'absolute', inset: 0 }}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.16, ease: [0.2, 0.7, 0.2, 1] }}
          >
            <ScreenBoundary key={route}>
              <Screen />
            </ScreenBoundary>
          </motion.div>
        </AnimatePresence>
      </main>
      <StatusBar />
      <Toasts />
      <AboutModal />
      <AnimatePresence>
        {dragging && route !== 'knowledge' && (
          <motion.div className="drop-overlay" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <div>
              <CloudUpload size={44} color="var(--blue)" />
              <h2>Drop to ingest</h2>
              <div className="muted">Files will be indexed into “{target}”</div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
