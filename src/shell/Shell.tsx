import {
  BookOpen,
  PanelLeftClose,
  PanelLeftOpen,
  Monitor,
  Moon,
  Sun,
  Cpu,
  Database,
  HelpCircle,
  Radio,
  Settings,
  SquareTerminal,
  UserRoundCog,
  Workflow,
  Zap,
  CheckCircle2,
  AlertTriangle,
  Info,
  XCircle,
  X,
} from 'lucide-react';
import { AnimatePresence, motion } from 'motion/react';
import type { ReactNode } from 'react';
import { useEngine } from '../llm/engine';
import { modelById } from '../llm/models';
import { useKb } from '../state/kb';
import { usePersona } from '../state/persona';
import { useLayout } from '../state/layout';
import { nextMode, useTheme } from '../state/theme';
import { useUi, type Route } from '../state/ui';
import { Modal } from '../components/ui';

const NAV: { route: Route; label: string; icon: ReactNode; key: string }[] = [
  { route: 'command', label: 'Command Center', icon: <SquareTerminal size={19} />, key: '⌘1' },
  { route: 'workflows', label: 'Workflows', icon: <Workflow size={19} />, key: '⌘2' },
  { route: 'knowledge', label: 'Knowledge', icon: <Database size={19} />, key: '⌘3' },
  { route: 'persona', label: 'Persona', icon: <UserRoundCog size={19} />, key: '⌘4' },
];

export function useCoreStatus() {
  const { status, progress, loadedId, generating } = useEngine();
  const pct = progress && progress.total ? Math.round((progress.loaded / progress.total) * 100) : 0;
  if (status === 'loading') return { color: 'var(--amber)', text: `Core loading ${pct}%`, pulse: true };
  if (status === 'error') return { color: 'var(--red)', text: 'Core fault', pulse: false };
  if (loadedId && generating) return { color: 'var(--violet)', text: 'Core processing', pulse: true };
  if (loadedId) return { color: 'var(--blue)', text: 'Core online', pulse: true };
  return { color: 'var(--text-3)', text: 'Core standby', pulse: false };
}

export function TopBar() {
  const core = useCoreStatus();
  const go = useUi((s) => s.go);
  const ug = useKb((s) => s.ug);
  const engine = useEngine();
  const name = usePersona((s) => s.agentName);
  return (
    <header className="topbar" data-tauri-drag-region>
      <div className="wordmark" data-tauri-drag-region>
        <span className="a">ANDAI</span>
        <span className="b">AGENT&nbsp;OS</span>
      </div>
      <span className="divider" />
      <div className="core-status" style={{ color: core.color }} data-tauri-drag-region>
        <span className={`dot${core.pulse ? ' pulse' : ''}`} />
        {core.text}
      </div>
      <div className="spacer" data-tauri-drag-region />
      <button
        className="icon-btn"
        data-state={ug?.found ? 'ok' : 'warn'}
        title={ug?.found ? `ug knowledge engine · ${ug.version}` : 'ug CLI not found'}
        onClick={() => go('knowledge')}
      >
        <Radio size={19} />
      </button>
      <button
        className="icon-btn"
        data-state={engine.loadedId ? 'ok' : 'off'}
        title={engine.info ? `${engine.info.backend} · ${engine.info.threads} threads` : 'No model loaded'}
        onClick={() => go('settings')}
      >
        <Cpu size={19} />
      </button>
      <button className="icon-btn" title="Persona" onClick={() => go('persona')}>
        <Zap size={19} />
      </button>
      <ThemeButton />
      <div className="avatar" title={name}>
        {name.slice(0, 1)}
      </div>
    </header>
  );
}

const THEME_ICON = { system: Monitor, light: Sun, dark: Moon };
const THEME_LABEL = { system: 'Match system', light: 'Light', dark: 'Dark' };

function ThemeButton() {
  const { mode, setMode } = useTheme();
  const Icon = THEME_ICON[mode];
  const next = nextMode(mode);
  return (
    <button
      className="icon-btn"
      title={`Appearance: ${THEME_LABEL[mode]} — click for ${THEME_LABEL[next]}`}
      aria-label={`Appearance: ${THEME_LABEL[mode]}`}
      onClick={() => setMode(next)}
    >
      <Icon size={19} />
    </button>
  );
}

export function Sidebar() {
  const { route, go, setAbout } = useUi();
  const { navCollapsed: collapsed, toggleNav } = useLayout();
  const name = usePersona((s) => s.agentName);
  const core = useCoreStatus();
  const current = route === 'workflow-detail' ? 'workflows' : route;
  // collapsed items keep an accessible name and a hover tooltip
  const tip = (label: string, key?: string) => (collapsed ? { title: key ? `${label} (${key})` : label } : {});
  return (
    <aside className={`sidebar${collapsed ? ' collapsed' : ''}`} aria-label="Main navigation">
      <div className="agent-card" onClick={() => go('persona')} {...tip(`${name} · Persona`)}>
        <div className="agent-tile" style={{ ['--dot' as string]: core.color }}>
          <SquareTerminal size={20} />
        </div>
        <div className="agent-meta">
          <div className="agent-name">{name}</div>
          <div className="agent-sub">Active Protocol</div>
        </div>
      </div>
      <nav className="nav">
        {NAV.map((n) => (
          <button
            key={n.route}
            className="nav-item"
            aria-label={n.label}
            aria-current={current === n.route ? 'page' : undefined}
            onClick={() => go(n.route)}
            {...tip(n.label, n.key)}
          >
            {n.icon}
            <span className="nav-label">{n.label}</span>
            <span className="kbd">{n.key}</span>
          </button>
        ))}
      </nav>
      <div className="grow" />
      <nav className="nav">
        <button
          className="nav-item"
          aria-label="Settings"
          aria-current={route === 'settings' ? 'page' : undefined}
          onClick={() => go('settings')}
          {...tip('Settings', '⌘,')}
        >
          <Settings size={19} />
          <span className="nav-label">Settings</span>
          <span className="kbd">⌘,</span>
        </button>
        <button className="nav-item" aria-label="Support" onClick={() => setAbout(true)} {...tip('Support')}>
          <HelpCircle size={19} />
          <span className="nav-label">Support</span>
        </button>
        <button
          className="nav-item nav-collapse"
          aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          aria-expanded={!collapsed}
          onClick={toggleNav}
          {...tip('Expand sidebar', '⌘B')}
        >
          {collapsed ? <PanelLeftOpen size={19} /> : <PanelLeftClose size={19} />}
          <span className="nav-label">Collapse</span>
          <span className="kbd">⌘B</span>
        </button>
      </nav>
    </aside>
  );
}

export function StatusBar() {
  const { loadedId, info, tokPerSec, caps } = useEngine();
  const model = modelById(loadedId);
  const kb = useKb((s) => s.kbs.find((k) => k.slug === s.grounding));
  const lastSearch = useKb((s) => s.lastSearch);
  return (
    <footer className="statusbar">
      <span className="item">
        <span className="dot" style={{ color: model ? 'var(--violet)' : 'var(--text-4)' }} />
        Model: {model ? model.name : 'none'}
      </span>
      <span className="item">
        <span className="dot" style={{ color: 'var(--blue)' }} />
        Throughput: {tokPerSec ? `${tokPerSec.toFixed(1)} tok/s` : '—'}
      </span>
      <span className="item">
        <span className="dot" style={{ color: kb ? 'var(--green)' : 'var(--text-4)' }} />
        Knowledge: {kb ? kb.name : 'off'}
        {lastSearch && kb ? ` · ${lastSearch.ms} ms` : ''}
      </span>
      <span className="spacer" />
      <span className="coords">
        {info ? `${info.backend.toUpperCase()} / ${info.threads.split(' ')[0]} THR` : `CPU / ${caps.cores} CORES`}
        {' / '}
        {caps.isolated ? 'ISOLATED' : 'NOT ISOLATED'}
      </span>
      <span className="badge-offline">LOCAL · ON-DEVICE</span>
    </footer>
  );
}

const TOAST_ICON = {
  ok: <CheckCircle2 size={18} color="var(--green)" />,
  info: <Info size={18} color="var(--blue)" />,
  warn: <AlertTriangle size={18} color="var(--amber)" />,
  error: <XCircle size={18} color="var(--red)" />,
};

export function Toasts() {
  const { toasts, dismiss } = useUi();
  return (
    <div className="toasts">
      <AnimatePresence initial={false}>
        {toasts.map((t) => (
          <motion.div
            key={t.id}
            className="toast"
            layout
            initial={{ opacity: 0, x: 24 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 24, transition: { duration: 0.15 } }}
            transition={{ duration: 0.22, ease: [0.2, 0.7, 0.2, 1] }}
          >
            {TOAST_ICON[t.tone]}
            <div style={{ flex: 1, minWidth: 0 }}>
              <div className="t-title">{t.title}</div>
              {t.body && <div className="t-body">{t.body}</div>}
            </div>
            <button className="faint" onClick={() => dismiss(t.id)} aria-label="Dismiss">
              <X size={15} />
            </button>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
}

export function AboutModal() {
  const { aboutOpen, setAbout } = useUi();
  const ug = useKb((s) => s.ug);
  const info = useEngine((s) => s.info);
  return (
    <Modal open={aboutOpen} onClose={() => setAbout(false)}>
      <div style={{ display: 'flex', gap: 16, alignItems: 'center', marginBottom: 18 }}>
        <div className="avatar" style={{ width: 52, height: 52, borderRadius: 14 }}>
          <Zap size={24} />
        </div>
        <div>
          <h3>
            <span className="grad-text">Andai</span> Agent OS
          </h3>
          <div className="muted">A local-first AI agent. Nothing leaves this machine.</div>
        </div>
      </div>
      <div className="logblock" style={{ marginBottom: 20 }}>
        <div>
          <span className="log-violet">inference </span> wllama · llama.cpp → WebAssembly {info ? `· ${info.libllama}` : ''}
        </div>
        <div>
          <span className="log-violet">knowledge </span> ug {ug?.version?.replace('ug version ', '') ?? '(not found)'} · GraphRAG
        </div>
        <div>
          <span className="log-violet">shell     </span> Tauri 2 · React 19
        </div>
      </div>
      <div className="label" style={{ marginBottom: 10 }}>
        Shortcuts
      </div>
      <div className="muted" style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '8px 16px', fontSize: 13 }}>
        <span className="kbd-hint">⌘K</span> Focus the command input
        <span className="kbd-hint">⌘1–4</span> Switch screens
        <span className="kbd-hint">⌘B</span> Collapse / expand the sidebar
        <span className="kbd-hint">⌘J</span> Show / hide the Execution Trace
        <span className="kbd-hint">⌘,</span> Settings &amp; models
        <span className="kbd-hint">Esc</span> Stop generating
      </div>
      <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 24, gap: 10 }}>
        <button className="btn ghost" onClick={() => setAbout(false)}>
          <BookOpen size={14} /> Close
        </button>
      </div>
    </Modal>
  );
}

