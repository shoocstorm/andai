import { create } from 'zustand';

export type Route = 'command' | 'workflows' | 'workflow-detail' | 'knowledge' | 'tools' | 'persona' | 'settings' | 'logs';

export type Toast = {
  id: number;
  tone: 'info' | 'ok' | 'warn' | 'error';
  title: string;
  body?: string;
};

type UiState = {
  route: Route;
  toasts: Toast[];
  aboutOpen: boolean;
  /** A Finder drag is hovering the window. */
  dragging: boolean;
  go: (route: Route) => void;
  toast: (t: Omit<Toast, 'id'>) => void;
  dismiss: (id: number) => void;
  setAbout: (open: boolean) => void;
};

let nextId = 1;

const ROUTES: Route[] = ['command', 'workflows', 'workflow-detail', 'knowledge', 'tools', 'persona', 'settings', 'logs'];
const initialRoute = (): Route => {
  const h = window.location.hash.slice(1) as Route;
  return ROUTES.includes(h) ? h : 'command';
};

export const useUi = create<UiState>((set, get) => ({
  route: initialRoute(),
  toasts: [],
  aboutOpen: false,
  dragging: false,
  go: (route) => {
    history.replaceState(null, '', `#${route}`);
    set({ route });
  },
  toast: (t) => {
    const id = nextId++;
    set({ toasts: [...get().toasts, { ...t, id }] });
    setTimeout(() => get().dismiss(id), t.tone === 'error' ? 7000 : 4000);
  },
  dismiss: (id) => set({ toasts: get().toasts.filter((t) => t.id !== id) }),
  setAbout: (aboutOpen) => set({ aboutOpen }),
}));

export const toast = (t: Omit<Toast, 'id'>) => useUi.getState().toast(t);
