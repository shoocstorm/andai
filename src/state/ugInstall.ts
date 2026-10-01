// The in-app ug install: one run at a time, shared by every place that offers
// it (the setup dialog, Knowledge, Settings), so they all show the same
// progress. Rust does the work (src-tauri/src/ug_install.rs).
import { listen } from '@tauri-apps/api/event';
import { create } from 'zustand';
import { inTauri, ugInstall, type UgInstallProgress } from '../kb/api';
import { useKb } from './kb';
import { toast } from './ui';

export type InstallStage = 'idle' | UgInstallProgress['stage'] | 'done' | 'error';

type UgInstallState = {
  stage: InstallStage;
  version: string | null;
  done: number;
  total: number;
  error: string | null;
  /** The setup dialog is open. */
  dialog: boolean;
  /** The user closed the launch prompt; it isn't shown again this session. */
  dismissed: boolean;
};

export const useUgInstall = create<UgInstallState>(() => ({
  stage: 'idle',
  version: null,
  done: 0,
  total: 0,
  error: null,
  dialog: false,
  dismissed: false,
}));

export const running = (stage: InstallStage) => !['idle', 'done', 'error'].includes(stage);

export const openUgSetup = () => useUgInstall.setState({ dialog: true });
export const closeUgSetup = () => useUgInstall.setState({ dialog: false, dismissed: true });

/** At launch: ug is the knowledge engine, so without it Andai asks once per session. */
export function promptUgSetupIfMissing() {
  const ug = useKb.getState().ug;
  if (inTauri && ug && !ug.found && !useUgInstall.getState().dismissed) openUgSetup();
}

const errText = (e: unknown) => (e instanceof Error ? e.message : String(e));

export async function installUg(): Promise<boolean> {
  if (running(useUgInstall.getState().stage)) return false;
  useUgInstall.setState({ stage: 'lookup', version: null, done: 0, total: 0, error: null });
  const unlisten = await listen<UgInstallProgress>('ug-install', (e) => {
    const p = e.payload;
    useUgInstall.setState((s) => ({ stage: p.stage, version: p.version ?? s.version, done: p.done, total: p.total }));
  }).catch(() => undefined);
  try {
    const ug = await ugInstall();
    useKb.setState({ ug });
    useUgInstall.setState((s) => ({ stage: 'done', version: ug.version?.replace(/^ug version\s*/, '') ?? s.version }));
    toast({ tone: 'ok', title: 'UltraGraph is installed', body: 'Knowledge bases are ready. Add a sample or your own files.' });
    return true;
  } catch (e) {
    useUgInstall.setState({ stage: 'error', error: errText(e) });
    return false;
  } finally {
    unlisten?.();
  }
}
