import { act, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { openUgWebsite, ugInstall, type UgInstallProgress, type UgStatus } from '../kb/api';
import { useKb } from '../state/kb';
import { closeUgSetup, promptUgSetupIfMissing, useUgInstall } from '../state/ugInstall';
import { useUi } from '../state/ui';
import { UgInstall, UgSetupDialog } from './UgSetup';

let emit: ((p: UgInstallProgress) => void) | null = null;
vi.mock('@tauri-apps/api/event', () => ({
  listen: vi.fn(async (_: string, cb: (e: { payload: UgInstallProgress }) => void) => {
    emit = (payload) => cb({ payload });
    return () => (emit = null);
  }),
}));

vi.mock('../kb/api', async (original) => ({
  ...(await original<typeof import('../kb/api')>()),
  inTauri: true,
  openUgWebsite: vi.fn().mockResolvedValue(undefined),
  ugInstall: vi.fn(),
}));

const CMD = 'curl -fsSL https://ultra-graph.web.app/install.sh | sh';
const missing: UgStatus = { found: false, path: null, version: null, canInstall: true, installCommand: CMD };
const installed: UgStatus = { found: true, path: '/Users/me/.local/bin/ug', version: 'ug version 0.1.22', canInstall: true, installCommand: CMD };

beforeEach(() => {
  useKb.setState({ ug: missing });
  useUgInstall.setState({ stage: 'idle', version: null, done: 0, total: 0, error: null, dialog: false, dismissed: false });
  useUi.setState({ route: 'command' });
  vi.mocked(ugInstall).mockReset();
  vi.mocked(openUgWebsite).mockClear();
});

describe('installing ug from inside Andai', () => {
  it('shows each stage with the download progress, then ready', async () => {
    const user = userEvent.setup();
    let finish!: (s: UgStatus) => void;
    vi.mocked(ugInstall).mockReturnValue(new Promise((r) => (finish = r)));
    render(<UgInstall variant="card" />);

    await user.click(screen.getByRole('button', { name: 'Install UltraGraph' }));
    expect(ugInstall).toHaveBeenCalledOnce();
    act(() => emit!({ stage: 'download', version: 'v0.1.22', done: 12_600_000, total: 25_292_946 }));
    expect(screen.getByText('v0.1.22')).toBeInTheDocument();
    expect(screen.getByText('12.6 / 25.3 MB')).toBeInTheDocument();
    expect(screen.getByRole('progressbar', { name: 'Download' })).toHaveAttribute('aria-valuenow', '50');
    expect(screen.getByText('Download').closest('li')).toHaveAttribute('aria-current', 'step');
    expect(screen.queryByRole('button', { name: 'Install UltraGraph' })).not.toBeInTheDocument();

    await act(async () => finish(installed));
    expect(await screen.findByText('UltraGraph 0.1.22 is ready')).toBeInTheDocument();
    expect(useKb.getState().ug).toEqual(installed);
    await user.click(screen.getByRole('button', { name: /Build a knowledge base/ }));
    expect(useUi.getState().route).toBe('knowledge');
  });

  it('says what went wrong, offers a retry and the terminal command', async () => {
    const user = userEvent.setup();
    vi.mocked(ugInstall).mockRejectedValueOnce(new Error("Couldn't reach GitHub. Check your internet connection and try again."));
    render(<UgInstall variant="card" />);
    await user.click(screen.getByRole('button', { name: 'Install UltraGraph' }));

    expect(await screen.findByRole('alert')).toHaveTextContent("Couldn't reach GitHub");
    expect(screen.getByText(CMD)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Copy the install command' })).toBeInTheDocument();
    vi.mocked(ugInstall).mockResolvedValueOnce(installed);
    await user.click(screen.getByRole('button', { name: 'Try again' }));
    expect(await screen.findByText(/is ready/)).toBeInTheDocument();
  });

  it('keeps the terminal command one click away before any failure', async () => {
    const user = userEvent.setup();
    render(<UgInstall variant="card" />);
    expect(screen.queryByText(CMD)).not.toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: /Prefer the terminal/ }));
    expect(screen.getByText(CMD)).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: /ultra-graph\.web\.app/ }));
    expect(openUgWebsite).toHaveBeenCalledOnce();
  });

  it('sends people to the website where Andai cannot install it (Windows)', async () => {
    const user = userEvent.setup();
    useKb.setState({ ug: { ...missing, canInstall: false } });
    render(<UgInstall variant="card" />);
    expect(screen.queryByRole('button', { name: 'Install UltraGraph' })).not.toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: /Get UltraGraph/ }));
    expect(openUgWebsite).toHaveBeenCalledOnce();
  });
});

describe('the launch prompt', () => {
  it('opens when ug is missing, and not again once dismissed this session', async () => {
    const user = userEvent.setup();
    render(<UgSetupDialog />);
    promptUgSetupIfMissing();
    expect(await screen.findByRole('dialog', { name: 'Install UltraGraph' })).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Not now' }));
    expect(useUgInstall.getState()).toMatchObject({ dialog: false, dismissed: true });
    promptUgSetupIfMissing();
    expect(useUgInstall.getState().dialog).toBe(false);
  });

  it('stays closed when ug is installed', () => {
    useKb.setState({ ug: installed });
    promptUgSetupIfMissing();
    expect(useUgInstall.getState().dialog).toBe(false);
    closeUgSetup();
  });
});
