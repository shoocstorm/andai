import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { act } from 'react';
import { describe, expect, it, vi } from 'vitest';
import { loadDecider, removeLaya, removeLegacyCopies, useEngine } from '../llm/engine';
import { MODELS } from '../llm/models';
import { useTheme } from '../state/theme';
import { Settings } from './Settings';

vi.mock('../llm/engine', async (original) => ({
  ...(await original<typeof import('../llm/engine')>()),
  removeLegacyCopies: vi.fn().mockResolvedValue(undefined),
  loadDecider: vi.fn().mockResolvedValue(undefined),
  refreshLaya: vi.fn().mockResolvedValue(undefined),
  removeLaya: vi.fn().mockResolvedValue(undefined),
}));

describe('Settings', () => {
  it('lists every catalog model with a download action', () => {
    render(<Settings />);
    const registry = within(screen.getByRole('region', { name: 'Model registry' }));
    for (const m of MODELS) expect(registry.getByText(m.name)).toBeInTheDocument();
    expect(registry.getAllByRole('button', { name: /download & load/i })).toHaveLength(MODELS.length);
  });

  it('offers only decision-capable models as the decision model, and loads the one picked', async () => {
    const user = userEvent.setup();
    render(<Settings />);
    const panel = within(screen.getByRole('region', { name: 'Decision model' }));
    const deciders = MODELS.filter((m) => m.decider);
    expect(panel.getAllByRole('button', { name: /use for decisions/i })).toHaveLength(deciders.length);
    expect(panel.queryByText(MODELS.find((m) => !m.decider)!.name)).toBeNull();
    expect(panel.getByText(/using chat model/i)).toBeInTheDocument();
    await user.click(panel.getAllByRole('button', { name: /use for decisions/i })[0]);
    expect(loadDecider).toHaveBeenCalledWith(deciders[0].id);
  });

  it('offers Laya only where Rust supports it, and removes a download only after confirming', async () => {
    const user = userEvent.setup();
    render(<Settings />);
    const panel = () => within(screen.getByRole('region', { name: 'Decision model' }));
    expect(panel().queryByText('Laya Multilingual')).toBeNull();

    const ckpt = (id: string, downloaded: boolean) => ({ id, repo: 'r/x', commit: 'c', bytes: 678_199_549, files: [], downloaded });
    act(() => useEngine.setState({ laya: { supported: true, loaded: null, checkpoints: [ckpt('laya-multilingual', true), ckpt('laya-en', false)] } }));
    expect(panel().getByText('Laya Multilingual')).toBeInTheDocument();
    expect(panel().getByText(/Laya English/)).toBeInTheDocument();
    expect(panel().getByText(/646\.8 MB · downloaded/)).toBeInTheDocument();
    // only a downloaded checkpoint can be removed
    expect(panel().queryByRole('button', { name: 'Remove Laya English' })).toBeNull();

    await user.click(panel().getByRole('button', { name: 'Remove Laya Multilingual' }));
    await user.click(screen.getByRole('button', { name: /cancel/i }));
    expect(removeLaya).not.toHaveBeenCalled();
    await user.click(panel().getByRole('button', { name: 'Remove Laya Multilingual' }));
    await user.click(screen.getByRole('button', { name: /^remove$/i }));
    expect(removeLaya).toHaveBeenCalledWith('laya-multilingual');

    const uses = panel().getAllByRole('button', { name: /use for decisions/i });
    await user.click(uses[uses.length - 1]);
    expect(loadDecider).toHaveBeenLastCalledWith('laya-en');
    act(() => useEngine.setState({ laya: { supported: false, loaded: null, checkpoints: [] } }));
  });

  it('switches appearance from the picker', async () => {
    const user = userEvent.setup();
    render(<Settings />);
    await user.click(screen.getByRole('radio', { name: /light/i }));
    expect(useTheme.getState().mode).toBe('light');
    expect(screen.getByRole('radio', { name: /light/i })).toHaveAttribute('aria-checked', 'true');
  });

  // Pinning changed the download URLs (AGENTS.md §9); copies cached under the
  // old ones are user data, so removing them needs a confirm step (§1.5).
  it('offers to remove a pre-pinning copy only after the user confirms', async () => {
    const user = userEvent.setup();
    act(() => useEngine.setState({ legacy: { [MODELS[0].id]: 639_446_688 } }));
    render(<Settings />);
    expect(screen.getByText(/older copy/i)).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: /remove old copy/i }));
    expect(removeLegacyCopies).not.toHaveBeenCalled();
    await user.click(screen.getByRole('button', { name: /cancel/i }));
    expect(removeLegacyCopies).not.toHaveBeenCalled();

    await user.click(screen.getByRole('button', { name: /remove old copy/i }));
    await user.click(screen.getByRole('button', { name: /^remove$/i }));
    expect(removeLegacyCopies).toHaveBeenCalledWith(MODELS[0].id);
    act(() => useEngine.setState({ legacy: {} }));
  });
});
