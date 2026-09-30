import { act, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { loadDecider, removeLaya, removeLegacyCopies, useEngine } from '../llm/engine';
import { DEFAULT_MODEL, isMlx, MODELS, setCustomModels, type WllamaDef } from '../llm/models';
import { removeCustomModel } from '../llm/custom';
import { useTheme } from '../state/theme';
import { Settings } from './Settings';

vi.mock('../llm/custom', () => ({ removeCustomModel: vi.fn().mockResolvedValue(undefined), addHubModel: vi.fn() }));

vi.mock('../llm/engine', async (original) => ({
  ...(await original<typeof import('../llm/engine')>()),
  removeLegacyCopies: vi.fn().mockResolvedValue(undefined),
  loadDecider: vi.fn().mockResolvedValue(undefined),
  refreshLaya: vi.fn().mockResolvedValue(undefined),
  refreshNative: vi.fn().mockResolvedValue(undefined),
  removeLaya: vi.fn().mockResolvedValue(undefined),
}));

const gguf = MODELS.filter((m) => !isMlx(m));
const setMlx = (supported: boolean) => act(() => useEngine.setState({ native: { supported, chat: null, decider: null, checkpoints: [], memory: null } }));

describe('Settings', () => {
  it('lists every portable model with a download action, and no MLX model where MLX can’t run', () => {
    render(<Settings />);
    const registry = within(screen.getByRole('region', { name: 'Model registry' }));
    for (const m of gguf) expect(registry.getByText(m.name)).toBeInTheDocument();
    expect(registry.getAllByRole('button', { name: /download & load/i })).toHaveLength(gguf.length);
    expect(registry.queryByText(/MLX/)).toBeNull();
    expect(screen.queryByRole('note')).toBeNull();
  });

  it('on Apple Silicon, lists the MLX models first and says they are faster here', () => {
    setMlx(true);
    render(<Settings />);
    const registry = within(screen.getByRole('region', { name: 'Model registry' }));
    const names = registry.getAllByText(/^Qwen3|^TinyStories/).map((n) => n.textContent);
    expect(names.slice(0, 2)).toEqual(['Qwen3 1.7B · MLX', 'Qwen3 0.6B · MLX']);
    expect(registry.getAllByText(/fastest on this Mac/)).toHaveLength(MODELS.filter(isMlx).length);
    expect(registry.getAllByText(/wllama · WebGPU/)).toHaveLength(gguf.length);
    expect(registry.getAllByRole('button', { name: /download & load/i })).toHaveLength(MODELS.length);
    expect(screen.getByRole('note')).toHaveTextContent(/MLX models are much faster on this Mac/);
    setMlx(false);
  });

  it('offers only decision-capable models as the decision model, and loads the one picked', async () => {
    const user = userEvent.setup();
    render(<Settings />);
    const panel = within(screen.getByRole('region', { name: 'Decision model' }));
    const deciders = gguf.filter((m) => m.decider);
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

    // Laya first, then the chat-model deciders.
    const uses = panel().getAllByRole('button', { name: /use for decisions/i });
    await user.click(uses[1]);
    expect(loadDecider).toHaveBeenLastCalledWith('laya-en');
    const rows = panel().getAllByText(/^(Laya|Qwen3)/).map((n) => n.textContent);
    expect(rows.slice(0, 2)).toEqual(['Laya Multilingual', 'Laya English']);
    act(() => useEngine.setState({ laya: { supported: false, loaded: null, checkpoints: [] } }));
  });

  it('switches appearance from the picker, which offers light and dark only', async () => {
    const user = userEvent.setup();
    useTheme.getState().setTheme('dark');
    render(<Settings />);
    const group = within(screen.getByRole('radiogroup', { name: /appearance/i }));
    const radios = group.getAllByRole('radio');
    expect(radios).toHaveLength(2);
    expect(radios[0]).toHaveTextContent(/light/i);
    expect(radios[1]).toHaveTextContent(/dark/i);
    await user.click(group.getByRole('radio', { name: /light/i }));
    expect(useTheme.getState().theme).toBe('light');
    expect(group.getByRole('radio', { name: /light/i })).toHaveAttribute('aria-checked', 'true');
  });

  // Pinning changed the download URLs (AGENTS.md §9); copies cached under the
  // old ones are user data, so removing them needs a confirm step (§1.5).
  it('offers to remove a pre-pinning copy only after the user confirms', async () => {
    const user = userEvent.setup();
    act(() => useEngine.setState({ legacy: { [DEFAULT_MODEL]: 639_446_688 } }));
    render(<Settings />);
    expect(screen.getByText(/older copy/i)).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: /remove old copy/i }));
    expect(removeLegacyCopies).not.toHaveBeenCalled();
    await user.click(screen.getByRole('button', { name: /cancel/i }));
    expect(removeLegacyCopies).not.toHaveBeenCalled();

    await user.click(screen.getByRole('button', { name: /remove old copy/i }));
    await user.click(screen.getByRole('button', { name: /^remove$/i }));
    expect(removeLegacyCopies).toHaveBeenCalledWith(DEFAULT_MODEL);
    act(() => useEngine.setState({ legacy: {} }));
  });

  it('opens the Hugging Face search, and removes an added model only after confirming', async () => {
    const user = userEvent.setup();
    const added: WllamaDef = {
      id: 'hf-gguf-someone--tiny-q4-d7f544ee',
      engine: 'wllama',
      name: 'Tiny · Q4_K_M',
      family: 'Q4_K_M · llama · Hugging Face',
      size: '600 MB',
      bytes: 6e8,
      url: 'https://huggingface.co/someone/tiny/resolve/d7f544eead698dbd1f15126ef60b45a1e1933222/tiny-Q4_K_M.gguf',
      sha256: 'b'.repeat(64),
      legacyUrls: [],
      note: 'Added from someone/tiny on Hugging Face.',
      thinking: false,
      n_ctx: 4096,
      source: { repo: 'someone/tiny', commit: 'd7f544eead698dbd1f15126ef60b45a1e1933222', license: 'mit' },
    };
    setCustomModels([added]);
    render(<Settings />);
    const registry = within(screen.getByRole('region', { name: 'Model registry' }));
    expect(registry.getByText('someone/tiny@d7f544e · mit', { exact: false })).toBeInTheDocument();
    // Catalog models have no Remove; the added one does.
    expect(registry.getAllByRole('button', { name: /^Remove / })).toHaveLength(1);
    await user.click(registry.getByRole('button', { name: 'Remove Tiny · Q4_K_M' }));
    await user.click(screen.getByRole('button', { name: /cancel/i }));
    expect(removeCustomModel).not.toHaveBeenCalled();
    await user.click(registry.getByRole('button', { name: 'Remove Tiny · Q4_K_M' }));
    await user.click(screen.getByRole('button', { name: /^remove$/i }));
    expect(removeCustomModel).toHaveBeenCalledWith(added.id);

    await user.click(screen.getByRole('button', { name: /add from hugging face/i }));
    expect(screen.getByRole('dialog', { name: 'Add a model from Hugging Face' })).toBeInTheDocument();
    setCustomModels([]);
  });

  it('marks the loaded chat model and the decision model in use with a green check', () => {
    act(() =>
      useEngine.setState((st) => ({
        status: 'ready',
        loadedId: DEFAULT_MODEL,
        decider: { ...st.decider, status: 'ready', loadedId: 'qwen3-1.7b', loadingId: null },
      })),
    );
    render(<Settings />);
    const registry = within(screen.getByRole('region', { name: 'Model registry' }));
    expect(registry.getAllByText('Loaded · in use')).toHaveLength(1);
    expect(registry.getByLabelText('Loaded')).toBeInTheDocument();
    const panel = within(screen.getByRole('region', { name: 'Decision model' }));
    expect(panel.getByLabelText('In use for decisions')).toBeInTheDocument();
    expect(panel.getByText('In use')).toBeInTheDocument();
    act(() => useEngine.setState((st) => ({ status: 'idle', loadedId: null, decider: { ...st.decider, status: 'idle', loadedId: null } })));
  });

  it('warns on an MLX model that likely doesn’t fit this Mac’s memory', () => {
    act(() => useEngine.setState({ native: { supported: true, chat: null, decider: null, checkpoints: [], memory: 16e9 } }));
    render(<Settings />);
    const notes = screen.getAllByRole('note').filter((n) => /Needs about/.test(n.textContent ?? ''));
    expect(notes.map((n) => n.textContent)).toEqual([expect.stringMatching(/Needs about 22 GB of memory; this Mac has 16 GB/)]);
    setMlx(false);
  });
});

describe('Settings · activity log', () => {
  it('is off until switched on, and says the desktop app writes it', async () => {
    const { useActivity } = await import('../state/activity');
    const user = userEvent.setup();
    render(<Settings />);
    const toggle = screen.getByRole('switch', { name: 'Keep an activity log' });
    expect(toggle).toHaveAttribute('aria-checked', 'false');
    await user.click(toggle);
    expect(useActivity.getState().enabled).toBe(true);
    expect(screen.getByRole('switch', { name: 'Keep an activity log' })).toHaveAttribute('aria-checked', 'true');
    expect(screen.getByText(/written by the desktop app/)).toBeInTheDocument();
    act(() => useActivity.setState({ enabled: false }));
  });
});
