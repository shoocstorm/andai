import { act, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { HubModel } from '../llm/hub';
import { useEngine } from '../llm/engine';
import { HubModels } from './HubModels';

const COMMIT = 'd7f544eead698dbd1f15126ef60b45a1e1933222';
const ok: HubModel = {
  repo: 'unsloth/Qwen3-1.7B-GGUF',
  commit: COMMIT,
  format: 'gguf',
  license: 'apache-2.0',
  downloads: 72415,
  likes: 10,
  lastModified: null,
  checks: [{ level: 'ok', text: '2 of 3 files fit the 2 GB limit.' }],
  ok: true,
  gguf: {
    architecture: 'qwen3',
    contextLength: 40960,
    thinking: true,
    recommended: 'q4.gguf',
    variants: [
      { path: 'q4.gguf', bytes: 1_107_409_472, sha256: 'b'.repeat(64), quant: 'Q4_K_M', fits: true },
      { path: 'q8.gguf', bytes: 1_834_000_000, sha256: 'c'.repeat(64), quant: 'Q8_0', fits: true },
      { path: 'bf16.gguf', bytes: 3_447_349_568, sha256: 'd'.repeat(64), quant: 'BF16', fits: false },
    ],
  },
};
const blocked: HubModel = { ...ok, repo: 'meta-llama/Llama-3.2-1B-GGUF', ok: false, gguf: undefined, checks: [{ level: 'block', text: 'It needs a Hugging Face login.' }] };

const hub = vi.hoisted(() => ({ searches: [] as string[] }));
vi.mock('../llm/hub', async (orig) => ({
  ...(await orig<typeof import('../llm/hub')>()),
  searchHub: vi.fn(async (q: string) => {
    hub.searches.push(q);
    return [
      { repo: 'unsloth/Qwen3-1.7B-GGUF', downloads: 72415, likes: 10, lastModified: '2025-06-08T08:08:38.000Z', license: 'apache-2.0' },
      { repo: 'meta-llama/Llama-3.2-1B-GGUF', downloads: 5, likes: 1, lastModified: null, license: null },
    ];
  }),
  inspectHub: vi.fn(async (repo: string) => (repo === ok.repo ? ok : blocked)),
}));
vi.mock('../llm/custom', () => ({ addHubModel: vi.fn(async () => 'hf-gguf-x') }));
vi.mock('../llm/engine', async (orig) => ({ ...(await orig<typeof import('../llm/engine')>()), loadModel: vi.fn(async () => {}) }));

const { addHubModel } = await import('../llm/custom');
const { loadModel } = await import('../llm/engine');

beforeEach(() => {
  hub.searches = [];
  vi.mocked(addHubModel).mockClear();
  act(() => useEngine.setState({ native: { supported: false, chat: null, decider: null, checkpoints: [], memory: null } }));
});

describe('Add from Hugging Face', () => {
  it('searches as you type, shows why a model can’t be added, and adds the recommended file', async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    render(<HubModels open onClose={onClose} />);
    expect(screen.queryByRole('group', { name: 'Model format' })).toBeNull(); // no MLX here: GGUF only
    await user.type(screen.getByRole('searchbox', { name: 'Search Hugging Face' }), 'qwen3');
    const results = within(screen.getByRole('region', { name: 'Search results' }));
    await waitFor(() => expect(results.getByText('unsloth/Qwen3-1.7B-GGUF')).toBeInTheDocument());
    expect(hub.searches).toEqual(['qwen3']); // debounced: one search, not one per key

    await user.click(results.getByText('meta-llama/Llama-3.2-1B-GGUF'));
    const detail = within(screen.getByRole('region', { name: 'Model details' }));
    await waitFor(() => expect(detail.getByText(/needs a Hugging Face login/)).toBeInTheDocument());
    expect(screen.getByRole('button', { name: /add & download/i })).toBeDisabled();

    await user.click(results.getByText('unsloth/Qwen3-1.7B-GGUF'));
    await waitFor(() => expect(detail.getByRole('radiogroup', { name: 'File to download' })).toBeInTheDocument());
    expect(detail.getByRole('radio', { name: /Q4_K_M/ })).toHaveAttribute('aria-checked', 'true');
    expect(detail.getAllByRole('radio')[0]).toHaveAccessibleName(/Q4_K_M/); // the recommended file is in view
    expect(detail.getByRole('radio', { name: /BF16/ })).toBeDisabled();
    expect(detail.getByRole('note')).toHaveTextContent(/third-party model/i);
    expect(detail.getByText(/Version d7f544e/)).toBeInTheDocument();

    await user.click(detail.getByRole('radio', { name: /Q8_0/ }));
    await user.click(screen.getByRole('button', { name: /add & download 1\.71 GB/i }));
    expect(addHubModel).toHaveBeenCalledWith(ok, ok.gguf!.variants[1]);
    expect(loadModel).toHaveBeenCalledWith('hf-gguf-x');
    expect(onClose).toHaveBeenCalled();
  });

  it('offers MLX first on a Mac that runs it', () => {
    act(() => useEngine.setState({ native: { supported: true, chat: null, decider: null, checkpoints: [], memory: null } }));
    render(<HubModels open onClose={() => {}} />);
    const format = within(screen.getByRole('group', { name: 'Model format' }));
    expect(format.getByRole('button', { name: /MLX/ })).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByText(/runs Qwen3 and Qwen3.5 models quantized for MLX/)).toBeInTheDocument();
  });
});
