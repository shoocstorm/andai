// The Logs screen: a day's activity grouped by question, newest first, each
// event's summary shown and its data opened as plain text; filters, search,
// and a day picker over the files Rust lists.
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const tauri = vi.hoisted(() => ({
  on: true,
  reads: [] as string[],
  days: {} as Record<string, unknown[]>,
}));
vi.mock('@tauri-apps/api/core', () => ({
  isTauri: () => tauri.on,
  invoke: async (cmd: string, args?: { name?: string }) => {
    if (cmd === 'activity_info') return { dir: '/logs', files: Object.keys(tauri.days).sort().reverse().map((name) => ({ name, bytes: 2048 })) };
    if (cmd === 'activity_read') {
      tauri.reads.push(args!.name!);
      const events = tauri.days[args!.name!] ?? [];
      return { events, total: events.length, bad: 0 };
    }
    return null;
  },
}));

const { Logs } = await import('./Logs');
const { useActivity } = await import('../state/activity');

const today = [
  { at: 1_000, kind: 'model', turn: 'app', summary: 'Loaded chat model Qwen3 1.7B (MLX) in 420 ms', level: 'info', data: { action: 'load' } },
  { at: 2_000, kind: 'turn', turn: 'm1', data: { question: 'Is there Wi-Fi on the Kestrel?', mode: 'agent', model: 'Qwen3 1.7B', kb: { name: 'Ferries' } } },
  { at: 2_100, kind: 'tool', turn: 'm1', data: { step: 0, tool: 'kb_search', status: 'error', error: 'ug timed out <img src=x>' } },
  { at: 3_000, kind: 'done', turn: 'm1', data: { outcome: 'answered', ms: 1000 } },
  { at: 4_000, kind: 'turn', turn: 'm2', data: { question: 'When does the Osprey sail?', mode: 'fixed', model: 'Qwen3 1.7B' } },
  { at: 4_500, kind: 'done', turn: 'm2', data: { outcome: 'answered', ms: 500 } },
];

beforeEach(() => {
  tauri.on = true;
  tauri.reads = [];
  tauri.days = { 'agent-2026-09-30.jsonl': today, 'agent-2026-09-29.jsonl': [{ at: 1, kind: 'turn', turn: 'old', data: { question: 'Yesterday’s question' } }] };
  useActivity.setState({ enabled: true });
});

describe('Logs', () => {
  it('shows the newest day’s questions newest first, with app events between them', async () => {
    render(<Logs />);
    const heads = await screen.findAllByRole('button', { expanded: false, name: /Kestrel|Osprey/ });
    expect(heads.map((h) => (h.textContent!.includes('Osprey') ? 'm2' : 'm1'))).toEqual(['m2', 'm1']);
    expect(screen.getByText('Loaded chat model Qwen3 1.7B (MLX) in 420 ms')).toBeInTheDocument();
    expect(screen.getByText(/1 errors/)).toBeInTheDocument();
    expect(tauri.reads).toEqual(['agent-2026-09-30.jsonl']);
  });

  it('opens a question to its steps, and an event to its data as plain text', async () => {
    const user = userEvent.setup();
    render(<Logs />);
    await user.click(await screen.findByRole('button', { name: /Kestrel/ }));
    const row = screen.getByRole('button', { name: /Step 1: kb_search failed · ug timed out/ });
    await user.click(row);
    const json = document.querySelector('.lg-json')!;
    expect(json.textContent).toContain('"error": "ug timed out <img src=x>"');
    expect(json.querySelector('img')).toBeNull();
  });

  it('filters to problems and searches the data', async () => {
    const user = userEvent.setup();
    render(<Logs />);
    await screen.findByRole('button', { name: /Osprey/ });
    await user.click(screen.getByRole('button', { name: 'Problems' }));
    expect(screen.queryByRole('button', { name: /Osprey/ })).toBeNull();
    expect(screen.getByRole('button', { name: /Kestrel/ })).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'All' }));
    await user.type(screen.getByRole('searchbox', { name: 'Search the log' }), 'osprey');
    expect(screen.queryByRole('button', { name: /Kestrel/ })).toBeNull();
    expect(screen.queryByText(/Loaded chat model/)).toBeNull();
  });

  it('reads another day when picked', async () => {
    const user = userEvent.setup();
    render(<Logs />);
    await screen.findByRole('button', { name: /Osprey/ });
    await user.selectOptions(screen.getByRole('combobox', { name: 'Day' }), 'agent-2026-09-29.jsonl');
    expect(await screen.findByRole('button', { name: /Yesterday’s question/ })).toBeInTheDocument();
    expect(tauri.reads.at(-1)).toBe('agent-2026-09-29.jsonl');
  });

  it('offers to turn the log on when it is off, and says the browser has none', async () => {
    const user = userEvent.setup();
    useActivity.setState({ enabled: false });
    const { unmount } = render(<Logs />);
    const off = await screen.findByRole('status');
    expect(within(off).getByText('The activity log is off')).toBeInTheDocument();
    await user.click(within(off).getByRole('switch', { name: 'Keep an activity log' }));
    expect(useActivity.getState().enabled).toBe(true);
    unmount();
    tauri.on = false;
    render(<Logs />);
    expect(screen.getByText(/written and read by the desktop app/)).toBeInTheDocument();
  });
});
