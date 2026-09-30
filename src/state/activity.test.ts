// The activity log's webview side: nothing is written while it's off or
// outside the desktop app; when on, events are batched to Rust, long texts
// are cut to what Rust accepts, and a failed write never throws.
import { beforeEach, describe, expect, it, vi } from 'vitest';

const tauri = vi.hoisted(() => ({ on: true, calls: [] as { cmd: string; args: { events: { kind: string; turn: string; data: unknown }[] } }[], fail: null as Error | null }));
vi.mock('@tauri-apps/api/core', () => ({
  isTauri: () => tauri.on,
  invoke: async (cmd: string, args: { events: { kind: string; turn: string; data: unknown }[] }) => {
    tauri.calls.push({ cmd, args });
    if (tauri.fail) throw tauri.fail;
    return args.events.length;
  },
}));

const { flushActivity, logActivity, MAX_TEXT, useActivity } = await import('./activity');

beforeEach(async () => {
  await flushActivity();
  tauri.on = true;
  tauri.calls = [];
  tauri.fail = null;
  useActivity.setState({ enabled: true });
});

describe('activity log', () => {
  it('is off by default', () => {
    expect(useActivity.getInitialState().enabled).toBe(false);
  });

  it('writes nothing while off, or outside the desktop app', async () => {
    useActivity.setState({ enabled: false });
    logActivity('m1', 'turn', { question: 'q' });
    useActivity.setState({ enabled: true });
    tauri.on = false;
    logActivity('m1', 'turn', { question: 'q' });
    await flushActivity();
    expect(tauri.calls).toEqual([]);
  });

  it('sends queued events to Rust in one batch, in order', async () => {
    logActivity('m1', 'turn', { question: 'Is there Wi-Fi?' });
    logActivity('m1', 'args', { tool: 'kb_search', args: { query: 'wifi' } });
    await flushActivity();
    expect(tauri.calls).toEqual([
      {
        cmd: 'activity_write',
        args: {
          events: [
            { kind: 'turn', turn: 'm1', data: { question: 'Is there Wi-Fi?' } },
            { kind: 'args', turn: 'm1', data: { tool: 'kb_search', args: { query: 'wifi' } } },
          ],
        },
      },
    ]);
  });

  it('writes on its own after a moment, and at 32 events without waiting', async () => {
    vi.useFakeTimers();
    try {
      logActivity('m1', 'step', {});
      expect(tauri.calls).toHaveLength(0);
      await vi.advanceTimersByTimeAsync(400);
      expect(tauri.calls).toHaveLength(1);
      for (let i = 0; i < 32; i++) logActivity('m1', 'step', { i });
      await Promise.resolve();
      expect(tauri.calls).toHaveLength(2);
      expect(tauri.calls[1].args.events).toHaveLength(32);
    } finally {
      vi.useRealTimers();
    }
  });

  it('cuts long texts to what Rust accepts, and drops what can’t be serialized', async () => {
    logActivity('m1', 'tool', { output: 'x'.repeat(MAX_TEXT + 10), fn: () => 1 });
    await flushActivity();
    const data = tauri.calls[0].args.events[0].data as { output: string };
    expect(data.output).toHaveLength(MAX_TEXT + '… (10 more characters)'.length);
    expect(data).not.toHaveProperty('fn');
  });

  it('never throws when a write fails', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    tauri.fail = new Error('disk full');
    logActivity('m1', 'turn', {});
    await expect(flushActivity()).resolves.toBeUndefined();
    warn.mockRestore();
  });
});
