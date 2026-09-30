// The activity log (src-tauri/src/activity.rs). Off by default (a product
// decision, 2026-09-30); when the user turns it on in Settings, every agent
// event (the turn, each step and decision, each argument writer call, each
// tool call, the context sent, the answer and its checks) becomes one JSON
// line in a daily file under the app's data folder, kept 7 days. Logging
// never slows or fails a turn: events are queued and written in batches, and
// a failed write is dropped. Besides the agent's events, it records what the
// app did to models and knowledge bases (`logApp`), and each line carries a
// one-line summary and a level (lib/activityText.ts), which the Logs screen
// (screens/Logs.tsx) reads back.
import { invoke, isTauri } from '@tauri-apps/api/core';
import { create } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';
import { describe, type Level } from '../lib/activityText';

/** What Rust accepts (activity.rs `KINDS`). */
export type ActivityKind = 'turn' | 'step' | 'args' | 'tool' | 'retrieve' | 'relevance' | 'context' | 'answer' | 'claims' | 'error' | 'done' | 'model' | 'kb';
/** The turn id of events that belong to no question: models and knowledge bases. */
export const APP_TURN = 'app';

type ActivityState = { enabled: boolean; setEnabled: (enabled: boolean) => void };

export const useActivity = create<ActivityState>()(
  persist((set) => ({ enabled: false, setEnabled: (enabled) => set({ enabled }) }), {
    name: 'andai.activity',
    storage: createJSONStorage(() => localStorage),
    partialize: (s) => ({ enabled: s.enabled }),
  }),
);

/** Events per write (Rust takes up to 64), and how long an event waits for company. */
const BATCH = 32;
const FLUSH_MS = 300;
/** Longer strings are cut: Rust refuses an event over 256 KB, and a tool's raw output can be larger. */
export const MAX_TEXT = 32_000;

type Pending = { kind: ActivityKind; turn: string; summary: string; level: Level; data: unknown };
let queue: Pending[] = [];
let timer: ReturnType<typeof setTimeout> | null = null;
let warned = false;

const cut = (_key: string, v: unknown) => (typeof v === 'string' && v.length > MAX_TEXT ? `${v.slice(0, MAX_TEXT)}… (${v.length - MAX_TEXT} more characters)` : v);

/** Queues one event for the log, when it's on. `turn` is the assistant message's id. */
export function logActivity(turn: string, kind: ActivityKind, data: unknown): void {
  if (!useActivity.getState().enabled || !isTauri()) return;
  let clean: unknown;
  try {
    clean = JSON.parse(JSON.stringify(data ?? null, cut)) as unknown;
  } catch {
    clean = { unserializable: String(data) };
  }
  queue.push({ kind, turn, ...describe(kind, clean), data: clean });
  if (queue.length >= BATCH) void flushActivity();
  else timer ??= setTimeout(() => void flushActivity(), FLUSH_MS);
}

/** Queues an event that belongs to no question: a model loaded, a knowledge base indexed. */
export const logApp = (kind: 'model' | 'kb', data: unknown) => logActivity(APP_TURN, kind, data);

/** Writes what's queued. Never throws: the log is a record, not part of the turn. */
export async function flushActivity(): Promise<void> {
  if (timer) clearTimeout(timer);
  timer = null;
  while (queue.length) {
    const events = queue.splice(0, BATCH);
    try {
      await invoke('activity_write', { events });
    } catch (e) {
      if (!warned) console.warn('Activity log write failed; events dropped:', e);
      warned = true;
    }
  }
}

export type ActivityInfo = { dir: string; files: { name: string; bytes: number }[] };

/** The logs folder and its files, newest first. */
export const activityInfo = () => invoke<ActivityInfo>('activity_info');
/** One logged line as read back. `summary` and `level` are missing on lines an older build wrote. */
export type ActivityEvent = { at: number; kind: string; turn: string; summary?: string; level?: Level; data: unknown };
export type ActivityDay = { events: ActivityEvent[]; total: number; bad: number };

/** One day's events (a name from `activityInfo`), oldest first; Rust returns the newest 5,000. */
export async function readActivity(name: string): Promise<ActivityDay> {
  await flushActivity();
  return invoke<ActivityDay>('activity_read', { name });
}
/** Deletes every log file; returns how many. The caller confirms first. */
export const clearActivity = () => invoke<number>('activity_clear');
/** Shows the logs folder in Finder. */
export const openActivityFolder = () => invoke<void>('activity_open');
