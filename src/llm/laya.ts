// The Laya decision model, run by Rust on MLX (src-tauri/src/laya/, Apple
// Silicon only). Typed wrappers over its commands, and the checkpoint
// download it shares with the native chat models (llm/native.ts): the Rust
// side has no HTTP client (AGENTS.md §9), so the webview fetches each file
// from its pinned Hugging Face commit and streams it to Rust in chunks. Rust
// keeps nothing it can't verify: sizes and sha256 are in its catalogs.

import { invoke, isTauri } from '@tauri-apps/api/core';
import { pinnedFileUrl } from './models';

export type LayaCheckpoint = {
  id: string;
  repo: string;
  commit: string;
  bytes: number;
  files: { path: string; bytes: number }[];
  downloaded: boolean;
};

export type LayaStatus = { supported: boolean; loaded: string | null; checkpoints: LayaCheckpoint[] };

/** `choice`: pick one of `options`. `noul`: does the statement in `question` hold, given the state? */
export type LayaQuestion =
  | { id: string; kind: 'choice'; question: string; options: { id: string; text: string }[] }
  | { id: string; kind: 'noul'; question: string };

export type LayaAnswer = {
  id: string;
  /** Calibrated probability per option, in the order given; `[P(false), P(true)]` for a noul. */
  probabilities: number[];
  inputTokens: number;
  /** An option, the question or the state was cut to fit the model's input. */
  truncated: boolean;
};

export type LayaAnswers = {
  answers: LayaAnswer[];
  /** Model time for the whole batch (Rust), ms. */
  ms: number;
  /** Checkpoint id. */
  model: string;
};

export type Progress = { loaded: number; total: number; speed: number; phase: string };

const UNSUPPORTED: LayaStatus = { supported: false, loaded: null, checkpoints: [] };

/** In a plain browser (no Tauri) there's no Laya: same answer as a non-Apple-Silicon build. */
export const layaStatus = (): Promise<LayaStatus> => (isTauri() ? invoke<LayaStatus>('laya_status') : Promise.resolve(UNSUPPORTED));
export const layaLoad = (id: string) => invoke<number>('laya_load', { checkpoint: id });
export const layaUnload = () => invoke<void>('laya_unload');
export const layaRemove = (id: string) => invoke<void>('laya_remove', { checkpoint: id });
/** Every question about `state` in one batched forward pass (up to 4). */
export const layaDecide = (state: string, questions: LayaQuestion[]) => invoke<LayaAnswers>('laya_decide', { state, questions });

/** How likely each passage helps answer `request` (one yes/no row per passage, batched in Rust). */
export const layaRelevance = (request: string, passages: { source: string; text: string }[]) =>
  invoke<{ scores: number[]; ms: number; model: string }>('laya_relevance', { request, passages });

/** How likely each cited passage supports the sentence that cites it (one yes/no row per claim, batched in Rust). */
export const layaSupport = (claims: { statement: string; source: string; text: string }[]) =>
  invoke<{ scores: number[]; ms: number; model: string }>('laya_support', { claims });

/** Bytes per IPC call; Rust accepts up to 16 MiB. */
export const CHUNK = 8 * 1024 * 1024;

/** A checkpoint in one of Rust's catalogs: Laya's, or the native chat models' (`llm`). */
export type Checkpoint = Omit<LayaCheckpoint, 'downloaded'>;

/**
 * Downloads every file of `c` and has Rust verify it (`<kind>_write_chunk`,
 * `<kind>_finish`). Rust discards the whole download if any file's size or
 * sha256 is off, and says which.
 */
export async function downloadCheckpoint(
  kind: 'laya' | 'llm',
  c: Checkpoint,
  onProgress: (p: Progress) => void,
  signal?: AbortSignal,
): Promise<void> {
  const started = performance.now();
  let loaded = 0;
  const report = (phase: string) => onProgress({ loaded, total: c.bytes, speed: loaded / Math.max(0.001, (performance.now() - started) / 1000), phase });
  for (const f of c.files) {
    report(`Downloading ${f.path}…`);
    const res = await fetch(pinnedFileUrl(c.repo, c.commit, f.path), { signal });
    if (!res.ok || !res.body) throw new Error(`Couldn't download ${f.path} (HTTP ${res.status}).`);
    const reader = res.body.getReader();
    let offset = 0;
    let buf = new Uint8Array(Math.min(CHUNK, f.bytes));
    let fill = 0;
    const flush = async () => {
      if (!fill) return;
      offset = await invoke<number>(`${kind}_write_chunk`, buf.subarray(0, fill), {
        headers: { 'x-checkpoint': c.id, 'x-file': f.path, 'x-offset': String(offset) },
      });
      fill = 0;
    };
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      let at = 0;
      while (at < value.length) {
        const n = Math.min(buf.length - fill, value.length - at);
        buf.set(value.subarray(at, at + n), fill);
        fill += n;
        at += n;
        loaded += n;
        if (fill === buf.length) {
          await flush();
          buf = new Uint8Array(Math.min(CHUNK, Math.max(1, f.bytes - offset)));
        }
      }
      report(`Downloading ${f.path}…`);
    }
    await flush();
  }
  report('Verifying checksums…');
  await invoke(`${kind}_finish`, { checkpoint: c.id });
}

export const downloadLaya = (c: LayaCheckpoint, onProgress: (p: Progress) => void, signal?: AbortSignal) =>
  downloadCheckpoint('laya', c, onProgress, signal);
