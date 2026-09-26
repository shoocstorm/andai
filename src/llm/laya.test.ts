// The Laya download: each file is fetched from its pinned URL and streamed to
// Rust in ordered chunks, then Rust is asked to verify (it holds the hashes).
import { beforeEach, describe, expect, it, vi } from 'vitest';

const calls = vi.hoisted(() => [] as { cmd: string; args: unknown; headers?: Record<string, string> }[]);
vi.mock('@tauri-apps/api/core', () => ({
  isTauri: () => true,
  invoke: vi.fn(async (cmd: string, args: unknown, opts?: { headers: Record<string, string> }) => {
    calls.push({ cmd, args: args instanceof Uint8Array ? Array.from(args) : args, headers: opts?.headers });
    if (cmd === 'laya_write_chunk') return Number(opts!.headers['x-laya-offset']) + (args as Uint8Array).length;
    return undefined;
  }),
}));

const { CHUNK, downloadLaya } = await import('./laya');

const ckpt = {
  id: 'laya-multilingual',
  repo: 'aac6fef/laya-multilingual-mlx',
  commit: 'f2b4faf51023039425946074e2cf1361d2db11d5',
  bytes: 5 + 3,
  files: [
    { path: 'model.safetensors', bytes: 5 },
    { path: 'tokenizer/tokenizer.json', bytes: 3 },
  ],
  downloaded: false,
};

/** A response whose body arrives in the given pieces. */
const body = (...parts: number[][]) =>
  new Response(
    new ReadableStream({
      start(c) {
        for (const p of parts) c.enqueue(new Uint8Array(p));
        c.close();
      },
    }),
  );

beforeEach(() => {
  calls.length = 0;
});

describe('downloadLaya', () => {
  it('fetches each file at its pinned commit, streams it in order and asks Rust to verify', async () => {
    const urls: string[] = [];
    vi.stubGlobal('fetch', async (url: string) => {
      urls.push(url);
      return url.endsWith('model.safetensors') ? body([1, 2], [3, 4, 5]) : body([7, 8, 9]);
    });
    const phases: string[] = [];
    await downloadLaya(ckpt, (p) => phases.push(`${p.phase} ${p.loaded}/${p.total}`));
    expect(urls).toEqual([
      'https://huggingface.co/aac6fef/laya-multilingual-mlx/resolve/f2b4faf51023039425946074e2cf1361d2db11d5/model.safetensors',
      'https://huggingface.co/aac6fef/laya-multilingual-mlx/resolve/f2b4faf51023039425946074e2cf1361d2db11d5/tokenizer/tokenizer.json',
    ]);
    // small files fit one chunk each; headers name checkpoint, file and offset
    expect(calls.map((c) => [c.cmd, c.args, c.headers])).toEqual([
      ['laya_write_chunk', [1, 2, 3, 4, 5], { 'x-laya-checkpoint': ckpt.id, 'x-laya-file': 'model.safetensors', 'x-laya-offset': '0' }],
      ['laya_write_chunk', [7, 8, 9], { 'x-laya-checkpoint': ckpt.id, 'x-laya-file': 'tokenizer/tokenizer.json', 'x-laya-offset': '0' }],
      ['laya_finish', { checkpoint: ckpt.id }, undefined],
    ]);
    expect(phases.at(-1)).toBe('Verifying checksums… 8/8');
  });

  it('splits a large file into chunks at increasing offsets', async () => {
    const big = { ...ckpt, bytes: CHUNK + 10, files: [{ path: 'model.safetensors', bytes: CHUNK + 10 }] };
    vi.stubGlobal('fetch', async () => body(Array(CHUNK - 1).fill(1), Array(11).fill(2)));
    await downloadLaya(big, () => {});
    const writes = calls.filter((c) => c.cmd === 'laya_write_chunk');
    expect(writes.map((w) => [w.headers!['x-laya-offset'], (w.args as number[]).length])).toEqual([
      ['0', CHUNK],
      [String(CHUNK), 10],
    ]);
  });

  it('stops on an HTTP error without asking Rust to finish', async () => {
    vi.stubGlobal('fetch', async () => new Response('nope', { status: 404 }));
    await expect(downloadLaya(ckpt, () => {})).rejects.toThrow(/model\.safetensors \(HTTP 404\)/);
    expect(calls.some((c) => c.cmd === 'laya_finish')).toBe(false);
  });
});
