import { createHash } from 'node:crypto';
import { describe, expect, it } from 'vitest';
import { Sha256, verifyBlobs } from './integrity';

const enc = (s: string) => new TextEncoder().encode(s);
const node = (d: Uint8Array) => createHash('sha256').update(d).digest('hex');

describe('Sha256', () => {
  it('matches the FIPS 180-2 test vectors', () => {
    expect(new Sha256().hex()).toBe('e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855');
    expect(new Sha256().update(enc('abc')).hex()).toBe('ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad');
    expect(new Sha256().update(enc('abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq')).hex()).toBe(
      '248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1',
    );
  });

  it('gives the same digest however the input is chunked, at every padding boundary', () => {
    for (const n of [55, 56, 63, 64, 65, 119, 120, 1000, 4099]) {
      const data = Uint8Array.from({ length: n }, (_, i) => (i * 31 + 7) & 0xff);
      const h = new Sha256();
      for (let i = 0; i < n; i += 13) h.update(data.subarray(i, i + 13));
      expect(h.hex(), `n=${n}`).toBe(node(data));
    }
  });
});

describe('verifyBlobs', () => {
  const data = Uint8Array.from({ length: 70_000 }, (_, i) => (i * 7) & 0xff);
  const pin = { bytes: data.length, sha256: node(data) };

  it('accepts a model whose shards match the pin, reporting progress', async () => {
    const seen: number[] = [];
    const blobs = [new Blob([data.subarray(0, 30_000)]), new Blob([data.subarray(30_000)])];
    expect(await verifyBlobs(blobs, pin, (d) => seen.push(d))).toEqual({ ok: true });
    expect(seen.at(-1)).toBe(data.length);
  });

  it('rejects a file of the wrong size without hashing it', async () => {
    const v = await verifyBlobs([new Blob([data.subarray(1)])], pin);
    expect(v).toEqual({ ok: false, reason: expect.stringContaining('size') });
  });

  it('rejects a same-size file with different bytes', async () => {
    const tampered = data.slice();
    tampered[12_345] ^= 1;
    const v = await verifyBlobs([new Blob([tampered])], pin);
    expect(v).toEqual({ ok: false, reason: expect.stringContaining('sha256') });
  });
});
