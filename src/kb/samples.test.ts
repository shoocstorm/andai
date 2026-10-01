import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { SAMPLES, sampleByProject } from './samples';

const root = join(__dirname, '../..');

describe('sample knowledge bases', () => {
  it('match the closed list Rust installs from, by id, project and name', () => {
    const rust = readFileSync(join(root, 'src-tauri/src/samples.rs'), 'utf8');
    const pinned = [...rust.matchAll(/Sample \{ id: "([\w-]+)", slug: "([\w-]+)", name: "([^"]+)"/g)].map((m) => [m[1], `andai-${m[2]}`, m[3]]);
    expect(pinned).toHaveLength(SAMPLES.length);
    expect(SAMPLES.map((s) => [s.id, s.project, s.name])).toEqual(pinned);
  });

  it('suggest only eval questions asked of the same kind of knowledge base', () => {
    const { cases } = JSON.parse(readFileSync(join(root, 'tests/fixtures/eval/cases.json'), 'utf8')) as { cases: { kb: string; prompt: string; history?: unknown[] }[] };
    // The documents sample holds the eval's `docs` and `large` files (src-tauri/src/samples.rs).
    const kb = { document: ['docs', 'large'], code: ['code'], mixed: ['mixed'] } as const;
    for (const s of SAMPLES) {
      for (const q of s.questions) {
        const c = cases.find((x) => x.prompt === q);
        expect(c, q).toBeDefined();
        expect(kb[s.kind], q).toContain(c!.kb);
        expect(c!.history ?? [], q).toEqual([]);
      }
    }
  });

  it('recognizes an added sample by its ug project', () => {
    expect(sampleByProject('andai-tidewater-ferries-code')?.id).toBe('tidewater-code');
    expect(sampleByProject('andai-my-docs')).toBeUndefined();
    expect(sampleByProject(null)).toBeUndefined();
  });
});
