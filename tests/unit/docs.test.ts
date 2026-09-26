// Keeps the docs and the marketing site honest (AGENTS.md §8):
// every local link/image resolves, in-page anchors exist, and the site
// never presents the simulated Workflows preview as a shipped feature.
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

const ROOT = join(__dirname, '../..');
const SITE = join(ROOT, 'docs/andai-website/index.html');
const html = readFileSync(SITE, 'utf8');
const isLocal = (u: string) => !/^(https?:|mailto:|#|data:)/.test(u);

describe('andai-website', () => {
  it('every local image, icon and link resolves to a file', () => {
    const refs = [...html.matchAll(/\s(?:src|href)="([^"]+)"/g)].map((m) => m[1]).filter(isLocal);
    const missing = refs.filter((r) => !existsSync(resolve(dirname(SITE), r)));
    expect(refs.length).toBeGreaterThan(5);
    expect(missing).toEqual([]);
  });

  it('every in-page #anchor has a target', () => {
    const ids = new Set([...html.matchAll(/\sid="([^"]+)"/g)].map((m) => m[1]));
    const anchors = [...html.matchAll(/href="#([^"]+)"/g)].map((m) => m[1]);
    expect(anchors.filter((a) => !ids.has(a))).toEqual([]);
  });

  it('labels Workflows as a simulated preview', () => {
    const section = html.slice(html.indexOf('Workflows, coming next'), html.indexOf('id="how"'));
    expect(section).toMatch(/simulated preview/i);
    expect(html).toMatch(/Are workflows and tools real\?/);
  });

  it('every screenshot has alt text', () => {
    const imgs = [...html.matchAll(/<img\b[^>]*>/g)].map((m) => m[0]);
    expect(imgs.filter((i) => !/\salt="[^"]*"/.test(i))).toEqual([]);
  });
});

describe('docs', () => {
  it('relative links in docs/*.md resolve', () => {
    const missing: string[] = [];
    for (const f of ['docs/README.md', 'docs/features.md']) {
      const md = readFileSync(join(ROOT, f), 'utf8');
      for (const [, target] of md.matchAll(/\]\(([^)#\s]+)(?:#[^)]*)?\)/g)) {
        if (!isLocal(target)) continue;
        if (!existsSync(resolve(dirname(join(ROOT, f)), target))) missing.push(`${f} → ${target}`);
      }
    }
    expect(missing).toEqual([]);
  });

  it('features.md marks the Workflows preview as simulated', () => {
    const md = readFileSync(join(ROOT, 'docs/features.md'), 'utf8');
    expect(md).toMatch(/## Workflows & tools · Preview/);
    expect(md).toMatch(/nothing is\s+executed/);
  });
});
