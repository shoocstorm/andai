// Keeps the docs and the marketing site honest (AGENTS.md §8):
// every local link/image resolves, in-page anchors exist, and the site
// never presents the simulated Workflows preview as a shipped feature.
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

const ROOT = join(__dirname, '../..');
const SITE_DIR = join(ROOT, 'docs/andai-website');
const SITE = join(SITE_DIR, 'index.html');
const html = readFileSync(SITE, 'utf8');
/** Every page of the site: the home page and the ones it links to (agent-loop.html, …). */
const pages = readdirSync(SITE_DIR)
  .filter((f) => f.endsWith('.html'))
  .map((f) => ({ file: f, html: readFileSync(join(SITE_DIR, f), 'utf8') }));
const isLocal = (u: string) => !/^(https?:|mailto:|#|data:)/.test(u);

describe('andai-website', () => {
  it('has more than the home page, and the home page links every page', () => {
    expect(pages.map((p) => p.file)).toContain('agent-loop.html');
    for (const p of pages.filter((x) => x.file !== 'index.html')) expect(html, p.file).toContain(`href="${p.file}"`);
  });

  it.each(pages.map((p) => [p.file, p.html]))('%s: every local image, icon and link resolves to a file', (_, page) => {
    const refs = [...page.matchAll(/\s(?:src|href)="([^"]+)"/g)].map((m) => m[1]).filter(isLocal);
    const missing = refs.filter((r) => !existsSync(resolve(SITE_DIR, r)));
    expect(refs.length).toBeGreaterThan(1);
    expect(missing).toEqual([]);
  });

  it.each(pages.map((p) => [p.file, p.html]))('%s: every in-page #anchor has a target', (_, page) => {
    const ids = new Set([...page.matchAll(/\sid="([^"]+)"/g)].map((m) => m[1]));
    const anchors = [...page.matchAll(/href="#([^"]+)"/g)].map((m) => m[1]);
    expect(anchors.filter((a) => !ids.has(a))).toEqual([]);
  });

  it('labels Workflows as a simulated preview', () => {
    const section = html.slice(html.indexOf('Workflows, coming next'), html.indexOf('id="how"'));
    expect(section).toMatch(/simulated preview/i);
    expect(html).toMatch(/Are workflows and tools real\?/);
  });

  it.each(pages.map((p) => [p.file, p.html]))('%s: every image has alt text, every diagram a title', (_, page) => {
    const imgs = [...page.matchAll(/<img\b[^>]*>/g)].map((m) => m[0]);
    expect(imgs.filter((i) => !/\salt="[^"]*"/.test(i))).toEqual([]);
    const diagrams = [...page.matchAll(/<svg\b[^>]*role="img"[^>]*>([\s\S]*?)<\/svg>/g)].map((m) => m[1]);
    expect(diagrams.filter((d) => !/<title\b/.test(d) || !/<desc\b/.test(d))).toEqual([]);
  });
});

describe('docs', () => {
  it('relative links in docs/*.md resolve', () => {
    const missing: string[] = [];
    const docs = readdirSync(join(ROOT, 'docs')).filter((f) => f.endsWith('.md'));
    expect(docs).toContain('performance.md');
    for (const f of docs.map((d) => `docs/${d}`)) {
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
