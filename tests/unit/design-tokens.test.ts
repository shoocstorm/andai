// Guards the theming contract (see AGENTS.md → Design system):
//  1. no color literals outside tokens.css (both themes must stay correct),
//  2. every var(--x) used is defined,
//  3. the light theme overrides every color token the dark theme defines.
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';

const SRC = join(__dirname, '../../src');
const walk = (dir: string): string[] =>
  readdirSync(dir).flatMap((f) => {
    const p = join(dir, f);
    return statSync(p).isDirectory() ? walk(p) : [p];
  });
const files = walk(SRC).filter((f) => /\.(css|tsx?)$/.test(f) && !/\.test\./.test(f));
const tokensCss = readFileSync(join(SRC, 'theme/tokens.css'), 'utf8');

const block = (selector: string) => {
  const start = tokensCss.indexOf(`${selector} {`);
  return tokensCss.slice(start, tokensCss.indexOf('\n}', start));
};
const defs = (css: string) => new Set([...css.matchAll(/^\s*(--[\w-]+)\s*:/gm)].map((m) => m[1]));
const darkTokens = defs(block(':root'));
const lightTokens = defs(block(":root[data-theme='light']"));

/** Strip regions and lines explicitly marked as intentional literals. */
function stripAllowed(src: string): string {
  return src
    .replace(/\/\* theme-literal: start \*\/[\s\S]*?\/\* theme-literal: end \*\//g, '')
    .split('\n')
    .filter((l) => !l.includes('theme-literal'))
    .join('\n');
}

describe('design tokens', () => {
  it('no hex color literals outside tokens.css (mark intentional ones with theme-literal)', () => {
    const offenders: string[] = [];
    for (const f of files) {
      if (f.endsWith('tokens.css')) continue;
      stripAllowed(readFileSync(f, 'utf8'))
        .split('\n')
        .forEach((line, i) => {
          // skip comments and CSS custom-property references like #root
          const code = line.replace(/\/\/.*$|\/\*.*?\*\//g, '');
          if (/#[0-9a-fA-F]{3,8}\b/.test(code) && !/#root\b/.test(code)) offenders.push(`${f.replace(SRC, 'src')}:${i + 1}: ${line.trim()}`);
        });
    }
    expect(offenders).toEqual([]);
  });

  it('every var(--token) used in src is defined in tokens.css', () => {
    const inline = new Set<string>();
    const used = new Set<string>();
    for (const f of files) {
      const src = readFileSync(f, 'utf8');
      for (const m of src.matchAll(/var\((--[\w-]+)/g)) used.add(m[1]);
      // tokens set inline at runtime, e.g. style={{ '--fill': ... }} or ['--dot' as string]
      for (const m of src.matchAll(/['"](--[\w-]+)['"]\s*(?:as string\])?\s*[:\]]/g)) inline.add(m[1]);
      for (const m of src.matchAll(/^\s*(--[\w-]+)\s*:/gm)) inline.add(m[1]);
    }
    const missing = [...used].filter((t) => !darkTokens.has(t) && !inline.has(t));
    expect(missing).toEqual([]);
  });

  it('the light theme overrides every color token of the dark theme', () => {
    const isColor = (t: string) => {
      const m = block(':root').match(new RegExp(`${t}\\s*:\\s*([^;]+);`));
      return !!m && /#|gradient|rgb/.test(m[1]) && !t.startsWith('--font');
    };
    // pastel orb gradient is intentionally shared
    const shared = new Set(['--grad-orb']);
    const missing = [...darkTokens].filter((t) => isColor(t) && !shared.has(t) && !lightTokens.has(t));
    expect(missing).toEqual([]);
  });

  it('light-theme text colors are dark enough to read on light surfaces', () => {
    const hex = (t: string) => block(":root[data-theme='light']").match(new RegExp(`${t}:\\s*(#[0-9a-f]{6})`))![1];
    const lum = (h: string) => {
      const [r, g, b] = [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16) / 255).map((c) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
      return 0.2126 * r + 0.7152 * g + 0.0722 * b;
    };
    const contrast = (a: string, b: string) => {
      const [x, y] = [lum(a), lum(b)].sort((m, n) => n - m);
      return (x + 0.05) / (y + 0.05);
    };
    const bg = hex('--panel');
    // WCAG AA: 4.5 for body text, 3 for large/secondary UI text
    expect(contrast(hex('--text'), bg)).toBeGreaterThanOrEqual(4.5);
    expect(contrast(hex('--text-2'), bg)).toBeGreaterThanOrEqual(4.5);
    expect(contrast(hex('--text-3'), bg)).toBeGreaterThanOrEqual(4.5);
    for (const accent of ['--blue', '--violet', '--amber', '--red', '--green']) {
      expect.soft(contrast(hex(accent), bg), accent).toBeGreaterThanOrEqual(4.5);
    }
  });
});
