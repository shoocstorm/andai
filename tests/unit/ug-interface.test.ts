// Andai talks to ug only through ug's command line (`ug list`, `ug files`,
// `ug search`, …), never by reading ug's data folder: its file names and
// formats are ug's to change (AGENTS.md §2, "A knowledge base is a ug project").
import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';

const RUST = join(__dirname, '../../src-tauri/src');
/** ug's own storage: per-project files and folders under ~/.ug (its data home), and the field naming that folder. ug's program folder (`~/.local/share/ultragraph/.ug`, ug_install.rs) is not data. */
const UG_INTERNALS = /project\.json|graph\.json|indexed-tree\.json|cache\.json|\bugdb\b|dataDir|join\("\.ug"\)|~\/\.ug\b/;

describe('ug is used through its interface only', () => {
  it('no Rust source names a file or folder from inside ug’s data', () => {
    const files = readdirSync(RUST, { recursive: true })
      .map((f) => String(f).replaceAll('\\', '/'))
      .filter((f) => f.endsWith('.rs'))
      // The installer lays out ug's program folder (`~/.local/share/ultragraph/.ug`), as
      // install.sh does; it never reads a project. That is installing ug, not using it.
      .filter((f) => f !== 'ug_install.rs');
    const hits = files.flatMap((f) =>
      readFileSync(join(RUST, f), 'utf8')
        .split(/\r?\n/)
        .map((line, i) => ({ line, at: `${f}:${i + 1}` }))
        // Comments may explain what ug does; only code counts.
        .filter(({ line }) => !line.trim().startsWith('//') && UG_INTERNALS.test(line.replace(/\/\/.*$/, '')))
        .map(({ at, line }) => `${at}: ${line.trim()}`),
    );
    expect(hits).toEqual([]);
  });
});
