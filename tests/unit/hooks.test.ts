// Guards the pre-push hook wiring (see AGENTS.md → Git): the hook is what
// keeps a failing `bun run check` off origin instead of failing CI. It lives
// in .githooks/ (not .git/hooks/) so every clone gets it via bun install.
import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';

const ROOT = join(__dirname, '../..');

describe('pre-push hook', () => {
  it('exists, is executable, and runs bun run check', () => {
    const path = join(ROOT, '.githooks/pre-push');
    expect(existsSync(path), '.githooks/pre-push is missing').toBe(true);
    // The invariant is the file's mode in git, not on the working tree's
    // filesystem: NTFS (Windows CI) carries no POSIX exec bits (AGENTS.md §2).
    const listed = execFileSync('git', ['ls-files', '-s', '.githooks/pre-push'], { cwd: ROOT }).toString();
    const mode = /^(\d{6}) /.exec(listed)?.[1];
    expect(mode && (parseInt(mode, 8) & 0o111) !== 0, 'the hook must carry the executable bit in git').toBe(true);
    expect(readFileSync(path, 'utf8'), 'the hook must run the full check').toContain('bun run check');
  });

  it('is installed by bun install, so every clone gets it', () => {
    const pkg = JSON.parse(readFileSync(join(ROOT, 'package.json'), 'utf8'));
    expect(pkg.scripts.postinstall).toContain('install-hooks.mjs');
    expect(existsSync(join(ROOT, 'scripts/install-hooks.mjs'))).toBe(true);
  });
});
