// Test-only cache of Laya checkpoints, so eval and e2e runs (each with a fresh
// ANDAI_DATA_DIR, AGENTS.md §6) don't download 0.6–0.8 GB every time. It only
// ever holds folders Rust already verified (their `.verified` marker is
// written by laya/store.rs after the sha256 check), copied out after a run and
// cloned back in before the next one. It's never the user's app data (§1.5).
import { constants, copyFileSync, existsSync, mkdirSync, readdirSync, rmSync, statSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';

const CACHE = join(homedir(), '.cache/andai-test/laya');
const isLaya = (id) => typeof id === 'string' && /^laya-[a-z]+$/.test(id);

function copyTree(from, to) {
  mkdirSync(to, { recursive: true });
  for (const name of readdirSync(from)) {
    const a = join(from, name);
    const b = join(to, name);
    if (statSync(a).isDirectory()) copyTree(a, b);
    // APFS clones the file (instant, no extra space); elsewhere it copies.
    else copyFileSync(a, b, constants.COPYFILE_FICLONE);
  }
}

/** Before a run: put a cached, verified checkpoint where the app looks (`<data>/models/laya/<id>`). */
export function seedLaya(appDataDir, id) {
  if (!isLaya(id) || !existsSync(join(CACHE, id, '.verified'))) return false;
  copyTree(join(CACHE, id), join(appDataDir, 'models/laya', id));
  console.log(`[laya-cache] ${id} seeded from ${CACHE}`);
  return true;
}

/** After a run: keep the checkpoint the app downloaded and verified. */
export function keepLaya(appDataDir, id) {
  const dir = join(appDataDir, 'models/laya', id);
  if (!isLaya(id) || !existsSync(join(dir, '.verified')) || existsSync(join(CACHE, id, '.verified'))) return;
  rmSync(join(CACHE, id), { recursive: true, force: true });
  copyTree(dir, join(CACHE, id));
  console.log(`[laya-cache] ${id} kept in ${CACHE}`);
}
