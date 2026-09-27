// Test-only cache of the checkpoints Rust downloads and verifies (Laya
// decision models, and MLX chat models), so eval, e2e and bench runs (each
// with a fresh ANDAI_DATA_DIR, AGENTS.md §6) don't download 0.6–1 GB every
// time. It only ever holds folders Rust already verified (their `.verified`
// marker is written by laya/store.rs after the sha256 check), copied out after
// a run and cloned back in before the next one. It's never the user's app
// data (§1.5).
import { constants, copyFileSync, existsSync, mkdirSync, readdirSync, rmSync, statSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';

const CACHE = join(homedir(), '.cache/andai-test');

/** Where the app keeps a checkpoint (`<data>/models/<kind>/<id>`), by id; null for a wllama model. */
function kindOf(id) {
  if (typeof id !== 'string') return null;
  if (/^laya-[a-z]+$/.test(id)) return 'laya';
  if (/^[a-z0-9.-]+-mlx$/.test(id)) return 'llm';
  return null;
}

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

/** Before a run: put a cached, verified checkpoint where the app looks. */
export function seedCheckpoint(appDataDir, id) {
  const kind = kindOf(id);
  if (!kind || !existsSync(join(CACHE, kind, id, '.verified'))) return false;
  copyTree(join(CACHE, kind, id), join(appDataDir, 'models', kind, id));
  console.log(`[checkpoint-cache] ${id} seeded from ${join(CACHE, kind)}`);
  return true;
}

/** After a run: keep the checkpoint the app downloaded and verified. */
export function keepCheckpoint(appDataDir, id) {
  const kind = kindOf(id);
  if (!kind) return;
  const dir = join(appDataDir, 'models', kind, id);
  const cached = join(CACHE, kind, id);
  if (!existsSync(join(dir, '.verified')) || existsSync(join(cached, '.verified'))) return;
  rmSync(cached, { recursive: true, force: true });
  copyTree(dir, cached);
  console.log(`[checkpoint-cache] ${id} kept in ${join(CACHE, kind)}`);
}
