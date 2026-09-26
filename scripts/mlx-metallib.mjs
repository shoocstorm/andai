// Copies the mlx.metallib that the release build of mlx-sys just produced to
// src-tauri/resources/, where tauri.laya.conf.json bundles it into
// Andai.app/Contents/Resources (laya/worker.rs points MLX at it). Runs as the
// overlay's beforeBundleCommand, after cargo and before bundling: Tauri's
// `bundle.resources` would be copied at build.rs time, before mlx-sys may
// have finished. Apple Silicon builds only (AGENTS.md §2).
import { copyFileSync, mkdirSync, readdirSync, statSync } from 'node:fs';
import { join, resolve } from 'node:path';

const root = resolve(import.meta.dirname, '..');
const target = join(root, 'src-tauri/target');
const found = [];
// target/release/… or target/<triple>/release/…
for (const base of [target, ...readdirSync(target).map((d) => join(target, d))]) {
  const build = join(base, 'release/build');
  let dirs = [];
  try {
    dirs = readdirSync(build).filter((d) => d.startsWith('mlx-sys-'));
  } catch {
    continue;
  }
  for (const d of dirs) {
    const lib = join(build, d, 'out/build/lib/mlx.metallib');
    try {
      found.push({ lib, mtime: statSync(lib).mtimeMs });
    } catch {}
  }
}
if (!found.length) {
  console.error('[mlx-metallib] no mlx.metallib under src-tauri/target/**/release/build/mlx-sys-*: build for aarch64-apple-darwin first');
  process.exit(1);
}
const newest = found.sort((a, b) => b.mtime - a.mtime)[0].lib;
mkdirSync(join(root, 'src-tauri/resources'), { recursive: true });
copyFileSync(newest, join(root, 'src-tauri/resources/mlx.metallib'));
console.log(`[mlx-metallib] ${newest} → src-tauri/resources/mlx.metallib (${(statSync(newest).size / 1e6).toFixed(0)} MB)`);
