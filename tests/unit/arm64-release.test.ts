// Guards the Apple Silicon release config (see AGENTS.md §2 → MLX). The Tauri
// CLI derives MACOSX_DEPLOYMENT_TARGET from bundle.macOS.minimumSystemVersion
// (default 10.13), and MLX's CMake refuses to configure against anything below
// 14.0 — which failed the v0.1.2 release build. The arm64 overlay is where the
// deployment target must be raised: only the Apple Silicon build compiles MLX,
// and every Apple Silicon Mac supports macOS 14. Intel and Windows builds have
// no MLX and keep Tauri's default.
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';

const ROOT = join(__dirname, '../..');
const overlay = JSON.parse(readFileSync(join(ROOT, 'src-tauri/tauri.laya.conf.json'), 'utf8'));

const major = (v: unknown) => Number(String(v).split('.')[0]);

describe('Apple Silicon release overlay (tauri.laya.conf.json)', () => {
  it('pins minimumSystemVersion to macOS 14+ (MLX refuses a lower deployment target)', () => {
    const v = overlay.bundle?.macOS?.minimumSystemVersion;
    expect(v).toBeTruthy();
    expect(major(v)).toBeGreaterThanOrEqual(14);
  });
  it('bundles mlx.metallib into the app (same overlay, checked again in release.yml)', () => {
    expect(overlay.bundle?.macOS?.files?.['Resources/mlx.metallib']).toBe('resources/mlx.metallib');
  });
});
