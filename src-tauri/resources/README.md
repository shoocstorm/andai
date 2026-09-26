# resources/

`mlx.metallib` is MLX's precompiled Metal kernel library (about 183 MB). The
Laya decision model (`src/laya/`) runs on MLX, and MLX needs this file at
runtime — without it, it compiles Metal kernels on the fly, which makes the
first decision ~42 ms P50 instead of ~9 ms (measured, see AGENTS.md §2).

MLX looks for it next to the binary (and in a `Resources/` folder beside it),
not in the app bundle's `Contents/Resources`, so the Apple Silicon release
build copies this file into the bundle via `tauri.laya.conf.json`
(`bun run build:mac-arm64`). The copy is refreshed by
`scripts/mlx-metallib.mjs` before bundling, because `bundle.resources` runs at
build.rs time, before mlx-sys may have built it.

Intel and Windows builds don't ship it (MLX is Apple Silicon only).
