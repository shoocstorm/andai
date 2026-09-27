// The models the user added from Hugging Face (llm/hub.ts finds them).
// - GGUF ones are kept here, in the webview's storage: their pinned URL and
//   sha256 guard the download exactly like a catalog model's (engine.ts
//   `openVerified`), and they run in wllama's sandbox.
// - MLX ones are kept by Rust (src-tauri/src/llm/custom.rs), which validated
//   them; this reads them from `useEngine().native`.
// Both reach `modelById`/`allModels` through `setCustomModels`.

import { invoke } from '@tauri-apps/api/core';
import { create } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';
import { evictModel, refreshCache, refreshNative, unloadDecider, unloadModel, useEngine } from './engine';
import type { GgufVariant, HubModel } from './hub';
import { isMlx, modelById, pinnedFileUrl, repoTreeUrl, setCustomModels, type MlxDef, type ModelDef, type WllamaDef } from './models';
import { nativeRemove, type NativeStatus } from './native';

const fmtSize = (bytes: number) => (bytes >= 1e9 ? `${(bytes / 1e9).toFixed(1)} GB` : `${Math.round(bytes / 1e6)} MB`);
const shortName = (repo: string) => repo.split('/')[1].replace(/[-_]?(GGUF|MLX)$/i, '');

/** A custom GGUF entry as stored; anything malformed is dropped on load. */
function sane(d: unknown): d is WllamaDef {
  const m = d as Partial<WllamaDef> | null;
  if (!m || m.engine !== 'wllama' || typeof m.id !== 'string' || !m.id.startsWith('hf-gguf-') || !m.source) return false;
  try {
    const path = m.url?.split(`/resolve/${m.source.commit}/`)[1] ?? '';
    return m.url === pinnedFileUrl(m.source.repo, m.source.commit, path) && /^[0-9a-f]{64}$/.test(m.sha256 ?? '') && typeof m.bytes === 'number' && m.bytes > 0;
  } catch {
    return false;
  }
}

type CustomState = { gguf: WllamaDef[] };

export const useCustomModels = create<CustomState>()(
  persist(() => ({ gguf: [] as WllamaDef[] }), {
    name: 'andai.customModels',
    storage: createJSONStorage(() => localStorage),
    merge: (persisted, current) => ({ ...current, gguf: (((persisted as CustomState | undefined)?.gguf ?? []) as unknown[]).filter(sane) }),
  }),
);

/** The id a GGUF file of a repo at a commit gets. */
export const ggufId = (repo: string, commit: string, path: string) =>
  `hf-gguf-${`${repo}--${path}`.toLowerCase().replace(/[^a-z0-9.-]+/g, '-')}-${commit.slice(0, 8)}`;

export function ggufDef(m: HubModel, v: GgufVariant): WllamaDef {
  const g = m.gguf!;
  return {
    id: ggufId(m.repo, m.commit, v.path),
    engine: 'wllama',
    name: `${shortName(m.repo)} · ${v.quant}`,
    family: `${v.quant} · ${g.architecture ?? 'GGUF'} · Hugging Face`,
    size: fmtSize(v.bytes),
    bytes: v.bytes,
    url: pinnedFileUrl(m.repo, m.commit, v.path),
    sha256: v.sha256,
    legacyUrls: [],
    note: `Added from ${m.repo} on Hugging Face. A third-party model Andai hasn’t reviewed.`,
    thinking: g.thinking,
    n_ctx: Math.max(1024, Math.min(4096, g.contextLength ?? 4096)),
    source: { repo: m.repo, commit: m.commit, license: m.license },
  };
}

/** The MLX models Rust keeps, as model definitions. */
export function mlxDefs(native: NativeStatus): MlxDef[] {
  return native.checkpoints.flatMap((c) =>
    c.custom
      ? [
          {
            id: c.id,
            engine: 'mlx' as const,
            native: c.id,
            name: `${shortName(c.repo)} · MLX`,
            family: `${c.custom.bits}-bit · ${c.custom.layers} layers · Hugging Face`,
            size: fmtSize(c.bytes),
            bytes: c.bytes,
            url: repoTreeUrl(c.repo, c.commit),
            note: `Added from ${c.repo} on Hugging Face. Runs natively with MLX. A third-party model Andai hasn’t reviewed.`,
            thinking: c.custom.thinking,
            n_ctx: 4096,
            source: { repo: c.repo, commit: c.commit, license: null },
          },
        ]
      : [],
  );
}

function sync() {
  setCustomModels([...useCustomModels.getState().gguf, ...mlxDefs(useEngine.getState().native)]);
}

let started = false;
/** Keeps `allModels` current; call once at startup, before restoring the last model. */
export function initCustomModels() {
  if (started) return;
  started = true;
  sync();
  useCustomModels.subscribe(sync);
  useEngine.subscribe((s, prev) => {
    if (s.native !== prev.native) sync();
  });
}

/** Adds a model the user picked; returns its id. MLX models are checked by Rust first. */
export async function addHubModel(m: HubModel, variant?: GgufVariant): Promise<string> {
  if (!m.ok) throw new Error('This model can’t be added.');
  if (m.format === 'gguf') {
    if (!variant?.fits) throw new Error('Pick a file up to 2 GB.');
    const def = ggufDef(m, variant);
    useCustomModels.setState((s) => ({ gguf: [...s.gguf.filter((d) => d.id !== def.id), def] }));
    await refreshCache();
    return def.id;
  }
  const added = await invoke<{ id: string }>('llm_add_custom', { spec: m.mlx!.spec });
  await refreshCache();
  return added.id;
}

/** Forgets a model the user added, and deletes its downloaded files. */
export async function removeCustomModel(id: string): Promise<void> {
  const def: ModelDef | undefined = modelById(id);
  if (!def?.source) throw new Error('Only models you added can be removed here.');
  const { loadedId, decider } = useEngine.getState();
  if (loadedId === id) await unloadModel();
  if (decider.loadedId === id) await unloadDecider();
  if (isMlx(def)) {
    await nativeRemove(def.native);
    await refreshNative();
  } else {
    await evictModel(id);
    useCustomModels.setState((s) => ({ gguf: s.gguf.filter((d) => d.id !== id) }));
  }
  await refreshCache();
}
