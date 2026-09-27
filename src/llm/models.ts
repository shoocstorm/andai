// Model catalog. Two engines:
// - `wllama`: a GGUF file run by llama.cpp compiled to WebAssembly, inside the
//   webview, on every platform. wllama loads single files up to 2 GB, so
//   larger models need gguf-split shards.
// - `mlx`: an MLX checkpoint run natively by Rust on the Mac's GPU
//   (src-tauri/src/llm/, Apple Silicon only). Its files, sizes and sha256
//   live in the Rust catalog, which verifies every download.

type Common = {
  id: string;
  name: string;
  family: string;
  size: string;
  bytes: number;
  note: string;
  /** Qwen3-style `<think>` support, toggled by `enable_thinking`. */
  thinking: boolean;
  n_ctx: number;
  /**
   * Can score decisions (llm/decide.ts): a chat model that answers a lettered
   * choice with the letter. Every readout still checks that each letter was
   * scored, so a model that can't fails closed.
   */
  decider?: boolean;
};

export type WllamaDef = Common & {
  engine: 'wllama';
  /** Pinned to an immutable commit: `resolve/<40-hex sha>/…` (AGENTS.md §9). */
  url: string;
  /** The file's sha256 (Hugging Face's LFS oid); checked after every download. */
  sha256: string;
  /** URLs earlier versions downloaded from; copies cached under them can be removed. */
  legacyUrls: string[];
};

export type MlxDef = Common & {
  engine: 'mlx';
  /** The checkpoint id in the Rust catalog (src-tauri/src/llm/catalog.rs). */
  native: string;
  /** The repository at its pinned commit: the cache key for `downloaded`, and shown to the user. */
  url: string;
};

export type ModelDef = WllamaDef | MlxDef;

// A commit URL always serves the same bytes; `main` can be moved under us.
const HF = (repo: string, commit: string, file: string) => `https://huggingface.co/${repo}/resolve/${commit}/${file}`;
const MAIN = (repo: string, file: string) => `https://huggingface.co/${repo}/resolve/main/${file}`;

/**
 * Measured on an Apple M5 Max (2026-09-27): Qwen3 1.7B generates about 350
 * tok/s natively on MLX against 30–65 tok/s in wllama on WebGPU, and reads a
 * prompt 50× faster (docs/performance.md, *Engine*).
 */
export const MODELS: ModelDef[] = [
  {
    id: 'qwen3-1.7b-mlx',
    engine: 'mlx',
    native: 'qwen3-1.7b-mlx',
    name: 'Qwen3 1.7B · MLX',
    family: '4-bit · 1.7B params',
    size: '980 MB',
    bytes: 979_513_507,
    url: 'https://huggingface.co/mlx-community/Qwen3-1.7B-4bit/tree/3b1b1768f8f8cf8351c712464f906e86c2b8269e',
    note: 'Smarter than 0.6B and fast on a Mac: runs natively on the GPU with MLX, about 350 tok/s on an M5 Max.',
    thinking: true,
    n_ctx: 4096,
    decider: true,
  },
  {
    id: 'qwen3-0.6b-mlx',
    engine: 'mlx',
    native: 'qwen3-0.6b-mlx',
    name: 'Qwen3 0.6B · MLX',
    family: '8-bit · 596M params',
    size: '645 MB',
    bytes: 644_876_804,
    url: 'https://huggingface.co/mlx-community/Qwen3-0.6B-8bit/tree/11de96878523501bcaa86104e3c186de07ff9068',
    note: 'Fastest: runs natively on the GPU with MLX, about 445 tok/s on an M5 Max. A good decision model.',
    thinking: true,
    n_ctx: 4096,
    decider: true,
  },
  {
    id: 'qwen3-0.6b',
    engine: 'wllama',
    name: 'Qwen3 0.6B',
    family: 'Q8_0 · 596M params',
    size: '639 MB',
    bytes: 639_446_688,
    url: HF('Qwen/Qwen3-0.6B-GGUF', '23749fefcc72300e3a2ad315e1317431b06b590a', 'Qwen3-0.6B-Q8_0.gguf'),
    sha256: '9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031',
    legacyUrls: [MAIN('Qwen/Qwen3-0.6B-GGUF', 'Qwen3-0.6B-Q8_0.gguf')],
    note: 'Fast default. Good for grounded Q&A over your knowledge base.',
    thinking: true,
    n_ctx: 4096,
    decider: true,
  },
  {
    id: 'qwen3-1.7b',
    engine: 'wllama',
    name: 'Qwen3 1.7B',
    family: 'Q4_K_M · 1.7B params',
    size: '1.1 GB',
    bytes: 1_107_409_472,
    url: HF('unsloth/Qwen3-1.7B-GGUF', 'd7f544eead698dbd1f15126ef60b45a1e1933222', 'Qwen3-1.7B-Q4_K_M.gguf'),
    sha256: 'b139949c5bd74937ad8ed8c8cf3d9ffb1e99c866c823204dc42c0d91fa181897',
    legacyUrls: [MAIN('unsloth/Qwen3-1.7B-GGUF', 'Qwen3-1.7B-Q4_K_M.gguf')],
    note: 'Noticeably smarter, about half the speed. Best answers on a fast computer.',
    thinking: true,
    n_ctx: 4096,
    decider: true,
  },
  {
    id: 'stories-260k',
    engine: 'wllama',
    name: 'TinyStories 260K',
    family: 'F32 · 260K params',
    size: '1.2 MB',
    bytes: 1_185_376,
    // ggml-org/models was renamed to models-moved; pin the new name directly.
    url: HF('ggml-org/models-moved', '499bc8821c6b12b4e53c5bffcb21ec206f212d81', 'tinyllamas/stories260K.gguf'),
    sha256: '270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d',
    legacyUrls: [MAIN('ggml-org/models', 'tinyllamas/stories260K.gguf')],
    note: 'Smoke test only — downloads in a second, talks nonsense.',
    thinking: false,
    n_ctx: 1024,
  },
];

/** The portable default (every platform, and the e2e runs, AGENTS.md §5). */
export const DEFAULT_MODEL = 'qwen3-0.6b';
export const modelById = (id: string | null | undefined) => MODELS.find((m) => m.id === id);
export const isMlx = (def: ModelDef | undefined): def is MlxDef => def?.engine === 'mlx';
/** The models this computer can run: MLX ones only where Rust reports MLX (Apple Silicon). */
export const availableModels = (mlx: boolean) => MODELS.filter((m) => mlx || m.engine !== 'mlx');
/** What to suggest first: natively on MLX where it runs (much faster), else the portable default. */
export const recommendedModel = (mlx: boolean): ModelDef => modelById(mlx ? 'qwen3-1.7b-mlx' : DEFAULT_MODEL)!;

/**
 * Laya decision models (src-tauri/src/laya/): encoders that score a choice in
 * one forward pass, Apple Silicon only. Files, sizes and sha256 live in the
 * Rust catalog, which verifies every download; this is how they're shown.
 * Timings measured with `bun run test:laya` on an M5 Max (2026-09-26).
 */
export type LayaDef = { id: string; name: string; family: string; note: string };

export const LAYA_MODELS: LayaDef[] = [
  {
    id: 'laya-multilingual',
    name: 'Laya Multilingual',
    family: 'mmBERT-base · 322M params · FP16',
    note: 'Decides in about 10 ms. Reads up to 1,024 tokens, in any language.',
  },
  {
    id: 'laya-en',
    name: 'Laya English',
    family: 'ModernBERT-large · 421M params · FP16',
    note: 'Decides in about 20 ms. English only, reads up to 512 tokens.',
  },
];

export const layaById = (id: string | null | undefined) => LAYA_MODELS.find((m) => m.id === id);

/** A checkpoint file at its pinned commit: the only URLs a Laya or MLX download fetches. */
export function pinnedFileUrl(repo: string, commit: string, path: string): string {
  if (!/^[\w.-]+\/[\w.-]+$/.test(repo) || !/^[0-9a-f]{40}$/.test(commit) || !/^[\w.-]+(\/[\w.-]+)*$/.test(path) || path.split('/').includes('..')) {
    throw new Error(`Not a pinned checkpoint file: ${repo}@${commit}/${path}`);
  }
  return HF(repo, commit, path);
}
