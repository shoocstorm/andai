// Model catalog — ported from wllama-chat/app.js. wllama loads single GGUF
// files up to 2 GB, so larger models need gguf-split shards.

export type ModelDef = {
  id: string;
  name: string;
  family: string;
  size: string;
  bytes: number;
  /** Pinned to an immutable commit: `resolve/<40-hex sha>/…` (AGENTS.md §9). */
  url: string;
  /** The file's sha256 (Hugging Face's LFS oid); checked after every download. */
  sha256: string;
  /** URLs earlier versions downloaded from; copies cached under them can be removed. */
  legacyUrls: string[];
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

// A commit URL always serves the same bytes; `main` can be moved under us.
const HF = (repo: string, commit: string, file: string) => `https://huggingface.co/${repo}/resolve/${commit}/${file}`;
const MAIN = (repo: string, file: string) => `https://huggingface.co/${repo}/resolve/main/${file}`;

export const MODELS: ModelDef[] = [
  {
    id: 'qwen3-0.6b',
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

export const DEFAULT_MODEL = MODELS[0].id;
export const modelById = (id: string | null | undefined) => MODELS.find((m) => m.id === id);

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

/** A Laya checkpoint file at its pinned commit: the only URLs a Laya download fetches. */
export function layaFileUrl(repo: string, commit: string, path: string): string {
  if (!/^[\w.-]+\/[\w.-]+$/.test(repo) || !/^[0-9a-f]{40}$/.test(commit) || !/^[\w.-]+(\/[\w.-]+)*$/.test(path) || path.split('/').includes('..')) {
    throw new Error(`Not a pinned Laya file: ${repo}@${commit}/${path}`);
  }
  return HF(repo, commit, path);
}
