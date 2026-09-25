// Model catalog — ported from wllama-chat/app.js. wllama loads single GGUF
// files up to 2 GB, so larger models need gguf-split shards.

export type ModelDef = {
  id: string;
  name: string;
  family: string;
  size: string;
  bytes: number;
  url: string;
  note: string;
  /** Qwen3-style `<think>` support, toggled by `enable_thinking`. */
  thinking: boolean;
  n_ctx: number;
};

const HF = (repo: string, file: string) => `https://huggingface.co/${repo}/resolve/main/${file}`;

export const MODELS: ModelDef[] = [
  {
    id: 'qwen3-0.6b',
    name: 'Qwen3 0.6B',
    family: 'Q8_0 · 596M params',
    size: '639 MB',
    bytes: 639_446_688,
    url: HF('Qwen/Qwen3-0.6B-GGUF', 'Qwen3-0.6B-Q8_0.gguf'),
    note: 'Fast default. Good for grounded Q&A over your knowledge base.',
    thinking: true,
    n_ctx: 4096,
  },
  {
    id: 'qwen3-1.7b',
    name: 'Qwen3 1.7B',
    family: 'Q4_K_M · 1.7B params',
    size: '1.1 GB',
    bytes: 1_107_409_472,
    url: HF('unsloth/Qwen3-1.7B-GGUF', 'Qwen3-1.7B-Q4_K_M.gguf'),
    note: 'Noticeably smarter, about half the speed. Best answers on a fast Mac.',
    thinking: true,
    n_ctx: 4096,
  },
  {
    id: 'stories-260k',
    name: 'TinyStories 260K',
    family: 'F32 · 260K params',
    size: '1.2 MB',
    bytes: 1_185_376,
    url: HF('ggml-org/models', 'tinyllamas/stories260K.gguf'),
    note: 'Smoke test only — downloads in a second, talks nonsense.',
    thinking: false,
    n_ctx: 1024,
  },
];

export const DEFAULT_MODEL = MODELS[0].id;
export const modelById = (id: string | null | undefined) => MODELS.find((m) => m.id === id);
