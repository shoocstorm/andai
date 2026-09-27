// Searching and inspecting Hugging Face (llm/hub.ts), against responses
// shaped like the real API's (probed 2026-09-27). What can't run is
// refused with a reason, and nothing but huggingface.co is ever requested.
import { afterEach, describe, expect, it, vi } from 'vitest';
import { ggufVariants, inspectHub, mlxChecks, readTree, recommend, searchHub, WLLAMA_MAX_BYTES } from './hub';
import { hubModelUrl, hubSearchUrl, hubTreeUrl } from './models';

const COMMIT = 'd7f544eead698dbd1f15126ef60b45a1e1933222';
const sha = (c: string) => c.repeat(64);
const lfs = (path: string, size: number, c = 'a') => ({ type: 'file', path, size, oid: 'x', lfs: { oid: sha(c), size } });
const small = (path: string, size = 900) => ({ type: 'file', path, size, oid: 'git-blob-sha1' });

const qwenConfig = { model_type: 'qwen3', num_hidden_layers: 28, tie_word_embeddings: true, quantization: { group_size: 64, bits: 4 } };
const qwenTokenizer = { eos_token: '<|im_end|>', chat_template: "{{ '<|im_start|>' }}{% if enable_thinking is false %}{% endif %}" };

type Routes = Record<string, unknown>;
let requested: string[] = [];
function serve(routes: Routes) {
  requested = [];
  vi.stubGlobal('fetch', async (url: string, init?: RequestInit) => {
    requested.push(url);
    expect(init?.credentials).toBe('omit');
    const hit = Object.entries(routes).find(([k]) => url.startsWith(k));
    if (!hit) return new Response('not found', { status: 404 });
    const body = hit[1];
    if (typeof body === 'number') return new Response('', { status: body });
    return new Response(typeof body === 'string' ? body : JSON.stringify(body));
  });
}

afterEach(() => vi.unstubAllGlobals());

describe('URL builders', () => {
  it('encode the search and refuse ids that could escape the API path', () => {
    const url = new URL(hubSearchUrl('qwen3 & <x>', 'gguf'));
    expect(url.origin + url.pathname).toBe('https://huggingface.co/api/models');
    expect(url.searchParams.get('search')).toBe('qwen3 & <x>');
    expect(url.searchParams.get('filter')).toBe('gguf');
    expect(url.searchParams.get('pipeline_tag')).toBe('text-generation');
    for (const bad of ['../x', 'a/../b', 'a/b/c', 'a', 'a/b?x=1', '-a/b']) {
      expect(() => hubModelUrl(bad), bad).toThrow();
    }
    expect(() => hubTreeUrl('a/b', 'main')).toThrow(/commit/);
  });
});

describe('searchHub', () => {
  it('lists public, ungated models with their downloads and license, and drops the rest', async () => {
    serve({
      'https://huggingface.co/api/models?': [
        { id: 'Qwen/Qwen3-1.7B-GGUF', downloads: 83844, likes: 70, gated: false, private: false, lastModified: '2025-05-09T07:16:01.000Z', tags: ['gguf', 'license:apache-2.0'] },
        { id: 'meta-llama/Llama-3.2-1B-GGUF', downloads: 9, likes: 1, gated: 'manual', private: false },
        { id: 'someone/private', downloads: 1, private: true },
        { id: '../../etc', downloads: 1 },
      ],
    });
    const r = await searchHub('qwen3', 'gguf');
    expect(r).toEqual([{ repo: 'Qwen/Qwen3-1.7B-GGUF', downloads: 83844, likes: 70, lastModified: '2025-05-09T07:16:01.000Z', license: 'apache-2.0' }]);
    expect(requested.every((u) => u.startsWith('https://huggingface.co/api/models?'))).toBe(true);
    expect(await searchHub(' q ', 'gguf')).toEqual([]);
  });

  it('says plainly when Hugging Face can’t be reached', async () => {
    vi.stubGlobal('fetch', async () => {
      throw new TypeError('Load failed');
    });
    await expect(searchHub('qwen3', 'mlx')).rejects.toThrow(/Couldn’t reach Hugging Face/);
  });
});

describe('GGUF files', () => {
  const files = readTree([
    small('README.md'),
    lfs('Qwen3-1.7B-Q8_0.gguf', 1_834_000_000),
    lfs('Qwen3-1.7B-Q4_K_M.gguf', 1_107_409_472),
    lfs('Qwen3-1.7B-BF16.gguf', 3_447_349_568),
    lfs('big-00001-of-00002.gguf', 1_000),
    lfs('mmproj-F16.gguf', 1_000),
    { type: 'directory', path: 'sub' },
    lfs('../evil.gguf', 5),
  ]);

  it('reads the tree, keeping the LFS sha256 and dropping odd paths', () => {
    expect(files.map((f) => f.path)).toEqual(['README.md', 'Qwen3-1.7B-Q8_0.gguf', 'Qwen3-1.7B-Q4_K_M.gguf', 'Qwen3-1.7B-BF16.gguf', 'big-00001-of-00002.gguf', 'mmproj-F16.gguf']);
    expect(files[0].sha256).toBeNull();
  });

  it('offers single files, marks the ones over wllama’s 2 GB, and recommends a 4-bit K-quant', () => {
    const v = ggufVariants(files);
    expect(v.map((x) => [x.quant, x.fits])).toEqual([
      ['Q4_K_M', true],
      ['Q8_0', true],
      ['BF16', false],
    ]);
    expect(recommend(v)).toBe('Qwen3-1.7B-Q4_K_M.gguf');
    expect(recommend(v.filter((x) => x.quant !== 'Q4_K_M'))).toBe('Qwen3-1.7B-Q8_0.gguf');
    expect(recommend([{ path: 'x.gguf', bytes: WLLAMA_MAX_BYTES + 1, sha256: sha('a'), quant: 'F16', fits: false }])).toBeNull();
  });
});

describe('inspectHub', () => {
  const model = (extra: object) => ({ id: 'x', sha: COMMIT, gated: false, private: false, downloads: 5, likes: 1, cardData: { license: 'apache-2.0' }, ...extra });

  it('pins a GGUF repo to its current commit and reads its architecture and template', async () => {
    serve({
      [`https://huggingface.co/api/models/unsloth/Qwen3-1.7B-GGUF?`]: model({ gguf: { architecture: 'qwen3', context_length: 40960, chat_template: '<|im_start|>{% if enable_thinking %}' } }),
      [`https://huggingface.co/api/models/unsloth/Qwen3-1.7B-GGUF/tree/${COMMIT}`]: [lfs('Qwen3-1.7B-Q4_K_M.gguf', 1_107_409_472)],
    });
    const m = await inspectHub('unsloth/Qwen3-1.7B-GGUF', 'gguf');
    expect(m).toMatchObject({ repo: 'unsloth/Qwen3-1.7B-GGUF', commit: COMMIT, ok: true, license: 'apache-2.0' });
    expect(m.gguf).toMatchObject({ architecture: 'qwen3', contextLength: 40960, thinking: true, recommended: 'Qwen3-1.7B-Q4_K_M.gguf' });
  });

  it('refuses a gated repo, and one whose files are all too large, saying why', async () => {
    serve({
      'https://huggingface.co/api/models/meta-llama/Llama-3.2-1B-GGUF?': model({ gated: 'manual', gguf: {} }),
      [`https://huggingface.co/api/models/meta-llama/Llama-3.2-1B-GGUF/tree/${COMMIT}`]: [lfs('a-Q4_K_M.gguf', 3e9)],
    });
    const m = await inspectHub('meta-llama/Llama-3.2-1B-GGUF', 'gguf');
    expect(m.ok).toBe(false);
    const blocks = m.checks.filter((c) => c.level === 'block').map((c) => c.text).join(' ');
    expect(blocks).toMatch(/login/);
    expect(blocks).toMatch(/2 GB/);
  });

  it('reads an MLX repo’s small files at the pinned commit and builds what Rust is sent', async () => {
    const repo = 'mlx-community/Qwen3-4B-4bit';
    serve({
      [`https://huggingface.co/api/models/${repo}?`]: model({}),
      [`https://huggingface.co/api/models/${repo}/tree/${COMMIT}`]: [small('config.json'), small('tokenizer_config.json'), lfs('tokenizer.json', 11_422_654, 'b'), lfs('model.safetensors', 2_263_022_000, 'c')],
      [`https://huggingface.co/${repo}/resolve/${COMMIT}/config.json`]: JSON.stringify(qwenConfig),
      [`https://huggingface.co/${repo}/resolve/${COMMIT}/tokenizer_config.json`]: JSON.stringify(qwenTokenizer),
    });
    const m = await inspectHub(repo, 'mlx');
    expect(m.ok).toBe(true);
    expect(m.mlx).toMatchObject({ layers: 28, bits: 4, thinking: true });
    expect(m.mlx!.spec.files).toEqual([
      { path: 'tokenizer.json', bytes: 11_422_654, sha256: sha('b') },
      { path: 'model.safetensors', bytes: 2_263_022_000, sha256: sha('c') },
    ]);
    expect(m.mlx!.spec.inline.map((f) => f.path)).toEqual(['config.json', 'tokenizer_config.json']);
    expect(m.mlx!.spec.commit).toBe(COMMIT);
    expect(requested.every((u) => u.startsWith('https://huggingface.co/'))).toBe(true);
  });

  it('says why an MLX repo can’t run natively', async () => {
    const repo = 'mlx-community/Llama-3.2-1B-4bit';
    serve({
      [`https://huggingface.co/api/models/${repo}?`]: model({}),
      [`https://huggingface.co/api/models/${repo}/tree/${COMMIT}`]: [small('config.json'), small('tokenizer_config.json'), lfs('tokenizer.json', 5, 'b'), lfs('model.safetensors', 5, 'c')],
      [`https://huggingface.co/${repo}/resolve/${COMMIT}/config.json`]: JSON.stringify({ ...qwenConfig, model_type: 'llama' }),
      [`https://huggingface.co/${repo}/resolve/${COMMIT}/tokenizer_config.json`]: JSON.stringify({ eos_token: '<|eot_id|>', chat_template: '<|start_header_id|>' }),
    });
    const m = await inspectHub(repo, 'mlx');
    expect(m.ok).toBe(false);
    expect(m.mlx).toBeUndefined();
    const why = m.checks.filter((c) => c.level === 'block').map((c) => c.text).join(' ');
    expect(why).toMatch(/llama model/);
    expect(why).toMatch(/ChatML/);
  });
});

describe('mlxChecks mirrors Rust’s config check', () => {
  it('accepts unquantized norms but not per-layer bit widths or other schemes', () => {
    expect(mlxChecks({ ...qwenConfig, quantization: { ...qwenConfig.quantization, 'model.norm': false } }, qwenTokenizer, null).checks.every((c) => c.level !== 'block')).toBe(true);
    const blocked = (q: object) => mlxChecks({ ...qwenConfig, quantization: q }, qwenTokenizer, null).checks.some((c) => c.level === 'block');
    expect(blocked({ group_size: 64, bits: 4, 'model.layers.0.mlp.down_proj': { bits: 8 } })).toBe(true);
    expect(blocked({ group_size: 64, bits: 4, mode: 'mxfp4' })).toBe(true);
    expect(blocked({})).toBe(true);
    expect(mlxChecks(qwenConfig, { eos_token: '<|im_end|>' }, '<|im_start|>').thinking).toBe(false);
  });
});
