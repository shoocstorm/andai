// Finding a model on Hugging Face to add (Settings → Models → Add from
// Hugging Face): search, then inspect one repository at its current commit
// and say whether Andai can run it, and how.
//
// Security (AGENTS.md §9): only huggingface.co's API and pinned file URLs are
// fetched (models.ts builds every URL and checks repo ids and commits first);
// responses are read as data, never rendered as HTML or Markdown (a model
// card isn't shown at all); only public, ungated repos; only GGUF,
// safetensors and JSON, formats that can't run code. What the user picks is
// pinned to the commit seen here and checked by sha256 before it's used:
// GGUF in the webview like the catalog's (integrity.ts), MLX in Rust, which
// re-checks everything this file concludes (src-tauri/src/llm/custom.rs).

import { HF_ORIGIN, hubModelUrl, hubSearchUrl, hubTreeUrl, isRepo, pinnedFileUrl, type HubFormat } from './models';

export type HubResult = {
  repo: string;
  downloads: number;
  likes: number;
  lastModified: string | null;
  license: string | null;
};

/** One line of the compatibility verdict. `block` means it can't be added. */
export type Check = { level: 'ok' | 'warn' | 'block'; text: string };

export type GgufVariant = {
  path: string;
  bytes: number;
  sha256: string;
  /** Quantization read from the file name, e.g. Q4_K_M. */
  quant: string;
  /** Within wllama's single-file limit. */
  fits: boolean;
};

/** What the Rust side is sent to add an MLX model (llm/custom.rs `Spec`). */
export type MlxSpec = {
  repo: string;
  commit: string;
  files: { path: string; bytes: number; sha256: string }[];
  inline: { path: string; content: string }[];
};

export type HubModel = {
  repo: string;
  /** The commit everything below was read at; what gets pinned. */
  commit: string;
  format: HubFormat;
  license: string | null;
  downloads: number;
  likes: number;
  lastModified: string | null;
  checks: Check[];
  /** No `block` check: it can be added. */
  ok: boolean;
  gguf?: { architecture: string | null; contextLength: number | null; thinking: boolean; variants: GgufVariant[]; recommended: string | null };
  mlx?: { spec: MlxSpec; bytes: number; layers: number; bits: number; thinking: boolean };
};

/** wllama loads one GGUF file up to 2 GB (AGENTS.md §2). */
export const WLLAMA_MAX_BYTES = 2 * 1024 ** 3;
/** Rust's limits for a file sent inline (llm/custom.rs). */
export const MAX_INLINE_BYTES = 16 * 1024 ** 2;

const SHA256_RE = /^[0-9a-f]{64}$/;
const COMMIT_RE = /^[0-9a-f]{40}$/;

async function get(url: string, signal?: AbortSignal): Promise<Response> {
  // models.ts built it; checked again here so nothing else can slip through.
  if (!url.startsWith(HF_ORIGIN)) throw new Error('Refused a request outside huggingface.co.');
  let res: Response;
  try {
    res = await fetch(url, { signal, credentials: 'omit', referrerPolicy: 'no-referrer' });
  } catch (e) {
    if (signal?.aborted) throw e;
    throw new Error('Couldn’t reach Hugging Face. Check your connection and try again.');
  }
  if (res.status === 401 || res.status === 403) throw new Error('Hugging Face refused: this model needs a login, which Andai doesn’t use.');
  if (res.status === 404) throw new Error('Hugging Face has no such model (or it’s private).');
  if (!res.ok) throw new Error(`Hugging Face answered ${res.status}. Try again later.`);
  return res;
}

const getJson = async (url: string, signal?: AbortSignal): Promise<unknown> => (await get(url, signal)).json();

const str = (v: unknown) => (typeof v === 'string' ? v : null);
const num = (v: unknown) => (typeof v === 'number' && Number.isFinite(v) && v >= 0 ? v : 0);
const obj = (v: unknown): Record<string, unknown> => (v && typeof v === 'object' && !Array.isArray(v) ? (v as Record<string, unknown>) : {});
const licenseOf = (tags: unknown) => (Array.isArray(tags) ? (tags.map(str).find((t) => t?.startsWith('license:'))?.slice(8) ?? null) : null);

/** Public, ungated text-generation models matching `query`, most downloaded first. */
export async function searchHub(query: string, format: HubFormat, signal?: AbortSignal): Promise<HubResult[]> {
  if (query.trim().length < 2) return [];
  const data = await getJson(hubSearchUrl(query, format), signal);
  if (!Array.isArray(data)) throw new Error('Hugging Face sent an unexpected answer.');
  return data.flatMap((raw) => {
    const m = obj(raw);
    const repo = str(m.id);
    if (!repo || !isRepo(repo) || m.private === true || (m.gated !== false && m.gated !== undefined)) return [];
    return [{ repo, downloads: num(m.downloads), likes: num(m.likes), lastModified: str(m.lastModified), license: licenseOf(m.tags) }];
  });
}

type TreeFile = { path: string; bytes: number; sha256: string | null };

/** The tree's files: path, size and the LFS sha256 (null for small files kept in git). */
export function readTree(data: unknown): TreeFile[] {
  if (!Array.isArray(data)) return [];
  return data.flatMap((raw) => {
    const f = obj(raw);
    const path = str(f.path);
    if (f.type !== 'file' || !path || !/^[\w.-]+$/.test(path)) return [];
    const lfs = str(obj(f.lfs).oid);
    return [{ path, bytes: num(f.size), sha256: lfs && SHA256_RE.test(lfs) ? lfs : null }];
  });
}

const QUANT_RE = /(IQ\d_[A-Z]+|Q\d_K_[SMLX]{1,2}|Q\d_K|Q\d_\d|Q\d|BF16|F16|F32)(?=\.gguf$|[-_.])/i;
const PREFERRED = ['Q4_K_M', 'Q4_K_S', 'Q5_K_M', 'Q4_0', 'Q5_K_S', 'Q6_K', 'Q8_0'];

/** Single-file GGUF weights (not shards, not vision projectors), smallest first. */
export function ggufVariants(files: TreeFile[]): GgufVariant[] {
  return files
    .filter((f) => f.path.toLowerCase().endsWith('.gguf') && !/-\d{5}-of-\d{5}\.gguf$/i.test(f.path) && !/mmproj/i.test(f.path) && f.sha256 && f.bytes > 0)
    .map((f) => ({ path: f.path, bytes: f.bytes, sha256: f.sha256!, quant: (QUANT_RE.exec(f.path)?.[1] ?? 'GGUF').toUpperCase(), fits: f.bytes <= WLLAMA_MAX_BYTES }))
    .sort((a, b) => a.bytes - b.bytes);
}

/** A sensible default: a 4–5-bit K-quant if one fits, else the largest that fits. */
export function recommend(variants: GgufVariant[]): string | null {
  const fitting = variants.filter((v) => v.fits);
  for (const q of PREFERRED) {
    const v = fitting.find((x) => x.quant === q);
    if (v) return v.path;
  }
  return fitting.at(-1)?.path ?? null;
}

/** The chat template's text (mirrors src-tauri/src/llm/config.rs `chat_template`). */
export function chatTemplate(tokenizerConfig: unknown, jinja: string | null): string | null {
  const t = obj(tokenizerConfig).chat_template;
  if (typeof t === 'string') return t;
  if (Array.isArray(t)) {
    const d = t.map(obj).find((x) => x.name === 'default');
    if (typeof d?.template === 'string') return d.template;
  }
  return jinja;
}

/** Bit widths MLX's quantized kernels take (mirrors config.rs `check_quant`). */
const MLX_BITS = [2, 3, 4, 5, 6, 8];
const MLX_GROUPS = [32, 64, 128];

/**
 * What the native engine can run (mirrors src-tauri/src/llm/config.rs, which
 * decides): Qwen3 or dense Qwen3.5, MLX-quantized (mixed precision allowed),
 * a ChatML template with an end-of-turn token.
 */
export function mlxChecks(config: unknown, tokenizerConfig: unknown, jinja: string | null): { checks: Check[]; layers: number; bits: number; thinking: boolean } {
  const c = obj(config);
  const qwen35 = c.model_type === 'qwen3_5';
  // Qwen3.5 checkpoints are vision-language: the text model's settings are nested.
  const t = qwen35 && c.text_config && typeof c.text_config === 'object' ? obj(c.text_config) : c;
  const q = obj(c.quantization ?? t.quantization);
  const checks: Check[] = [];
  const block = (text: string) => checks.push({ level: 'block', text });
  if (c.model_type === 'qwen3') checks.push({ level: 'ok', text: 'Qwen3, which the native engine runs.' });
  else if (qwen35) {
    if (num(t.num_experts) > 0) block('It’s a mixture-of-experts Qwen3.5, which the native engine doesn’t run.');
    else checks.push({ level: 'ok', text: 'Qwen3.5 (text only), which the native engine runs.' });
    const rope = obj(t.rope_parameters);
    const kind = str(rope.rope_type) ?? str(rope.type) ?? 'default';
    if (kind !== 'default') block(`It uses ${kind} RoPE scaling, which the native engine doesn’t support.`);
    if (t.attn_output_gate === false) block('Its attention has no output gate, which the Qwen3.5 engine expects.');
  } else block(`It’s a ${str(c.model_type) ?? 'unknown'} model; the native engine runs Qwen3 and Qwen3.5.`);
  const bits = num(q.bits);
  const layerQuant = Object.entries(q).filter(([k, v]) => !['group_size', 'bits', 'mode'].includes(k) && typeof v !== 'boolean');
  const quantOk = (b: number, g: number) => MLX_BITS.includes(b) && MLX_GROUPS.includes(g);
  if (!bits || !num(q.group_size)) block('Its weights aren’t MLX-quantized (2- to 8-bit).');
  else if (bits === 1) block('It has 1-bit weights, which need a patched MLX; the native engine runs 2- to 8-bit weights.');
  else if ((q.mode && q.mode !== 'affine') || !quantOk(bits, num(q.group_size))) block('It uses a quantization scheme the native engine doesn’t support.');
  else if (layerQuant.some(([, v]) => (obj(v).mode && obj(v).mode !== 'affine') || !quantOk(num(obj(v).bits), num(obj(v).group_size)))) {
    block('It uses a quantization scheme the native engine doesn’t support.');
  } else checks.push({ level: 'ok', text: layerQuant.length ? `${bits}-bit weights, some layers at other widths.` : `${bits}-bit weights.` });
  if (!qwen35 && t.rope_scaling != null) block('It uses scaled RoPE, which the native engine doesn’t support.');
  if (t.attention_bias === true) block('It uses attention bias, which the native engine doesn’t support.');
  const eos = obj(tokenizerConfig).eos_token;
  if (!str(eos) && !str(obj(eos).content)) block('Its tokenizer names no end-of-turn token.');
  const template = chatTemplate(tokenizerConfig, jinja);
  const thinking = !!template?.includes('enable_thinking');
  if (!template?.includes('<|im_start|>')) block('Its chat template isn’t ChatML, the format the native engine writes.');
  else checks.push({ level: 'ok', text: thinking ? 'Chat template with a thinking switch.' : 'Chat template (no thinking switch).' });
  return { checks, layers: num(t.num_hidden_layers), bits, thinking };
}

function access(info: Record<string, unknown>): Check[] {
  if (info.private === true) return [{ level: 'block', text: 'It’s private.' }];
  if (info.gated !== false && info.gated !== undefined) return [{ level: 'block', text: 'It needs a Hugging Face login and license agreement (gated); Andai downloads only public models.' }];
  return [];
}

/** Reads `repo` at its current commit and says whether and how it can be added. */
export async function inspectHub(repo: string, format: HubFormat, signal?: AbortSignal): Promise<HubModel> {
  const info = obj(await getJson(hubModelUrl(repo), signal));
  const commit = str(info.sha);
  if (!commit || !COMMIT_RE.test(commit)) throw new Error('Hugging Face didn’t say which version this is.');
  const files = readTree(await getJson(hubTreeUrl(repo, commit), signal));
  const license = str(obj(info.cardData).license) ?? licenseOf(info.tags);
  const base = { repo, commit, format, license, downloads: num(info.downloads), likes: num(info.likes), lastModified: str(info.lastModified) };
  const checks = access(info);

  if (format === 'gguf') {
    const gguf = obj(info.gguf);
    const variants = ggufVariants(files);
    const template = str(gguf.chat_template);
    const architecture = str(gguf.architecture);
    if (!variants.length) checks.push({ level: 'block', text: 'It has no single-file GGUF weights.' });
    else if (!variants.some((v) => v.fits)) checks.push({ level: 'block', text: 'Every file is over 2 GB, the in-app engine’s limit (wllama). Try a smaller model or quantization.' });
    else checks.push({ level: 'ok', text: `${variants.filter((v) => v.fits).length} of ${variants.length} files fit the 2 GB limit.` });
    if (architecture) checks.push({ level: 'ok', text: `Architecture ${architecture}; llama.cpp supports most. If it doesn’t, loading says so.` });
    if (!template) checks.push({ level: 'warn', text: 'It has no chat template, so replies may be poorly formatted.' });
    const contextLength = num(gguf.context_length) || null;
    return {
      ...base,
      checks,
      ok: !checks.some((c) => c.level === 'block'),
      gguf: { architecture, contextLength, thinking: !!template?.includes('enable_thinking'), variants, recommended: recommend(variants) },
    };
  }

  // MLX: the small files decide compatibility; fetch them at the same commit.
  const has = (p: string) => files.find((f) => f.path === p);
  const need = ['config.json', 'tokenizer_config.json', 'tokenizer.json'].filter((p) => !has(p));
  if (need.length) {
    checks.push({ level: 'block', text: `It’s missing ${need.join(', ')}.` });
    return { ...base, checks, ok: false };
  }
  const text = async (p: string) => {
    const f = has(p)!;
    if (f.bytes > MAX_INLINE_BYTES) throw new Error(`${p} is unexpectedly large.`);
    return (await get(pinnedFileUrl(repo, commit, p), signal)).text();
  };
  const parse = (p: string, t: string) => {
    try {
      return JSON.parse(t) as unknown;
    } catch {
      throw new Error(`${p} isn’t valid JSON.`);
    }
  };
  const inline: MlxSpec['inline'] = [];
  const add = async (p: string) => {
    const content = await text(p);
    inline.push({ path: p, content });
    return content;
  };
  const config = parse('config.json', await add('config.json'));
  const tokenizerConfig = parse('tokenizer_config.json', await add('tokenizer_config.json'));
  const jinja = has('chat_template.jinja') ? await add('chat_template.jinja') : null;
  const verdict = mlxChecks(config, tokenizerConfig, jinja);
  checks.push(...verdict.checks);

  const download: MlxSpec['files'] = [];
  const tokenizer = has('tokenizer.json')!;
  if (tokenizer.sha256) download.push({ path: tokenizer.path, bytes: tokenizer.bytes, sha256: tokenizer.sha256 });
  else await add('tokenizer.json');
  const shards = files.filter((f) => /^model-\d{5}-of-\d{5}\.safetensors$/.test(f.path));
  const single = has('model.safetensors');
  if (single?.sha256 && !shards.length) download.push({ path: single.path, bytes: single.bytes, sha256: single.sha256 });
  else if (shards.length && has('model.safetensors.index.json') && shards.every((f) => f.sha256)) {
    await add('model.safetensors.index.json');
    for (const f of shards) download.push({ path: f.path, bytes: f.bytes, sha256: f.sha256! });
  } else checks.push({ level: 'block', text: 'Its weights aren’t in safetensors files Andai can verify.' });
  const bytes = [...download.map((f) => f.bytes), ...inline.map((f) => f.content.length)].reduce((a, b) => a + b, 0);
  const ok = !checks.some((c) => c.level === 'block');
  return {
    ...base,
    checks,
    ok,
    mlx: ok ? { spec: { repo, commit, files: download, inline }, bytes, layers: verdict.layers, bits: verdict.bits, thinking: verdict.thinking } : undefined,
  };
}
