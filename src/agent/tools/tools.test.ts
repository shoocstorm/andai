// Tool framework: schema validation, the registry's filtering and policies,
// argument filling (engine mocked), and every ug output reader against the
// shapes probed from ug 0.1.21.
import { beforeEach, describe, expect, it, vi } from 'vitest';

const eng = vi.hoisted(() => ({ replies: [] as string[], seen: [] as Record<string, unknown>[], throwNext: null as Error | null }));
vi.mock('../../llm/engine', async () => {
  const { MODELS } = await import('../../llm/models');
  return {
    complete: async (_: string, params: Record<string, unknown>) => {
      eng.seen.push(params);
      if (eng.throwNext) {
        const e = eng.throwNext;
        eng.throwNext = null;
        throw e;
      }
      return { slot: 'chat', def: MODELS[0], response: { choices: [{ message: { content: eng.replies.shift() ?? '' } }] } };
    },
  };
});

const { schemaGrammar, validate } = await import('./validate');
const { available, defaultPolicy, policyOf, toolById, TOOLS } = await import('./registry');
const { fillArgs, MAX_FILE_ENUM, parseObject, schemaFor } = await import('./argfill');

const tool = (id: string) => toolById(id)!;

beforeEach(() => {
  eng.replies = [];
  eng.seen = [];
});

describe('validate', () => {
  const schema = tool('kb_search').schema!;
  it('accepts a matching object and trims strings', () => {
    expect(validate(schema, { query: '  wllama headers ', scope: 'broad' })).toEqual({ ok: true, value: { query: 'wllama headers', scope: 'broad' } });
  });
  it('rejects unknown, missing, mistyped and out-of-range arguments', () => {
    const bad = validate(schema, { query: 'x'.repeat(301), scope: 'everything', extra: 1 });
    expect(bad.ok).toBe(false);
    if (!bad.ok) expect(bad.errors.join('|')).toMatch(/unknown argument extra.*longer than 300.*must be one of/);
    expect(validate(schema, { scope: 'broad' }).ok).toBe(false);
    expect(validate(schema, 'query').ok).toBe(false);
    expect(validate(schema, [1]).ok).toBe(false);
    const lines = tool('kb_read_lines').schema!;
    expect(validate(lines, { range: 5 }).ok).toBe(false);
    expect(validate(lines, { range: 'a.md:1-20' }).ok).toBe(true);
    const syms = tool('kb_find_symbols').schema!;
    expect(validate(syms, { names: [], node_type: 'any' }).ok).toBe(false);
    expect(validate(syms, { names: ['a', ''], node_type: 'any' }).ok).toBe(false);
    expect(validate(syms, { names: ['a', 'b*'], node_type: 'Function' }).ok).toBe(true);
  });
});

describe('schemaGrammar', () => {
  it('fixes key order, enum values and value types', () => {
    const g = schemaGrammar(tool('kb_search').schema!);
    expect(g.split('\n')[0]).toBe('root ::= "{" ws "\\"query\\"" ws ":" ws string ws "," ws "\\"scope\\"" ws ":" ws ("\\"broad\\"" | "\\"focused\\"") ws "}"');
    const counted = { type: 'object' as const, properties: { n: { type: 'integer' as const, minimum: 1, maximum: 9 } }, required: ['n'], additionalProperties: false as const };
    expect(schemaGrammar(counted)).toMatch(/"\\"n\\"" ws ":" ws uint/);
    expect(schemaGrammar(tool('kb_find_symbols').schema!)).toContain('"[" ws string (ws "," ws string){0,4} ws "]"');
  });
  it('covers every property, since the grammar makes each one present', () => {
    for (const t of TOOLS) if (t.schema) expect([...t.schema.required].sort()).toEqual(Object.keys(t.schema.properties).sort());
  });
});

describe('registry', () => {
  it('has unique ids, one line options and read-only risk for every built-in tool', () => {
    expect(new Set(TOOLS.map((t) => t.id)).size).toBe(TOOLS.length);
    for (const t of TOOLS) {
      expect(t.risk).toBe('read');
      expect(t.option).not.toContain('\n');
    }
  });
  it('offers code navigation only for code and mixed knowledge bases', () => {
    const ids = (kind: 'document' | 'code' | 'mixed') => available(kind, {}).map((t) => t.id);
    expect(ids('document')).toEqual(['kb_search', 'kb_read_lines', 'kb_file_context', 'kb_overview']);
    expect(ids('code')).toContain('kb_find_usages');
    expect(ids('mixed')).toHaveLength(TOOLS.length);
    // answer + clarify + every tool must fit the decision's 16 letters
    expect(TOOLS.length + 2).toBeLessThanOrEqual(16);
  });
  it('defaults read tools to Auto and anything riskier to Ask; Off hides a tool', () => {
    expect(defaultPolicy('read')).toBe('auto');
    expect(defaultPolicy('write')).toBe('ask');
    expect(defaultPolicy('device')).toBe('ask');
    expect(policyOf(tool('kb_search'), { kb_search: 'ask' })).toBe('ask');
    expect(available('document', { kb_search: 'off' }).map((t) => t.id)).not.toContain('kb_search');
  });
  it('maps arguments to the call Rust expects', () => {
    const ctx = { kind: 'document' as const, k: 6, maxChars: 5000 };
    expect(tool('kb_search').toCall({ query: 'q', scope: 'focused' }, ctx)).toEqual({ tool: 'kb_search', query: 'q', k: 6, expand: false, max_chars: 5000 });
    expect(tool('kb_find_symbols').toCall({ names: ['a*'], node_type: 'any' }, ctx)).toEqual({
      tool: 'kb_find_symbols',
      names: ['a*'],
      node_type: null,
      file_prefix: null,
    });
    expect(tool('kb_read_lines').toCall({ range: 'notes/a:b.md:3-9' }, ctx)).toEqual({ tool: 'kb_get_code', symbol: null, file: 'notes/a:b.md', start: 3, end: 9 });
    expect(() => tool('kb_read_lines').toCall({ range: 'a.md lines 3 to 9' }, ctx)).toThrow(/file:start-end/);
  });
});

describe('fillArgs', () => {
  const ctx = { state: 'User request:\nwhat headers does wllama need?', kind: 'document' as const };

  it('constrains the reply with a grammar from the tool schema and returns validated arguments', async () => {
    eng.replies = ['<think>\n\n</think>\n\n{"query":"wllama COOP COEP headers","scope":"broad"}'];
    const fill = await fillArgs(tool('kb_search'), ctx);
    expect(fill).toMatchObject({ ok: true, args: { query: 'wllama COOP COEP headers', scope: 'broad' }, attempts: 1 });
    const p = eng.seen[0] as { grammar: string; temperature: number; messages: { content: string }[] };
    // measured: wllama can't build a grammar from response_format json_schema
    expect(p).not.toHaveProperty('response_format');
    expect(p.grammar).toBe(schemaGrammar(tool('kb_search').schema!));
    expect(p.temperature).toBe(0);
    expect(p.messages[1].content).toContain('what headers does wllama need?');
    expect(p.messages[1].content).toContain('not the user’s literal wording');
  });

  it('retries once with the validation errors, then gives up', async () => {
    eng.replies = ['{"query":"x"}', 'not json'];
    const fill = await fillArgs(tool('kb_search'), ctx);
    expect(fill).toMatchObject({ ok: false, attempts: 2 });
    const retry = eng.seen[1] as { messages: { role: string; content: string }[] };
    expect(retry.messages.at(-1)!.content).toMatch(/That was not valid: .*missing scope/);
  });

  it('recovers on the retry', async () => {
    eng.replies = ['{"query":"x"}', '{"query":"headers","scope":"focused"}'];
    expect(await fillArgs(tool('kb_search'), ctx)).toMatchObject({ ok: true, attempts: 2 });
  });

  it('reports an engine failure as a failed fill, so the loop can fall back', async () => {
    eng.throwNext = new Error('Failed to initialize samplers');
    expect(await fillArgs(tool('kb_search'), ctx)).toMatchObject({ ok: false, errors: ['Failed to initialize samplers'], attempts: 1 });
  });

  it('holds a file argument to the files that exist, and lists them', async () => {
    const files = ['notes.md', 'src/app.ts'];
    eng.replies = ['{"file":"kb1"}', '{"file":"notes.md"}'];
    const fill = await fillArgs(tool('kb_file_context'), { ...ctx, known: { files } });
    expect(fill).toMatchObject({ ok: true, args: { file: 'notes.md' }, attempts: 2 });
    const p = eng.seen[0] as { grammar: string; messages: { content: string }[] };
    expect(p.grammar.split('\n')[0]).toBe('root ::= "{" ws "\\"file\\"" ws ":" ws ("\\"notes.md\\"" | "\\"src/app.ts\\"") ws "}"');
    expect(p.messages[1].content).toContain('Files in the knowledge base: notes.md, src/app.ts');
    const retry = eng.seen[1] as { messages: { content: string }[] };
    expect(retry.messages.at(-1)!.content).toMatch(/file must be one of notes.md, src\/app.ts/);
  });

  it('leaves the file argument free when no files are known or there are too many to list', () => {
    const t = tool('kb_file_context');
    expect(schemaFor(t, { files: [] })).toBe(t.schema);
    expect(schemaFor(t, { files: Array.from({ length: MAX_FILE_ENUM + 1 }, (_, i) => `f${i}.md`) })).toBe(t.schema);
    expect(schemaFor(tool('kb_search'), { files: ['notes.md'] })).toBe(tool('kb_search').schema);
  });

  it('holds a symbol argument to the symbols seen so far, and lists them', async () => {
    const symbols = ['computeFare', 'VEHICLE_SURCHARGE'];
    eng.replies = ['{"symbol":"computeFare"}'];
    const fill = await fillArgs(tool('kb_get_code'), { ...ctx, kind: 'code', known: { symbols } });
    expect(fill).toMatchObject({ ok: true, args: { symbol: 'computeFare' } });
    const p = eng.seen[0] as { grammar: string; messages: { content: string }[] };
    expect(p.grammar.split('\n')[0]).toContain('("\\"computeFare\\"" | "\\"VEHICLE_SURCHARGE\\"")');
    expect(p.messages[1].content).toContain('Symbols seen so far: computeFare, VEHICLE_SURCHARGE');
    for (const id of ['kb_symbol_context', 'kb_find_usages']) expect(schemaFor(tool(id), { symbols })!.properties.symbol).toMatchObject({ enum: symbols });
  });

  it('leaves the symbol argument free when none or too many are known', () => {
    const t = tool('kb_get_code');
    expect(schemaFor(t, { symbols: [] })).toBe(t.schema);
    expect(schemaFor(t, { symbols: Array.from({ length: MAX_FILE_ENUM + 1 }, (_, i) => `s${i}`) })).toBe(t.schema);
    expect(schemaFor(tool('kb_file_context'), { symbols: ['add'] })).toBe(tool('kb_file_context').schema);
  });

  it('holds Read lines to the line ranges seen so far, and lists them', async () => {
    const ranges = ['operations.md:1-45', 'refund-policy.md:1-31'];
    eng.replies = ['{"range":"operations.md:1-45"}'];
    const fill = await fillArgs(tool('kb_read_lines'), { ...ctx, known: { ranges } });
    expect(fill).toMatchObject({ ok: true, args: { range: 'operations.md:1-45' } });
    const p = eng.seen[0] as { grammar: string; messages: { content: string }[] };
    expect(p.grammar.split('\n')[0]).toContain('("\\"operations.md:1-45\\"" | "\\"refund-policy.md:1-31\\"")');
    expect(p.messages[1].content).toContain('Line ranges seen so far: operations.md:1-45, refund-policy.md:1-31');
  });

  it('needs no model for a tool without arguments', async () => {
    expect(await fillArgs(tool('kb_overview'), ctx)).toEqual({ ok: true, args: {}, raw: '{}', attempts: 0, model: null });
    expect(eng.seen).toHaveLength(0);
  });

  it('finds the object inside prose or a fence', () => {
    expect(parseObject('Sure:\n```json\n{"a":1}\n```')).toEqual({ a: 1 });
    expect(() => parseObject('no')).toThrow();
  });
});

describe('ug output readers', () => {
  it('search: dedupes and keeps ug order', () => {
    const ev = tool('kb_search').observe({
      items: [
        { id: 'file:a.md', name: 'a.md', node_type: 'File', file: 'a.md', start_line: 1, end_line: 50, snippet: 'whole' },
        { id: 's', name: 'Isolation', node_type: 'Concept', file: 'a.md', start_line: 3, end_line: 9, snippet: 'COOP' },
      ],
    });
    expect(ev.hits.map((h) => h.id)).toEqual(['s']);
    expect(ev.summary).toBe('1 passage(s): Isolation @ a.md:3-9');
  });

  it('find_symbols, context, get_code, usages, file context, overview', () => {
    const syms = tool('kb_find_symbols').observe({
      queries: [{ query: 'build*', items: [{ id: 'f:p:buildSystem', name: 'buildSystem', node_type: 'Function', file: 'prompt.ts', start_line: 49, end_line: 78, doc: 'System prompt.' }] }],
    });
    expect(syms.hits[0]).toMatchObject({ file: 'prompt.ts', start_line: 49, snippet: 'Function buildSystem\nSystem prompt.' });

    const ctx = tool('kb_symbol_context').observe({
      items: [{ role: 'dependency', why: 'target —Calls→ this', id: 'd', name: 'defang', node_type: 'Function', file: 'prompt.ts', start_line: 42, end_line: 42 }],
    });
    expect(ctx.hits[0].snippet).toBe('[dependency: target —Calls→ this] Function defang');

    const code = tool('kb_get_code').observe({ slices: [{ title: 'Function keywords', file: 'prompt.ts', start_line: 21, end_line: 32, code: 'export function keywords' }] });
    expect(code.hits[0]).toMatchObject({ name: 'Function keywords', snippet: 'export function keywords' });
    const missing = tool('kb_get_code').observe({ slices: [{ title: 'zzz', error: "No symbol named 'zzz'. try find_symbols" }] });
    expect(missing).toEqual({ hits: [], summary: "No symbol named 'zzz'. try find_symbols" });

    const uses = tool('kb_find_usages').observe({
      nodes: [
        {
          subject: { name: 'call' },
          users: [{ id: 'u', name: 'kbCreate', node_type: 'Function', file: 'api.ts', start_line: 54, end_line: 54, via_edge: 'Calls', call_sites: [{ line: 54, text: 'call(...)' }] }],
        },
      ],
    });
    expect(uses.hits[0].snippet).toBe('Function kbCreate → Calls call\n  54: call(...)');

    const file = tool('kb_file_context').observe({
      files: [{ file: 'api.ts', id: 'file:api.ts', facts: { lines: 87 }, items: [{ role: 'outline', name: 'Source', node_type: 'Interface', file: 'api.ts', start_line: 4 }] }],
    });
    expect(file.hits[0]).toMatchObject({ file: 'api.ts', start_line: 1, end_line: 87, snippet: '- [outline] Interface Source (api.ts:4)' });

    const ov = tool('kb_overview').observe({ kb_type: 'mixed', index: { files: 4, symbols: 29, lines: 211 }, languages: [{ name: 'typescript', count: 2 }] });
    expect(ov.hits[0].snippet).toContain('Kind: mixed; 4 files, 29 symbols, 211 lines.');
    expect(ov.summary).toContain('typescript (2)');
  });

  it('tolerates any output shape without throwing', () => {
    for (const t of TOOLS) {
      for (const junk of [null, 'text', 42, [], { items: 'x', slices: [null], nodes: [{ users: [7] }], files: [{}] }]) {
        expect(() => t.observe(junk)).not.toThrow();
      }
    }
  });
});
