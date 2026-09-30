// The model writes a tool's arguments, and for kb_search that is where the
// question becomes a search phrase. A GBNF grammar built from the tool's
// schema holds the reply to its shape (validate.ts), then `validate` checks
// the values anyway: a grammar is a UX aid, not a guarantee.

import { complete, type ChatMessage } from '../../llm/engine';
import { splitThink } from '../../state/chat';
import type { ToolDef } from './types';
import { schemaGrammar, validate, type ObjectSchema } from './validate';

export type FillContext = {
  /** The decision state: request, recent turns, knowledge base, results so far (loop.ts). */
  state: string;
  kind: 'document' | 'code' | 'mixed';
  /** What the tools have shown so far (loop.ts); a matching argument must be one of them. */
  known?: Known;
  /** Enum arguments already picked (by Laya, loop.ts); each is held to its one value. */
  fixed?: Record<string, string>;
  signal?: AbortSignal;
};

export type Known = {
  /** Paths ug knows: the KB's indexed sources and files seen in results so far. */
  files?: string[];
  /** Code symbols the tools have shown this turn. */
  symbols?: string[];
  /** `file:start-end` around each passage with lines, for Read lines. */
  ranges?: string[];
};

/** Argument name → what it's held to, and how the prompt lists the choices. */
const HELD = [
  ['file', 'files', 'Files in the knowledge base'],
  ['symbol', 'symbols', 'Symbols seen so far'],
  ['range', 'ranges', 'Line ranges seen so far'],
] as const;

/**
 * One request to the argument writer, exactly as sent, and what came back:
 * the Execution Trace shows it (tool call dialog, "What the argument writer saw").
 */
export type ArgCall = {
  model: string | null;
  messages: ChatMessage[];
  /** Sampling and decoding parameters as sent, grammar included. */
  params: Record<string, unknown>;
  /** The reply as written, before parsing; empty when the call failed. */
  reply: string;
  /** Why the reply was rejected (parse or schema errors, or the engine's); empty when it was accepted. */
  errors: string[];
  ms: number;
  promptTokens: number | null;
};

/** Every request of one fill, and the schema the arguments were held to. */
export type FillIO = { schema: ObjectSchema | null; calls: ArgCall[] };

export type Fill =
  | { ok: true; args: Record<string, unknown>; raw: string; attempts: number; model: string | null; io: FillIO }
  | { ok: false; errors: string[]; raw: string; attempts: number; model: string | null; io: FillIO };

const MAX_ATTEMPTS = 2;
/** Up to this many known files (or symbols) become an enum for a `file` (or `symbol`) argument; past it they're only listed. */
export const MAX_FILE_ENUM = 64;

/**
 * The tool's schema, with a `file`, `symbol` or `range` argument held to what
 * the knowledge base and the results so far have shown. Left free, Qwen3 0.6B
 * wrote the knowledge base's name ("kb1") as the file for File outline, and
 * every retry repeated it; a free symbol name fails the same way ("No symbol
 * named …"), and free line numbers read lines nothing pointed to.
 */
export function schemaFor(tool: ToolDef, known: Known = {}, fixed: Record<string, string> = {}): ObjectSchema | null {
  if (!tool.schema) return null;
  let schema = tool.schema;
  // A picked value narrows its enum to one, so the grammar and `validate` hold it as they hold any enum.
  for (const [key, value] of Object.entries(fixed)) {
    const prop = schema.properties[key];
    if (prop?.type !== 'string' || (prop.enum && !prop.enum.includes(value))) continue;
    schema = { ...schema, properties: { ...schema.properties, [key]: { ...prop, enum: [value] } } };
  }
  for (const [key, from] of HELD) {
    // A picked value stays the only one allowed.
    if (key in fixed) continue;
    const prop = schema.properties[key];
    const values = known[from] ?? [];
    if (prop?.type !== 'string' || !values.length || values.length > MAX_FILE_ENUM) continue;
    schema = { ...schema, properties: { ...schema.properties, [key]: { ...prop, enum: values } } };
  }
  return schema;
}

/** The first `{…}` in a reply, in case the model wrapped the object in prose or a fence. */
export function parseObject(text: string): unknown {
  const body = splitThink(text).answer.trim();
  try {
    return JSON.parse(body);
  } catch {
    const start = body.indexOf('{');
    const end = body.lastIndexOf('}');
    if (start !== -1 && end > start) return JSON.parse(body.slice(start, end + 1));
    throw new Error('the reply is not a JSON object');
  }
}

export function fillMessages(tool: ToolDef, ctx: FillContext, previous?: { raw: string; errors: string[] }): ChatMessage[] {
  const schema = schemaFor(tool, ctx.known, ctx.fixed);
  const files = HELD.map(([key, from, label]) => {
    const values = ctx.known?.[from];
    return schema?.properties[key] && values?.length ? `${label}: ${values.slice(0, MAX_FILE_ENUM).join(', ')}${values.length > MAX_FILE_ENUM ? ', …' : ''}\n` : '';
  }).join('');
  const msgs: ChatMessage[] = [
    {
      role: 'system',
      content: 'You fill in the arguments for one tool call. Reply with only a JSON object that matches the schema: no prose, no markdown.',
    },
    {
      role: 'user',
      content:
        `${ctx.state}\n\n` +
        `Tool: ${tool.title}. ${tool.description}\n` +
        `Arguments schema:\n${JSON.stringify(schema)}\n\n` +
        files +
        `${tool.guide(ctx.kind)}\nReply with only the JSON object.`,
    },
  ];
  if (previous) {
    msgs.push(
      { role: 'assistant', content: previous.raw },
      { role: 'user', content: `That was not valid: ${previous.errors.join('; ')}. Reply again with only the corrected JSON object.` },
    );
  }
  return msgs;
}

/** The writer's decoding parameters: greedy, short, held to the schema's grammar, no thinking. */
export const fillParams = (grammar: string) => ({ max_tokens: 200, temperature: 0, grammar, chat_template_kwargs: { enable_thinking: false } });

export async function fillArgs(tool: ToolDef, ctx: FillContext): Promise<Fill> {
  const schema = schemaFor(tool, ctx.known, ctx.fixed);
  const io: FillIO = { schema, calls: [] };
  if (!schema) return { ok: true, args: {}, raw: '{}', attempts: 0, model: null, io };
  let previous: { raw: string; errors: string[] } | undefined;
  let model: string | null = null;
  const params = fillParams(schemaGrammar(schema));
  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
    const messages = fillMessages(tool, ctx, previous);
    const started = performance.now();
    const call: ArgCall = { model, messages, params, reply: '', errors: [], ms: 0, promptTokens: null };
    io.calls.push(call);
    let raw: string;
    try {
      const { response, def } = await complete('chat', { messages, ...params, abortSignal: ctx.signal });
      model = def.name;
      raw = response.choices?.[0]?.message?.content ?? '';
      Object.assign(call, { model, reply: raw, ms: Math.round(performance.now() - started), promptTokens: response.usage?.prompt_tokens ?? null });
    } catch (e) {
      // An engine failure is a failed fill, so the loop's fallback still applies; an abort is not.
      if (ctx.signal?.aborted) throw e;
      const errors = [e instanceof Error ? e.message : String(e)];
      Object.assign(call, { errors, ms: Math.round(performance.now() - started) });
      return { ok: false, errors, raw: '', attempts: attempt, model, io };
    }
    let errors: string[];
    try {
      const v = validate(schema, parseObject(raw));
      if (v.ok) return { ok: true, args: v.value, raw, attempts: attempt, model, io };
      errors = v.errors;
    } catch (e) {
      errors = [e instanceof Error ? e.message : String(e)];
    }
    call.errors = errors;
    if (attempt === MAX_ATTEMPTS) return { ok: false, errors, raw, attempts: attempt, model, io };
    previous = { raw, errors };
  }
  throw new Error('unreachable');
}
