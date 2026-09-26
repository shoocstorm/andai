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
  /** Paths ug knows: the KB's indexed sources and files seen in results so far (loop.ts). */
  files?: string[];
  signal?: AbortSignal;
};

export type Fill =
  | { ok: true; args: Record<string, unknown>; raw: string; attempts: number; model: string | null }
  | { ok: false; errors: string[]; raw: string; attempts: number; model: string | null };

const MAX_ATTEMPTS = 2;
/** Up to this many known files become an enum for a `file` argument; past it they're only listed. */
export const MAX_FILE_ENUM = 64;

/**
 * The tool's schema, with a `file` argument held to the files that exist.
 * Left free, Qwen3 0.6B wrote the knowledge base's name ("kb1") as the file
 * for File outline, and every retry repeated it.
 */
export function schemaFor(tool: ToolDef, files: string[] = []): ObjectSchema | null {
  const file = tool.schema?.properties.file;
  if (!tool.schema || file?.type !== 'string' || !files.length || files.length > MAX_FILE_ENUM) return tool.schema;
  return { ...tool.schema, properties: { ...tool.schema.properties, file: { ...file, enum: files } } };
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
  const schema = schemaFor(tool, ctx.files);
  const files = schema?.properties.file && ctx.files?.length
    ? `Files in the knowledge base: ${ctx.files.slice(0, MAX_FILE_ENUM).join(', ')}${ctx.files.length > MAX_FILE_ENUM ? ', …' : ''}\n`
    : '';
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

export async function fillArgs(tool: ToolDef, ctx: FillContext): Promise<Fill> {
  const schema = schemaFor(tool, ctx.files);
  if (!schema) return { ok: true, args: {}, raw: '{}', attempts: 0, model: null };
  let previous: { raw: string; errors: string[] } | undefined;
  let model: string | null = null;
  const grammar = schemaGrammar(schema);
  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
    let raw: string;
    try {
      const { response, def } = await complete('chat', {
        messages: fillMessages(tool, ctx, previous),
        max_tokens: 200,
        temperature: 0,
        grammar,
        chat_template_kwargs: { enable_thinking: false },
        abortSignal: ctx.signal,
      });
      model = def.name;
      raw = response.choices?.[0]?.message?.content ?? '';
    } catch (e) {
      // An engine failure is a failed fill, so the loop's fallback still applies; an abort is not.
      if (ctx.signal?.aborted) throw e;
      return { ok: false, errors: [e instanceof Error ? e.message : String(e)], raw: '', attempts: attempt, model };
    }
    let errors: string[];
    try {
      const v = validate(schema, parseObject(raw));
      if (v.ok) return { ok: true, args: v.value, raw, attempts: attempt, model };
      errors = v.errors;
    } catch (e) {
      errors = [e instanceof Error ? e.message : String(e)];
    }
    if (attempt === MAX_ATTEMPTS) return { ok: false, errors, raw, attempts: attempt, model };
    previous = { raw, errors };
  }
  throw new Error('unreachable');
}
