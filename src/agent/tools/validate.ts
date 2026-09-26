// A JSON-Schema subset, enough for tool arguments: the same schema constrains
// the model's output (llama.cpp turns it into a grammar) and then checks it
// here, because a grammar is a UX aid and not a guarantee. Rust validates the
// resulting call again (src-tauri/src/tools.rs).

export type PropSchema =
  | { type: 'string'; description?: string; minLength?: number; maxLength?: number; enum?: string[] }
  | { type: 'integer'; description?: string; minimum?: number; maximum?: number }
  | { type: 'boolean'; description?: string }
  | { type: 'array'; description?: string; items: { type: 'string'; maxLength?: number }; minItems?: number; maxItems?: number };

export type ObjectSchema = {
  type: 'object';
  properties: Record<string, PropSchema>;
  required: string[];
  additionalProperties: false;
};

export type Validation = { ok: true; value: Record<string, unknown> } | { ok: false; errors: string[] };

function checkProp(name: string, p: PropSchema, v: unknown, errors: string[]): unknown {
  switch (p.type) {
    case 'string': {
      if (typeof v !== 'string') return void errors.push(`${name} must be a string`);
      const s = v.trim();
      if (p.minLength !== undefined && s.length < p.minLength) errors.push(`${name} is too short`);
      if (p.maxLength !== undefined && s.length > p.maxLength) errors.push(`${name} is longer than ${p.maxLength} characters`);
      if (p.enum && !p.enum.includes(s)) errors.push(`${name} must be one of ${p.enum.join(', ')}`);
      return s;
    }
    case 'integer': {
      if (typeof v !== 'number' || !Number.isInteger(v)) return void errors.push(`${name} must be an integer`);
      if (p.minimum !== undefined && v < p.minimum) errors.push(`${name} must be at least ${p.minimum}`);
      if (p.maximum !== undefined && v > p.maximum) errors.push(`${name} must be at most ${p.maximum}`);
      return v;
    }
    case 'boolean':
      if (typeof v !== 'boolean') errors.push(`${name} must be true or false`);
      return v;
    case 'array': {
      if (!Array.isArray(v)) return void errors.push(`${name} must be a list`);
      if (p.minItems !== undefined && v.length < p.minItems) errors.push(`${name} needs at least ${p.minItems} item(s)`);
      if (p.maxItems !== undefined && v.length > p.maxItems) errors.push(`${name} allows at most ${p.maxItems} items`);
      const out: string[] = [];
      for (const item of v) {
        if (typeof item !== 'string' || !item.trim()) errors.push(`${name} items must be non-empty strings`);
        else if (p.items.maxLength !== undefined && item.length > p.items.maxLength) errors.push(`${name} item is too long`);
        else out.push(item.trim());
      }
      return out;
    }
  }
}

/** Checks `value` against `schema`; strings come back trimmed. */
export function validate(schema: ObjectSchema, value: unknown): Validation {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return { ok: false, errors: ['arguments must be a JSON object'] };
  const obj = value as Record<string, unknown>;
  const errors: string[] = [];
  for (const key of Object.keys(obj)) if (!(key in schema.properties)) errors.push(`unknown argument ${key}`);
  for (const key of schema.required) if (!(key in obj)) errors.push(`missing ${key}`);
  const out: Record<string, unknown> = {};
  for (const [key, prop] of Object.entries(schema.properties)) {
    if (key in obj) out[key] = checkProp(key, prop, obj[key], errors);
  }
  return errors.length ? { ok: false, errors } : { ok: true, value: out };
}

// ── schema → GBNF ────────────────────────────────────────────────────────
// wllama can't convert a JSON Schema itself (engine.ts), so the argument
// grammar is built here. It fixes the key order and shape; lengths and ranges
// that GBNF can't express cheaply are left to `validate`.

const lit = (s: string) => JSON.stringify(s);
/** A GBNF literal that matches the JSON text of `s`, e.g. "\"broad\"". */
const jsonLit = (s: string) => lit(JSON.stringify(s));

function propGrammar(p: PropSchema): string {
  switch (p.type) {
    case 'string':
      return p.enum ? `(${p.enum.map(jsonLit).join(' | ')})` : 'string';
    case 'integer':
      return p.minimum !== undefined && p.minimum >= 0 ? 'uint' : 'int';
    case 'boolean':
      return '("true" | "false")';
    case 'array': {
      const extra = Math.max(0, (p.maxItems ?? 8) - 1);
      const first = (p.minItems ?? 0) > 0 ? 'string' : '';
      return first
        ? `"[" ws string (ws "," ws string){0,${extra}} ws "]"`
        : `"[" ws (string (ws "," ws string){0,${extra}})? ws "]"`;
    }
  }
}

/** GBNF for one JSON object with every property of `schema`, in order. */
export function schemaGrammar(schema: ObjectSchema): string {
  const fields = Object.entries(schema.properties).map(([k, p]) => `${jsonLit(k)} ws ":" ws ${propGrammar(p)}`);
  return [
    `root ::= "{" ws ${fields.join(' ws "," ws ')} ws "}"`,
    'string ::= "\\"" ( [^"\\\\\\x7F\\x00-\\x1F] | "\\\\" ["\\\\/bfnrt] ){0,300} "\\""',
    'uint ::= [0-9]{1,7}',
    'int ::= "-"? [0-9]{1,7}',
    'ws ::= [ \\t\\n]{0,3}',
  ].join('\n');
}
