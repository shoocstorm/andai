// The native engine enforces argument grammars itself (src-tauri/src/llm/
// grammar.rs), and its tests compile every grammar argument filling can send,
// from tests/fixtures/llm/grammars.json. This keeps that fixture in step with
// `schemaGrammar`: when it fails, write the printed JSON to the fixture.
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { schemaFor } from './argfill';
import { TOOLS } from './registry';
import { schemaGrammar } from './validate';

describe('argument grammars fixture', () => {
  it('holds exactly the grammars argument filling sends, with and without known values', () => {
    const known = { files: ['booking.ts', 'refund-policy.md'], symbols: ['computeFare', 'withRetry'], ranges: ['booking.ts:22-32'] };
    const now: Record<string, string> = {};
    for (const t of TOOLS) {
      const s = schemaFor(t, {});
      if (s) now[t.id] = schemaGrammar(s);
      const k = schemaFor(t, known as never);
      if (k && JSON.stringify(k) !== JSON.stringify(s)) now[`${t.id}+known`] = schemaGrammar(k);
    }
    const fixture = JSON.parse(readFileSync(join(__dirname, '../../../src-tauri/tests/fixtures/llm/grammars.json'), 'utf8'));
    expect(fixture, `update src-tauri/tests/fixtures/llm/grammars.json to:\n${JSON.stringify(now, null, 1)}`).toEqual(now);
  });
});
