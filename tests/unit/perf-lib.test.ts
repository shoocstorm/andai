// The harness runners read the app's output line by line (scripts/perf-lib.mjs).
import { describe, expect, it } from 'vitest';
// @ts-expect-error — plain ESM script without type declarations
import { byLine } from '../../scripts/perf-lib.mjs';

describe('byLine', () => {
  it('joins a line split across chunks, and holds back an unfinished one', () => {
    // A CASE record split across two chunks was lost, so eval runs came up a question short.
    const got: string[] = [];
    const onData = byLine((lines: string[]) => got.push(...lines));
    onData(Buffer.from('[webview] START {}\n[webview] CASE {"id":'));
    expect(got).toEqual(['[webview] START {}']);
    onData(Buffer.from('"q1"}\n[webview] O'));
    onData(Buffer.from('K\n'));
    expect(got).toEqual(['[webview] START {}', '[webview] CASE {"id":"q1"}', '[webview] OK']);
  });
});
