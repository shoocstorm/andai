// explainCall: the plain sentences the tool call dialog shows.
import { describe, expect, it } from 'vitest';
import type { ToolCallRecord } from '../state/chat';
import { explainCall } from './AgentTrace';

const call = (p: Partial<ToolCallRecord> = {}): ToolCallRecord => ({
  tool: 'kb_search',
  title: 'Knowledge search',
  args: { query: 'group discount', scope: 'focused' },
  argsRaw: null,
  argModel: 'Qwen3 1.7B',
  argAttempts: 1,
  policy: 'auto',
  startedAt: 0,
  status: 'done',
  ...p,
});

describe('explainCall', () => {
  it('says which argument Laya picked, and with what probability', () => {
    const out = explainCall(call({ argChoices: [{ arg: 'scope', value: 'focused', probability: 0.87, model: 'Laya Multilingual' }] }));
    expect(out).toContain('Laya Multilingual picked scope “focused” (87%) as a typed choice, and the chat model kept it.');
  });

  it('says nothing about picks when the chat model wrote every argument', () => {
    expect(explainCall(call()).some((l) => l.includes('typed choice'))).toBe(false);
  });
});
