import { describe, expect, it } from 'vitest';
import type { Message } from '../state/chat';
import { debugReport, shellCommand, type ReportEnv } from './debugReport';

const env: ReportEnv = {
  chatModel: 'Qwen3 0.6B',
  deciderModel: null,
  engine: 'WebGPU · 4 (multi) threads',
  ug: 'ug version 0.1.21',
  agent: { agentMode: true, maxSteps: 20, minConfidence: 0.3, policies: { kb_find_usages: 'ask' } },
  kb: { name: 'Docs', kind: 'mixed', kindOverride: null, nodes: 34, sources: [{ file: 'notes.md' }, { file: 'api.ts' }] as never },
};

const msg: Message = {
  id: 'a',
  role: 'assistant',
  content: '<think>\n\n</think>\n\nUse COOP [1].',
  createdAt: Date.UTC(2026, 8, 26, 12),
  kbName: 'Docs',
  steps: [{ kind: 'plan', title: 'Plan', detail: '1 tool call', status: 'done', ms: 2210 }],
  agent: [
    {
      id: 's1',
      index: 0,
      at: 0,
      action: 'kb_search',
      decision: {
        question: 'q',
        chosen: 'kb_search',
        confidence: 0.96,
        bounded: [],
        model: 'Qwen3 0.6B',
        slot: 'chat',
        ms: 612,
        seed: 1,
        promptTokens: 400,
        options: [
          { id: 'answer_now', label: 'A', text: '', probability: 0.02 },
          { id: 'kb_search', label: 'B', text: '', probability: 0.96 },
          { id: 'kb_overview', label: 'C', text: '', probability: 0.02 },
        ],
      },
      call: {
        tool: 'kb_search',
        title: 'Knowledge search',
        args: { query: 'wllama headers', scope: 'focused' },
        argsRaw: null,
        argModel: 'Qwen3 0.6B',
        argAttempts: 1,
        policy: 'auto',
        argv: ['search', 'wllama headers', '-n', 'andai-docs', '--json'],
        startedAt: 0,
        ms: 94,
        status: 'done',
        output: 'SECRET-OUTPUT',
        outputBytes: 1480,
        observation: '1 passage(s): Isolation @ notes.md:3-9',
      },
    },
    { id: 's2', index: 1, at: 0, action: 'answer_now', decision: null, note: 'Low confidence, answering.' },
  ],
  sources: [{ id: 'h', name: 'Isolation', node_type: 'Concept', file: 'notes.md', start_line: 3, end_line: 9 }],
  stats: { tokens: 9, tokPerSec: 30, promptTokens: 400, nCtx: 4096, totalMs: 3000, firstTokenMs: 900, model: 'Qwen3 0.6B' },
};

describe('shellCommand', () => {
  it('quotes only what a shell would split or expand, safely', () => {
    expect(shellCommand(['search', 'wllama headers', '-k', '8'])).toBe("ug search 'wllama headers' -k 8");
    expect(shellCommand(['find_symbols', 'build*'])).toBe("ug find_symbols 'build*'");
    expect(shellCommand(['search', "it's $HOME `x`"])).toBe(`ug search 'it'\\''s $HOME \`x\`'`);
    expect(shellCommand(['search', ''])).toBe("ug search ''");
  });
});

describe('debugReport', () => {
  const r = debugReport(msg, 'What headers?', env);

  it('states the setup a turn ran with', () => {
    expect(r).toContain('## Andai turn · 2026-09-26T12:00:00.000Z');
    expect(r).toContain('- models: chat Qwen3 0.6B · decider (chat model) · WebGPU');
    expect(r).toContain('- agent: on · max 20 calls · min confidence 30% · policies kb_find_usages=ask');
    expect(r).toContain('- kb: “Docs” · mixed · 34 nodes · 2 files: notes.md, api.ts');
  });

  it('lists each decision, call, command and result, but not the raw output', () => {
    expect(r).toContain('1. kb_search @ 96% (Qwen3 0.6B, chat, 612 ms; top: kb_search 96%, answer_now 2%, kb_overview 2%)');
    expect(r).toContain('   args: {"query":"wllama headers","scope":"focused"} (by Qwen3 0.6B, 1 attempt)');
    expect(r).toContain("   $ ug search 'wllama headers' -n andai-docs --json");
    expect(r).toContain('   → done · 94 ms · 1480 bytes');
    expect(r).toContain('2. answer_now (no decision scored)\n   note: Low confidence, answering.');
    expect(r).not.toContain('SECRET-OUTPUT');
  });

  it('ends with the sources, the answer without reasoning, and stats', () => {
    expect(r).toContain('[1] notes.md:3-9 — Isolation');
    expect(r).toContain('### Answer\nUse COOP [1].');
    expect(r).toContain('9 tokens · 30.0 tok/s · prompt 400 / ctx 4096 · first token 900 ms');
    expect(r).not.toMatch(/\n\n\n/);
  });
});
