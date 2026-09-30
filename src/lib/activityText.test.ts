// The activity log's one-line summaries: what a person reads in the log file
// and on the Logs screen, and how bad each event is.
import { describe as group, expect, it } from 'vitest';
import { describe, fmtMs, groupEvents } from './activityText';

group('activity summaries', () => {
  it('says what was asked, how, and against what', () => {
    expect(describe('turn', { question: 'Is there Wi-Fi on the Kestrel?', mode: 'agent', model: 'Qwen3 1.7B', decider: 'Laya English', kb: { name: 'Ferries' } })).toEqual({
      level: 'info',
      summary: 'Asked “Is there Wi-Fi on the Kestrel?” · agent mode · Qwen3 1.7B · decider Laya English · knowledge base “Ferries”',
    });
  });

  it('says what a step chose, how sure and why it did something else', () => {
    const step = { index: 1, action: 'kb_read_lines', decision: { model: 'Laya English', confidence: 0.82, ms: 21, stop: { probability: 0.3 } }, fallback: 'read-whole' };
    expect(describe('step', step).summary).toBe('Step 2: kb_read_lines · 82% by Laya English in 21 ms · results suffice 30% · read a clipped passage whole');
    expect(describe('step', { index: 0, action: 'answer_now', decision: null }).summary).toBe('Step 1: answer · no decision needed');
    expect(describe('step', { index: 0, action: 'kb_search', failedDecision: { error: 'bad letter' } })).toMatchObject({ level: 'warn', summary: expect.stringContaining('decision failed: bad letter') });
  });

  it('shows the arguments written, and a failed fill as an error', () => {
    expect(describe('args', { step: 0, tool: 'kb_search', args: { query: 'wifi', scope: 'focused' }, model: 'Qwen3 1.7B', attempts: 1 }).summary).toBe(
      'Step 1: kb_search {"query":"wifi","scope":"focused"} · written by Qwen3 1.7B',
    );
    expect(describe('args', { step: 2, tool: 'kb_symbol', args: null, error: 'No valid JSON', attempts: 2 })).toEqual({ level: 'error', summary: 'Step 3: couldn’t write arguments for kb_symbol · No valid JSON · 2 attempts' });
  });

  it('says how a tool call went, with what the next decision saw', () => {
    expect(describe('tool', { step: 0, tool: 'kb_search', status: 'done', ms: 42, hits: 3, outputBytes: 5000, observation: 'Found 3 passages; most useful: “Fares”' }).summary).toBe(
      'Step 1: kb_search ran in 42 ms · 3 passages added · 5 KB out · → Found 3 passages; most useful: “Fares”',
    );
    expect(describe('tool', { step: 0, tool: 'kb_symbol', status: 'error', error: 'No symbol named x', ms: 5 }).level).toBe('error');
    expect(describe('tool', { step: 0, tool: 'kb_search', status: 'denied' })).toEqual({ level: 'warn', summary: 'Step 1: kb_search denied by you' });
  });

  it('reads the fixed pipeline’s search, the context, the answer and the checks', () => {
    expect(describe('retrieve', { kb: 'Docs', query: 'headers?', hits: 0, ms: 12 })).toEqual({ level: 'warn', summary: 'Searched “Docs” for “headers?” · nothing found · 12 ms' });
    expect(describe('context', { tokens: 1234, passages: [{}, {}], history: [{}], nCtx: 4096 }).summary).toBe('Sent ~1,234 tokens to the model · 2 passages · 1 earlier message · context 4,096');
    expect(describe('answer', { text: '<think>\nhmm\n</think>\nYes, on deck 2 [1].', stats: { tokens: 9, tokPerSec: 44.44, firstTokenMs: 1500 }, sources: ['a:1-2'] }).summary).toBe(
      'Answered “Yes, on deck 2 [1].” · 9 tokens at 44.4 tok/s · first token 1.50 s · 1 source',
    );
    expect(describe('relevance', { model: 'Laya English', items: [{ kept: true }, { kept: false }], tokensSaved: 80, ms: 30 }).summary).toBe('Relevance check by Laya English: kept 1 of 2 passages · ~80 tokens saved · 30 ms');
    expect(describe('claims', { items: [{ sentence: 'a', flagged: true }, { sentence: 'b', flagged: false }], ms: 9 })).toEqual({ level: 'warn', summary: 'Claim check: 1 of 2 cited sentences may not be supported · 9 ms' });
  });

  it('marks stops as warnings and failures as errors', () => {
    expect(describe('error', { stopped: true })).toEqual({ level: 'warn', summary: 'Stopped by you' });
    expect(describe('error', { message: 'kv cache full' })).toEqual({ level: 'error', summary: 'Failed: kv cache full' });
    expect(describe('done', { outcome: 'answered', ms: 3200, toolCalls: 2, passages: 4 })).toEqual({ level: 'info', summary: 'Turn answered in 3.20 s · 2 tool calls · 4 passages' });
    expect(describe('done', { outcome: 'failed', ms: 10 }).level).toBe('error');
  });

  it('reads model and knowledge base events', () => {
    expect(describe('model', { action: 'load', slot: 'chat', model: 'Qwen3 1.7B', engine: 'MLX', ms: 420, backend: 'MLX · Metal (native)' }).summary).toBe('Loaded chat model Qwen3 1.7B (MLX) in 420 ms · MLX · Metal (native)');
    expect(describe('model', { action: 'load', slot: 'chat', model: 'Qwen3 1.7B · MLX', engine: 'MLX', ms: 5 }).summary).toBe('Loaded chat model Qwen3 1.7B · MLX in 5 ms');
    expect(describe('model', { action: 'unload', slot: 'decider', model: 'Laya English' }).summary).toBe('Unloaded decision model Laya English');
    expect(describe('kb', { action: 'index', kb: 'Ferries', sources: 7, nodes: 120, edges: 300, ms: 4100 }).summary).toBe('Indexed “Ferries” · 7 sources · 120 nodes · 300 edges · 4.10 s');
    expect(describe('kb', { action: 'add', kb: 'Ferries', files: 2, skipped: ['big.pdf is over 100 MB'] })).toEqual({ level: 'warn', summary: 'Added 2 files to “Ferries” · 1 skipped' });
    expect(describe('kb', { action: 'index', kb: 'Ferries', error: 'ug not found' })).toEqual({ level: 'error', summary: 'Indexing “Ferries” failed: ug not found' });
  });

  it('never throws on data of any shape, and keeps a summary to one bounded line', () => {
    for (const kind of ['turn', 'step', 'args', 'tool', 'retrieve', 'relevance', 'context', 'answer', 'claims', 'error', 'done', 'model', 'kb', 'mystery']) {
      for (const data of [null, undefined, 42, 'x', [], { items: 'no', stats: 3, decision: [1] }]) {
        const d = describe(kind, data);
        expect(typeof d.summary).toBe('string');
        expect(['info', 'warn', 'error']).toContain(d.level);
      }
    }
    const long = describe('error', { message: `line one\nline two ${'x'.repeat(2000)}` }).summary;
    expect(long).not.toContain('\n');
    expect(long.length).toBeLessThanOrEqual(400);
  });

  it('formats durations', () => {
    expect([fmtMs(12.4), fmtMs(1500), fmtMs(42_000)]).toEqual(['12 ms', '1.50 s', '42.0 s']);
  });
});

group('grouping a day for the Logs screen', () => {
  it('groups by question, newest first, keeps app events apart and fills in missing summaries', () => {
    const day = [
      { at: 1, kind: 'model', turn: 'app', summary: 'Loaded chat model Qwen3', level: 'info', data: {} },
      { at: 2, kind: 'turn', turn: 'm1', data: { question: 'First?' } },
      { at: 3, kind: 'tool', turn: 'm1', data: { step: 0, tool: 'kb_search', status: 'error', error: 'boom' } },
      { at: 4, kind: 'turn', turn: 'm2', data: { question: 'Second?' } },
      { at: 5, kind: 'done', turn: 'm1', data: { outcome: 'answered' } },
      'junk',
      { at: 6, kind: 'done', turn: 'm2', data: { outcome: 'answered' } },
    ];
    const groups = groupEvents(day);
    expect(groups.map((g) => g.id)).toEqual(['m2', 'm1', 'app-0']);
    const m1 = groups[1];
    expect(m1).toMatchObject({ type: 'turn', question: 'First?', at: 2, end: 5, level: 'error' });
    expect(m1.events.map((e) => e.kind)).toEqual(['turn', 'tool', 'done']);
    expect(m1.events[1].summary).toBe('Step 1: kb_search failed · boom');
    expect(groups[2]).toMatchObject({ type: 'app', level: 'info', events: [{ summary: 'Loaded chat model Qwen3' }] });
  });
});
