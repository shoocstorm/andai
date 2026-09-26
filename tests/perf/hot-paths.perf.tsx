// Hot paths of a chat turn, measured in isolation (docs/performance.md).
// Inputs are sized like a real long session with Qwen3 0.6B, so a change
// that makes one of them grow worse than linearly shows up here first.
import { renderToString } from 'react-dom/server';
import { expect, it } from 'vitest';
import { budgets, buildHistory, buildSystem, keywords } from '../../src/agent/prompt';
import { dedupeHits, type SearchHit } from '../../src/kb/api';
import { Markdown } from '../../src/components/ui';
import { Sha256 } from '../../src/llm/integrity';
import { modelById } from '../../src/llm/models';
import { addMessage, clearChat, patchMessage, useChat, type Message } from '../../src/state/chat';
import { msPerOp, record, reportTo } from './measure';

reportTo(import.meta.filename);

const qwen = modelById('qwen3-0.6b')!;
const prose = (n: number) =>
  'wllama needs cross-origin isolation for SharedArrayBuffer, so the release build serves the UI from a loopback origin. '
    .repeat(Math.ceil(n / 116))
    .slice(0, n);

/** A long, persisted conversation: 200 turns, answers with think blocks and sources. */
function longChat(turns = 200): Message[] {
  const out: Message[] = [];
  for (let i = 0; i < turns; i++) {
    out.push({ id: `u${i}`, role: 'user', content: prose(160), createdAt: i });
    out.push({
      id: `a${i}`,
      role: 'assistant',
      content: `<think>${prose(400)}</think>\n\n${prose(900)}`,
      createdAt: i,
      steps: [],
      sources: hits(4),
    });
  }
  return out;
}

/** ug-shaped hits: each document node next to its nested sections (see dedupeHits). */
function hits(n: number): SearchHit[] {
  return Array.from({ length: n }, (_, i) => ({
    id: `h${i}`,
    name: `Section ${i}`,
    node_type: 'Section',
    file: `doc-${i % 8}.md`,
    start_line: (i % 4) * 10,
    end_line: (i % 4) * 10 + (i % 3 === 0 ? 200 : 8),
    snippet: prose(1500),
  }));
}

it('sha256 throughput (model verification before load)', () => {
  const chunk = new Uint8Array(8 * 2 ** 20).map((_, i) => i * 31);
  const mb = 64;
  const ms = msPerOp(
    () => {
      const h = new Sha256();
      for (let i = 0; i < mb / 8; i++) h.update(chunk);
      return h.hex();
    },
    { samples: 5, sampleMs: 0 },
  );
  record('sha256-throughput', { value: (mb / ms) * 1000, unit: 'MB/s', better: 'higher' });
});

it('keywords of a long prompt', () => {
  const q = prose(2000);
  record('keywords-2k', { value: msPerOp(() => keywords(q)) * 1000, unit: 'µs', better: 'lower' });
});

it('dedupe 64 search hits', () => {
  const h = hits(64);
  expect(dedupeHits(h).length).toBeGreaterThan(0);
  record('dedupe-hits-64', { value: msPerOp(() => dedupeHits(h)) * 1000, unit: 'µs', better: 'lower' });
});

it('build the system prompt from 8 passages', () => {
  const b = budgets(qwen.n_ctx, 1024);
  const h = hits(8);
  const persona = { systemPrompt: prose(600), tone: 'professional' as const };
  record('build-system-8', {
    value: msPerOp(() => buildSystem(persona, h, b.context, 'Docs')) * 1000,
    unit: 'µs',
    better: 'lower',
  });
});

it('build history from a 400-message chat', () => {
  const msgs = longChat();
  const b = budgets(qwen.n_ctx, 1024);
  expect(buildHistory(msgs, b.history).length).toBeGreaterThan(0);
  record('build-history-400', {
    value: msPerOp(() => buildHistory(msgs, b.history, 'a199')) * 1000,
    unit: 'µs',
    better: 'lower',
  });
});

it('one streaming repaint of the chat store with 400 messages (persisted)', () => {
  // turn.ts patches the streaming message every 33 ms, and the persist
  // middleware writes the whole chat to localStorage each time.
  clearChat();
  useChat.setState({ messages: longChat() });
  addMessage({ id: 'live', role: 'assistant', content: '', createdAt: 0, streaming: true });
  let n = 0;
  const ms = msPerOp(() => patchMessage('live', { content: prose(200 + (n++ % 800)) }));
  expect(localStorage.getItem('andai.chat')?.length).toBeGreaterThan(500_000);
  record('stream-patch-400', { value: ms, unit: 'ms', better: 'lower', slack: 0.5 });
  clearChat();
});

it('render a long markdown answer', () => {
  const answer = [
    '## Why the headers matter',
    prose(600),
    '- COOP: `same-origin`\n- COEP: `require-corp`\n- CORP on every subresource',
    '```ts\nconst w = new Wllama(paths);\nawait w.loadModel(model, { n_ctx: 4096 });\n```',
    '| Header | Value |\n|---|---|\n| COOP | same-origin |\n| COEP | require-corp |',
    prose(1200),
    'See [1] and [the docs](https://github.com/ngxson/wllama).',
  ]
    .join('\n\n')
    .repeat(3);
  expect(renderToString(<Markdown text={answer} />)).toContain('<table>');
  record('markdown-render-8k', {
    value: msPerOp(() => renderToString(<Markdown text={answer} />)),
    unit: 'ms',
    better: 'lower',
    slack: 0.5,
  });
});
