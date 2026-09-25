import { beforeEach, describe, expect, it } from 'vitest';
import { addMessage, clearChat, patchStep, splitThink, useChat, type Message } from './chat';

describe('splitThink', () => {
  it('passes through text without a think block', () => {
    expect(splitThink('plain')).toEqual({ thinking: null, answer: 'plain', open: false });
  });
  it('separates closed reasoning from the answer', () => {
    expect(splitThink('<think>\nstep 1\n</think>\n\nThe answer')).toEqual({
      thinking: 'step 1',
      answer: 'The answer',
      open: false,
    });
  });
  it('reports an open block while the model is still thinking', () => {
    expect(splitThink('<think>partial')).toEqual({ thinking: 'partial', answer: '', open: true });
  });
  it('treats the empty pair Qwen3 emits with thinking off as no reasoning', () => {
    expect(splitThink('<think>\n\n</think>\n\nHi').thinking).toBeNull();
  });
});

describe('chat store', () => {
  beforeEach(() => clearChat());

  it('clearChat empties messages and rotates the session id', () => {
    addMessage({ id: 'a', role: 'user', content: 'x', createdAt: 0 });
    clearChat();
    expect(useChat.getState().messages).toEqual([]);
    expect(useChat.getState().session).toMatch(/^X-\d{3}$/);
  });

  it('patchStep records duration from running → done', async () => {
    const m: Message = {
      id: 't',
      role: 'assistant',
      content: '',
      createdAt: 0,
      steps: [{ kind: 'retrieve', title: 'r', detail: '', status: 'queued' }],
    };
    addMessage(m);
    patchStep('t', 'retrieve', { status: 'running' });
    await new Promise((r) => setTimeout(r, 15));
    patchStep('t', 'retrieve', { status: 'done', detail: 'ok' });
    const step = useChat.getState().messages[0].steps![0];
    expect(step.status).toBe('done');
    expect(step.detail).toBe('ok');
    expect(step.ms).toBeGreaterThanOrEqual(10);
  });
});

describe('chat persistence', () => {
  it('marks a turn interrupted by quitting the app as stopped, never stuck streaming', async () => {
    localStorage.setItem(
      'andai.chat',
      JSON.stringify({
        state: {
          session: 'X-111',
          messages: [
            {
              id: 'z',
              role: 'assistant',
              content: 'half',
              createdAt: 0,
              streaming: true,
              steps: [{ kind: 'generate', title: 'g', detail: '', status: 'running' }],
            },
          ],
        },
        version: 0,
      }),
    );
    await useChat.persist.rehydrate();
    const m = useChat.getState().messages[0];
    expect(m.streaming).toBe(false);
    expect(m.stopped).toBe(true);
    expect(m.steps![0].status).toBe('skipped');
  });
});
