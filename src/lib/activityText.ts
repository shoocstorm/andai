// PURE: one line a person can read for each activity log event (state/activity.ts),
// and how bad it is. Written into the log with the event, so the files read
// without the app, and recomputed by the Logs screen for lines that have none.
// The data is whatever was logged, possibly by an older build or edited by
// hand, so every reader here tolerates any shape.

export type Level = 'info' | 'warn' | 'error';
export type Described = { summary: string; level: Level };

type Obj = Record<string, unknown>;
const obj = (v: unknown): Obj => (v && typeof v === 'object' && !Array.isArray(v) ? (v as Obj) : {});
const str = (v: unknown): string | null => (typeof v === 'string' && v.trim() ? v : null);
const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const arr = (v: unknown): unknown[] => (Array.isArray(v) ? v : []);

/** One line, at most `n` characters. */
export const clip = (s: string, n: number) => {
  const flat = s.replace(/\s+/g, ' ').trim();
  return flat.length > n ? `${flat.slice(0, n - 1)}…` : flat;
};
const plural = (n: number, word: string) => `${n.toLocaleString()} ${word}${n === 1 ? '' : 's'}`;
export const fmtMs = (ms: number) => (ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(ms < 10_000 ? 2 : 1)} s`);
const pct = (p: number) => `${Math.round(p * 100)}%`;
const join = (...parts: (string | number | null | false | undefined)[]) => parts.filter(Boolean).join(' · ');
const quote = (s: string, n = 80) => `“${clip(s, n)}”`;

const ACTION: Record<string, string> = { answer_now: 'answer', ask_clarification: 'ask to clarify' };
const actionName = (a: unknown) => ACTION[String(a)] ?? String(a ?? '?');
const stepNo = (d: Obj) => {
  const i = num(d.step) ?? num(d.index);
  return i === null ? 'Step' : `Step ${i + 1}`;
};

const FALLBACK: Record<string, string> = {
  'low-confidence': 'decision below the confidence floor',
  'decision-failed': 'the decision failed',
  'needs-symbol': 'looked the symbol up first',
  'needs-range': 'searched first for a line range',
  'read-whole': 'read a clipped passage whole',
  'named-symbol': 'fetched a symbol the request names',
  'whole-repo': 'analyzed the whole codebase first',
};

function turn(d: Obj): Described {
  const kb = obj(d.kb);
  return {
    level: 'info',
    summary: join(
      `Asked ${quote(str(d.question) ?? '', 120)}`,
      d.mode === 'agent' ? 'agent mode' : d.mode === 'fixed' ? 'one search' : null,
      str(d.model),
      str(d.decider) && `decider ${str(d.decider)}`,
      str(kb.name) ? `knowledge base “${str(kb.name)}”` : 'no knowledge base',
    ),
  };
}

function step(d: Obj): Described {
  const dec = obj(d.decision);
  const failed = obj(d.failedDecision);
  const chosen = num(dec.confidence);
  const how = str(dec.model)
    ? `${chosen !== null ? `${pct(chosen)} ` : ''}by ${str(dec.model)}${num(dec.ms) !== null ? ` in ${fmtMs(num(dec.ms)!)}` : ''}`
    : d.planned
      ? 'by rule'
      : 'no decision needed';
  const stop = obj(dec.stop);
  return {
    level: str(failed.error) ? 'warn' : 'info',
    summary: join(
      `${stepNo(d)}: ${actionName(d.action)}`,
      how,
      num(stop.probability) !== null && `results suffice ${pct(num(stop.probability)!)}`,
      str(d.fallback) && (FALLBACK[String(d.fallback)] ?? String(d.fallback)),
      str(failed.error) && `decision failed: ${clip(String(failed.error), 120)}`,
      str(d.note) && clip(String(d.note), 140),
    ),
  };
}

function args(d: Obj): Described {
  const io = obj(d.io);
  const attempts = num(d.attempts) ?? 0;
  if (str(d.error)) return { level: 'error', summary: join(`${stepNo(d)}: couldn’t write arguments for ${d.tool}`, clip(String(d.error), 160), str(d.model), attempts && plural(attempts, 'attempt')) };
  const a = d.args && typeof d.args === 'object' ? clip(JSON.stringify(d.args), 160) : '—';
  return {
    level: 'info',
    summary: join(`${stepNo(d)}: ${d.tool} ${a}`, str(io.note) ? clip(String(io.note), 80) : str(d.model) && `written by ${str(d.model)}`, attempts > 1 && plural(attempts, 'attempt')),
  };
}

function tool(d: Obj): Described {
  const name = `${stepNo(d)}: ${d.tool}`;
  const ms = num(d.ms);
  const obs = str(d.observation) && `→ ${clip(String(d.observation), 160)}`;
  switch (d.status) {
    case 'error':
      return { level: 'error', summary: join(`${name} failed`, clip(String(d.error ?? 'unknown error'), 160), ms !== null && fmtMs(ms)) };
    case 'denied':
      return { level: 'warn', summary: `${name} denied by you` };
    case 'skipped':
      return { level: 'warn', summary: join(`${name} skipped`, str(d.error) && clip(String(d.error), 160)) };
    default: {
      const hits = num(d.hits);
      const bytes = num(d.outputBytes);
      return {
        level: 'info',
        summary: join(`${name} ran${ms !== null ? ` in ${fmtMs(ms)}` : ''}`, hits !== null && `${plural(hits, 'passage')} added`, bytes !== null && `${Math.ceil(bytes / 1024)} KB out`, d.truncated === true && 'output cut', obs),
      };
    }
  }
}

function retrieve(d: Obj): Described {
  if (str(d.error)) return { level: 'error', summary: join(`Search in “${d.kb}” failed`, clip(String(d.error), 160)) };
  const hits = num(d.hits) ?? arr(d.sources).length;
  return { level: hits ? 'info' : 'warn', summary: join(`Searched “${d.kb}” for ${quote(str(d.query) ?? '')}`, hits ? plural(hits, 'passage') : 'nothing found', num(d.ms) !== null && fmtMs(num(d.ms)!)) };
}

function relevance(d: Obj): Described {
  const items = arr(d.items).map(obj);
  const kept = items.filter((i) => i.kept !== false).length;
  return {
    level: 'info',
    summary: join(
      `Relevance check${str(d.model) ? ` by ${str(d.model)}` : ''}: kept ${kept} of ${plural(items.length, 'passage')}`,
      (num(d.tokensSaved) ?? 0) > 0 && `~${num(d.tokensSaved)!.toLocaleString()} tokens saved`,
      num(d.ms) !== null && fmtMs(num(d.ms)!),
    ),
  };
}

function context(d: Obj): Described {
  const tokens = num(d.tokens);
  const passages = arr(d.passages).length;
  const history = Array.isArray(d.history) ? d.history.length : num(obj(d.history).sent);
  return {
    level: 'info',
    summary: join(`Sent ${tokens !== null ? `~${tokens.toLocaleString()} tokens` : 'the prompt'} to the model`, plural(passages, 'passage'), history !== null && plural(history, 'earlier message'), num(d.nCtx) !== null && `context ${num(d.nCtx)!.toLocaleString()}`),
  };
}

/** The answer without its reasoning (Qwen3's `<think>` block, empty or not). */
const answerOf = (text: string) => text.replace(/<think>[\s\S]*?(<\/think>|$)/g, '').trim();

function answer(d: Obj): Described {
  const s = obj(d.stats);
  const text = answerOf(str(d.text) ?? '');
  const tokens = num(s.tokens);
  const first = num(s.firstTokenMs);
  return {
    level: text ? 'info' : 'warn',
    summary: join(
      text ? `Answered ${quote(text, 100)}` : 'Answered with no text',
      tokens !== null && `${plural(tokens, 'token')}${num(s.tokPerSec) ? ` at ${num(s.tokPerSec)!.toFixed(1)} tok/s` : ''}`,
      first !== null && `first token ${fmtMs(first)}`,
      plural(arr(d.sources).length, 'source'),
    ),
  };
}

function claims(d: Obj): Described {
  const items = arr(d.items).map(obj);
  const n = new Set(items.map((i) => i.sentence)).size;
  const flagged = new Set(items.filter((i) => i.flagged).map((i) => i.sentence)).size;
  return {
    level: flagged ? 'warn' : 'info',
    summary: join(flagged ? `Claim check: ${flagged} of ${plural(n, 'cited sentence')} may not be supported` : `Claim check: ${n === 1 ? 'the cited sentence looks' : `all ${n} cited sentences look`} supported`, num(d.ms) !== null && fmtMs(num(d.ms)!)),
  };
}

function error(d: Obj): Described {
  if (d.stopped) return { level: 'warn', summary: 'Stopped by you' };
  return { level: 'error', summary: `Failed: ${clip(String(d.message ?? 'unknown error'), 200)}` };
}

function done(d: Obj): Described {
  const outcome = str(d.outcome) ?? 'finished';
  return {
    level: outcome === 'failed' ? 'error' : outcome === 'stopped' ? 'warn' : 'info',
    summary: join(`Turn ${outcome}${num(d.ms) !== null ? ` in ${fmtMs(num(d.ms)!)}` : ''}`, num(d.toolCalls) !== null && plural(num(d.toolCalls)!, 'tool call'), num(d.passages) !== null && plural(num(d.passages)!, 'passage')),
  };
}

function model(d: Obj): Described {
  const role = d.slot === 'decider' ? 'decision model' : 'chat model';
  const name = str(d.model) ?? str(d.id) ?? 'a model';
  // MLX catalog names already say so ("Qwen3 1.7B · MLX").
  const engine = str(d.engine) && !name.includes(String(d.engine)) && ` (${d.engine})`;
  switch (d.action) {
    case 'load':
      return { level: 'info', summary: join(`Loaded ${role} ${name}${engine || ''}${num(d.ms) !== null ? ` in ${fmtMs(num(d.ms)!)}` : ''}`, num(d.verifyMs) !== null && `verified in ${fmtMs(num(d.verifyMs)!)}`, str(d.backend)) };
    case 'unload':
      return { level: 'info', summary: `Unloaded ${role} ${name}` };
    case 'load-failed':
      return { level: 'error', summary: `Couldn’t load ${role} ${name}: ${clip(String(d.error ?? 'unknown error'), 200)}` };
    case 'remove':
      return { level: 'info', summary: `Deleted ${name} from this computer` };
    default:
      return { level: 'info', summary: join(`Model ${String(d.action ?? 'event')}`, name) };
  }
}

function kb(d: Obj): Described {
  const name = `“${str(d.kb) ?? '?'}”`;
  if (str(d.error)) {
    const what: Record<string, string> = { create: 'Creating', index: 'Indexing', add: 'Adding files to', sample: 'Adding the sample', remove: 'Removing a source from', delete: 'Deleting', kind: 'Changing the kind of' };
    return { level: 'error', summary: `${what[String(d.action)] ?? String(d.action)} ${name} failed: ${clip(String(d.error), 200)}` };
  }
  switch (d.action) {
    case 'create':
      return { level: 'info', summary: `Created knowledge base ${name}` };
    case 'index':
      return {
        level: 'info',
        summary: join(`Indexed ${name}`, plural(num(d.sources) ?? 0, 'source'), num(d.nodes) !== null && plural(num(d.nodes)!, 'node'), num(d.edges) !== null && plural(num(d.edges)!, 'edge'), num(d.ms) !== null && fmtMs(num(d.ms)!)),
      };
    case 'add': {
      const skipped = arr(d.skipped).length;
      return { level: skipped ? 'warn' : 'info', summary: join(`Added ${plural(num(d.files) ?? 0, 'file')} to ${name}`, skipped && `${skipped} skipped`) };
    }
    case 'sample':
      return { level: 'info', summary: `Added the sample knowledge base ${name}` };
    case 'remove':
      return { level: 'info', summary: `Removed “${d.file}” from ${name}` };
    case 'delete':
      return { level: 'info', summary: `Deleted knowledge base ${name}` };
    case 'kind':
      return { level: 'info', summary: `Set ${name} to ${d.kind ?? 'automatic'}` };
    default:
      return { level: 'info', summary: `Knowledge base ${name}: ${String(d.action ?? 'event')}` };
  }
}

const DESCRIBE: Record<string, (d: Obj) => Described> = { turn, step, args, tool, retrieve, relevance, context, answer, claims, error, done, model, kb };

/** The line for one event; never throws. */
export function describe(kind: string, data: unknown): Described {
  try {
    const f = DESCRIBE[kind];
    if (f) {
      const d = f(obj(data));
      return { ...d, summary: clip(d.summary, 400) };
    }
  } catch {
    // fall through: a line that can't be described is still shown
  }
  return { level: 'info', summary: `${kind} event` };
}

/** What each kind is called on the Logs screen. */
export const KIND_LABEL: Record<string, string> = {
  turn: 'Question',
  step: 'Step',
  args: 'Arguments',
  tool: 'Tool call',
  retrieve: 'Search',
  relevance: 'Relevance',
  context: 'Context',
  answer: 'Answer',
  claims: 'Claim check',
  error: 'Error',
  done: 'Finished',
  model: 'Model',
  kb: 'Knowledge',
};

// ── The Logs screen's view of a day ─────────────────────────────────────

/** A logged line as read back (state/activity.ts `ActivityEvent`), with its summary and level filled in. */
export type LogEvent = { at: number; kind: string; turn: string; summary: string; level: Level; data: unknown };
/** A question and everything the agent did for it, or one event that belongs to no question. */
export type LogGroup =
  | { type: 'turn'; id: string; at: number; end: number; question: string; level: Level; events: LogEvent[] }
  | { type: 'app'; id: string; at: number; end: number; level: Level; events: [LogEvent] };

const RANK: Record<Level, number> = { info: 0, warn: 1, error: 2 };
export const worst = (levels: Level[]): Level => levels.reduce<Level>((a, b) => (RANK[b] > RANK[a] ? b : a), 'info');

/** Fills in a line's summary and level when an older build didn't write them. */
export function readEvent(raw: unknown): LogEvent | null {
  const e = obj(raw);
  if (typeof e.kind !== 'string' || typeof e.turn !== 'string') return null;
  const d = describe(e.kind, e.data);
  const level = e.level === 'info' || e.level === 'warn' || e.level === 'error' ? e.level : d.level;
  return { at: num(e.at) ?? 0, kind: e.kind, turn: e.turn, summary: str(e.summary) ?? d.summary, level, data: e.data ?? null };
}

/** Groups a day's events by question, newest first; events within a question stay in order. */
export function groupEvents(raw: unknown[], appTurn = 'app'): LogGroup[] {
  const groups: LogGroup[] = [];
  const turns = new Map<string, Extract<LogGroup, { type: 'turn' }>>();
  raw.forEach((r, i) => {
    const e = readEvent(r);
    if (!e) return;
    if (e.turn === appTurn) {
      groups.push({ type: 'app', id: `app-${i}`, at: e.at, end: e.at, level: e.level, events: [e] });
      return;
    }
    let g = turns.get(e.turn);
    if (!g) {
      g = { type: 'turn', id: e.turn, at: e.at, end: e.at, question: '', level: 'info', events: [] };
      turns.set(e.turn, g);
      groups.push(g);
    }
    g.events.push(e);
    g.end = Math.max(g.end, e.at);
    if (e.kind === 'turn') g.question = str(obj(e.data).question) ?? '';
    g.level = worst([g.level, e.level]);
  });
  return groups.sort((a, b) => b.at - a.at);
}
