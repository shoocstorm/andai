// The activity log, read back (state/activity.ts, src-tauri/src/activity.rs):
// one day at a time, each question with every step the agent took for it,
// and what the app did to models and knowledge bases in between. Each event
// shows its one-line summary (lib/activityText.ts) and opens to the raw data
// it was logged with. The data is the user's own, from their own disk, but it
// holds model output and passages from documents, so it is shown as plain
// text only (AGENTS.md §9).

import { isTauri } from '@tauri-apps/api/core';
import { ChevronRight, FolderOpen, RefreshCw, ScrollText, Search } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { CopyButton, Toggle, fmtBytes } from '../components/ui';
import { KIND_LABEL, clip, fmtMs, groupEvents, type Level, type LogEvent, type LogGroup } from '../lib/activityText';
import { activityInfo, openActivityFolder, readActivity, useActivity, type ActivityDay, type ActivityInfo } from '../state/activity';
import { toast, useUi } from '../state/ui';

type Filter = 'all' | 'questions' | 'models' | 'knowledge' | 'problems';
const FILTERS: { id: Filter; label: string }[] = [
  { id: 'all', label: 'All' },
  { id: 'questions', label: 'Questions' },
  { id: 'models', label: 'Models' },
  { id: 'knowledge', label: 'Knowledge' },
  { id: 'problems', label: 'Problems' },
];
/** Groups shown before "Show more". */
const PAGE = 100;
/** How often today's file is read again while Live is on. */
const LIVE_MS = 2000;

const LEVEL_TONE: Record<Level, string> = { info: '', warn: 'amber', error: 'red' };
const KIND_TONE: Record<string, string> = { turn: 'blue', step: 'violet', args: 'violet', tool: 'blue', retrieve: 'blue', answer: 'green', done: 'green', model: 'violet', kb: 'green', error: 'red' };

/** "Wed 30 Sep" for `agent-2026-09-30.jsonl`, in UTC like the file name. */
function dayLabel(name: string): string {
  const m = /^agent-(\d{4})-(\d{2})-(\d{2})\.jsonl$/.exec(name);
  if (!m) return name;
  const d = new Date(Date.UTC(+m[1], +m[2] - 1, +m[3]));
  return d.toLocaleDateString(undefined, { weekday: 'short', day: 'numeric', month: 'short', timeZone: 'UTC' });
}
const clock = (at: number) => new Date(at).toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });

function matches(g: LogGroup, filter: Filter, q: string): boolean {
  const kinds = g.events.map((e) => e.kind);
  if (filter === 'questions' && g.type !== 'turn') return false;
  if (filter === 'models' && !kinds.includes('model')) return false;
  if (filter === 'knowledge' && !kinds.includes('kb')) return false;
  if (filter === 'problems' && g.level === 'info') return false;
  if (!q) return true;
  return g.events.some((e) => e.summary.toLowerCase().includes(q) || JSON.stringify(e.data).toLowerCase().includes(q));
}

export function Logs() {
  const desktop = isTauri();
  const { enabled, setEnabled } = useActivity();
  const go = useUi((s) => s.go);
  const [info, setInfo] = useState<ActivityInfo | null>(null);
  const [name, setName] = useState<string | null>(null);
  const [day, setDay] = useState<ActivityDay | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<Filter>('all');
  const [query, setQuery] = useState('');
  const [live, setLive] = useState(true);
  const [shown, setShown] = useState(PAGE);
  const [open, setOpen] = useState<Set<string>>(new Set());

  const refresh = async (file = name) => {
    if (!desktop) return;
    try {
      const next = await activityInfo();
      setInfo(next);
      const pick = file && next.files.some((f) => f.name === file) ? file : (next.files[0]?.name ?? null);
      setName(pick);
      setDay(pick ? await readActivity(pick) : null);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };
  // Once on open; Live and the day picker read again.
  useEffect(() => void refresh(), []);

  // Live: re-read the newest day while it's the one on screen.
  const newest = info?.files[0]?.name ?? null;
  const following = live && enabled && !!name && name === newest;
  const refreshRef = useRef(refresh);
  refreshRef.current = refresh;
  useEffect(() => {
    if (!following) return;
    const t = setInterval(() => void refreshRef.current(), LIVE_MS);
    return () => clearInterval(t);
  }, [following]);

  const groups = useMemo(() => groupEvents(day?.events ?? []), [day]);
  const q = query.trim().toLowerCase();
  const visible = useMemo(() => groups.filter((g) => matches(g, filter, q)), [groups, filter, q]);
  const counts = useMemo(() => {
    const all = groups.flatMap((g) => g.events);
    return { events: all.length, questions: groups.filter((g) => g.type === 'turn').length, warn: all.filter((e) => e.level === 'warn').length, error: all.filter((e) => e.level === 'error').length };
  }, [groups]);
  const toggle = (id: string) =>
    setOpen((s) => {
      const n = new Set(s);
      if (n.has(id)) n.delete(id);
      else n.add(id);
      return n;
    });
  const file = info?.files.find((f) => f.name === name);

  return (
    <div className="screen">
      <div className="page lg">
        <div className="page-hero" style={{ marginBottom: 26 }}>
          <h1>
            Activity <span className="grad-text">Log</span>
          </h1>
          <p>
            What the agent did for each question, step by step, and what the app did to your models and knowledge bases.
            Kept 7 days on this computer; nothing leaves it.
          </p>
        </div>

        {!desktop ? (
          <div className="panel pad empty">
            <ScrollText size={28} />
            <div>The activity log is written and read by the desktop app.</div>
          </div>
        ) : (
          <>
            {!enabled && (
              <div className="panel pad lg-off" role="status">
                <div>
                  <div className="lg-off-title">The activity log is off</div>
                  <div className="faint">Turn it on to record each question from now on. It stores your questions and passages from your documents.</div>
                </div>
                <Toggle checked={enabled} onChange={setEnabled} label="Keep an activity log" />
              </div>
            )}

            <div className="panel lg-bar">
              <select className="field lg-day" aria-label="Day" value={name ?? ''} onChange={(e) => void refresh(e.target.value).then(() => setShown(PAGE))} disabled={!info?.files.length}>
                {info?.files.length ? (
                  info.files.map((f) => (
                    <option key={f.name} value={f.name}>
                      {dayLabel(f.name)} · {fmtBytes(f.bytes)}
                    </option>
                  ))
                ) : (
                  <option value="">No log files yet</option>
                )}
              </select>
              <div className="lg-filters" role="group" aria-label="Show">
                {FILTERS.map((f) => (
                  <button key={f.id} className="lg-chip" aria-pressed={filter === f.id} onClick={() => setFilter(f.id)}>
                    {f.label}
                  </button>
                ))}
              </div>
              <label className="lg-search">
                <Search size={14} />
                <input className="field" type="search" placeholder="Search summaries and data" aria-label="Search the log" value={query} onChange={(e) => setQuery(e.target.value)} />
              </label>
              <div className="lg-actions">
                <span className="lg-live">
                  <Toggle checked={live} onChange={setLive} label="Live" />
                  <span className={following ? 'lg-live-on' : 'faint'}>Live</span>
                </span>
                <button className="btn ghost sm" onClick={() => void refresh()} aria-label="Refresh">
                  <RefreshCw size={13} />
                </button>
                <button className="btn ghost sm" onClick={() => void openActivityFolder().catch((err: unknown) => toast({ tone: 'error', title: 'Couldn’t open the logs folder', body: String(err) }))}>
                  <FolderOpen size={13} /> Folder
                </button>
              </div>
            </div>

            <div className="lg-counts mono">
              <span>{counts.events.toLocaleString()} events</span>
              <span>{counts.questions.toLocaleString()} questions</span>
              <span className={counts.warn ? 'log-warn' : ''}>{counts.warn} warnings</span>
              <span className={counts.error ? 'log-err' : ''}>{counts.error} errors</span>
              {file && <span>{fmtBytes(file.bytes)}</span>}
              {day && day.total > day.events.length + day.bad && <span className="log-warn">showing the newest {day.events.length.toLocaleString()} of {day.total.toLocaleString()}</span>}
              {!!day?.bad && <span className="log-warn">{day.bad} unreadable lines skipped</span>}
              <span className="lg-grow" />
              <button className="lg-link" onClick={() => go('settings')}>
                Log settings
              </button>
            </div>

            {error && (
              <div className="panel pad log-err" role="alert">
                Couldn’t read the log: {error}
              </div>
            )}

            {!visible.length ? (
              <div className="panel pad empty">
                <ScrollText size={28} />
                <div>{groups.length ? 'Nothing matches.' : enabled ? 'Nothing logged yet on this day. Ask a question and it appears here.' : 'Nothing logged on this day.'}</div>
              </div>
            ) : (
              <div className="lg-list">
                {visible.slice(0, shown).map((g) =>
                  g.type === 'turn' ? (
                    <TurnCard key={g.id} group={g} open={open.has(g.id)} onToggle={() => toggle(g.id)} />
                  ) : (
                    <div key={g.id} className="panel lg-app">
                      <EventRow event={g.events[0]} start={null} />
                    </div>
                  ),
                )}
                {visible.length > shown && (
                  <button className="btn ghost sm lg-more" onClick={() => setShown((n) => n + PAGE)}>
                    Show {Math.min(PAGE, visible.length - shown)} more of {visible.length - shown}
                  </button>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

function TurnCard({ group, open, onToggle }: { group: Extract<LogGroup, { type: 'turn' }>; open: boolean; onToggle: () => void }) {
  const first = group.events.find((e) => e.kind === 'turn');
  const done = group.events.find((e) => e.kind === 'done');
  const d = (first?.data ?? {}) as { mode?: string; model?: string; kb?: { name?: string } | null };
  const outcome = (done?.data as { outcome?: string } | undefined)?.outcome;
  const tools = group.events.filter((e) => e.kind === 'tool').length;
  return (
    <div className={`panel lg-turn${open ? ' open' : ''}`}>
      <button className="lg-turn-head" aria-expanded={open} onClick={onToggle}>
        <ChevronRight size={16} className="lg-chev" />
        <span className="mono faint lg-time">{clock(group.at)}</span>
        <span className={`lg-dot ${group.level}`} aria-label={group.level === 'info' ? undefined : group.level} />
        <span className="lg-question">{group.question ? clip(group.question, 200) : <span className="faint">(question not logged)</span>}</span>
        <span className="lg-pills">
          {d.mode && <span className="pill">{d.mode === 'agent' ? 'agent' : 'one search'}</span>}
          {d.model && <span className="pill violet">{d.model}</span>}
          {d.kb?.name && <span className="pill green">{d.kb.name}</span>}
          {tools > 0 && <span className="pill blue">{tools} tool{tools === 1 ? '' : 's'}</span>}
          <span className={`pill ${outcome === 'failed' ? 'red' : outcome === 'stopped' ? 'amber' : ''}`}>{outcome ? `${outcome} · ${fmtMs(group.end - group.at)}` : 'unfinished'}</span>
        </span>
      </button>
      {open && (
        <div className="lg-events">
          {group.events.map((e, i) => (
            <EventRow key={i} event={e} start={group.at} />
          ))}
        </div>
      )}
    </div>
  );
}

function EventRow({ event, start }: { event: LogEvent; start: number | null }) {
  const [open, setOpen] = useState(false);
  const json = useMemo(() => (open ? JSON.stringify(event.data, null, 2) : ''), [open, event.data]);
  return (
    <div className={`lg-event ${event.level}`}>
      <button className="lg-event-head" aria-expanded={open} onClick={() => setOpen(!open)}>
        <ChevronRight size={14} className="lg-chev" />
        <span className="mono faint lg-time">{start === null ? clock(event.at) : `+${fmtMs(event.at - start)}`}</span>
        <span className={`pill ${event.level === 'info' ? (KIND_TONE[event.kind] ?? '') : LEVEL_TONE[event.level]} lg-kind`}>{KIND_LABEL[event.kind] ?? event.kind}</span>
        <span className="lg-summary">{event.summary}</span>
      </button>
      {open && (
        <div className="lg-data">
          <div className="lg-data-head faint mono">
            {new Date(event.at).toLocaleString()} · {event.kind}
            <CopyButton text={() => JSON.stringify({ at: event.at, kind: event.kind, turn: event.turn, summary: event.summary, level: event.level, data: event.data }, null, 2)} label="Copy this event as JSON" />
          </div>
          <pre className="logblock lg-json">{json}</pre>
        </div>
      )}
    </div>
  );
}
