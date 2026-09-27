import {
  BarChart3,
  Check,
  ChevronDown,
  CircleDashed,
  Code2,
  Database,
  FileText,
  ListFilter,
  Loader2,
  MessagesSquare,
  Network,
  PanelRightClose,
  PanelRightOpen,
  Plus,
  RotateCw,
  Search,
  Send,
  Sparkles,
  Square,
  Cpu,
  Wrench,
  X,
} from 'lucide-react';
import { AnimatePresence, motion } from 'motion/react';
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { available, policyOf } from '../agent/tools/registry';
import { runTurn, stopTurn } from '../agent/turn';
import { Bar, CopyButton, Markdown, Stat, fmtTime } from '../components/ui';
import { debugReport } from '../agent/debugReport';
import { inTauri } from '../kb/api';
import { loadModel, useEngine } from '../llm/engine';
import { layaById, MODELS, modelById } from '../llm/models';
import { clearChat, splitThink, useChat, type Message, type TraceStep } from '../state/chat';
import { addFiles, createKb, useKb } from '../state/kb';
import { useLayout } from '../state/layout';
import { usePersona } from '../state/persona';
import { useTools } from '../state/tools';
import { toast, useUi } from '../state/ui';
import { AgentStepCard, ApprovalCard, CopyTraceButton, DecisionSummary, RelevanceList, ToolChips, decisionTiming, fmtMs } from './AgentTrace';
import { pickFiles } from './Knowledge';
import { shortcut } from '../lib/platform';

export const composerRef: { current: HTMLTextAreaElement | null } = { current: null };

const STEP_STYLE: Record<TraceStep['kind'], { color: string; icon: typeof Search }> = {
  analyze: { color: 'var(--blue)', icon: BarChart3 },
  retrieve: { color: 'var(--violet)', icon: Search },
  plan: { color: 'var(--violet)', icon: Wrench },
  filter: { color: 'var(--violet)', icon: ListFilter },
  build: { color: 'var(--blue)', icon: Network },
  generate: { color: 'var(--amber)', icon: Code2 },
};

export function CommandCenter() {
  const messages = useChat((s) => s.messages);
  const [focusId, setFocusId] = useState<string | null>(null);
  const lastAssistant = [...messages].reverse().find((m) => m.role === 'assistant');
  const focused = messages.find((m) => m.id === focusId && m.role === 'assistant') ?? lastAssistant;

  const traceOpen = useLayout((s) => s.traceOpen);

  return (
    <div className={`screen cc${traceOpen ? '' : ' no-trace'}`}>
      <section className="cc-hub">
        <HubHeader latest={lastAssistant} />
        <Thread messages={messages} focusedId={focused?.id} onFocus={setFocusId} />
        <Composer />
      </section>
      <AnimatePresence initial={false}>
        {traceOpen && (
          <motion.aside
            className="cc-trace"
            aria-label="Execution trace"
            initial={{ opacity: 0, x: 24 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 24 }}
            transition={{ duration: 0.18, ease: [0.2, 0.7, 0.2, 1] }}
          >
            <TracePanel msg={focused} />
          </motion.aside>
        )}
      </AnimatePresence>
    </div>
  );
}

// ── hub ──────────────────────────────────────────────────────────────────

function HubHeader({ latest }: { latest?: Message }) {
  const session = useChat((s) => s.session);
  const count = useChat((s) => s.messages.length);
  const busy = useChat((s) => s.messages.some((m) => m.streaming));
  const { traceOpen, toggleTrace } = useLayout();
  return (
    <div className="cc-head">
      <MessagesSquare size={24} color="var(--blue)" />
      <h1 className="display">Communication Hub</h1>
      <span className="label cc-session">SESSION: {session}</span>
      {!traceOpen && latest?.streaming && <MiniTrace msg={latest} />}
      <button
        className="btn ghost sm"
        disabled={!count || busy}
        onClick={() => {
          clearChat();
          toast({ tone: 'info', title: 'New session started' });
        }}
        title="Clear the conversation"
      >
        <RotateCw size={13} /> New session
      </button>
      <button
        className={`icon-btn cc-trace-toggle${traceOpen ? ' on' : ''}`}
        aria-label={traceOpen ? 'Hide execution trace' : 'Show execution trace'}
        aria-pressed={traceOpen}
        title={`${traceOpen ? 'Hide' : 'Show'} execution trace (${shortcut('J')})`}
        onClick={toggleTrace}
      >
        {traceOpen ? <PanelRightClose size={18} /> : <PanelRightOpen size={18} />}
      </button>
    </div>
  );
}

/** Compact live progress while the trace panel is hidden. */
function MiniTrace({ msg }: { msg: Message }) {
  const steps = msg.steps ?? [];
  const i = steps.findIndex((s) => s.status === 'running');
  const step = steps[i];
  if (!step) return null;
  return (
    <span className="cc-minitrace" role="status" title={step.detail}>
      <span className="dot pulse" style={{ color: STEP_STYLE[step.kind].color }} />
      Step {i + 1}/{steps.length} · {step.title}
    </span>
  );
}

function Thread({
  messages,
  focusedId,
  onFocus,
}: {
  messages: Message[];
  focusedId?: string;
  onFocus: (id: string) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const stick = useRef(true);
  const last = messages[messages.length - 1];

  useLayoutEffect(() => {
    const el = ref.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  }, [messages.length, last?.content, last?.steps, last?.agent]);

  if (!messages.length) return <EmptyHub />;

  return (
    <div
      className="cc-thread"
      ref={ref}
      onScroll={(e) => {
        const el = e.currentTarget;
        stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 90;
      }}
    >
      <div className="cc-thread-inner">
        {messages.map((m) =>
          m.role === 'user' ? (
            <UserMsg key={m.id} m={m} />
          ) : m.role === 'error' ? (
            <div key={m.id} className="cc-error">
              <Markdown text={m.content} />
              <CopyButton text={m.content} label="Copy error" />
            </div>
          ) : (
            <AssistantMsg key={m.id} m={m} focused={m.id === focusedId} onFocus={() => onFocus(m.id)} />
          ),
        )}
      </div>
    </div>
  );
}

function UserMsg({ m }: { m: Message }) {
  return (
    <motion.div className="cc-user" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}>
      <div className="cc-user-row">
        <CopyButton text={m.content} label="Copy question" />
        <div className="cc-user-bubble selectable">{m.content}</div>
      </div>
      <div className="cc-meta">{fmtTime(m.createdAt)} · Operator</div>
    </motion.div>
  );
}

function AssistantMsg({ m, focused, onFocus }: { m: Message; focused: boolean; onFocus: () => void }) {
  const { thinking, answer, open } = splitThink(m.content);
  const visibleSteps = (m.steps ?? []).filter((s) => s.status !== 'queued' && s.status !== 'skipped');
  const waiting = m.streaming && !answer && !thinking;
  return (
    <div className={`cc-turn${focused ? ' focused' : ''}`} onClick={onFocus}>
      <div className="cc-divider">
        <span className="label blue">Processing reasoning</span>
      </div>
      <div className="cc-chips">
        <AnimatePresence initial={false}>
          {visibleSteps.map((s) => {
            const st = STEP_STYLE[s.kind];
            const Icon = st.icon;
            return (
              <motion.div
                key={s.kind}
                className="cc-chip-row"
                initial={{ opacity: 0, x: -8 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.2 }}
              >
                <span className={`dot${s.status === 'running' ? ' pulse' : ''}`} style={{ color: st.color }} />
                <div className="cc-chip" style={{ color: s.status === 'error' ? 'var(--red)' : st.color }}>
                  {s.status === 'running' ? <Loader2 size={14} className="spin" /> : <Icon size={14} />}
                  <span>
                    {chipText(s)}
                  </span>
                </div>
              </motion.div>
            );
          })}
        </AnimatePresence>
        <ToolChips m={m} />
      </div>
      {m.agent
        ?.filter((a) => a.call?.status === 'awaiting')
        .map((a) => <ApprovalCard key={a.id} s={a} />)}

      {(answer || thinking || waiting || m.stopped) && (
        <motion.div className="cc-answer" initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }}>
          {thinking && (
            <details className="cc-think" open={open}>
              <summary>
                <Sparkles size={13} /> {open ? 'Thinking…' : 'Thought process'}
              </summary>
              <div className="selectable">{thinking}</div>
            </details>
          )}
          {waiting ? (
            <div className="cc-typing">
              <i />
              <i />
              <i />
            </div>
          ) : (
            answer && <Markdown text={answer + (m.streaming ? ' ▍' : '')} />
          )}
          {m.stopped && !answer && <div className="faint">Stopped before an answer was produced.</div>}

          {!!m.sources?.length && (
            <details className="logblock cc-log">
              <summary className="head">
                <span>retrieval.log — {m.kbName}</span>
                {m.streaming ? <span className="live">LIVE</span> : <span>{m.sources.length} hits</span>}
              </summary>
              {m.sources.map((h, i) => (
                <div key={h.id + i} className="cc-log-line">
                  <span className="log-violet">[{i + 1}]</span> <span className="log-info">{h.file}</span>
                  <span className="faint">
                    :{h.start_line}-{h.end_line}
                  </span>{' '}
                  <span className="faint">{h.name !== h.file ? `— ${h.name}` : ''}</span>
                </div>
              ))}
            </details>
          )}
          <div className="cc-answer-meta">
            {fmtTime(m.createdAt)} · {m.stats?.model ?? 'Andai'}
            {m.stats && ` · ${m.stats.tokens} tok · ${m.stats.tokPerSec.toFixed(1)} tok/s`}
            {m.stats?.promptTokPerSec ? ` · prompt ${Math.round(m.stats.promptTokPerSec)} tok/s` : ''}
            {decisionLine(m)}
            {m.stopped && ' · stopped'}
            <CopyButton text={() => reportFor(m)} label="Copy debug report" />
          </div>
        </motion.div>
      )}
    </div>
  );
}

/** ` · decided 2× in 1.62 s`, when the turn scored any decisions. */
function decisionLine(m: Message): string {
  const t = decisionTiming(m.agent);
  return t ? ` · decided ${t.count}× in ${fmtMs(t.totalMs)}` : '';
}

/** The user question a reply answers. */
function questionFor(m: Message): string {
  const msgs = useChat.getState().messages;
  const i = msgs.findIndex((x) => x.id === m.id);
  for (let j = i - 1; j >= 0; j--) if (msgs[j].role === 'user') return msgs[j].content;
  return '';
}

/** A turn as plain text with the current setup (agent/debugReport.ts), read at click time. */
function reportFor(m: Message): string {
  const e = useEngine.getState();
  const t = useTools.getState();
  const k = useKb.getState();
  const kb = k.kbs.find((x) => x.name === m.kbName) ?? null;
  return debugReport(m, questionFor(m), {
    chatModel: modelById(e.loadedId)?.name ?? null,
    deciderModel: (layaById(e.decider.loadedId) ?? modelById(e.decider.loadedId))?.name ?? null,
    engine: e.info ? `${e.info.backend} · ${e.info.threads} threads${e.info.compat ? ' · compat' : ''}` : null,
    ug: k.ug?.version ?? null,
    agent: { agentMode: t.agentMode, maxSteps: t.maxSteps, minConfidence: t.minConfidence, policies: t.policies },
    kb,
  });
}

function chipText(s: TraceStep) {
  switch (s.kind) {
    case 'analyze':
      return `Analyzing query: ${s.detail.replace(/^Identifying/, 'identifying')}`;
    case 'retrieve':
      return s.status === 'running' ? s.detail : `Searching knowledge base: ${s.detail}`;
    case 'plan':
      return s.status === 'running' ? `Planning: ${s.detail}` : `Plan complete: ${s.detail}`;
    case 'filter':
      return s.status === 'running' ? `Checking relevance: ${s.detail}` : `Relevance check: ${s.detail}`;
    case 'build':
      return `Assembling context: ${s.detail}`;
    case 'generate':
      return s.status === 'running' ? `Generating response: ${s.detail}` : `Response complete: ${s.detail}`;
  }
}

const SUGGESTIONS = [
  { icon: Database, title: 'Brief me on my knowledge base', prompt: 'Give me a structured briefing of the key topics in my knowledge base.' },
  { icon: FileText, title: 'Draft a status update', prompt: 'Draft a concise weekly status update template with sections for wins, risks and next steps.' },
  { icon: Code2, title: 'Explain a concept', prompt: 'Explain how retrieval-augmented generation works, in five bullet points.' },
  { icon: Sparkles, title: 'What can you do offline?', prompt: 'What can you help me with while running fully offline on this machine?' },
];

function EmptyHub() {
  const { loadedId, status, cached, progress, loadingId } = useEngine();
  const kbs = useKb((s) => s.kbs);
  const go = useUi((s) => s.go);
  const name = usePersona((s) => s.agentName);
  const def = MODELS[0];
  const loading = status === 'loading';
  const pct = progress && progress.total ? progress.loaded / progress.total : 0;
  return (
    <div className="cc-empty">
      <motion.div className="cc-orb" initial={{ scale: 0.9, opacity: 0 }} animate={{ scale: 1, opacity: 1 }}>
        <div className="cc-orb-inner">
          <Sparkles size={30} />
        </div>
      </motion.div>
      <h2 className="display">
        {name} <span className="grad-text">standing by</span>
      </h2>
      <p className="muted">
        A private agent running entirely on this machine — local inference with wllama, grounded in your own
        knowledge graphs built by ug.
      </p>

      {!loadedId ? (
        <div className="panel cc-setup">
          <div className="label amber">Step 1 · Neural core</div>
          <div className="cc-setup-row">
            <div>
              <div style={{ fontWeight: 600 }}>{loading ? `Loading ${modelById(loadingId)?.name}` : `Load ${def.name}`}</div>
              <div className="faint" style={{ fontSize: 12.5 }}>
                {loading
                  ? progress?.phase
                  : cached[def.url]
                    ? 'Cached on this computer — loads in about a second.'
                    : `${def.size} one-time download from Hugging Face, then fully offline.`}
              </div>
            </div>
            <button className="btn primary" disabled={loading} onClick={() => void loadModel(def.id)}>
              {loading ? <Loader2 size={14} className="spin" /> : <Sparkles size={14} />}
              {loading ? `${Math.round(pct * 100)}%` : cached[def.url] ? 'Load' : 'Download'}
            </button>
          </div>
          {loading && <Bar value={pct} indeterminate={!pct} />}
          <button className="cc-link" onClick={() => go('settings')}>
            Choose another model →
          </button>
        </div>
      ) : !kbs.length ? (
        <div className="panel cc-setup">
          <div className="label violet">Step 2 · Knowledge (optional)</div>
          <div className="cc-setup-row">
            <div>
              <div style={{ fontWeight: 600 }}>Ground answers in your documents</div>
              <div className="faint" style={{ fontSize: 12.5 }}>
                Drop PDFs, Markdown, text or code — ug turns them into a searchable knowledge graph.
              </div>
            </div>
            <button className="btn secondary" onClick={() => go('knowledge')}>
              <Database size={14} /> Build
            </button>
          </div>
        </div>
      ) : null}

      <div className="cc-suggest">
        {SUGGESTIONS.map((s) => (
          <button
            key={s.title}
            className="cc-suggest-card"
            disabled={!loadedId}
            onClick={() => void runTurn(s.prompt)}
          >
            <s.icon size={17} color="var(--blue)" />
            <span>{s.title}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

// ── composer ─────────────────────────────────────────────────────────────

function Composer() {
  const [draft, setDraft] = useState('');
  const [menu, setMenu] = useState<'tools' | 'kb' | null>(null);
  const busy = useChat((s) => s.messages.some((m) => m.streaming));
  const loaded = useEngine((s) => s.loadedId);
  const name = usePersona((s) => s.agentName);
  const { kbs, grounding } = useKb();
  const kb = kbs.find((k) => k.slug === grounding);
  const policies = useTools((s) => s.policies);
  const agentMode = useTools((s) => s.agentMode);
  const go = useUi((s) => s.go);
  const kbTools = available(kb?.kind ?? 'document', policies);
  const ta = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    composerRef.current = ta.current;
  }, []);

  useLayoutEffect(() => {
    const el = ta.current;
    if (!el) return;
    el.style.height = '0px';
    el.style.height = `${Math.min(180, Math.max(26, el.scrollHeight))}px`;
  }, [draft]);

  const send = () => {
    if (!draft.trim() || busy) return;
    void runTurn(draft);
    setDraft('');
  };

  const addFile = async () => {
    if (!inTauri) return toast({ tone: 'warn', title: 'Desktop app required', body: 'Files are indexed by ug natively.' });
    const paths = await pickFiles('Add to knowledge base');
    if (!paths.length) return;
    let target = kb;
    if (!target) {
      target = kbs[0] ?? (await createKb('Quick Drop')) ?? undefined;
      if (!target) return;
      useKb.setState({ grounding: target.slug });
    }
    toast({ tone: 'info', title: `Indexing ${paths.length} file${paths.length > 1 ? 's' : ''}`, body: `into “${target.name}”` });
    await addFiles(target.slug, paths);
  };

  const groundingLabel = useMemo(() => (kb ? kb.name : 'No knowledge'), [kb]);

  return (
    <div className="cc-composer panel">
      <div className="cc-tools">
        <button className="btn ghost sm" onClick={() => void addFile()}>
          <Plus size={13} /> Add file
        </button>
        <div style={{ position: 'relative' }}>
          <button className="btn ghost sm" onClick={() => setMenu(menu === 'tools' ? null : 'tools')}>
            <Wrench size={13} /> Choose tool
          </button>
          {menu === 'tools' && (
            <Menu onClose={() => setMenu(null)}>
              <div className="menu-item sub" aria-hidden>
                {agentMode
                  ? kb
                    ? `The agent picks from these tools for “${kb.name}” (${kb.kind})`
                    : 'Select a knowledge base to give the agent its tools'
                  : 'Agent mode is off: each question runs one knowledge search'}
              </div>
              {kbTools.map((t) => (
                <button
                  key={t.id}
                  className="menu-item"
                  onClick={() => {
                    setMenu(null);
                    go('tools');
                  }}
                >
                  <Wrench size={16} color="var(--violet)" />
                  <div style={{ flex: 1 }}>
                    <div>{t.title}</div>
                    <div className="sub">{t.option}</div>
                  </div>
                  <span className="pill violet">{policyOf(t, policies) === 'ask' ? 'Ask' : 'Auto'}</span>
                </button>
              ))}
              <button
                className="menu-item"
                onClick={() => {
                  setMenu(null);
                  go('tools');
                }}
              >
                <Database size={16} color="var(--blue)" />
                <div style={{ flex: 1 }}>Manage tools…</div>
              </button>
            </Menu>
          )}
        </div>
        <div style={{ position: 'relative', marginLeft: 'auto' }}>
          <button
            className={`cc-ground${kb ? ' on' : ''}`}
            onClick={() => setMenu(menu === 'kb' ? null : 'kb')}
            title="Knowledge base used to ground answers"
          >
            <Database size={13} />
            {groundingLabel}
            {kb && kb.status === 'indexing' && <Loader2 size={12} className="spin" />}
            <ChevronDown size={13} />
          </button>
          {menu === 'kb' && (
            <Menu onClose={() => setMenu(null)} align="right">
              <button
                className="menu-item"
                aria-selected={!grounding}
                onClick={() => {
                  useKb.setState({ grounding: null });
                  setMenu(null);
                }}
              >
                <X size={15} />
                <div style={{ flex: 1 }}>No knowledge base</div>
                {!grounding && <Check size={14} color="var(--blue)" />}
              </button>
              {kbs.map((k) => (
                <button
                  key={k.slug}
                  className="menu-item"
                  aria-selected={k.slug === grounding}
                  onClick={() => {
                    useKb.setState({ grounding: k.slug });
                    setMenu(null);
                  }}
                >
                  <Database size={15} color="var(--violet)" />
                  <div style={{ flex: 1 }}>
                    <div>{k.name}</div>
                    <div className="sub">
                      {k.sources.length} sources · {k.nodes.toLocaleString()} nodes · {k.status}
                    </div>
                  </div>
                  {k.slug === grounding && <Check size={14} color="var(--blue)" />}
                </button>
              ))}
              {!kbs.length && <div className="menu-item sub">No knowledge bases yet — create one in Knowledge.</div>}
            </Menu>
          )}
        </div>
      </div>
      <div className="cc-input">
        <textarea
          ref={ta}
          value={draft}
          rows={1}
          placeholder={loaded ? `Direct transmission to ${name}…` : 'Load a model to begin transmission…'}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault();
              send();
            } else if (e.key === 'Escape' && busy) stopTurn();
          }}
        />
        {busy ? (
          <button className="cc-send stop" onClick={stopTurn} title="Stop (Esc)">
            <Square size={16} fill="currentColor" />
          </button>
        ) : (
          <button className="cc-send" onClick={send} disabled={!draft.trim()} title="Send (Enter)">
            <Send size={19} />
          </button>
        )}
      </div>
    </div>
  );
}

function Menu({ children, onClose, align }: { children: React.ReactNode; onClose: () => void; align?: 'right' }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose();
    const onDown = (e: MouseEvent) => !(e.target as HTMLElement).closest('.menu') && onClose();
    window.addEventListener('keydown', onKey);
    setTimeout(() => window.addEventListener('mousedown', onDown), 0);
    return () => {
      window.removeEventListener('keydown', onKey);
      window.removeEventListener('mousedown', onDown);
    };
  }, [onClose]);
  return (
    <motion.div
      className="menu"
      style={{ bottom: 38, ...(align === 'right' ? { right: 0 } : { left: 0 }) }}
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.14 }}
    >
      {children}
    </motion.div>
  );
}

// ── execution trace ──────────────────────────────────────────────────────

const STATUS_PILL: Record<TraceStep['status'], [string, string]> = {
  done: ['Completed', ''],
  running: ['Processing', 'blue'],
  queued: ['Queued', ''],
  skipped: ['Skipped', ''],
  error: ['Failed', 'red'],
};

function TracePanel({ msg }: { msg?: Message }) {
  const engine = useEngine();
  const maxTokens = usePersona((s) => s.maxTokens);
  const steps = msg?.steps ?? [];
  const done = steps.filter((s) => s.status === 'done' || s.status === 'skipped').length;
  const pct = steps.length ? Math.round((done / steps.length) * 100) : 0;
  const active = !!msg?.streaming;
  const title = useChat((s) => {
    if (!msg) return '';
    const idx = s.messages.findIndex((m) => m.id === msg.id);
    for (let i = idx - 1; i >= 0; i--) if (s.messages[i].role === 'user') return s.messages[i].content;
    return '';
  });
  const genTokens = Number(steps.find((s) => s.kind === 'generate')?.detail.match(/^(\d+) tokens/)?.[1] ?? 0);
  const stats = msg?.stats;
  const ctxPct = stats?.promptTokens != null ? ((stats.promptTokens + stats.tokens) / stats.nCtx) * 100 : null;

  return (
    <>
      <div className="cc-head">
        <Network size={22} color="var(--blue)" />
        <h1 className="display">Execution Trace</h1>
        {msg && <CopyButton text={() => reportFor(msg)} label="Copy debug report" size={14} />}
        <span className="label cc-session" style={{ color: active ? 'var(--blue)' : undefined }}>
          <span className={`dot${active ? ' pulse' : ''}`} style={{ display: 'inline-block', marginRight: 8, color: active ? 'var(--blue)' : 'var(--text-4)' }} />
          {active ? 'Active task' : msg ? 'Last task' : 'Idle'}
        </span>
      </div>
      <div className="panel trace">
        {!msg ? (
          <div className="empty" style={{ height: '100%' }}>
            <CircleDashed size={30} />
            <div>No task yet</div>
            <div style={{ fontSize: 12.5, maxWidth: 240 }}>
              Each transmission is traced here — every decision and tool call, context assembly and generation, step by step.
            </div>
          </div>
        ) : (
          <div className="trace-rail">
            <TraceNode state={active ? 'active' : 'done'}>
              <div className="trace-card root">
                <div className="trace-top">
                  <span className="label blue">Root task</span>
                  <span className="trace-root-side">
                    <span className="pill">{pct}%</span>
                    <CopyTraceButton m={msg} question={title} />
                  </span>
                </div>
                <div className="trace-title">{title.length > 70 ? `${title.slice(0, 70)}…` : title || 'Transmission'}</div>
              </div>
            </TraceNode>
            {steps.map((s, i) => (
              <TraceNode key={s.kind} state={s.status === 'running' ? 'active' : s.status === 'queued' ? 'queued' : s.status === 'skipped' ? 'queued' : 'done'}>
                <div className={`trace-card${s.status === 'running' ? ' running' : ''}${s.status === 'queued' || s.status === 'skipped' ? ' dim' : ''}`}>
                  <div className="trace-top">
                    <span className="label blue">Step {String(i + 1).padStart(2, '0')}</span>
                    <span className={`pill ${STATUS_PILL[s.status][1]}`}>{STATUS_PILL[s.status][0]}</span>
                  </div>
                  <div className="trace-title">{s.title}</div>
                  <div className="trace-detail">
                    {s.detail}
                    {s.ms != null && s.status === 'done' && <span className="faint"> · {(s.ms / 1000).toFixed(2)}s</span>}
                  </div>
                  {s.kind === 'generate' && s.status === 'running' && (
                    <div style={{ marginTop: 12 }}>
                      <Bar value={genTokens / maxTokens} indeterminate={!genTokens} />
                    </div>
                  )}
                </div>
                {s.kind === 'plan' && !!msg.agent?.length && (
                  <div className="trace-subs ag-subs">
                    <DecisionSummary agent={msg.agent} />
                    {msg.agent.map((a) => (
                      <AgentStepCard key={a.id} s={a} />
                    ))}
                  </div>
                )}
                {s.kind === 'filter' && msg.relevance && <RelevanceList r={msg.relevance} />}
                {(s.kind === 'retrieve' || s.kind === 'plan') && !!msg.sources?.length && (
                  <div className="trace-subs">
                    {msg.sources.slice(0, 5).map((h, j) => (
                      <div key={h.id + j} className="trace-sub">
                        <FileText size={13} color="var(--text-3)" />
                        <span className="ellipsis">
                          [{j + 1}] {h.file}
                          <span className="faint">
                            :{h.start_line}-{h.end_line}
                          </span>
                        </span>
                        <Check size={14} color="var(--violet)" />
                      </div>
                    ))}
                    {msg.sources.length > 5 && <div className="trace-sub faint">+{msg.sources.length - 5} more</div>}
                  </div>
                )}
              </TraceNode>
            ))}
          </div>
        )}
      </div>
      <div className="trace-stats">
        <Stat
          label="Throughput"
          value={(active ? engine.tokPerSec : stats?.tokPerSec)?.toFixed(1) ?? '—'}
          unit="TOK/S"
          color="var(--blue)"
        />
        <Stat
          label="Context pool"
          value={ctxPct != null ? `${Math.max(1, Math.round(ctxPct))}%` : '—'}
          unit="UTIL"
          color="var(--violet)"
        />
      </div>
      {msg && steps.some((s) => s.status === 'error') && (
        <button className="btn ghost sm" style={{ marginTop: 10 }} onClick={() => useUi.getState().go('settings')}>
          <Cpu size={12} /> Check engine settings
        </button>
      )}
    </>
  );
}

function TraceNode({ state, children }: { state: 'done' | 'active' | 'queued'; children: React.ReactNode }) {
  return (
    <div className={`trace-node ${state}`}>
      <span className="trace-pin" />
      <div className="trace-body">{children}</div>
    </div>
  );
}
