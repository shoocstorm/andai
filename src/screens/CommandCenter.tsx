import {
  BarChart3,
  Check,
  ChevronDown,
  CircleDashed,
  Code2,
  Database,
  Download,
  FileText,
  FolderGit2,
  HelpCircle,
  ListFilter,
  ShieldCheck,
  Loader2,
  MessageSquareText,
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
  TriangleAlert,
  Cpu,
  Wrench,
  X,
} from 'lucide-react';
import { AnimatePresence, motion } from 'motion/react';
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { available, policyOf } from '../agent/tools/registry';
import { runTurn, stopTurn } from '../agent/turn';
import { Bar, CopyButton, Markdown, Pager, Stat, fmtTime, usePaged } from '../components/ui';
import { NeuralCore } from '../components/NeuralCore';
import { debugReport } from '../agent/debugReport';
import { inTauri } from '../kb/api';
import { sampleByProject } from '../kb/samples';
import { loadModel, useEngine } from '../llm/engine';
import { layaById, modelById, recommendedModel } from '../llm/models';
import { clearChat, splitThink, useChat, type Message, type TraceStep } from '../state/chat';
import { addFiles, addSample, createKb, useKb } from '../state/kb';
import { useLayout } from '../state/layout';
import { usePersona } from '../state/persona';
import { useTools } from '../state/tools';
import { toast, useUi } from '../state/ui';
import { openUgSetup } from '../state/ugInstall';
import { AgentStepCard, ApprovalCard, callChipText, callLive, ContextDialog, traceCallId, CopyTraceButton, DecisionSummary, MatchBadge, RelevanceList, SourceDialog, SupportList, SupportNote, ToolChips, decisionTiming, fmtMs } from './AgentTrace';
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
  verify: { color: 'var(--violet)', icon: ShieldCheck },
};

export function CommandCenter() {
  const messages = useChat((s) => s.messages);
  const [focusId, setFocusId] = useState<string | null>(null);
  const lastAssistant = [...messages].reverse().find((m) => m.role === 'assistant');
  const focused = messages.find((m) => m.id === focusId && m.role === 'assistant') ?? lastAssistant;

  const traceOpen = useLayout((s) => s.traceOpen);

  return (
    <div className={`screen cc${traceOpen ? '' : ' no-trace'}`}>
      <AnimatePresence>
        {!messages.length && (
          <motion.div className="cc-ambient" aria-hidden="true" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.6 }}>
            <i className="cc-aurora a1" />
            <i className="cc-aurora a2" />
            <i className="cc-aurora a3" />
            <i className="cc-grid" />
          </motion.div>
        )}
      </AnimatePresence>
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
        // Hidden trace, agent working: the toggle breathes so the work stays one click away.
        className={`icon-btn cc-trace-toggle${traceOpen ? ' on' : busy ? ' breathing' : ''}`}
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
  // The source whose dialog is open, from a citation or the passages list.
  const [source, setSource] = useState<number | null>(null);
  const nSources = m.sources?.length ?? 0;
  return (
    <div className={`cc-turn${focused ? ' focused' : ''}`} onClick={onFocus}>
      <ReasoningStrip m={m} steps={visibleSteps} />
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
            answer && <Markdown text={answer + (m.streaming ? ' ▍' : '')} sources={nSources} onCite={setSource} />
          )}
          {m.stopped && !answer && <div className="faint">Stopped before an answer was produced.</div>}

          {!!m.sources?.length && (
            <details className="logblock cc-log">
              <summary className="head">
                <span>Passages found{m.kbName ? ` in “${m.kbName}”` : ''}</span>
                {m.streaming ? <span className="live">LIVE</span> : <span>{m.sources.length} passage{m.sources.length === 1 ? '' : 's'}</span>}
              </summary>
              {m.sources.map((h, i) => (
                <button key={h.id + i} type="button" className="cc-log-line linklike" aria-haspopup="dialog" aria-label={`Source ${i + 1}: ${h.file}:${h.start_line}-${h.end_line}`} onClick={() => setSource(i + 1)}>
                  <span className="cc-log-main">
                    <span className="log-violet">[{i + 1}]</span> <span className="log-info">{h.file}</span>
                    <span className="faint">
                      :{h.start_line}-{h.end_line}
                    </span>{' '}
                    <span className="faint">{h.name !== h.file ? `— ${h.name}` : ''}</span>
                  </span>
                  <MatchBadge h={h} peers={m.sources!} />
                </button>
              ))}
            </details>
          )}
          {m.support && <SupportNote r={m.support} sources={m.sources ?? []} />}
          {source != null && <SourceDialog m={m} n={source} onClose={() => setSource(null)} />}
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

/** The "Assemble context" card's button to what the chat model was sent; the dialog is built only while open. */
function ContextButton({ m }: { m: Message }) {
  const [open, setOpen] = useState(false);
  const model = m.stats?.model ?? m.steps?.find((s) => s.kind === 'generate')?.title.replace(/^Generate · /, '') ?? 'The chat model';
  return (
    <>
      <div className="ag-step-buttons">
        <button className="btn ghost sm ag-toggle" aria-haspopup="dialog" onClick={() => setOpen(true)}>
          <HelpCircle size={12} /> <span className="ellipsis">What went in?</span>
        </button>
      </div>
      {open && <ContextDialog c={m.context!} model={model} onClose={() => setOpen(false)} />}
    </>
  );
}

/** The DOM id of a step's card in the Execution Trace, so a chip can point at it. */
const traceStepId = (msgId: string, kind: TraceStep['kind']) => `trace-${msgId}-${kind}`;

/** A step's number as the trace shows it: its place among all the turn's steps, skipped ones included. */
const stepNo = (m: Message, s: TraceStep) => (m.steps ?? []).indexOf(s) + 1;
const pad2 = (n: number) => String(n).padStart(2, '0');

/**
 * Scrolls the Execution Trace to a card and flashes it, after the click has
 * focused the turn (two frames: the trace re-renders for it first). Does
 * nothing when the trace is hidden.
 */
export function flashTrace(domId: string) {
  const run = () => {
    const el = document.getElementById(domId);
    if (!el) return;
    const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    el.scrollIntoView?.({ block: 'center', behavior: reduce ? 'auto' : 'smooth' });
    el.classList.remove('flash');
    void el.offsetWidth; // restart the animation on a second click
    el.classList.add('flash');
    window.setTimeout(() => el.classList.remove('flash'), 1600);
  };
  requestAnimationFrame(() => requestAnimationFrame(run));
}

/** Every step as a chip, one per line, and the tool calls: the full "how it got here". Each points at its card in the trace. */
function StepChips({ m, steps }: { m: Message; steps: TraceStep[] }) {
  const traceOpen = useLayout((st) => st.traceOpen);
  const hint = traceOpen ? undefined : `Open the Execution Trace (${shortcut('J')}) to see this step`;
  const plan = steps.find((s) => s.kind === 'plan');
  return (
    <div className="cc-chips">
      <AnimatePresence initial={false}>
        {steps.map((s) => {
          const st = STEP_STYLE[s.kind];
          const Icon = st.icon;
          const no = pad2(stepNo(m, s));
          return (
            <motion.div key={s.kind} className="cc-chip-row" initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.2 }}>
              <span className={`dot${s.status === 'running' ? ' pulse' : ''}`} style={{ color: st.color }} />
              <button
                type="button"
                className="cc-chip"
                style={{ color: s.status === 'error' ? 'var(--red)' : st.color }}
                title={hint}
                aria-label={`Show step ${no} in the trace: ${chipText(s)}`}
                onClick={() => flashTrace(traceStepId(m.id, s.kind))}
              >
                <span className="step-no">{no}</span>
                {s.status === 'running' ? <Loader2 size={14} className="spin" /> : <Icon size={14} />}
                <span>{chipText(s)}</span>
              </button>
              {/* the tool calls belong to the plan step: listed right under it, not after the answer's steps */}
              {s.kind === 'plan' && (
                <div className="cc-chip-subs">
                  <ToolChips m={m} planNo={stepNo(m, s)} onPick={flashTrace} />
                </div>
              )}
            </motion.div>
          );
        })}
      </AnimatePresence>
      {!plan && <ToolChips m={m} onPick={flashTrace} />}
    </div>
  );
}

/** One line for a finished turn's reasoning: how long, what it used, and anything worth a look. Pure. */
export function reasoningSummary(m: Message): { text: string; warn: string | null } {
  const calls = (m.agent ?? []).filter((a) => a.call).length;
  const passages = m.sources?.length ?? 0;
  const secs = m.stats?.totalMs != null ? m.stats.totalMs / 1000 : null;
  const failed = (m.steps ?? []).filter((s) => s.status === 'error').length;
  const flagged = new Set((m.support?.items ?? []).filter((x) => x.flagged).map((x) => x.sentence)).size;
  const parts = [
    m.stopped ? 'Stopped' : secs != null ? `Reasoned in ${secs < 10 ? secs.toFixed(1) : Math.round(secs)} s` : 'Reasoning',
    calls ? `${calls} tool call${calls === 1 ? '' : 's'}` : null,
    passages ? `${passages} passage${passages === 1 ? '' : 's'}` : m.kbName ? 'no passages' : null,
  ].filter(Boolean);
  const warn = failed ? `${failed} step${failed === 1 ? '' : 's'} failed` : flagged ? `${flagged} claim${flagged === 1 ? '' : 's'} to check` : null;
  return { text: parts.join(' · '), warn };
}

/**
 * The reasoning above an answer, at the height it deserves. While the turn
 * runs: one live line, a dot per step and what the current one is doing (a
 * running tool call, if any). Once it's done: folded to one line of step
 * icons and a summary, which opens to every chip. Details live in the
 * Execution Trace; this is only the gist.
 */
function ReasoningStrip({ m, steps }: { m: Message; steps: TraceStep[] }) {
  if (!steps.length) return null;
  const calls = (m.agent ?? []).flatMap((a) => (a.call ? [a.call] : []));
  const busy = m.streaming || steps.some((s) => s.status === 'running');
  if (busy) {
    const running = [...steps].reverse().find((s) => s.status === 'running') ?? steps[steps.length - 1];
    const liveCall = [...calls].reverse().find(callLive);
    const st = STEP_STYLE[running.kind];
    return (
      <div className="cc-live">
        <span className="cc-live-dots" aria-hidden="true">
          {steps.map((s) => (
            <i key={s.kind} className={s.status} style={{ color: s.status === 'error' ? 'var(--red)' : STEP_STYLE[s.kind].color }} />
          ))}
        </span>
        <span className="step-no" style={{ color: st.color }}>
          {pad2(stepNo(m, running))}
        </span>
        <Loader2 size={13} className="spin" style={{ color: st.color }} />
        <span className="cc-live-text ellipsis" style={{ color: st.color }}>
          {running.kind === 'plan' && liveCall ? callChipText(liveCall) : chipText(running)}
        </span>
      </div>
    );
  }
  const { text, warn } = reasoningSummary(m);
  return (
    <details className="cc-reasoning">
      <summary>
        <span className="cc-reasoning-icons" aria-hidden="true">
          {steps.map((s) => {
            const Icon = STEP_STYLE[s.kind].icon;
            return <Icon key={s.kind} size={12} style={{ color: s.status === 'error' ? 'var(--red)' : STEP_STYLE[s.kind].color }} />;
          })}
        </span>
        <span className="cc-reasoning-text">{text}</span>
        {warn && <span className="cc-reasoning-warn">· {warn}</span>}
        <ChevronDown size={13} className="cc-reasoning-chev" aria-hidden="true" />
      </summary>
      <StepChips m={m} steps={steps} />
    </details>
  );
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
    case 'verify':
      return s.status === 'running' ? `Checking claims: ${s.detail}` : `Claim check: ${s.detail}`;
  }
}

const SUGGESTIONS = [
  { icon: Database, title: 'Brief me on my knowledge base', prompt: 'Give me a structured briefing of the key topics in my knowledge base.' },
  { icon: FileText, title: 'Draft a status update', prompt: 'Draft a concise weekly status update template with sections for wins, risks and next steps.' },
  { icon: Code2, title: 'Explain a concept', prompt: 'Explain how retrieval-augmented generation works, in five bullet points.' },
  { icon: Sparkles, title: 'What can you do offline?', prompt: 'What can you help me with while running fully offline on this machine?' },
];

function EmptyHub() {
  const { loadedId, status, cached, progress, loadingId, native } = useEngine();
  const kbs = useKb((s) => s.kbs);
  const grounding = useKb((s) => s.grounding);
  // Grounded in a bundled sample: suggest questions it can answer.
  const sample = sampleByProject(grounding);
  const [addingSample, setAddingSample] = useState(false);
  const go = useUi((s) => s.go);
  const name = usePersona((s) => s.agentName);
  // On Apple Silicon, the MLX model: native on the GPU, several times faster.
  const def = recommendedModel(native.supported);
  const loading = status === 'loading';
  const pct = progress && progress.total ? progress.loaded / progress.total : 0;
  return (
    <div className="cc-empty">
      <motion.div className="cc-core" initial={{ scale: 0.85, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ duration: 0.6, ease: [0.2, 0.7, 0.2, 1] }}>
        <NeuralCore />
      </motion.div>
      <h2 className="display">
        {name} <span className="grad-text">standing by</span>
      </h2>
      <p className="muted">
        A private agent running entirely on this machine — local inference{native.supported ? ' on your Mac’s GPU with MLX' : ' with wllama'},
        grounded in your own knowledge graphs built by ug.
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
                Drop PDFs, Markdown, text or code — ug turns them into a searchable knowledge graph. Or try a sample: a
                fictional ferry operator’s handbook and policies.
              </div>
            </div>
            <div className="cc-setup-actions">
              {inTauri && (
                <button
                  className="btn primary"
                  disabled={addingSample}
                  onClick={() => {
                    setAddingSample(true);
                    void addSample('tidewater-docs').finally(() => setAddingSample(false));
                  }}
                >
                  {addingSample ? <Loader2 size={14} className="spin" /> : <Sparkles size={14} />}
                  {addingSample ? 'Indexing…' : 'Try a sample'}
                </button>
              )}
              <button className="btn secondary" onClick={() => go('knowledge')}>
                <Database size={14} /> Build
              </button>
            </div>
          </div>
        </div>
      ) : null}

      {sample && (
        <div className="label" style={{ marginTop: 6 }}>
          Try asking {sample.name}
        </div>
      )}
      <div className="cc-suggest">
        {(sample ? sample.questions.map((q) => ({ icon: MessageSquareText, title: q, prompt: q })) : SUGGESTIONS).map((s) => (
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
  const { kbs, grounding, ug } = useKb();
  const kb = kbs.find((k) => k.slug === grounding);
  const kbPage = usePaged(kbs, 6, (k) => k.slug === grounding);
  const policies = useTools((s) => s.policies);
  const agentMode = useTools((s) => s.agentMode);
  const go = useUi((s) => s.go);
  const kbTools = available(kb?.kind ?? 'document', policies);
  const ta = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    composerRef.current = ta.current;
  }, []);

  // A question handed over from another screen (Insights' Ask) becomes the draft, for the user to edit or send.
  const prefill = useUi((s) => s.prefill);
  useEffect(() => {
    if (prefill === null) return;
    setDraft(prefill);
    useUi.setState({ prefill: null });
    setTimeout(() => ta.current?.focus(), 30);
  }, [prefill]);

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
    // Files go into one of Andai's own knowledge bases; a ug project indexes the user's folder as is.
    let target = kb?.managed ? kb : undefined;
    if (!target) {
      target = kbs.find((k) => k.managed) ?? (await createKb('Quick Drop')) ?? undefined;
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
            {kb?.status === 'offline' ? <TriangleAlert size={13} color="var(--amber)" /> : <Database size={13} />}
            {groundingLabel}
            {kb?.status === 'offline' && <span className="pill amber">needs ug</span>}
            {kb && kb.status === 'indexing' && <Loader2 size={12} className="spin" />}
            <ChevronDown size={13} />
          </button>
          {menu === 'kb' && (
            <Menu onClose={() => setMenu(null)} align="right">
              {ug && !ug.found && inTauri && (
                <button
                  className="menu-item"
                  onClick={() => {
                    setMenu(null);
                    openUgSetup();
                  }}
                >
                  <Download size={15} color="var(--amber)" />
                  <div style={{ flex: 1 }}>
                    <div>Install UltraGraph</div>
                    <div className="sub">Knowledge bases need ug to be searched</div>
                  </div>
                </button>
              )}
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
              {kbPage.items.map((k) => (
                <button
                  key={k.slug}
                  className="menu-item"
                  aria-selected={k.slug === grounding}
                  onClick={() => {
                    useKb.setState({ grounding: k.slug });
                    setMenu(null);
                  }}
                >
                  {k.managed ? <Database size={15} color="var(--violet)" /> : <FolderGit2 size={15} color="var(--violet)" />}
                  <div style={{ flex: 1 }}>
                    <div>{k.name}</div>
                    <div className="sub">
                      {k.sourceCount} sources · {k.nodes.toLocaleString()} nodes · {k.status === 'offline' ? 'needs ug' : k.status}
                      {!k.managed && ' · ug project'}
                    </div>
                  </div>
                  {k.slug === grounding && <Check size={14} color="var(--blue)" />}
                </button>
              ))}
              {!kbs.length && <div className="menu-item sub">No knowledge bases yet — create one in Knowledge.</div>}
              <Pager paged={kbPage} label="Knowledge bases" className="menu-pager" />
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

const IDLE_STAGES = ['Decide', 'Tool', 'Observe', 'Answer'];

/** The empty trace: a scanner at rest and the loop's stages, waiting for a task. */
function TraceIdle() {
  return (
    <div className="trace-idle">
      <div className="ti-scope" aria-hidden="true">
        <i className="ti-ring r1" />
        <i className="ti-ring r2" />
        <i className="ti-ring r3" />
        <i className="ti-sweep" />
        <span className="ti-core">
          <CircleDashed size={22} />
        </span>
      </div>
      <div className="label ti-label">Awaiting transmission</div>
      <div className="ti-title">No task yet</div>
      <div className="ti-text">
        Each transmission is traced here — every decision and tool call, context assembly and generation, step by step.
      </div>
      <ol className="ti-stages" aria-hidden="true">
        {IDLE_STAGES.map((s, i) => (
          <li key={s} style={{ animationDelay: `${i * 0.6}s` }}>
            <i />
            {s}
          </li>
        ))}
      </ol>
    </div>
  );
}

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
          <TraceIdle />
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
            {steps.map((s, i) => {
              const Icon = STEP_STYLE[s.kind].icon;
              return (
                <TraceNode key={s.kind} state={s.status === 'running' ? 'active' : s.status === 'queued' ? 'queued' : s.status === 'skipped' ? 'queued' : 'done'}>
                  <div id={traceStepId(msg.id, s.kind)} className={`trace-card${s.status === 'running' ? ' running' : ''}${s.status === 'queued' || s.status === 'skipped' ? ' dim' : ''}`}>
                    <div className="trace-top">
                      <span className="label blue">Step {pad2(i + 1)}</span>
                      <span className={`pill ${STATUS_PILL[s.status][1]}`}>{STATUS_PILL[s.status][0]}</span>
                    </div>
                    <div className="trace-title trace-step-title">
                      {/* the same icon as the step's chip above the answer, so the two can be matched at a glance */}
                      <Icon size={15} className="trace-step-icon" style={{ color: s.status === 'error' ? 'var(--red)' : STEP_STYLE[s.kind].color }} aria-hidden="true" />
                      {s.title}
                    </div>
                    <div className="trace-detail">
                      {s.detail}
                      {s.ms != null && s.status === 'done' && <span className="faint"> · {(s.ms / 1000).toFixed(2)}s</span>}
                    </div>
                    {s.kind === 'build' && msg.context && <ContextButton m={msg} />}
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
                        <AgentStepCard key={a.id} s={a} domId={traceCallId(msg.id, a.id)} />
                      ))}
                    </div>
                  )}
                  {s.kind === 'filter' && msg.relevance && <RelevanceList r={msg.relevance} sources={msg.sources ?? []} question={questionFor(msg)} />}
                  {s.kind === 'verify' && msg.support && <SupportList r={msg.support} sources={msg.sources ?? []} />}
                  {/* In agent mode the tool calls and the relevance check already list the passages. */}
                  {s.kind === 'retrieve' && !!msg.sources?.length && (
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
                          {/* the narrow trace keeps room for the file name: the tag only, its score in the tooltip */}
                          <MatchBadge h={h} peers={msg.sources!} compact />
                          <Check size={14} color="var(--violet)" />
                        </div>
                      ))}
                      {msg.sources.length > 5 && <div className="trace-sub faint">+{msg.sources.length - 5} more</div>}
                    </div>
                  )}
                </TraceNode>
              );
            })}
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
