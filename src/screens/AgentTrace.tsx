// How agent mode shows its work: every decision with each option's
// probability, and every tool call with its arguments, approval, the ug
// command that ran, timing and output. Tool output and arguments come from
// the model and the user's files, so they render as plain text only.

import { Check, ChevronRight, Copy, Hand, ShieldQuestion, Wrench, X } from 'lucide-react';
import { useState } from 'react';
import { toolById } from '../agent/tools/registry';
import { CopyButton, fmtTime } from '../components/ui';
import { shellCommand } from '../agent/debugReport';
import type { DecisionIO } from '../llm/decide';
import type { AgentStep, CallStatus, DecisionRecord, Message, ToolCallRecord } from '../state/chat';
import { resolveApproval } from '../state/tools';
import { toast } from '../state/ui';

export const ACTION_LABEL: Record<string, string> = {
  answer_now: 'Answer now',
  ask_clarification: 'Ask to clarify',
};

export const actionLabel = (id: string) => ACTION_LABEL[id] ?? toolById(id)?.title ?? id;

const CALL_PILL: Record<CallStatus, [string, string]> = {
  filling: ['Writing args', 'blue'],
  awaiting: ['Needs approval', 'amber'],
  running: ['Running', 'blue'],
  done: ['Done', 'green'],
  error: ['Failed', 'red'],
  denied: ['Denied', 'amber'],
  skipped: ['Skipped', ''],
};

/** `{"query":"x"}` → `query: "x"`, short enough for a chip. */
export function argsInline(args: Record<string, unknown> | null, max = 60): string {
  if (!args) return '';
  const text = Object.entries(args)
    .map(([k, v]) => `${k}: ${JSON.stringify(v)}`)
    .join(', ');
  return text.length > max ? `${text.slice(0, max)}…` : text;
}

const pct = (p: number) => `${(p * 100).toFixed(p < 0.1 ? 1 : 0)}%`;

/** `812 ms`, or `1.62 s` from a second on. */
export const fmtMs = (ms: number) => (ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(2)} s`);

/** Every decision call in a turn, scored or failed: how many, and the time they took. */
export function decisionTiming(agent: AgentStep[] | undefined): { count: number; totalMs: number; avgMs: number; models: string[] } | null {
  const timed = (agent ?? []).flatMap((s) =>
    s.decision ? [{ ms: s.decision.ms, model: s.decision.model }] : s.failedDecision?.io ? [{ ms: s.failedDecision.ms, model: s.failedDecision.model }] : [],
  );
  if (!timed.length) return null;
  const totalMs = timed.reduce((a, t) => a + t.ms, 0);
  const models = [...new Set(timed.flatMap((t) => (t.model ? [t.model] : [])))];
  return { count: timed.length, totalMs, avgMs: totalMs / timed.length, models };
}

/** The turn's decision time, above the steps in the Execution Trace. */
export function DecisionSummary({ agent }: { agent: AgentStep[] | undefined }) {
  const t = decisionTiming(agent);
  if (!t) return null;
  return (
    <div className="ag-summary faint" aria-label="Decision time">
      {t.count} decision{t.count === 1 ? '' : 's'} · {fmtMs(t.totalMs)} total · {fmtMs(t.avgMs)} each on average
      {t.models.length ? ` · ${t.models.join(', ')}` : ''}
    </div>
  );
}

/** The prompt as plain text, one block per message: easy to read and to paste back. */
export const ioPromptText = (io: DecisionIO) => io.request.messages.map((m) => `[${m.role}]\n${m.content}`).join('\n\n');

/** What the decision model was sent and what it returned. Text from the model and the user's files, so plain text only. */
export function DecisionCall({ io, labels }: { io: DecisionIO; labels?: Record<string, string> }) {
  const r = io.response;
  const total = r ? r.topLogprobs.reduce((a, t) => a + Math.exp(t.logprob), 0) : 0;
  return (
    <div className="ag-call ag-io">
      <Field k="Decision call" copy={JSON.stringify(io, null, 2)}>
        <span className="faint ag-small">What the decision model was sent, and its raw reply. Copy it to report an odd decision.</span>
      </Field>
      <Field k="Input: prompt" copy={ioPromptText(io)}>
        {io.request.messages.map((m, i) => (
          <div key={i} className="ag-msg">
            <div className="mono faint ag-msg-role">{m.role}</div>
            <pre className="ag-pre selectable">{m.content}</pre>
          </div>
        ))}
      </Field>
      <Field k="Input: parameters" copy={JSON.stringify(io.request.params)}>
        <pre className="ag-pre selectable">{JSON.stringify(io.request.params, null, 2)}</pre>
      </Field>
      <Field k="Output" copy={r ? JSON.stringify(r, null, 2) : undefined}>
        {!r ? (
          <div className="ag-small ag-err">No reply: the call failed before the model answered.</div>
        ) : (
          <>
            <div className="ag-small">
              Sampled token <span className="mono">{JSON.stringify(r.sampled)}</span> · first token’s top {r.topLogprobs.length} candidates
            </div>
            {r.topLogprobs.length > 0 && (
              <table className="ag-logprobs mono">
                <thead>
                  <tr>
                    <th scope="col">Token</th>
                    <th scope="col">Logprob</th>
                    <th scope="col">Share of listed</th>
                  </tr>
                </thead>
                <tbody>
                  {r.topLogprobs.map((t, i) => (
                    <tr key={i} className={labels?.[t.token.trim()] ? 'option' : undefined}>
                      <td className="selectable">
                        {JSON.stringify(t.token)}
                        {labels?.[t.token.trim()] && <span className="faint"> {labels[t.token.trim()]}</span>}
                      </td>
                      <td>{t.logprob.toFixed(3)}</td>
                      <td>{total ? pct(Math.exp(t.logprob) / total) : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </>
        )}
      </Field>
    </div>
  );
}

export function DecisionBars({ d }: { d: DecisionRecord }) {
  const sorted = [...d.options].sort((a, b) => b.probability - a.probability);
  return (
    <div className="ag-decision">
      <div className="ag-decision-head faint">
        {d.model} · {d.slot === 'decider' ? 'decision model' : 'chat model'} · {d.ms} ms
        {d.promptTokens != null && ` · ${d.promptTokens} tok`}
      </div>
      <ul className="ag-bars" aria-label="Decision options and their probabilities">
        {sorted.map((o) => (
          <li key={o.id} className={o.id === d.chosen ? 'chosen' : undefined} title={o.text}>
            <span className="ag-bar-label">
              <span className="mono faint">{o.label}</span> {actionLabel(o.id)}
            </span>
            <span className="ag-bar-track">
              <i style={{ width: `${Math.max(1, o.probability * 100)}%` }} />
            </span>
            <span className="ag-bar-value mono">
              {d.bounded?.includes(o.id) ? '≤' : ''}
              {pct(o.probability)}
            </span>
          </li>
        ))}
      </ul>
      <div className="faint ag-seed">
        Option order seed {d.seed} · probabilities are uncalibrated scores
        {!!d.bounded?.length && ' · ≤ marks options below the readout, scored at its lowest value'}
      </div>
    </div>
  );
}

function Field({ k, copy, children }: { k: string; copy?: string; children: React.ReactNode }) {
  return (
    <div className="ag-field">
      <div className="label ag-field-head">
        {k}
        {copy != null && <CopyButton text={copy} label={`Copy ${k.toLowerCase()}`} size={11} />}
      </div>
      {children}
    </div>
  );
}

export function CallDetail({ c }: { c: ToolCallRecord }) {
  const [full, setFull] = useState(false);
  const out = c.output ?? '';
  const preview = out.length > 1200 && !full ? `${out.slice(0, 1200)}\n…` : out;
  return (
    <div className="ag-call">
      <Field k="Arguments" copy={c.args ? JSON.stringify(c.args) : undefined}>
        <pre className="ag-pre selectable">{c.args ? JSON.stringify(c.args, null, 2) : '—'}</pre>
        {c.argsRaw && (
          <details className="ag-raw">
            <summary>Model’s raw reply{c.argAttempts > 1 ? ` (${c.argAttempts} attempts)` : ''}</summary>
            <pre className="ag-pre selectable">{c.argsRaw}</pre>
          </details>
        )}
        {c.argModel && <div className="faint ag-small">Written by {c.argModel}</div>}
      </Field>
      <Field k="Policy">
        <span className="ag-small">
          {c.policy === 'ask' ? 'Ask' : c.policy === 'auto' ? 'Auto (read-only)' : 'Off'}
          {c.approval && ` · ${c.approval}`}
        </span>
      </Field>
      {c.argv && (
        <Field k="Command" copy={shellCommand(c.argv)}>
          <pre className="ag-pre selectable">{shellCommand(c.argv)}</pre>
        </Field>
      )}
      <Field k="Timing">
        <span className="ag-small">
          {fmtTime(c.startedAt)}
          {c.ms != null && ` · ${c.ms} ms`}
          {c.outputBytes != null && ` · ${c.outputBytes.toLocaleString()} bytes`}
          {c.truncated && ' · output cut at the size cap'}
        </span>
      </Field>
      {c.error && (
        <Field k={c.status === 'done' ? 'Note' : 'Error'} copy={c.error}>
          <div className="ag-small ag-err selectable">{c.error}</div>
        </Field>
      )}
      {c.observation && (
        <Field k="What the agent saw">
          <div className="ag-small selectable">{c.observation}</div>
        </Field>
      )}
      {out && (
        <Field k="Output" copy={out}>
          <pre className="ag-pre ag-out selectable">{preview}</pre>
          {out.length > 1200 && (
            <button className="btn ghost sm" onClick={() => setFull(!full)}>
              {full ? 'Show less' : `Show all ${out.length.toLocaleString()} characters`}
            </button>
          )}
        </Field>
      )}
    </div>
  );
}

/** One loop iteration in the Execution Trace. */
export function AgentStepCard({ s }: { s: AgentStep }) {
  const c = s.call;
  const [open, setOpen] = useState(false);
  const pill = c ? CALL_PILL[c.status] : ['Decided', 'violet'];
  return (
    <div className={`trace-card ag-step${c?.status === 'running' || c?.status === 'awaiting' ? ' running' : ''}`}>
      <div className="trace-top">
        <span className="label violet">
          Step {s.index + 1} · {fmtTime(s.at)}
        </span>
        <span className="ag-step-actions">
          <CopyButton text={() => JSON.stringify(s, null, 2)} label={`Copy step ${s.index + 1} as JSON`} />
          <span className={`pill ${pill[1]}`}>{pill[0]}</span>
        </span>
      </div>
      <div className="trace-title">
        {c ? <Wrench size={14} /> : <ChevronRight size={14} />} {actionLabel(s.action)}
        {c?.args && <span className="ag-args mono"> {argsInline(c.args)}</span>}
      </div>
      {s.decision && (
        <div className="trace-detail">
          Chosen with {pct(s.decision.confidence)} of {s.decision.options.length} options ·{' '}
          <span className="ag-ms">decided in {fmtMs(s.decision.ms)}</span> <span className="faint">· {s.decision.model}</span>
        </div>
      )}
      {!s.decision && s.failedDecision?.io && (
        <div className="trace-detail">
          <span className="ag-ms">Decision failed after {fmtMs(s.failedDecision.ms)}</span>
        </div>
      )}
      {s.note && <div className="trace-detail ag-note">{s.note}</div>}
      {c?.observation && <div className="trace-detail">{c.observation}</div>}
      <button className="btn ghost sm ag-toggle" aria-expanded={open} onClick={() => setOpen(!open)}>
        {open ? 'Hide details' : 'Details'}
      </button>
      {open && (
        <div className="ag-details">
          {s.decision ? <DecisionBars d={s.decision} /> : <div className="faint ag-small">No decision was scored for this step.</div>}
          {s.decision?.io && <DecisionCall io={s.decision.io} labels={Object.fromEntries(s.decision.options.map((o) => [o.label, actionLabel(o.id)]))} />}
          {!s.decision && s.failedDecision?.io && (
            <>
              <div className="ag-small ag-err selectable">{s.failedDecision.error}</div>
              <DecisionCall io={s.failedDecision.io} />
            </>
          )}
          {c && <CallDetail c={c} />}
        </div>
      )}
    </div>
  );
}

/** Inline in the chat while a call waits under the Ask policy. */
export function ApprovalCard({ s }: { s: AgentStep }) {
  const c = s.call!;
  const def = toolById(c.tool);
  return (
    <div className="ag-approval" role="group" aria-label={`Approve ${c.title}?`}>
      <div className="ag-approval-top">
        <ShieldQuestion size={18} color="var(--amber)" />
        <span className="label amber">Approval needed</span>
      </div>
      <p>
        The agent wants to run <b>{c.title}</b>
        {def ? ` (${def.risk === 'read' ? 'read-only' : def.risk}, on this knowledge base only)` : ''} with:
      </p>
      <pre className="ag-pre selectable">{JSON.stringify(c.args, null, 2)}</pre>
      <div className="ag-approval-actions">
        <CopyButton text={JSON.stringify({ tool: c.tool, args: c.args })} label="Copy the requested call" />
        <button className="btn secondary sm" onClick={() => resolveApproval(s.id, false)}>
          <X size={13} /> Deny
        </button>
        <button className="btn primary sm" onClick={() => resolveApproval(s.id, true)}>
          <Check size={13} /> Approve
        </button>
      </div>
    </div>
  );
}

/** Tool calls as chips under "Processing reasoning". */
export function ToolChips({ m }: { m: Message }) {
  const calls = (m.agent ?? []).filter((s) => s.call);
  if (!calls.length) return null;
  return (
    <>
      {calls.map((s) => {
        const c = s.call!;
        const bad = c.status === 'error' || c.status === 'denied';
        const live = c.status === 'running' || c.status === 'filling' || c.status === 'awaiting';
        return (
          <div key={s.id} className="cc-chip-row">
            <span className={`dot${live ? ' pulse' : ''}`} style={{ color: 'var(--violet)' }} />
            <div className="cc-chip" style={{ color: bad ? 'var(--red)' : 'var(--violet)' }}>
              {c.status === 'awaiting' ? <Hand size={14} /> : <Wrench size={14} />}
              <span>
                {c.title}
                {c.args ? ` (${argsInline(c.args, 48)})` : ''} · {CALL_PILL[c.status][0].toLowerCase()}
                {c.status === 'done' && c.observation ? ` — ${c.observation.slice(0, 60)}${c.observation.length > 60 ? '…' : ''}` : ''}
              </span>
            </div>
          </div>
        );
      })}
    </>
  );
}

/** The whole turn's trace as JSON: decisions, calls, sources, stats. */
export function copyTrace(m: Message, question: string) {
  const data = {
    question,
    at: new Date(m.createdAt).toISOString(),
    knowledgeBase: m.kbName,
    steps: m.steps,
    agent: m.agent,
    sources: m.sources?.map((h, i) => ({ n: i + 1, file: h.file, lines: [h.start_line, h.end_line], name: h.name })),
    stats: m.stats,
  };
  void navigator.clipboard
    ?.writeText(JSON.stringify(data, null, 2))
    .then(() => toast({ tone: 'ok', title: 'Trace copied as JSON' }))
    .catch(() => toast({ tone: 'error', title: 'Could not copy the trace' }));
}

export function CopyTraceButton({ m, question }: { m: Message; question: string }) {
  return (
    <button className="btn ghost sm" onClick={() => copyTrace(m, question)} title="Copy this turn’s full trace as JSON">
      <Copy size={12} /> Copy trace
    </button>
  );
}
