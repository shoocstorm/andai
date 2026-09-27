// How agent mode shows its work: every decision with each option's
// probability, and every tool call with its arguments, approval, the ug
// command that ran, timing, the passages it found and its raw output. Tool output and arguments come from
// the model and the user's files, so they render as plain text only.

import { Check, ChevronRight, Copy, Hand, HelpCircle, ShieldQuestion, Wrench, X } from 'lucide-react';
import { useState } from 'react';
import { toolById } from '../agent/tools/registry';
import { CopyButton, Modal, fmtTime } from '../components/ui';
import { shellCommand } from '../agent/debugReport';
import type { RelevanceRecord } from '../agent/relevance';
import type { SearchHit } from '../kb/api';
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

/** The exact call: prompt, parameters and raw reply. Text from the model and the user's files, so plain text only. */
export function RawDecisionCall({ io, labels }: { io: DecisionIO; labels?: Record<string, string> }) {
  const r = io.response;
  const total = r ? r.topLogprobs.reduce((a, t) => a + Math.exp(t.logprob), 0) : 0;
  return (
    <div className="ag-call ag-io">
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
        ) : r.laya ? (
          <>
            <div className="ag-small">
              Scored {r.laya.scores.length} options{r.laya.stop != null ? ' and a yes/no question in one pass' : ''} from {r.laya.inputTokens} input
              tokens in {fmtMs(r.laya.ms)} (model time)
              {r.laya.truncated && <span className="ag-note"> · input cut to fit</span>}
            </div>
            {r.laya.stop != null && (
              <div className="ag-small">
                Yes/no: P(true) <span className="mono">{pct(r.laya.stop)}</span>
              </div>
            )}
            <table className="ag-logprobs mono">
              <thead>
                <tr>
                  <th scope="col">Option</th>
                  <th scope="col">Probability</th>
                </tr>
              </thead>
              <tbody>
                {r.laya.scores.map((t) => (
                  <tr key={t.id} className="option">
                    <td className="selectable">{t.id}</td>
                    <td>{pct(t.probability)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
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
        {d.truncated && <span className="ag-note"> · input cut to fit the model</span>}
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
      {d.stop && (
        <div className="ag-small ag-stop" title={d.stop.statement}>
          Results suffice? <b>{pct(d.stop.probability)} yes</b> <span className="faint">(yes/no question, asked with the choice)</span>
        </div>
      )}
      <div className="faint ag-seed">
        Option order seed {d.seed} · {d.io?.response?.laya ? 'calibrated probabilities' : 'probabilities are uncalibrated scores'}
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

/** The passages a call's output holds, read the way the agent read them; null when the stored output can't be parsed (clipped, or not JSON). */
export function callResults(c: ToolCallRecord): SearchHit[] | null {
  const def = toolById(c.tool);
  if (!def || !c.output) return null;
  try {
    return def.observe(JSON.parse(c.output)).hits;
  } catch {
    return null;
  }
}

const RISK_LABEL: Record<string, string> = { read: 'read-only', write: 'writes', device: 'reaches the device' };

/**
 * What a tool call did, in plain sentences, from its record: who wrote the
 * arguments, whether it ran on its own or waited for approval, how ug
 * answered, and what it added to the answer. Pure, for the call dialog and
 * its tests.
 */
export function explainCall(c: ToolCallRecord): string[] {
  const out: string[] = [];
  const writer = c.argModel ?? 'The chat model';
  const tries = c.argAttempts > 1 ? ` in ${c.argAttempts} attempts` : '';
  if (c.args) {
    const n = Object.keys(c.args).length;
    out.push(n ? `${writer} filled in the arguments${tries}: ${argsInline(c.args, 160)}.` : `${c.title} takes no arguments.`);
  } else if (c.status === 'filling') out.push(`${writer} is writing the arguments.`);
  else out.push(`${writer} could not write valid arguments${tries}, so the tool never ran.`);
  if (c.approval === 'pending') out.push('It is waiting for your approval before it runs.');
  else if (c.approval === 'approved') out.push('The tool is set to Ask, and you approved this call.');
  else if (c.approval === 'denied') out.push('The tool is set to Ask, and you denied this call, so it did not run.');
  else if (c.args && c.policy === 'auto') out.push('It ran without asking: the tool is set to Auto, and it only reads the selected knowledge base.');
  if (c.status === 'running') out.push('It is running now.');
  else if (c.status === 'skipped' && c.error) out.push(c.error);
  else if (c.status === 'error' && c.args) out.push(`The call failed${c.ms != null ? ` after ${fmtMs(c.ms)}` : ''}: ${c.error ?? 'no reason given'}`);
  else if (c.status === 'done') {
    if (c.error) out.push(c.error);
    const size = c.outputBytes != null ? ` with ${c.outputBytes.toLocaleString()} bytes of results` : '';
    out.push(`ug answered in ${c.ms != null ? fmtMs(c.ms) : 'an unknown time'}${size}${c.truncated ? ', cut at the size cap' : ''}.`);
    if (c.hits != null)
      out.push(
        c.hits
          ? `${c.hits} passage${c.hits === 1 ? '' : 's'} went into the answer’s context, new or with text the context didn’t have yet.`
          : 'Nothing new went into the answer’s context: it found nothing, or only passages already there.',
      );
  }
  return out;
}

function ResultList({ hits }: { hits: SearchHit[] }) {
  if (!hits.length) return <div className="ag-small faint">No passages in the output.</div>;
  return (
    <ol className="tc-hits" aria-label="Passages found">
      {hits.map((h, i) => (
        <li key={`${h.id}:${i}`}>
          <div className="tc-hit-head">
            <span className="mono tc-hit-loc selectable" title={h.file}>
              {h.file}
              {h.start_line ? <span className="faint">:{h.start_line}-{h.end_line}</span> : null}
            </span>
            <span className="tc-hit-kind faint">
              {h.name && h.name !== h.file ? `${h.name} · ` : ''}
              {h.node_type}
            </span>
          </div>
          {h.snippet && <pre className="tc-hit-snippet selectable">{h.snippet}</pre>}
        </li>
      ))}
    </ol>
  );
}

/** Everything about one tool call: why in plain words, what it was given, what it ran, and what came back. */
export function ToolCallDialog({ s, open, onClose }: { s: AgentStep; open: boolean; onClose: () => void }) {
  const c = s.call!;
  const def = toolById(c.tool);
  const [full, setFull] = useState(false);
  const out = c.output ?? '';
  const preview = out.length > 1200 && !full ? `${out.slice(0, 1200)}\n…` : out;
  const results = callResults(c);
  const pill = CALL_PILL[c.status];
  const props = def?.schema?.properties ?? {};
  return (
    <Modal open={open} onClose={onClose} wide label={`Tool call, step ${s.index + 1}`}>
      <div className="dd-head">
        <div>
          <div className="label violet">Step {s.index + 1} · tool call</div>
          <h3 className="tc-title">
            <Wrench size={17} /> {c.title} <span className={`pill ${pill[1]}`}>{pill[0]}</span>
          </h3>
          {def && <p className="ag-small faint tc-desc">{def.description}</p>}
          <div className="faint ag-small">
            {[
              def && RISK_LABEL[def.risk],
              c.policy === 'ask' ? 'Ask before running' : c.policy === 'auto' ? 'Runs automatically' : 'Off',
              c.ms != null && `took ${fmtMs(c.ms)}`,
              c.outputBytes != null && `${c.outputBytes.toLocaleString()} bytes`,
            ]
              .filter(Boolean)
              .join(' · ')}
          </div>
        </div>
        <button className="btn ghost sm" aria-label="Close dialog" onClick={onClose}>
          <X size={15} />
        </button>
      </div>

      <section className="dd-section" aria-label="What happened">
        <h4 className="label">What happened</h4>
        <ol className="dd-why">
          {explainCall(c).map((line, i) => (
            <li key={i} className="selectable">
              {line}
            </li>
          ))}
        </ol>
      </section>

      <div className="dd-grid">
        <section className="dd-section" aria-label="What it was given">
          <h4 className="label">What it was given</h4>
          {c.args && Object.keys(c.args).length ? (
            <dl className="tc-args">
              {Object.entries(c.args).map(([k, v]) => (
                <div key={k} className="tc-arg">
                  <dt>
                    <span className="mono">{k}</span>
                    {props[k]?.description && <span className="faint"> {props[k].description}</span>}
                  </dt>
                  <dd className="selectable">{typeof v === 'string' ? v : JSON.stringify(v)}</dd>
                </div>
              ))}
            </dl>
          ) : (
            <div className="ag-small faint">{c.args ? 'No arguments.' : 'No valid arguments.'}</div>
          )}
          {c.argModel && (
            <div className="faint ag-small tc-by">
              Written by {c.argModel}
              {c.argAttempts > 1 ? ` in ${c.argAttempts} attempts` : ''}
            </div>
          )}
          {c.argsRaw && (
            <details className="ag-raw">
              <summary>Model’s raw reply{c.argAttempts > 1 ? ` (${c.argAttempts} attempts)` : ''}</summary>
              <pre className="ag-pre selectable">{c.argsRaw}</pre>
            </details>
          )}
        </section>
        <section className="dd-section" aria-label="What it ran">
          <h4 className="label">What it ran</h4>
          {c.argv ? (
            <Field k="Command" copy={shellCommand(c.argv)}>
              <pre className="ag-pre selectable">{shellCommand(c.argv)}</pre>
            </Field>
          ) : (
            <div className="ag-small faint">No command ran.</div>
          )}
          <dl className="tc-facts ag-small">
            <dt>Policy</dt>
            <dd>
              {c.policy === 'ask' ? 'Ask' : c.policy === 'auto' ? 'Auto (read-only)' : 'Off'}
              {c.approval && ` · ${c.approval}`}
            </dd>
            <dt>Started</dt>
            <dd>{fmtTime(c.startedAt)}</dd>
            {c.ms != null && (
              <>
                <dt>Took</dt>
                <dd>{fmtMs(c.ms)}</dd>
              </>
            )}
            {c.outputBytes != null && (
              <>
                <dt>Output</dt>
                <dd>
                  {c.outputBytes.toLocaleString()} bytes{c.truncated && <span className="ag-note"> · cut at the size cap</span>}
                </dd>
              </>
            )}
          </dl>
        </section>
      </div>

      {(c.observation || c.error || results) && (
        <section className="dd-section tc-found" aria-label="What it found">
          <h4 className="label">What it found</h4>
          {c.error && c.status !== 'skipped' && (
            <Field k={c.status === 'done' ? 'Note' : 'Error'} copy={c.error}>
              <div className={`ag-small selectable${c.status === 'done' ? ' ag-note' : ' ag-err'}`}>{c.error}</div>
            </Field>
          )}
          {c.observation && (
            <Field k="What the agent saw next">
              <div className="ag-small selectable">{c.observation}</div>
            </Field>
          )}
          {results && (
            <Field k={`Passages (${results.length})`}>
              <ResultList hits={results} />
            </Field>
          )}
        </section>
      )}

      {out && (
        <details className="ag-raw dd-raw">
          <summary>Raw output from ug</summary>
          <Field k="Output" copy={out}>
            <pre className="ag-pre ag-out selectable">{preview}</pre>
            {out.length > 1200 && (
              <button className="btn ghost sm" onClick={() => setFull(!full)}>
                {full ? 'Show less' : `Show all ${out.length.toLocaleString()} characters`}
              </button>
            )}
          </Field>
        </details>
      )}
      <div className="dd-foot">
        <CopyButton text={JSON.stringify({ step: s.index + 1, ...c }, null, 2)} label="Copy tool call">
          Copy tool call
        </CopyButton>
        <button className="btn primary sm" onClick={onClose}>
          Close
        </button>
      </div>
    </Modal>
  );
}

const FINAL: Record<string, string> = {
  answer_now: 'So the agent stopped using tools and answered.',
  ask_clarification: 'So the agent asked the user a clarifying question.',
};

/**
 * Why a step did what it did, in plain sentences, from its record: what the
 * decision model was asked and how it scored the options, the yes/no stop
 * question when Laya asked one, any override by the loop (`note`), and the
 * action taken. Pure, for the decision dialog and its tests.
 */
export function explainStep(s: AgentStep): string[] {
  const out: string[] = [];
  const d = s.decision;
  if (d) {
    const [top, next] = [...d.options].sort((a, b) => b.probability - a.probability);
    out.push(`The agent asked ${d.model} “${d.question}” and it scored ${d.options.length} options in ${fmtMs(d.ms)}.`);
    if (d.stop) {
      const yes = d.stop.probability >= 0.5;
      out.push(`With tool results in hand, it also asked whether they already answer the request: ${pct(d.stop.probability)} yes, so ${yes ? 'no more tools were needed' : 'more looking was needed'}.`);
    }
    out.push(`${actionLabel(top.id)} scored highest at ${pct(top.probability)}${next ? `, ahead of ${actionLabel(next.id)} at ${pct(next.probability)}` : ''}.`);
  } else if (s.failedDecision) {
    out.push(`The agent asked the decision model what to do next, but the call failed: ${s.failedDecision.error}`);
  }
  // The loop's own reason when it overrode or skipped the decision; the stop rule is already said above.
  if (s.note && !(d?.stop && s.note.startsWith('The tool results cover'))) out.push(s.note);
  const c = s.call;
  out.push(c ? `So the agent ran ${c.title}${c.args && Object.keys(c.args).length ? ` with ${argsInline(c.args, 160)}` : ''}.` : (FINAL[s.action] ?? `So the agent chose ${actionLabel(s.action)}.`));
  return out;
}

/** Everything about one decision, apart from the tool call it led to. */
export function DecisionDialog({ s, open, onClose }: { s: AgentStep; open: boolean; onClose: () => void }) {
  const d = s.decision;
  const io = d?.io ?? s.failedDecision?.io;
  const laya = io?.response?.laya;
  const labels = d ? Object.fromEntries(d.options.map((o) => [o.label, actionLabel(o.id)])) : undefined;
  return (
    <Modal open={open} onClose={onClose} wide label={`Why step ${s.index + 1}`}>
      <div className="dd-head">
        <div>
          <div className="label violet">Step {s.index + 1} · decision</div>
          <h3>Why: {actionLabel(s.action)}</h3>
          <div className="faint ag-small">
            {d
              ? `${d.model} · ${d.slot === 'decider' ? 'decision model' : 'chat model'} · decided in ${fmtMs(d.ms)}${laya ? ` (model ${fmtMs(laya.ms)})` : ''}${d.promptTokens != null ? ` · ${d.promptTokens} input tokens` : ''}`
              : s.failedDecision
                ? `Decision failed after ${fmtMs(s.failedDecision.ms)}`
                : 'No decision was scored'}
            {d?.truncated && <span className="ag-note"> · input cut to fit the model</span>}
          </div>
        </div>
        <button className="btn ghost sm" aria-label="Close dialog" onClick={onClose}>
          <X size={15} />
        </button>
      </div>

      <section className="dd-section" aria-label="Why this step">
        <h4 className="label">Why this step</h4>
        <ol className="dd-why">
          {explainStep(s).map((line, i) => (
            <li key={i} className="selectable">
              {line}
            </li>
          ))}
        </ol>
      </section>

      {(io || d) && (
        <div className="dd-grid">
          <section className="dd-section" aria-label="What the model saw">
            <h4 className="label">What the model saw</h4>
            {io?.request.state != null && (
              <Field k="State" copy={io.request.state}>
                <pre className="ag-pre selectable dd-state">{io.request.state}</pre>
              </Field>
            )}
            {d && (
              <>
                <Field k="Question">
                  <div className="ag-small selectable">{d.question}</div>
                </Field>
                <Field k="Options">
                  <ol className="dd-options">
                    {d.options.map((o) => (
                      <li key={o.id} className={o.id === d.chosen ? 'chosen' : undefined}>
                        <span className="mono faint">{o.label}</span>
                        <span>
                          <b>{actionLabel(o.id)}</b> <span className="faint selectable">{o.text}</span>
                        </span>
                      </li>
                    ))}
                  </ol>
                </Field>
                {d.stop && (
                  <Field k="Yes/no question">
                    <div className="ag-small selectable">{d.stop.statement}</div>
                  </Field>
                )}
              </>
            )}
          </section>
          <section className="dd-section" aria-label="What it returned">
            <h4 className="label">What it returned</h4>
            {d ? <DecisionBars d={d} /> : <div className="ag-small ag-err selectable">{s.failedDecision?.error ?? 'No reply.'}</div>}
          </section>
        </div>
      )}

      {io && (
        <details className="ag-raw dd-raw">
          <summary>Exact call: prompt, parameters and raw reply</summary>
          <RawDecisionCall io={io} labels={labels} />
        </details>
      )}
      <div className="dd-foot">
        {io && (
          <CopyButton text={JSON.stringify({ step: s.index + 1, action: s.action, note: s.note, decision: d, failed: s.failedDecision }, null, 2)} label="Copy decision call">
            Copy decision call
          </CopyButton>
        )}
        <button className="btn primary sm" onClick={onClose}>
          Close
        </button>
      </div>
    </Modal>
  );
}

/** One loop iteration in the Execution Trace: the decision and the tool call each open in a dialog. */
export function AgentStepCard({ s }: { s: AgentStep }) {
  const c = s.call;
  const [open, setOpen] = useState(false);
  const [why, setWhy] = useState(false);
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
          {s.decision.stop && `Results suffice: ${pct(s.decision.stop.probability)} · `}
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
      <div className="ag-step-buttons">
        <button className="btn ghost sm ag-toggle" onClick={() => setWhy(true)}>
          <HelpCircle size={12} /> <span className="ellipsis">Why this step?</span>
        </button>
        {c && (
          <button className="btn ghost sm ag-toggle" aria-haspopup="dialog" onClick={() => setOpen(true)}>
            <Wrench size={12} /> <span className="ellipsis">Tool call</span>
          </button>
        )}
      </div>
      <DecisionDialog s={s} open={why} onClose={() => setWhy(false)} />
      {c && <ToolCallDialog s={s} open={open} onClose={() => setOpen(false)} />}
    </div>
  );
}

/**
 * The relevance check under its trace step: every retrieved passage, its
 * score, and whether it went into the prompt (numbered as the answer cites
 * it) or was dropped, and why.
 */
export function RelevanceList({ r }: { r: RelevanceRecord }) {
  let n = 0;
  const dropped = r.items.filter((x) => !x.kept).length;
  return (
    <section className="trace-subs ag-rel" aria-label="Relevance check">
      <div className="faint ag-small">
        {r.model} scored how likely each passage helps answer the request. The top {r.keepTop} search results are always kept; the
        rest are dropped below {pct(r.dropBelow)}.
      </div>
      <ul className="ag-rel-list">
        {r.items.map((x, i) => {
          const cite = x.kept ? ++n : null;
          return (
            <li key={i} className={x.kept ? 'kept' : 'dropped'} aria-label={`${x.file}: ${pct(x.score)}, ${x.kept ? 'kept' : 'dropped'}`}>
              {x.kept ? <Check size={13} color="var(--violet)" /> : <X size={13} color="var(--text-4)" />}
              <span className="ag-rel-file ellipsis" title={x.file}>
                {cite != null && <span className="mono">[{cite}] </span>}
                {x.file.split('/').pop()}
                <span className="faint">
                  :{x.start_line}-{x.end_line}
                </span>
              </span>
              <span className="ag-bar-track">
                <i style={{ width: `${Math.max(1, x.score * 100)}%` }} />
              </span>
              <span className="ag-bar-value mono">{pct(x.score)}</span>
              <span className="ag-rel-why faint">{x.reason === 'top' ? 'top result' : x.reason === 'score' ? 'relevant' : 'dropped'}</span>
            </li>
          );
        })}
      </ul>
      <div className="faint ag-small">
        {dropped ? `${dropped} dropped · ~${r.tokensSaved.toLocaleString()} fewer prompt tokens for the chat model to read` : 'Nothing dropped'} ·{' '}
        {fmtMs(r.ms)}
      </div>
    </section>
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
