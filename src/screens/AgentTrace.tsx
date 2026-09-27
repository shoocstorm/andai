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
import { claimState, MEASURED, SUPPORTS, type SupportItem, type SupportRecord } from '../agent/claims';
import { LAYA_MODELS } from '../llm/models';
import { matchOf } from '../kb/match';
import { CHARS_PER_TOKEN, type ContextRecord } from '../agent/prompt';
import { MEASURED_RELEVANCE, passageState, passageText, RELEVANT, type RelevanceItem } from '../agent/relevance';
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
            {r.laya.choices?.map((c) => (
              <div key={c.id} className="ag-small">
                Argument <span className="mono">{c.id}</span>:{' '}
                <span className="mono">{c.scores.map((t) => `${t.id} ${pct(t.probability)}`).join(' · ')}</span>
              </div>
            ))}
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
    for (const p of c.argChoices ?? []) out.push(`${p.model} picked ${p.arg} “${p.value}” (${pct(p.probability)}) as a typed choice, and the chat model kept it.`);
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

/**
 * How ug search found a passage: the channel as a tag (semantic, keyword, or
 * a graph walk) and its ranking score as a bar, relative to the best match
 * among `peers`. Nothing for a passage no search returned (Read lines, …).
 */
export function MatchBadge({ h, peers, compact = false }: { h: SearchHit; peers: SearchHit[]; compact?: boolean }) {
  const m = matchOf(h, peers);
  if (!m) return null;
  const tip = `${m.help}${m.score != null ? ` ug rank score ${m.score.toFixed(4)} (lower ranks higher); ${pct(m.strength ?? 1)} of the best match here.` : ''}`;
  return (
    <span className="match" title={tip} aria-label={[m.label, m.strength != null ? `${pct(m.strength)} of the best match` : ''].filter(Boolean).join(', ')}>
      {m.how && <span className={`match-tag match-${m.how}`}>{m.label}</span>}
      {m.strength != null && !compact && (
        <span className="ag-bar-track match-bar" aria-hidden="true">
          <i style={{ width: `${Math.max(4, m.strength * 100)}%` }} />
        </span>
      )}
    </span>
  );
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
            <MatchBadge h={h} peers={hits} />
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
          {c.argChoices?.map((p) => (
            <div key={p.arg} className="faint ag-small tc-by">
              <span className="mono">{p.arg}</span> picked by {p.model} ({pct(p.probability)})
            </div>
          ))}
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
export function AgentStepCard({ s, domId }: { s: AgentStep; domId?: string }) {
  const c = s.call;
  const [open, setOpen] = useState(false);
  const [why, setWhy] = useState(false);
  const pill = c ? CALL_PILL[c.status] : ['Decided', 'violet'];
  return (
    <div id={domId} className={`trace-card ag-step${c?.status === 'running' || c?.status === 'awaiting' ? ' running' : ''}`}>
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

/** Plain text that opens a dialog: the item itself is the link, with no extra button. */
function TextLink({ label, onClick, children }: { label: string; onClick: () => void; children: React.ReactNode }) {
  return (
    <button type="button" className="linklike" aria-haspopup="dialog" aria-label={label} onClick={onClick}>
      {children}
    </button>
  );
}

/** A short line on how rough the Laya checks are, shown wherever their verdicts are. */
export const ROUGH_CHECK = 'A rough automatic check: it can miss real problems and flag correct ones.';

/**
 * Why a passage was kept or dropped, in plain sentences: what Laya was asked,
 * its answer against the rules (top results always kept, then the cut), and
 * how far the check can be trusted. Pure, for the relevance dialog and its tests.
 */
export function explainRelevance(x: RelevanceItem, r: RelevanceRecord): string[] {
  const out = [`The agent asked ${r.model} whether this passage (${x.file}:${x.start_line}-${x.end_line}) helps answer the request, as a yes/no question: “${RELEVANT}”`];
  if (x.reason === 'top') {
    out.push(`It answered ${pct(x.score)} yes. The passage is one of the top ${r.keepTop} search results, which are always kept, so it went into the prompt whatever it scored.`);
  } else if (x.kept) {
    out.push(`It answered ${pct(x.score)} yes, at or above the ${pct(r.dropBelow)} cut, so the passage went into the prompt.`);
  } else {
    out.push(`It answered ${pct(x.score)} yes, below the ${pct(r.dropBelow)} cut, so the passage was left out of the prompt and the chat model never read it.`);
  }
  if (x.truncated) out.push('The request or the passage was cut to fit the model’s input, so part of it wasn’t read.');
  const m = MEASURED_RELEVANCE[r.modelId];
  out.push(
    m
      ? `How far to trust it: on our test passages ${r.model} ranked a passage holding the answer above one without it ${Math.round(m.auc * 100)}% of the time, so the cut is kept low and only clear misses are dropped.`
      : 'How far to trust it: this checkpoint wasn’t measured on the relevance check, so treat its scores as rough.',
  );
  return out;
}

/**
 * A relevance record with what an older one lacks filled in from the message
 * (saved before Andai kept the request, the passage text and the model id):
 * the request is the question, and a kept passage is one of the answer's
 * sources, with the same text Laya read. A dropped passage's text can't be
 * recovered, and is left undefined so the dialog says so.
 */
export function resolveRelevance(r: RelevanceRecord, sources: SearchHit[], question: string) {
  const modelId = r.modelId ?? LAYA_MODELS.find((m) => m.name === r.model)?.id ?? '';
  const request = r.request ?? question;
  const textOf = (x: RelevanceItem): string | undefined => {
    if (typeof x.text === 'string') return x.text;
    const h = x.kept ? sources.find((s) => s.file === x.file && s.start_line === x.start_line && s.end_line === x.end_line) : undefined;
    return h ? passageText(h) : undefined;
  };
  return { r: { ...r, modelId, request }, textOf };
}

const NOT_RECORDED = 'Not recorded: this check ran before Andai kept the text of dropped passages.';

/** One passage's relevance check: the request, the passage, Laya's answer and why it was kept or dropped, and the exact call. */
export function RelevanceDialog({ x, r, text, cite, onClose }: { x: RelevanceItem; r: RelevanceRecord; text: string | undefined; cite: number | null; onClose: () => void }) {
  const source = `${x.file}:${x.start_line}-${x.end_line}`;
  const state = text == null ? null : passageState(r.request, source, text);
  return (
    <Modal open onClose={onClose} wide label={`Relevance check for ${source}`}>
      <div className="dd-head">
        <div>
          <div className="label violet">Relevance check · {cite != null ? `source [${cite}]` : 'dropped passage'}</div>
          <h3>{x.reason === 'top' ? 'Kept: a top search result' : x.kept ? 'Kept: likely to help' : 'Dropped: unlikely to help'}</h3>
          <div className="faint ag-small">
            {r.model} · decision model{x.inputTokens != null ? ` · ${x.inputTokens} input tokens` : ''} · the check took {fmtMs(r.ms)} for {r.items.length}{' '}
            passage{r.items.length === 1 ? '' : 's'} (model {fmtMs(r.modelMs)})
            {x.truncated && <span className="ag-note"> · input cut to fit the model</span>}
          </div>
        </div>
        <button className="btn ghost sm" aria-label="Close dialog" onClick={onClose}>
          <X size={15} />
        </button>
      </div>

      <section className="dd-section" aria-label="Why this verdict">
        <h4 className="label">Why this verdict</h4>
        <ol className="dd-why">
          {explainRelevance(x, r).map((line, i) => (
            <li key={i} className="selectable">
              {line}
            </li>
          ))}
        </ol>
      </section>

      <div className="dd-grid">
        <section className="dd-section" aria-label="What the model saw">
          <h4 className="label">What the model saw</h4>
          <Field k="Request" copy={r.request}>
            <div className="ag-small selectable">{r.request}</div>
          </Field>
          <Field k={`Passage · ${source}`} copy={text}>
            {text == null ? <div className="ag-small faint">{NOT_RECORDED}</div> : <pre className="ag-pre selectable dd-state">{text || '(empty)'}</pre>}
          </Field>
          <Field k="Yes/no question">
            <div className="ag-small selectable">{RELEVANT}</div>
          </Field>
        </section>
        <section className="dd-section" aria-label="What it returned">
          <h4 className="label">What it returned</h4>
          <YesNoBars yes="Helps" no="Doesn’t help" p={x.score} chosenYes={x.score >= r.dropBelow} />
          <div className="faint ag-seed">
            Dropped below {pct(r.dropBelow)} “helps”, except the top {r.keepTop} search results · {ROUGH_CHECK}
          </div>
        </section>
      </div>

      <details className="ag-raw dd-raw">
        <summary>Exact call: Laya’s input for this passage</summary>
        <Field k="Input: state" copy={state ?? undefined}>
          {state == null ? <div className="ag-small faint">{NOT_RECORDED}</div> : <pre className="ag-pre selectable">{state}</pre>}
        </Field>
        <Field k="Input: question">
          <pre className="ag-pre selectable">{`noul · ${RELEVANT}`}</pre>
        </Field>
        <Field k="Output">
          <pre className="ag-pre selectable">{JSON.stringify({ probabilities: [1 - x.score, x.score], inputTokens: x.inputTokens, truncated: x.truncated }, null, 2)}</pre>
        </Field>
      </details>
      <div className="dd-foot">
        <CopyButton text={JSON.stringify({ model: r.model, question: RELEVANT, state, score: x.score, kept: x.kept, reason: x.reason, dropBelow: r.dropBelow }, null, 2)} label="Copy relevance check call">
          Copy relevance check call
        </CopyButton>
        <button className="btn primary sm" onClick={onClose}>
          Close
        </button>
      </div>
    </Modal>
  );
}

/** Two bars for a yes/no answer; the side the verdict went to is highlighted. */
function YesNoBars({ yes, no, p, chosenYes }: { yes: string; no: string; p: number; chosenYes: boolean }) {
  return (
    <ul className="ag-bars" aria-label="Laya's answer">
      {(
        [
          [yes, p, chosenYes],
          [no, 1 - p, !chosenYes],
        ] as const
      ).map(([label, v, chosen]) => (
        <li key={label} className={chosen ? 'chosen' : undefined}>
          <span className="ag-bar-label">{label}</span>
          <span className="ag-bar-track">
            <i style={{ width: `${Math.max(1, v * 100)}%` }} />
          </span>
          <span className="ag-bar-value mono">{pct(v)}</span>
        </li>
      ))}
    </ul>
  );
}

/**
 * The relevance check under its trace step, folded to one line: every
 * retrieved passage, its score, and whether it went into the prompt
 * (numbered as the answer cites it) or was dropped. Each passage opens its
 * check; a dialog is built only while open.
 */
export function RelevanceList({ r: stored, sources, question }: { r: RelevanceRecord; sources: SearchHit[]; question: string }) {
  const [open, setOpen] = useState<number | null>(null);
  const { r, textOf } = resolveRelevance(stored, sources, question);
  const dropped = r.items.filter((x) => !x.kept).length;
  let n = 0;
  const cites = r.items.map((x) => (x.kept ? ++n : null));
  return (
    <section className="trace-subs ag-rel" aria-label="Relevance check">
      <details className="ag-fold">
        <summary className="faint ag-small">
          {r.items.length} passages scored · {dropped ? `${dropped} dropped` : 'none dropped'}
        </summary>
        <div className="faint ag-small">
          {r.model} scored how likely each passage helps answer the request. The top {r.keepTop} search results are always kept; the
          rest are dropped below {pct(r.dropBelow)}. {ROUGH_CHECK}
        </div>
        <ul className="ag-rel-list">
          {r.items.map((x, i) => (
            <li key={i} className={x.kept ? 'kept' : 'dropped'} aria-label={`${x.file}: ${pct(x.score)}, ${x.kept ? 'kept' : 'dropped'}`}>
              {x.kept ? <Check size={13} color="var(--violet)" /> : <X size={13} color="var(--text-4)" />}
              <span className="ag-rel-file ellipsis" title={x.file}>
                <TextLink label={`Relevance check: ${x.file}:${x.start_line}-${x.end_line}`} onClick={() => setOpen(i)}>
                  {cites[i] != null && <span className="mono">[{cites[i]}] </span>}
                  {x.file.split('/').pop()}
                  <span className="faint">
                    :{x.start_line}-{x.end_line}
                  </span>
                </TextLink>
              </span>
              <span className="ag-bar-track">
                <i style={{ width: `${Math.max(1, x.score * 100)}%` }} />
              </span>
              <span className="ag-bar-value mono">{pct(x.score)}</span>
              <span className="ag-rel-why faint">{x.reason === 'top' ? 'top result' : x.reason === 'score' ? 'relevant' : 'dropped'}</span>
            </li>
          ))}
        </ul>
        <div className="faint ag-small">
          {dropped ? `${dropped} dropped · ~${r.tokensSaved.toLocaleString()} fewer prompt tokens for the chat model to read` : 'Nothing dropped'} ·{' '}
          {fmtMs(r.ms)}
        </div>
      </details>
      {open != null && <RelevanceDialog x={r.items[open]} r={r} text={textOf(r.items[open])} cite={cites[open]} onClose={() => setOpen(null)} />}
    </section>
  );
}

/**
 * Why a claim got its verdict, in plain sentences: what Laya was asked, what
 * it answered against the cut, what may have skewed it (other citations,
 * input cut to fit), and how far the check can be trusted at all. Pure, for
 * the claim dialog and its tests.
 */
export function explainClaim(x: SupportItem, r: SupportRecord): string[] {
  const out = [
    `The agent asked ${r.model} whether passage [${x.n}] (${x.source}) supports the sentence, as a yes/no question: “${SUPPORTS}”`,
    x.flagged
      ? `It answered ${pct(x.score)} yes, below the ${pct(r.flagBelow)} cut, so the sentence is marked as possibly unsupported by [${x.n}].`
      : `It answered ${pct(x.score)} yes, at or above the ${pct(r.flagBelow)} cut, so nothing is marked.`,
  ];
  const others = x.cites.filter((n) => n !== x.n);
  if (others.length) {
    out.push(
      `The sentence also cites ${others.map((n) => `[${n}]`).join(', ')}, and each source is checked on its own: a sentence that combines facts from several can score low against each.`,
    );
  }
  if (x.truncated) out.push('The sentence or the passage was cut to fit the model’s input, so part of it wasn’t read.');
  const m = MEASURED[r.modelId];
  out.push(
    m
      ? `How far to trust it: on our test set ${r.model} scored a sentence’s own passage above a passage from another answer ${Math.round(m.auc * 100)}% of the time; at this cut it flagged about ${Math.round(m.caught * 100)}% of those wrong pairings and wrongly flagged ${m.falseAlarms} right ones. It’s a hint to read the source, not a verdict.`
      : 'How far to trust it: this checkpoint wasn’t measured on the claim check. It’s a hint to read the source, not a verdict.',
  );
  return out;
}

/** One claim's check: the sentence, the passage it was checked against, Laya's answer and why, and the exact call. */
export function ClaimDialog({ x, r, passage, open, onClose }: { x: SupportItem; r: SupportRecord; passage: string; open: boolean; onClose: () => void }) {
  const state = claimState(x.sentence, x.source, passage);
  return (
    <Modal open={open} onClose={onClose} wide label={`Claim check for source ${x.n}`}>
      <div className="dd-head">
        <div>
          <div className="label violet">Claim check · source [{x.n}]</div>
          <h3>{x.flagged ? 'May not be supported by its source' : 'Looks supported by its source'}</h3>
          <div className="faint ag-small">
            {r.model} · decision model{x.inputTokens != null ? ` · ${x.inputTokens} input tokens` : ''} · the check took {fmtMs(r.ms)} for {r.items.length} claim{r.items.length === 1 ? '' : 's'} (model{' '}
            {fmtMs(r.modelMs)})
            {x.truncated && <span className="ag-note"> · input cut to fit the model</span>}
          </div>
        </div>
        <button className="btn ghost sm" aria-label="Close dialog" onClick={onClose}>
          <X size={15} />
        </button>
      </div>

      <section className="dd-section" aria-label="Why this verdict">
        <h4 className="label">Why this verdict</h4>
        <ol className="dd-why">
          {explainClaim(x, r).map((line, i) => (
            <li key={i} className="selectable">
              {line}
            </li>
          ))}
        </ol>
      </section>

      <div className="dd-grid">
        <section className="dd-section" aria-label="What the model saw">
          <h4 className="label">What the model saw</h4>
          <Field k="Sentence (from the answer)" copy={x.sentence}>
            <div className="ag-small selectable">{x.sentence}</div>
          </Field>
          <Field k={`Passage [${x.n}] · ${x.source}`} copy={passage}>
            <pre className="ag-pre selectable dd-state">{passage || '(empty)'}</pre>
          </Field>
          <Field k="Yes/no question">
            <div className="ag-small selectable">{SUPPORTS}</div>
          </Field>
        </section>
        <section className="dd-section" aria-label="What it returned">
          <h4 className="label">What it returned</h4>
          <YesNoBars yes="Supported" no="Not supported" p={x.score} chosenYes={!x.flagged} />
          <div className="faint ag-seed">
            Flagged below {pct(r.flagBelow)} “supported” · {ROUGH_CHECK}
          </div>
        </section>
      </div>

      <details className="ag-raw dd-raw">
        <summary>Exact call: Laya’s input for this claim</summary>
        <Field k="Input: state" copy={state}>
          <pre className="ag-pre selectable">{state}</pre>
        </Field>
        <Field k="Input: question">
          <pre className="ag-pre selectable">{`noul · ${SUPPORTS}`}</pre>
        </Field>
        <Field k="Output">
          <pre className="ag-pre selectable">{JSON.stringify({ probabilities: [1 - x.score, x.score], inputTokens: x.inputTokens, truncated: x.truncated }, null, 2)}</pre>
        </Field>
      </details>
      <div className="dd-foot">
        <CopyButton text={JSON.stringify({ model: r.model, question: SUPPORTS, state, score: x.score, flagged: x.flagged, flagBelow: r.flagBelow, inputTokens: x.inputTokens, truncated: x.truncated }, null, 2)} label="Copy claim check call">
          Copy claim check call
        </CopyButton>
        <button className="btn primary sm" onClick={onClose}>
          Close
        </button>
      </div>
    </Modal>
  );
}

/**
 * A claim's text as a link to its dialog. The dialog is built only while
 * open: an answer can list 24 claims, and a closed dialog would still run its
 * explanation on every render (one that threw blanked the whole app).
 */
function ClaimLink({ x, r, sources, children }: { x: SupportItem; r: SupportRecord; sources: SearchHit[]; children: React.ReactNode }) {
  const [open, setOpen] = useState(false);
  const h = sources[x.n - 1];
  // A claim saved before these were kept has no `cites` or `source`: both follow from the answer's sources.
  const full: SupportItem = { ...x, cites: x.cites ?? [x.n], source: x.source ?? (h ? `${h.file}:${h.start_line}-${h.end_line}` : `source [${x.n}]`) };
  const rr: SupportRecord = { ...r, modelId: r.modelId ?? LAYA_MODELS.find((m) => m.name === r.model)?.id ?? '' };
  return (
    <>
      <TextLink label={`Claim check: [${x.n}] ${x.sentence}`} onClick={() => setOpen(true)}>
        {children}
      </TextLink>
      {open && <ClaimDialog x={full} r={rr} passage={h ? passageText(h) : ''} open onClose={() => setOpen(false)} />}
    </>
  );
}

/**
 * The claim check under its trace step: every cited sentence, the source it
 * cites, and how likely that passage supports it; each opens its dialog.
 */
export function SupportList({ r, sources }: { r: SupportRecord; sources: SearchHit[] }) {
  const flagged = new Set(r.items.filter((x) => x.flagged).map((x) => x.sentence)).size;
  const sentences = new Set(r.items.map((x) => x.sentence)).size;
  return (
    <section className="trace-subs ag-rel" aria-label="Claim check">
      <details className="ag-fold">
        <summary className="faint ag-small">
          {sentences} cited sentence{sentences === 1 ? '' : 's'} checked · {flagged ? `${flagged} may not be supported` : 'none flagged'}
        </summary>
        <div className="faint ag-small">
          {r.model} scored how likely each cited passage supports the sentence that cites it. Below {pct(r.flagBelow)}, a sentence is
          marked as possibly unsupported; the answer itself is never changed. {ROUGH_CHECK}
        </div>
        <ul className="ag-rel-list ag-claims">
          {r.items.map((x, i) => (
            <li key={i} className={x.flagged ? 'dropped' : 'kept'} aria-label={`[${x.n}] ${x.sentence}: ${pct(x.score)}, ${x.flagged ? 'may not be supported' : 'supported'}`}>
              {x.flagged ? <X size={13} color="var(--text-4)" /> : <Check size={13} color="var(--violet)" />}
              <span className="ag-rel-file">
                <ClaimLink x={x} r={r} sources={sources}>
                  <span className="mono">[{x.n}] </span>
                  {x.sentence}
                </ClaimLink>
              </span>
              <span className="ag-bar-track">
                <i style={{ width: `${Math.max(1, x.score * 100)}%` }} />
              </span>
              <span className="ag-bar-value mono">{pct(x.score)}</span>
              <span className="ag-rel-why faint">{x.flagged ? 'unsupported?' : 'supported'}</span>
            </li>
          ))}
        </ul>
        <div className="faint ag-small">{fmtMs(r.ms)}</div>
      </details>
    </section>
  );
}

/**
 * Under an answer, only when the claim check flagged something: which cited
 * sentences may not be supported by the source they cite. Plain text.
 */
export function SupportNote({ r, sources }: { r: SupportRecord; sources: SearchHit[] }) {
  const flagged = r.items.filter((x) => x.flagged);
  if (!flagged.length) return null;
  return (
    <details className="cc-support">
      <summary>
        {flagged.length} of {r.items.length} cited sentence{r.items.length === 1 ? '' : 's'} may not be supported by the source {flagged.length === 1 ? 'it cites' : 'they cite'}
      </summary>
      <ul>
        {flagged.map((x, i) => (
          <li key={i}>
            <ClaimLink x={x} r={r} sources={sources}>
              <span className="mono">[{x.n}]</span> {x.sentence}
            </ClaimLink>{' '}
            <span className="faint mono">({pct(x.score)} supported)</span>
          </li>
        ))}
      </ul>
      <div className="faint">{ROUGH_CHECK} Click a sentence to see the passage it was checked against.</div>
    </details>
  );
}

/**
 * One source of an answer, `[n]`: where it's from, the passage the chat model
 * read, and what the Laya checks said about it (its relevance score, the
 * cited sentences checked against it). Opened from the answer's citations and
 * from retrieval.log.
 */
export function SourceDialog({ m, n, onClose }: { m: Message; n: number; onClose: () => void }) {
  const h = m.sources?.[n - 1];
  const where = h ? `${h.file}:${h.start_line}-${h.end_line}` : `source [${n}]`;
  const rel = h ? m.relevance?.items.find((x) => x.file === h.file && x.start_line === h.start_line && x.end_line === h.end_line) : undefined;
  const claims = (m.support?.items ?? []).filter((x) => x.n === n);
  const text = h ? passageText(h) : '';
  return (
    <Modal open onClose={onClose} wide label={`Source ${n}`}>
      <div className="dd-head">
        <div>
          <div className="label violet">Source [{n}]</div>
          <h3>{h ? (h.name && h.name !== h.file ? h.name : h.file.split('/').pop()) : 'Not in this answer’s sources'}</h3>
          <div className="faint ag-small">
            {where}
            {h?.node_type ? ` · ${h.node_type}` : ''}
            {m.kbName ? ` · from “${m.kbName}”` : ''}
          </div>
          {h && matchOf(h) && (
            <div className="ag-small dd-match">
              <MatchBadge h={h} peers={m.sources ?? []} /> <span className="faint">{matchOf(h)!.help}</span>
            </div>
          )}
        </div>
        <button className="btn ghost sm" aria-label="Close dialog" onClick={onClose}>
          <X size={15} />
        </button>
      </div>
      {h && (
        <section className="dd-section" aria-label="Passage">
          <Field k="Passage the chat model read" copy={text}>
            <pre className="ag-pre selectable dd-state">{text || '(empty)'}</pre>
          </Field>
        </section>
      )}
      {(rel || claims.length > 0) && (
        <section className="dd-section" aria-label="What the checks said">
          <h4 className="label">What the checks said</h4>
          {rel && (
            <div className="ag-small">
              Relevance check: <b>{pct(rel.score)}</b> likely to help answer the request (
              {rel.reason === 'top' ? `a top ${m.relevance!.keepTop} search result, always kept` : 'kept'}).
            </div>
          )}
          {claims.length > 0 && (
            <ul className="dd-why dd-claims">
              {claims.map((x, i) => (
                <li key={i} className="selectable">
                  {x.flagged ? <X size={12} color="var(--text-4)" /> : <Check size={12} color="var(--violet)" />} {x.sentence}{' '}
                  <span className="faint mono">
                    {pct(x.score)} supported{x.flagged ? ' · may not be supported' : ''}
                  </span>
                </li>
              ))}
            </ul>
          )}
          <div className="faint ag-small">{ROUGH_CHECK}</div>
        </section>
      )}
      <div className="dd-foot">
        {h && (
          <CopyButton text={text} label={`Copy source ${n}`}>
            Copy passage
          </CopyButton>
        )}
        <button className="btn primary sm" onClick={onClose}>
          Close
        </button>
      </div>
    </Modal>
  );
}

const nf = (n: number) => n.toLocaleString('en-US');
const toks = (chars: number) => Math.round(chars / CHARS_PER_TOKEN);

/**
 * What "Assemble context" did, in plain sentences: how the window was split,
 * which passages went in whole, cut or not at all, how much conversation was
 * kept, and the total. Pure, for the context dialog and its tests.
 */
export function explainContext(c: ContextRecord, model: string): string[] {
  const out = [
    `${model} reads at most ${nf(c.nCtx)} tokens at once. Andai kept ${nf(c.replyTokens)} of them for the reply and 256 spare, and gave about ${nf(toks(c.budget.context))} tokens of the rest to retrieved passages and ${nf(toks(c.budget.history))} to earlier conversation.`,
  ];
  const whole = c.passages.filter((p) => p.status === 'in').length;
  const cut = c.passages.filter((p) => p.status === 'clipped').length;
  const out_ = c.passages.filter((p) => p.status === 'left out').length;
  if (!c.passages.length) out.push('No passages went in: the knowledge base wasn’t searched for this turn, or found nothing (the system prompt says which).');
  else if (!cut && !out_) out.push(`All ${c.passages.length} passage${c.passages.length === 1 ? '' : 's'} went in whole, fenced as untrusted text the model must not take instructions from.`);
  else
    out.push(
      `${whole + cut} of ${c.passages.length} passages went in${cut ? ` (${cut} cut short to fit)` : ''}${out_ ? `; ${out_} ${out_ === 1 ? 'was' : 'were'} left out because the passage budget ran out, so the model never saw ${out_ === 1 ? 'it' : 'them'}` : ''}. They go in the order shown, so later ones give way first.`,
    );
  const h = c.history;
  out.push(
    !h.of
      ? 'There was no earlier conversation to include.'
      : h.sent === h.of
        ? `All ${h.of} earlier message${h.of === 1 ? '' : 's'} went in (about ${nf(toks(h.chars))} tokens), so the model can follow up on them.`
        : `The last ${h.sent} of ${h.of} earlier messages went in; older ones didn’t fit the conversation budget, so the model doesn’t see them.`,
  );
  out.push(`In all the prompt is about ${nf(c.tokens)} tokens, ${Math.round((c.tokens / c.nCtx) * 100)}% of the window, and the model reads it before writing its first word.`);
  return out;
}

/** A bar for how much of a budget was used. */
function BudgetBar({ label, used, of, unit }: { label: string; used: number; of: number; unit: string }) {
  const share = of > 0 ? Math.min(1, used / of) : 0;
  return (
    <li>
      <span className="ag-bar-label">{label}</span>
      <span className="ag-bar-track">
        <i style={{ width: `${Math.max(1, share * 100)}%` }} />
      </span>
      <span className="ag-bar-value mono">
        {nf(used)} / {nf(of)} {unit}
      </span>
    </li>
  );
}

const PASSAGE_STATUS: Record<ContextRecord['passages'][number]['status'], string> = { in: 'whole', clipped: 'cut to fit', 'left out': 'left out' };

/** The "Assemble context" step: what the chat model was sent, what was cut or left out, and the exact system prompt. */
export function ContextDialog({ c, model, onClose }: { c: ContextRecord; model: string; onClose: () => void }) {
  const passageChars = c.passages.reduce((n, p) => n + p.used, 0);
  return (
    <Modal open onClose={onClose} wide label="Assemble context">
      <div className="dd-head">
        <div>
          <div className="label violet">Assemble context</div>
          <h3>What the chat model read</h3>
          <div className="faint ag-small">
            {model} · {c.messages.length} message{c.messages.length === 1 ? '' : 's'} · ~{nf(c.tokens)} tokens of a {nf(c.nCtx)}-token window
          </div>
        </div>
        <button className="btn ghost sm" aria-label="Close dialog" onClick={onClose}>
          <X size={15} />
        </button>
      </div>

      <section className="dd-section" aria-label="What it did">
        <h4 className="label">What it did</h4>
        <ol className="dd-why">
          {explainContext(c, model).map((line, i) => (
            <li key={i} className="selectable">
              {line}
            </li>
          ))}
        </ol>
      </section>

      <div className="dd-grid">
        <section className="dd-section" aria-label="Budgets">
          <h4 className="label">Budgets</h4>
          <ul className="ag-bars dd-budgets">
            <BudgetBar label="Passages" used={passageChars} of={c.budget.context} unit="chars" />
            <BudgetBar label="Conversation" used={c.history.chars} of={c.budget.history} unit="chars" />
            <BudgetBar label="Whole prompt" used={c.tokens} of={c.nCtx - c.replyTokens} unit="tokens" />
          </ul>
          <div className="faint ag-seed">Tokens are estimated at {CHARS_PER_TOKEN} characters each; the reply’s {nf(c.replyTokens)} tokens are kept free.</div>
        </section>
        <section className="dd-section" aria-label="Passages">
          <h4 className="label">Passages</h4>
          {c.passages.length ? (
            <ul className="ag-rel-list dd-passages">
              {c.passages.map((p) => (
                <li key={p.n} className={p.status === 'left out' ? 'dropped' : 'kept'}>
                  {p.status === 'left out' ? <X size={13} color="var(--text-4)" /> : <Check size={13} color="var(--violet)" />}
                  <span className="ag-rel-file ellipsis" title={p.source}>
                    <span className="mono">[{p.n}] </span>
                    {p.source}
                  </span>
                  <span className="ag-bar-value mono">{nf(p.used)}</span>
                  <span className="ag-rel-why faint">{PASSAGE_STATUS[p.status]}</span>
                </li>
              ))}
            </ul>
          ) : (
            <div className="ag-small faint">None.</div>
          )}
        </section>
      </div>

      <section className="dd-section" aria-label="Messages sent">
        <h4 className="label">Messages sent, in order</h4>
        <ol className="dd-messages mono ag-small">
          {c.messages.map((m, i) => (
            <li key={i}>
              <span className="faint">{i === 0 ? 'system' : i === c.messages.length - 1 ? `${m.role} (the question)` : m.role}</span> · {nf(m.chars)} chars
            </li>
          ))}
        </ol>
      </section>

      <details className="ag-raw dd-raw">
        <summary>Exact system prompt ({nf(c.system.length)} characters)</summary>
        <Field k="System prompt" copy={c.system}>
          <pre className="ag-pre selectable">{c.system}</pre>
        </Field>
      </details>
      <div className="dd-foot">
        <CopyButton text={c.system} label="Copy system prompt">
          Copy system prompt
        </CopyButton>
        <button className="btn primary sm" onClick={onClose}>
          Close
        </button>
      </div>
    </Modal>
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
/** A tool call in one line: title, arguments, status and what it found. */
export function callChipText(c: ToolCallRecord): string {
  return `${c.title}${c.args ? ` (${argsInline(c.args, 48)})` : ''} · ${CALL_PILL[c.status][0].toLowerCase()}${
    c.status === 'done' && c.observation ? ` — ${c.observation.slice(0, 60)}${c.observation.length > 60 ? '…' : ''}` : ''
  }`;
}

/** Whether a call is still under way (writing arguments, running, or waiting for approval). */
export const callLive = (c: ToolCallRecord) => c.status === 'running' || c.status === 'filling' || c.status === 'awaiting';

/** The DOM id of an agent step's card in the Execution Trace, so a chip can point at it. */
export const traceCallId = (msgId: string, stepId: string) => `trace-${msgId}-call-${stepId}`;

/**
 * The tool calls as chips, numbered under the plan step (`planNo`.n, as the
 * trace nests them); `onPick` makes each one point at its card in the trace.
 */
export function ToolChips({ m, planNo, onPick }: { m: Message; planNo?: number; onPick?: (domId: string) => void }) {
  const calls = (m.agent ?? []).filter((s) => s.call);
  if (!calls.length) return null;
  return (
    <>
      {calls.map((s) => {
        const c = s.call!;
        const bad = c.status === 'error' || c.status === 'denied';
        const live = callLive(c);
        const no = planNo != null ? `${String(planNo).padStart(2, '0')}.${s.index + 1}` : null;
        const chip = (
          <>
            {no && <span className="step-no">{no}</span>}
            {c.status === 'awaiting' ? <Hand size={14} /> : <Wrench size={14} />}
            <span>{callChipText(c)}</span>
          </>
        );
        return (
          <div key={s.id} className="cc-chip-row">
            <span className={`dot${live ? ' pulse' : ''}`} style={{ color: 'var(--violet)' }} />
            {onPick ? (
              <button type="button" className="cc-chip" style={{ color: bad ? 'var(--red)' : 'var(--violet)' }} aria-label={`Show step ${no ?? s.index + 1} in the trace: ${c.title}`} onClick={() => onPick(traceCallId(m.id, s.id))}>
                {chip}
              </button>
            ) : (
              <div className="cc-chip" style={{ color: bad ? 'var(--red)' : 'var(--violet)' }}>
                {chip}
              </div>
            )}
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
    <button className="btn ghost sm" onClick={() => copyTrace(m, question)} title="Copy this turn’s full trace as JSON" aria-label="Copy trace">
      <Copy size={12} />
    </button>
  );
}
