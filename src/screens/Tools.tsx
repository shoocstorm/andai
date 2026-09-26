// Agent tools: what the agent can use, what each one runs, and the user's
// policy for it. Everything shown here is the live registry
// (agent/tools/registry.ts), not a description of it.

import { Cpu, ShieldCheck, Wrench } from 'lucide-react';
import { TOOLS, policyOf } from '../agent/tools/registry';
import type { Policy } from '../agent/tools/types';
import { Segmented, Slider, Toggle, fmtTime } from '../components/ui';
import { useEngine } from '../llm/engine';
import { modelById } from '../llm/models';
import { useKb } from '../state/kb';
import { setAgent, setPolicy, TOOL_DEFAULTS, useTools } from '../state/tools';
import { useUi } from '../state/ui';

const KIND_LABEL = { document: 'Documents', code: 'Code', mixed: 'Mixed' } as const;

export function Tools() {
  const t = useTools();
  const go = useUi((s) => s.go);
  const kb = useKb((s) => s.kbs.find((k) => k.slug === s.grounding));
  const chatId = useEngine((s) => s.loadedId);
  const decider = useEngine((s) => s.decider);
  const deciderName = modelById(decider.loadedId)?.name;
  const chatName = modelById(chatId)?.name;

  return (
    <div className="screen">
      <div className="page tl">
        <div className="page-hero" style={{ marginBottom: 30 }}>
          <h1>
            Agent <span className="grad-text">Tools</span>
          </h1>
          <p>
            The tools the agent may use to answer from your knowledge base. At each step it chooses one (or answers), and
            every choice and call is in the Execution Trace.
          </p>
        </div>

        <div className="tl-top">
          <div className="panel pad">
            <div className="panel-head">
              <Cpu size={20} color="var(--blue)" />
              <h3>Agent loop</h3>
              <span className={`right pill ${t.agentMode ? 'blue' : ''}`}>{t.agentMode ? 'on' : 'off'}</span>
            </div>
            <div className="tl-row">
              <div>
                <div className="tl-row-title">Agent mode</div>
                <div className="faint tl-small">
                  Off: each question runs one knowledge search with its own wording, then answers.
                </div>
              </div>
              <Toggle checked={t.agentMode} onChange={(agentMode) => setAgent({ agentMode })} label="Agent mode" />
            </div>
            <div className="tl-row col">
              <div className="tl-row-head">
                <span className="tl-row-title">Tool calls per question</span>
                <span className="mono">{t.maxSteps}</span>
              </div>
              <Slider value={t.maxSteps} min={1} max={20} step={1} onChange={(maxSteps) => setAgent({ maxSteps })} label="Tool calls per question" />
            </div>
            <div className="tl-row col">
              <div className="tl-row-head">
                <span className="tl-row-title">Minimum confidence</span>
                <span className="mono">{Math.round(t.minConfidence * 100)}%</span>
              </div>
              <Slider
                value={t.minConfidence}
                min={0}
                max={0.9}
                step={0.05}
                onChange={(minConfidence) => setAgent({ minConfidence })}
                label="Minimum confidence"
              />
              <div className="faint tl-small">
                Below this, a tool choice isn’t acted on: the agent searches once if it hasn’t yet, or answers.
              </div>
            </div>
            <div className="tl-row">
              <div>
                <div className="tl-row-title">Decisions made by</div>
                <div className="faint tl-small">
                  {deciderName
                    ? `${deciderName} (decision model)`
                    : chatName
                      ? `${chatName} (the chat model; no decision model is loaded)`
                      : 'No model loaded'}
                </div>
              </div>
              <button className="btn ghost sm" onClick={() => go('settings')}>
                Models
              </button>
            </div>
            {(t.maxSteps !== TOOL_DEFAULTS.maxSteps || t.minConfidence !== TOOL_DEFAULTS.minConfidence) && (
              <button className="btn ghost sm" onClick={() => setAgent({ maxSteps: TOOL_DEFAULTS.maxSteps, minConfidence: TOOL_DEFAULTS.minConfidence })}>
                Reset to defaults
              </button>
            )}
          </div>

          <div className="panel pad">
            <div className="panel-head">
              <ShieldCheck size={20} color="var(--green)" />
              <h3>What tools can do</h3>
            </div>
            <ul className="tl-rules">
              <li>
                <b>Read only.</b> Every tool reads the knowledge base the chat is grounded in, and nothing else. None can
                write, delete, or reach the network.
              </li>
              <li>
                <b>Checked on this computer.</b> The app re-validates every call before ug runs it, stops it after 20 s and caps
                its output.
              </li>
              <li>
                <b>You decide.</b> <i>Auto</i> runs a tool without asking, <i>Ask</i> shows an approval card in the chat,
                and <i>Off</i> never offers it to the model.
              </li>
              <li>
                <b>Nothing hidden.</b> The trace shows each decision’s options and scores, and each call’s arguments,
                command, timing and output.
              </li>
            </ul>
            <div className="faint tl-small">
              {kb
                ? `The chat is grounded in “${kb.name}”, a ${KIND_LABEL[kb.kind].toLowerCase()} knowledge base.`
                : 'No knowledge base is selected in the chat, so the agent has no tools to use.'}
            </div>
          </div>
        </div>

        <div className="section-title">
          <span className="label" style={{ color: 'var(--text)', letterSpacing: '0.2em' }}>
            Tools · {TOOLS.length}
          </span>
        </div>
        <div className="tl-grid">
          {TOOLS.map((tool) => {
            const policy = policyOf(tool, t.policies);
            const stats = t.stats[tool.id];
            const fits = kb ? tool.kinds.includes(kb.kind) : true;
            return (
              <article key={tool.id} className={`panel tl-card${policy === 'off' ? ' off' : ''}`} aria-label={tool.title}>
                <div className="tl-card-top">
                  <Wrench size={18} color="var(--violet)" />
                  <h3>{tool.title}</h3>
                  <span className="pill green">{tool.risk === 'read' ? 'Read-only' : tool.risk}</span>
                </div>
                <p className="muted">{tool.description}</p>
                <div className="tl-kinds">
                  {tool.kinds.map((k) => (
                    <span key={k} className={`pill${kb?.kind === k ? ' violet' : ''}`}>
                      {KIND_LABEL[k]}
                    </span>
                  ))}
                  {kb && !fits && <span className="faint tl-small">not offered for “{kb.name}”</span>}
                </div>
                <div className="tl-field">
                  <div className="label">Runs</div>
                  <code className="tl-cmd">{tool.command}</code>
                </div>
                <div className="tl-field">
                  <div className="label">Offered to the model as</div>
                  <div className="tl-small">“{tool.option}”</div>
                </div>
                <div className="tl-foot">
                  <Segmented<Policy>
                    label={`${tool.title} policy`}
                    value={policy}
                    onChange={(p) => setPolicy(tool.id, p)}
                    options={[
                      { value: 'auto', label: 'Auto' },
                      { value: 'ask', label: 'Ask' },
                      { value: 'off', label: 'Off' },
                    ]}
                  />
                  <span className="faint tl-small mono">
                    {stats
                      ? `${stats.calls} call${stats.calls === 1 ? '' : 's'} · ${stats.errors} failed · avg ${Math.round(stats.totalMs / stats.calls)} ms${stats.lastAt ? ` · last ${fmtTime(stats.lastAt)}` : ''}`
                      : 'not used yet'}
                  </span>
                </div>
              </article>
            );
          })}
        </div>
      </div>
    </div>
  );
}
