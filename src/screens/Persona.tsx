import { BrainCircuit, CheckCircle2, FileText, RotateCcw, ShieldCheck, Sparkles, Square, Zap } from 'lucide-react';
import { motion } from 'motion/react';
import { useRef, useState } from 'react';
import { Slider, Toggle } from '../components/ui';
import { chat, isAbort, loadedModel, useEngine } from '../llm/engine';
import { splitThink } from '../state/chat';
import { DEFAULT_PROMPT, TONES, usePersona, type Tone } from '../state/persona';
import { toast } from '../state/ui';

export function Persona() {
  const p = usePersona();
  const model = useEngine((s) => s.loadedId);
  const generating = useEngine((s) => s.generating);
  const [optimizing, setOptimizing] = useState(false);
  const abort = useRef<AbortController | null>(null);
  const [editingName, setEditingName] = useState(false);

  const optimize = async () => {
    if (optimizing) return abort.current?.abort();
    if (!loadedModel()) return toast({ tone: 'warn', title: 'Load a model first', body: 'Auto-optimize uses the local model.' });
    const before = p.systemPrompt;
    abort.current = new AbortController();
    setOptimizing(true);
    let out = '';
    try {
      for await (const ev of chat(
        [
          {
            role: 'system',
            content:
              'You rewrite system prompts for AI assistants. Return ONLY the improved prompt text — no preamble, no quotes, no markdown headings. Keep it under 90 words, second person ("You are…"), and preserve the agent name and every constraint.',
          },
          { role: 'user', content: `Improve this system prompt for clarity and precision:\n\n${before}` },
        ],
        { temperature: 0.4, maxTokens: 220, thinking: false, signal: abort.current.signal },
      )) {
        if (ev.type === 'delta') {
          out += ev.text;
          const answer = splitThink(out).answer.trim();
          if (answer) p.set({ systemPrompt: answer });
        }
      }
      const final = splitThink(out).answer.trim().replace(/^["“]|["”]$/g, '');
      if (final.length > 20) {
        p.set({ systemPrompt: final });
        toast({ tone: 'ok', title: 'Prompt optimized', body: 'Review it — Undo restores the previous version.' });
        setUndo(before);
      } else {
        p.set({ systemPrompt: before });
        toast({ tone: 'warn', title: 'The model returned nothing usable', body: 'Your prompt was left unchanged.' });
      }
    } catch (e) {
      p.set({ systemPrompt: before });
      if (!isAbort(e)) toast({ tone: 'error', title: 'Optimize failed', body: e instanceof Error ? e.message : String(e) });
    } finally {
      setOptimizing(false);
      abort.current = null;
    }
  };
  const [undo, setUndo] = useState<string | null>(null);

  return (
    <div className="screen">
      <div className="page ps">
        <div className="ps-hero">
          <motion.div
            className="ps-orb-frame"
            initial={{ opacity: 0, scale: 0.96 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.4 }}
          >
            <div className="ps-orb-well">
              <div className={`ps-orb${generating || optimizing ? ' live' : ''}`}>
                <Zap size={64} fill="currentColor" strokeWidth={1.2} />
              </div>
            </div>
          </motion.div>
          <h1 className="display">
            {editingName ? (
              <input
                autoFocus
                className="ps-name-input"
                defaultValue={p.agentName}
                maxLength={20}
                onBlur={(e) => {
                  const v = e.target.value.trim().toUpperCase().replace(/\s+/g, '_');
                  if (v) p.set({ agentName: v });
                  setEditingName(false);
                }}
                onKeyDown={(e) => e.key === 'Enter' && (e.target as HTMLInputElement).blur()}
              />
            ) : (
              <span className="ps-name" onClick={() => setEditingName(true)} title="Rename agent">
                {p.agentName}
              </span>
            )}{' '}
            <span className="ps-accent">Persona</span>
          </h1>
          <p>Configure the neural architecture and communicative heuristics of your active agent core.</p>
        </div>

        <div className="ps-grid">
          <div className="ps-left">
            <div className="panel pad ps-prompt">
              <div className="panel-head">
                <FileText size={22} color="var(--blue)" />
                <h3>System Prompt</h3>
                <div className="right" style={{ display: 'flex', gap: 8 }}>
                  {undo && !optimizing && (
                    <button
                      className="btn ghost sm"
                      onClick={() => {
                        p.set({ systemPrompt: undo });
                        setUndo(null);
                      }}
                    >
                      <RotateCcw size={12} /> Undo
                    </button>
                  )}
                  <button
                    className="btn ghost sm"
                    disabled={optimizing}
                    onClick={() => {
                      p.set({ systemPrompt: DEFAULT_PROMPT });
                      setUndo(null);
                    }}
                  >
                    Reset
                  </button>
                </div>
              </div>
              <div className="ps-editor">
                <textarea
                  value={p.systemPrompt}
                  onChange={(e) => {
                    p.set({ systemPrompt: e.target.value });
                    setUndo(null);
                  }}
                  readOnly={optimizing}
                  spellCheck={false}
                />
                <div className="ps-editor-foot">
                  <span className="faint mono" style={{ fontSize: 11 }}>
                    {p.systemPrompt.length} chars · ~{Math.ceil(p.systemPrompt.length / 3.2)} tokens
                  </span>
                  <button
                    className="ps-optimize"
                    onClick={() => void optimize()}
                    disabled={!model || (generating && !optimizing)}
                    title={model ? 'Rewrite with the local model' : 'Load a model to use auto-optimize'}
                  >
                    {optimizing ? <Square size={11} fill="currentColor" /> : <Sparkles size={12} />}
                    {optimizing ? 'Stop' : 'Auto-optimize'}
                  </button>
                </div>
              </div>
            </div>

            <div className="ps-toggles">
              <ToggleCard
                icon={<ShieldCheck size={22} />}
                title="Always Ask Approval"
                body="Verify actions before execution"
                checked={p.approval}
                onChange={(v) => p.set({ approval: v })}
                color="var(--blue)"
              />
              <ToggleCard
                icon={<BrainCircuit size={22} />}
                title="Verbose Reasoning"
                body="Expose step-by-step thinking"
                checked={p.verbose}
                onChange={(v) => p.set({ verbose: v })}
                color="var(--violet)"
              />
            </div>
          </div>

          <div className="ps-right">
            <div className="panel pad">
              <div className="label violet" style={{ fontSize: 14, marginBottom: 20 }}>
                Tone of voice
              </div>
              <div className="ps-tones">
                {(Object.keys(TONES) as Tone[]).map((t) => (
                  <button key={t} className="ps-tone" aria-pressed={p.tone === t} onClick={() => p.set({ tone: t })}>
                    <div>
                      <div className="ps-tone-name">{TONES[t].label}</div>
                      <div className="ps-tone-blurb">{TONES[t].blurb}</div>
                    </div>
                    {p.tone === t && <CheckCircle2 size={18} fill="var(--blue)" color="var(--panel)" />}
                  </button>
                ))}
              </div>
            </div>

            <div className="panel pad">
              <div className="ps-slider-head">
                <span className="label blue" style={{ fontSize: 14 }}>
                  Temperature
                </span>
                <span className="ps-value">{p.temperature.toFixed(2)}</span>
              </div>
              <Slider label="Temperature" min={0} max={1.5} step={0.01} value={p.temperature} onChange={(v) => p.set({ temperature: v })} />
              <div className="ps-scale">
                <span>Deterministic</span>
                <span>Chaotic</span>
              </div>

              <div className="ps-slider-head" style={{ marginTop: 26 }}>
                <span className="label blue" style={{ fontSize: 14 }}>
                  Max response
                </span>
                <span className="ps-value">{p.maxTokens}</span>
              </div>
              <Slider label="Max tokens" min={128} max={2048} step={64} value={p.maxTokens} onChange={(v) => p.set({ maxTokens: v })} />
              <div className="ps-scale">
                <span>Terse</span>
                <span>Expansive</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function ToggleCard({
  icon,
  title,
  body,
  checked,
  onChange,
  color,
}: {
  icon: React.ReactNode;
  title: string;
  body: string;
  checked: boolean;
  onChange: (v: boolean) => void;
  color: string;
}) {
  return (
    <div className="panel ps-toggle">
      <div className="ps-toggle-icon" style={{ color }}>
        {icon}
      </div>
      <div style={{ flex: 1 }}>
        <div className="ps-toggle-title">{title}</div>
        <div className="ps-toggle-body">{body}</div>
      </div>
      <Toggle checked={checked} onChange={onChange} label={title} />
    </div>
  );
}
