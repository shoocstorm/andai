import { Check, Cpu, Download, GitFork, HardDrive, Loader2, Monitor, Moon, Palette, Power, Radio, Sun, Trash2, X } from 'lucide-react';
import { useState } from 'react';
import { Bar, Modal, fmtBytes } from '../components/ui';
import { evictModel, loadDecider, loadModel, removeLegacyCopies, unloadDecider, unloadModel, useEngine } from '../llm/engine';
import { MODELS, modelById, type ModelDef } from '../llm/models';
import { clearChat } from '../state/chat';
import { useKb } from '../state/kb';
import { useTheme, type ThemeMode } from '../state/theme';
import { toast } from '../state/ui';

/**
 * The optional second model that scores the agent's next action
 * (llm/decide.ts); without one, the chat model decides. SemIf measured bigger
 * deciders picking better, but in Andai's eval Qwen3 1.7B deciding asked for
 * clarification instead of answering after most searches, while 1.7B
 * answering with 0.6B deciding did best (docs/agentic-rag-improvements.md,
 * item 8), so the copy recommends a small decider.
 */
function DecisionModel() {
  const e = useEngine();
  const d = e.decider;
  const chat = modelById(e.loadedId);
  const candidates = MODELS.filter((m) => m.decider);
  return (
    <section className="panel pad st-decider" aria-label="Decision model">
      <div className="panel-head">
        <GitFork size={20} color="var(--violet)" />
        <h3>Decision model</h3>
        <span className={`right pill ${d.status === 'ready' ? 'violet' : ''}`}>
          {d.status === 'ready' ? modelById(d.loadedId)?.name : d.status === 'loading' ? 'loading' : 'using chat model'}
        </span>
      </div>
      <p className="muted" style={{ marginTop: 0, fontSize: 13.5 }}>
        In agent mode, a model picks each next step (a tool, answer or ask) by scoring the options in one pass. A separate,
        small model can decide while a larger one writes: Qwen3 0.6B deciding with Qwen3 1.7B as the chat model gave the best
        answers in our tests. Without one, {chat ? chat.name : 'the chat model'} decides. It runs in addition to the chat model,
        so it needs its own memory.
      </p>
      <div className="st-decider-list">
        {candidates.map((m) => {
          const loaded = d.loadedId === m.id;
          const loading = d.loadingId === m.id;
          const pct = loading && d.progress?.total ? d.progress.loaded / d.progress.total : 0;
          return (
            <div key={m.id} className="st-decider-row">
              <div style={{ flex: 1 }}>
                <div style={{ fontWeight: 600 }}>{m.name}</div>
                <div className="faint mono" style={{ fontSize: 11.5 }}>
                  {m.size} · {e.cached[m.url] ? 'cached' : 'download'}
                  {loading && d.progress ? ` · ${d.progress.phase} ${Math.round(pct * 100)}%` : ''}
                </div>
              </div>
              {loaded ? (
                <button className="btn secondary sm" onClick={() => void unloadDecider()}>
                  <Power size={13} /> Unload
                </button>
              ) : (
                <button className="btn secondary sm" disabled={d.status === 'loading'} onClick={() => void loadDecider(m.id)}>
                  {loading ? <Loader2 size={13} className="spin" /> : <Power size={13} />}
                  {loading ? 'Loading' : `Use for decisions`}
                </button>
              )}
            </div>
          );
        })}
      </div>
      {d.error && (
        <div className="kn-error" style={{ marginTop: 12 }}>
          <X size={15} /> Decision model failed to load: {d.error}
        </div>
      )}
    </section>
  );
}

export function Settings() {
  const e = useEngine();
  const ug = useKb((s) => s.ug);
  const kbs = useKb((s) => s.kbs);
  const [legacyFor, setLegacyFor] = useState<ModelDef | null>(null);

  return (
    <div className="screen">
      <div className="page st">
        <div className="page-hero" style={{ marginBottom: 34 }}>
          <h1>
            Neural <span className="grad-text">Core</span>
          </h1>
          <p>Models run inside Andai with wllama — llama.cpp compiled to WebAssembly. Each downloads once, then works offline.</p>
        </div>

        <div className="section-title">
          <span className="label" style={{ color: 'var(--text)', letterSpacing: '0.2em' }}>
            Model Registry
          </span>
        </div>
        <section className="st-models" aria-label="Model registry">
          {MODELS.map((m) => {
            const cached = e.cached[m.url];
            const loaded = e.loadedId === m.id;
            const loading = e.loadingId === m.id;
            const p = loading ? e.progress : null;
            const pct = p && p.total ? p.loaded / p.total : 0;
            return (
              <div key={m.id} className={`panel st-model${loaded ? ' loaded' : ''}`}>
                <div className="st-model-top">
                  <div className="st-model-icon">
                    <Cpu size={20} />
                  </div>
                  {loaded ? (
                    <span className="pill blue">
                      <span className="dot pulse" /> Online
                    </span>
                  ) : cached ? (
                    <span className="pill violet">Cached · {fmtBytes(cached)}</span>
                  ) : (
                    <span className="pill">{m.size}</span>
                  )}
                </div>
                <div className="st-model-name display">{m.name}</div>
                <div className="mono faint" style={{ fontSize: 11.5, letterSpacing: '0.04em' }}>
                  {m.family} · ctx {m.n_ctx.toLocaleString()}
                  {m.thinking ? ' · reasoning' : ''}
                </div>
                <p className="muted">{m.note}</p>
                {e.legacy[m.id] && (
                  <div className="st-legacy">
                    <span>
                      An older copy ({fmtBytes(e.legacy[m.id])}) from before downloads were pinned and verified is still
                      cached.
                    </span>
                    <button className="btn ghost sm" onClick={() => setLegacyFor(m)}>
                      <Trash2 size={13} /> Remove old copy
                    </button>
                  </div>
                )}
                {loading && (
                  <div className="st-progress">
                    <Bar value={pct} indeterminate={!pct} />
                    <div className="st-progress-text mono">
                      <span>{p?.phase}</span>
                      <span>
                        {fmtBytes(p?.loaded ?? 0)} / {fmtBytes(p?.total ?? m.bytes)}
                        {p && p.speed > 0 && pct < 1 ? ` · ${fmtBytes(p.speed)}/s` : ''}
                      </span>
                    </div>
                  </div>
                )}
                <div className="st-model-actions">
                  {loaded ? (
                    <button className="btn secondary sm" onClick={() => void unloadModel()}>
                      <Power size={13} /> Unload
                    </button>
                  ) : (
                    <button
                      className={`btn sm ${cached ? 'secondary' : 'primary'}`}
                      disabled={e.status === 'loading'}
                      onClick={() => void loadModel(m.id)}
                    >
                      {loading ? <Loader2 size={13} className="spin" /> : cached ? <Power size={13} /> : <Download size={13} />}
                      {loading ? 'Loading' : cached ? 'Load' : 'Download & load'}
                    </button>
                  )}
                  {cached && !loading && (
                    <button
                      className="btn ghost sm"
                      title="Delete the cached model file"
                      onClick={() => {
                        void evictModel(m.id).then(() => toast({ tone: 'info', title: `${m.name} removed from cache` }));
                      }}
                    >
                      <Trash2 size={13} /> Evict
                    </button>
                  )}
                </div>
              </div>
            );
          })}
        </section>
        <Modal open={!!legacyFor} onClose={() => setLegacyFor(null)}>
          <h3>Remove the old copy of {legacyFor?.name}?</h3>
          <p className="muted" style={{ margin: '4px 0 22px' }}>
            It was downloaded before Andai pinned and verified model files, so it isn’t used anymore. Removing it frees{' '}
            {fmtBytes(legacyFor ? (e.legacy[legacyFor.id] ?? 0) : 0)}. The verified copy, if you have one, stays.
          </p>
          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10 }}>
            <button className="btn ghost" onClick={() => setLegacyFor(null)}>
              Cancel
            </button>
            <button
              className="btn danger"
              onClick={() => {
                const m = legacyFor;
                setLegacyFor(null);
                if (m) void removeLegacyCopies(m.id).then(() => toast({ tone: 'info', title: `Old copy of ${m.name} removed` }));
              }}
            >
              Remove
            </button>
          </div>
        </Modal>
        {e.error && (
          <div className="kn-error" style={{ marginTop: 16 }}>
            <X size={15} /> Model load failed: {e.error}
          </div>
        )}

        <DecisionModel />

        <div className="st-grid">
          <Appearance />
          <div className="panel pad">
            <div className="panel-head">
              <Cpu size={20} color="var(--blue)" />
              <h3>Inference engine</h3>
              <span className="right pill">{e.info ? (e.info.compat ? 'compat build' : 'standard build') : 'idle'}</span>
            </div>
            <dl className="st-dl">
              <Row k="Backend" v={e.info?.backend ?? '—'} />
              <Row k="Threads" v={e.info?.threads ?? `${e.caps.cores} available`} />
              <Row k="Context" v={e.info?.context ?? '—'} />
              <Row k="Layers" v={e.info?.layers ?? '—'} />
              <Row k="Architecture" v={e.info?.arch ?? '—'} />
              <Row k="libllama" v={e.info?.libllama ?? '—'} />
              <Row k="Last load" v={e.lastLoadMs ? `${(e.lastLoadMs / 1000).toFixed(1)} s` : '—'} />
            </dl>
            <div className="st-caps">
              <Cap on={e.caps.isolated} label="Cross-origin isolated" />
              <Cap on={e.caps.sharedArrayBuffer} label="SharedArrayBuffer" />
              <Cap on={e.caps.opfs} label="OPFS cache" />
              <Cap on={e.caps.webgpu} label="WebGPU" />
            </div>
          </div>

          <div className="panel pad">
            <div className="panel-head">
              <Radio size={20} color="var(--violet)" />
              <h3>Knowledge engine</h3>
              <span className={`right pill ${ug?.found ? 'green' : 'amber'}`}>{ug?.found ? 'connected' : 'missing'}</span>
            </div>
            <dl className="st-dl">
              <Row k="Engine" v="ug · GraphRAG" />
              <Row k="Version" v={ug?.version?.replace('ug version ', '') ?? '—'} />
              <Row k="Binary" v={ug?.path ?? 'not found'} mono />
              <Row k="Embedder" v="bge-small-en-v1.5 (local)" />
              <Row k="Knowledge bases" v={kbs.length} />
              <Row k="Graph nodes" v={kbs.reduce((n, k) => n + k.nodes, 0).toLocaleString()} />
              <Row k="On disk" v={fmtBytes(kbs.reduce((n, k) => n + k.sizeBytes, 0))} />
            </dl>
          </div>

          <div className="panel pad">
            <div className="panel-head">
              <HardDrive size={20} color="var(--amber)" />
              <h3>Data</h3>
            </div>
            <p className="muted" style={{ marginTop: 0, fontSize: 13.5 }}>
              Conversations, persona and settings live in this app's local storage. Models are cached in the webview's
              private file system; knowledge graphs in <span className="mono">~/.ug/andai-*</span>.
            </p>
            <button
              className="btn danger sm"
              onClick={() => {
                clearChat();
                toast({ tone: 'info', title: 'Conversation history cleared' });
              }}
            >
              <Trash2 size={13} /> Clear conversation history
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function Row({ k, v, mono }: { k: string; v: React.ReactNode; mono?: boolean }) {
  return (
    <>
      <dt>{k}</dt>
      <dd className={mono ? 'mono' : undefined} title={typeof v === 'string' ? v : undefined}>
        {v}
      </dd>
    </>
  );
}

function Cap({ on, label }: { on: boolean; label: string }) {
  return (
    <span className={`pill ${on ? 'green' : 'red'}`}>
      {on ? <Check size={11} /> : <X size={11} />} {label}
    </span>
  );
}

const APPEARANCE: { mode: ThemeMode; label: string; icon: typeof Sun; blurb: string }[] = [
  { mode: 'system', label: 'System', icon: Monitor, blurb: 'Follow the system' },
  { mode: 'light', label: 'Light', icon: Sun, blurb: 'Bright workspace' },
  { mode: 'dark', label: 'Dark', icon: Moon, blurb: 'Command deck' },
];

function Appearance() {
  const { mode, resolved, setMode } = useTheme();
  return (
    <div className="panel pad">
      <div className="panel-head">
        <Palette size={20} color="var(--blue)" />
        <h3>Appearance</h3>
        <span className="right pill">{resolved}</span>
      </div>
      <div className="st-themes" role="radiogroup" aria-label="Appearance">
        {APPEARANCE.map((a) => (
          <button
            key={a.mode}
            role="radio"
            aria-checked={mode === a.mode}
            className="st-theme"
            data-preview={a.mode}
            onClick={() => setMode(a.mode)}
          >
            <span className="st-theme-swatch" data-preview={a.mode}>
              <i />
              <i />
              <i />
            </span>
            <span className="st-theme-label">
              <a.icon size={14} /> {a.label}
            </span>
            <span className="faint" style={{ fontSize: 11.5 }}>
              {a.blurb}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}
