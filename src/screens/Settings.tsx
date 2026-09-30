import { AlertTriangle, Check, CheckCircle2, Cpu, Download, FolderOpen, GitFork, HardDrive, Loader2, Moon, Palette, Plus, Power, Radio, ScrollText, Sun, Trash2, X, Zap } from 'lucide-react';
import { useEffect, useState } from 'react';
import { isTauri } from '@tauri-apps/api/core';
import { Bar, Modal, Toggle, fmtBytes } from '../components/ui';
import { evictModel, loadDecider, loadModel, refreshLaya, refreshNative, removeLaya, removeLegacyCopies, unloadDecider, unloadModel, useEngine } from '../llm/engine';
import { removeCustomModel } from '../llm/custom';
import { availableModels, isMlx, LAYA_MODELS, layaById, memoryFit, modelById, type LayaDef, type ModelDef } from '../llm/models';
import { activityInfo, clearActivity, openActivityFolder, useActivity, type ActivityInfo } from '../state/activity';
import { clearChat } from '../state/chat';
import { useKb } from '../state/kb';
import { useTheme, type Theme } from '../state/theme';
import { toast, useUi } from '../state/ui';
import { HubModels } from './HubModels';
import { GetUltraGraph } from './Knowledge';

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
  const candidates = availableModels(e.native.supported).filter((m) => m.decider);
  const [removing, setRemoving] = useState<LayaDef | null>(null);
  useEffect(() => {
    void refreshLaya();
  }, []);
  // Laya runs on MLX: only offered where Rust reports it (Apple Silicon).
  const layas = e.laya.supported ? LAYA_MODELS.flatMap((m) => {
    const c = e.laya.checkpoints.find((x) => x.id === m.id);
    return c ? [{ m, c }] : [];
  }) : [];
  const loadedName = layaById(d.loadedId)?.name ?? modelById(d.loadedId)?.name;
  const row = (id: string, name: string, meta: string, extra?: React.ReactNode) => {
    const loaded = d.loadedId === id;
    const loading = d.loadingId === id;
    const pct = loading && d.progress?.total ? d.progress.loaded / d.progress.total : 0;
    return (
      <div key={id} className={`st-decider-row${loaded ? ' loaded' : ''}`}>
        {loaded && <CheckCircle2 size={18} className="st-check" aria-label="In use for decisions" />}
        <div style={{ flex: 1 }}>
          <div style={{ fontWeight: 600 }}>
            {name}
            {loaded && <span className="pill green st-inuse">In use</span>}
          </div>
          <div className="faint mono" style={{ fontSize: 11.5 }}>
            {meta}
            {loading && d.progress ? ` · ${d.progress.phase} ${Math.round(pct * 100)}%` : ''}
          </div>
        </div>
        {extra}
        {loaded ? (
          <button className="btn secondary sm" onClick={() => void unloadDecider()}>
            <Power size={13} /> Unload
          </button>
        ) : (
          <button className="btn secondary sm" disabled={d.status === 'loading'} onClick={() => void loadDecider(id)}>
            {loading ? <Loader2 size={13} className="spin" /> : <Power size={13} />}
            {loading ? 'Loading' : `Use for decisions`}
          </button>
        )}
      </div>
    );
  };
  return (
    <section className="panel pad st-decider" aria-label="Decision model">
      <div className="panel-head">
        <GitFork size={20} color="var(--violet)" />
        <h3>Decision model</h3>
        <span className={`right pill ${d.status === 'ready' ? 'green' : ''}`}>
          {d.status === 'ready' && <Check size={11} />}
          {d.status === 'ready' ? loadedName : d.status === 'loading' ? 'loading' : 'using chat model'}
        </span>
      </div>
      <p className="muted" style={{ marginTop: 0, fontSize: 13.5 }}>
        In agent mode, a model picks each next step (a tool, answer or ask) by scoring the options in one pass. A separate,
        small model can decide while a larger one writes: Qwen3 0.6B deciding with Qwen3 1.7B as the chat model gave the best
        answers in our tests. Without one, {chat ? chat.name : 'the chat model'} decides. It runs in addition to the chat model,
        so it needs its own memory.
      </p>
      <div className="st-decider-list">
        {/* Laya first: built for exactly this choice, and the fastest. */}
        {layas.map(({ m, c }) =>
          row(
            m.id,
            m.name,
            `${m.family} · ${fmtBytes(c.bytes)} · ${c.downloaded ? 'downloaded' : 'download'} — ${m.note}`,
            c.downloaded && d.loadingId !== m.id ? (
              <button className="btn ghost sm" aria-label={`Remove ${m.name}`} onClick={() => setRemoving(m)}>
                <Trash2 size={13} />
              </button>
            ) : null,
          ),
        )}
        {candidates.map((m) => row(m.id, m.name, `${m.size} · ${e.cached[m.url] ? 'cached' : 'download'}`))}
      </div>
      {layas.length > 0 && (
        <p className="faint" style={{ margin: '10px 0 0', fontSize: 12 }}>
          Laya models are small encoders built for this kind of choice, run natively on your Mac’s GPU with MLX.
        </p>
      )}
      {d.error && (
        <div className="kn-error" style={{ marginTop: 12 }}>
          <X size={15} /> Decision model failed to load: {d.error}
        </div>
      )}
      <Modal open={!!removing} onClose={() => setRemoving(null)}>
        <h3>Remove {removing?.name}?</h3>
        <p className="muted" style={{ margin: '4px 0 22px' }}>
          This deletes its downloaded files ({fmtBytes(e.laya.checkpoints.find((c) => c.id === removing?.id)?.bytes ?? 0)}). You can
          download it again later.
        </p>
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10 }}>
          <button className="btn ghost" onClick={() => setRemoving(null)}>
            Cancel
          </button>
          <button
            className="btn danger"
            onClick={() => {
              const m = removing;
              setRemoving(null);
              if (m)
                void removeLaya(m.id)
                  .then(() => toast({ tone: 'info', title: `${m.name} removed` }))
                  .catch((err: unknown) => toast({ tone: 'error', title: `Couldn’t remove ${m.name}`, body: String(err) }));
            }}
          >
            Remove
          </button>
        </div>
      </Modal>
    </section>
  );
}

/** Which engine runs a model, on its card: MLX is the fast one on a Mac. */
function EngineChip({ def }: { def: ModelDef }) {
  return isMlx(def) ? (
    <span className="pill green" title="Runs natively on your Mac’s GPU with MLX">
      <Zap size={11} /> MLX · Metal · fastest on this Mac
    </span>
  ) : (
    <span className="pill" title="llama.cpp compiled to WebAssembly, in the app’s webview, on WebGPU">
      wllama · WebGPU
    </span>
  );
}

export function Settings() {
  const e = useEngine();
  const ug = useKb((s) => s.ug);
  const kbs = useKb((s) => s.kbs);
  const [legacyFor, setLegacyFor] = useState<ModelDef | null>(null);
  const [hubOpen, setHubOpen] = useState(false);
  const [removing, setRemoving] = useState<ModelDef | null>(null);
  const mlx = e.native.supported;
  useEffect(() => {
    void refreshNative();
  }, []);

  return (
    <div className="screen">
      <div className="page st">
        <div className="page-hero" style={{ marginBottom: 34 }}>
          <h1>
            Neural <span className="grad-text">Core</span>
          </h1>
          <p>
            Models run inside Andai: natively on your Mac’s GPU with MLX on Apple Silicon, or with wllama (llama.cpp compiled
            to WebAssembly) anywhere. Each downloads once, then works offline.
          </p>
        </div>

        <div className="section-title">
          <span className="label" style={{ color: 'var(--text)', letterSpacing: '0.2em' }}>
            Model Registry
          </span>
          <button className="btn secondary sm" onClick={() => setHubOpen(true)}>
            <Plus size={13} /> Add from Hugging Face
          </button>
        </div>
        {mlx && (
          <div className="st-mlx-note" role="note">
            <Zap size={16} />
            <span>
              <b>MLX models are much faster on this Mac.</b> They run natively on the GPU instead of in the webview: Qwen3 1.7B
              writes about 350 tokens/s with MLX against 30–65 with wllama, and reads a prompt about 50× faster (measured on
              an M5 Max).
            </span>
          </div>
        )}
        <section className="st-models" aria-label="Model registry">
          {availableModels(mlx).map((m) => {
            const cached = e.cached[m.url];
            const loaded = e.loadedId === m.id;
            const loading = e.loadingId === m.id;
            const p = loading ? e.progress : null;
            const pct = p && p.total ? p.loaded / p.total : 0;
            return (
              <div key={m.id} className={`panel st-model${loaded ? ' loaded' : ''}`}>
                <div className="st-model-top">
                  <div className="st-model-icon">
                    {loaded ? <CheckCircle2 size={20} aria-label="Loaded" /> : <Cpu size={20} />}
                  </div>
                  {loaded ? (
                    <span className="pill green">
                      <Check size={11} /> Loaded · in use
                    </span>
                  ) : cached ? (
                    <span className="pill violet">Cached · {fmtBytes(cached)}</span>
                  ) : (
                    <span className="pill">{m.size}</span>
                  )}
                </div>
                <div className="st-model-name display">{m.name}</div>
                <div className="st-model-engine">
                  <EngineChip def={m} />
                </div>
                <div className="mono faint" style={{ fontSize: 11.5, letterSpacing: '0.04em' }}>
                  {m.family} · ctx {m.n_ctx.toLocaleString()}
                  {m.thinking ? ' · reasoning' : ''}
                </div>
                <p className="muted">{m.note}</p>
                {isMlx(m) && memoryFit(m.bytes, e.native.memory).fits === false && (
                  <div className="st-memory" role="note">
                    <AlertTriangle size={13} /> Needs about {memoryFit(m.bytes, e.native.memory).needGb} GB of memory; this Mac has{' '}
                    {Math.round((e.native.memory ?? 0) / 1e9)} GB, so it may not load, or run very slowly.
                  </div>
                )}
                {m.source && (
                  <div className="st-source faint mono" title={`${m.source.repo} at commit ${m.source.commit}`}>
                    <span className="pill">Hugging Face</span> {m.source.repo}@{m.source.commit.slice(0, 7)}
                    {m.source.license ? ` · ${m.source.license}` : ''}
                  </div>
                )}
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
                  {m.source && !loading && (
                    <button className="btn ghost sm" aria-label={`Remove ${m.name}`} onClick={() => setRemoving(m)}>
                      <Trash2 size={13} /> Remove
                    </button>
                  )}
                  {cached && !loading && !m.source && (
                    <button
                      className="btn ghost sm"
                      title={isMlx(m) ? 'Delete the downloaded model files' : 'Delete the cached model file'}
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
        <HubModels open={hubOpen} onClose={() => setHubOpen(false)} />
        <Modal open={!!removing} onClose={() => setRemoving(null)}>
          <h3>Remove {removing?.name}?</h3>
          <p className="muted" style={{ margin: '4px 0 22px' }}>
            This deletes its downloaded files ({removing?.size}) and removes it from the list. You can add it again from Hugging
            Face.
          </p>
          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10 }}>
            <button className="btn ghost" onClick={() => setRemoving(null)}>
              Cancel
            </button>
            <button
              className="btn danger"
              onClick={() => {
                const m = removing;
                setRemoving(null);
                if (m)
                  void removeCustomModel(m.id)
                    .then(() => toast({ tone: 'info', title: `${m.name} removed` }))
                    .catch((err: unknown) => toast({ tone: 'error', title: `Couldn’t remove ${m.name}`, body: String(err) }));
              }}
            >
              Remove
            </button>
          </div>
        </Modal>
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
              <span className="right pill">
                {e.info ? (e.info.backend.startsWith('MLX') ? 'native' : e.info.compat ? 'compat build' : 'standard build') : 'idle'}
              </span>
            </div>
            <dl className="st-dl">
              <Row k="Backend" v={e.info?.backend ?? '—'} />
              <Row k="Threads" v={e.info?.threads ?? `${e.caps.cores} available`} />
              <Row k="Context" v={e.info?.context ?? '—'} />
              <Row k="Layers" v={e.info?.layers ?? '—'} />
              <Row k="Architecture" v={e.info?.arch ?? '—'} />
              <Row k="Runtime" v={e.info?.libllama ?? '—'} />
              <Row k="Last load" v={e.lastLoadMs ? `${(e.lastLoadMs / 1000).toFixed(1)} s` : '—'} />
            </dl>
            <div className="st-caps">
              <Cap on={e.caps.isolated} label="Cross-origin isolated" />
              <Cap on={e.caps.sharedArrayBuffer} label="SharedArrayBuffer" />
              <Cap on={e.caps.opfs} label="OPFS cache" />
              <Cap on={e.caps.webgpu} label="WebGPU" />
              <Cap on={mlx} label="MLX (native)" />
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
            {ug && !ug.found && (
              <div className="st-ug-missing">
                <p className="muted" style={{ margin: '12px 0 10px', fontSize: 13.5 }}>
                  Knowledge bases need <b>ug (UltraGraph)</b>, a fast local knowledge-graph engine. Install it from the
                  UltraGraph website, then restart Andai.
                </p>
                <GetUltraGraph />
              </div>
            )}
          </div>

          <div className="panel pad">
            <div className="panel-head">
              <HardDrive size={20} color="var(--amber)" />
              <h3>Data</h3>
            </div>
            <p className="muted" style={{ marginTop: 0, fontSize: 13.5 }}>
              Conversations, persona and settings live in this app's local storage. wllama models are cached in the webview's
              private file system, MLX models in the app's data folder; knowledge graphs in{' '}
              <span className="mono">~/.ug/andai-*</span>.
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

          <ActivityLog />
        </div>
      </div>
    </div>
  );
}

/**
 * The activity log (state/activity.ts, src-tauri/src/activity.rs): off by
 * default, since it writes the user's questions and passages from their
 * documents to disk, where they outlive a cleared conversation.
 */
function ActivityLog() {
  const { enabled, setEnabled } = useActivity();
  const desktop = isTauri();
  const [info, setInfo] = useState<ActivityInfo | null>(null);
  const [confirming, setConfirming] = useState(false);
  const refresh = () => {
    if (desktop) void activityInfo().then(setInfo, () => setInfo(null));
  };
  useEffect(refresh, [desktop, enabled]);
  const bytes = info?.files.reduce((n, f) => n + f.bytes, 0) ?? 0;
  return (
    <div className="panel pad">
      <div className="panel-head">
        <ScrollText size={20} color="var(--violet)" />
        <h3>Activity log</h3>
        <span className={`right pill ${enabled ? 'green' : ''}`}>{enabled ? 'on' : 'off'}</span>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12 }}>
        <span style={{ fontSize: 14 }}>Keep an activity log</span>
        <Toggle checked={enabled} onChange={setEnabled} label="Keep an activity log" />
      </div>
      <p className="muted" style={{ margin: '10px 0 12px', fontSize: 13.5 }}>
        Writes each step the agent takes (its decisions, every argument writer call, tool calls, the context sent to the
        model and the answer) as JSON lines, one file per day, on this computer. It includes your questions and passages
        from your documents. Files are kept 7 days, and nothing leaves the machine.
      </p>
      {desktop ? (
        <>
          <dl className="st-dl">
            <Row k="Folder" v={info?.dir ?? '—'} mono />
            <Row k="Files" v={info ? `${info.files.length} · ${fmtBytes(bytes)}` : '—'} />
          </dl>
          <div style={{ display: 'flex', gap: 10, marginTop: 12 }}>
            <button className="btn secondary sm" onClick={() => useUi.getState().go('logs')}>
              <ScrollText size={13} /> View log
            </button>
            <button className="btn ghost sm" onClick={() => void openActivityFolder().catch((err: unknown) => toast({ tone: 'error', title: 'Couldn’t open the logs folder', body: String(err) }))}>
              <FolderOpen size={13} /> Open folder
            </button>
            <button className="btn danger sm" disabled={!info?.files.length} onClick={() => setConfirming(true)}>
              <Trash2 size={13} /> Delete logs
            </button>
          </div>
        </>
      ) : (
        <p className="faint" style={{ margin: 0, fontSize: 12.5 }}>
          The log is written by the desktop app.
        </p>
      )}
      <Modal open={confirming} onClose={() => setConfirming(false)}>
        <h3>Delete the activity logs?</h3>
        <p className="muted" style={{ margin: '4px 0 22px' }}>
          This deletes {info?.files.length ?? 0} log file{info?.files.length === 1 ? '' : 's'} ({fmtBytes(bytes)}). Your
          conversations aren’t affected.
        </p>
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10 }}>
          <button className="btn ghost" onClick={() => setConfirming(false)}>
            Cancel
          </button>
          <button
            className="btn danger"
            onClick={() => {
              setConfirming(false);
              void clearActivity()
                .then((n) => toast({ tone: 'info', title: `${n} log file${n === 1 ? '' : 's'} deleted` }))
                .catch((err: unknown) => toast({ tone: 'error', title: 'Couldn’t delete the logs', body: String(err) }))
                .finally(refresh);
            }}
          >
            Delete
          </button>
        </div>
      </Modal>
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

const APPEARANCE: { theme: Theme; label: string; icon: typeof Sun; blurb: string }[] = [
  { theme: 'light', label: 'Light', icon: Sun, blurb: 'Bright workspace' },
  { theme: 'dark', label: 'Dark', icon: Moon, blurb: 'Command deck' },
];

function Appearance() {
  const { theme, setTheme } = useTheme();
  return (
    <div className="panel pad">
      <div className="panel-head">
        <Palette size={20} color="var(--blue)" />
        <h3>Appearance</h3>
      </div>
      <div className="st-themes" role="radiogroup" aria-label="Appearance">
        {APPEARANCE.map((a) => (
          <button
            key={a.theme}
            role="radio"
            aria-checked={theme === a.theme}
            className="st-theme"
            data-preview={a.theme}
            onClick={() => setTheme(a.theme)}
          >
            <span className="st-theme-swatch" data-preview={a.theme}>
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
