import { AlertTriangle, Check, Download, Heart, Loader2, Search, ShieldAlert, X, Zap } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { CopyButton, Modal, Segmented, fmtBytes } from '../components/ui';
import { addHubModel } from '../llm/custom';
import { loadModel, useEngine } from '../llm/engine';
import { inspectHub, searchHub, type Check as Verdict, type HubModel, type HubResult } from '../llm/hub';
import { hubPageUrl, type HubFormat } from '../llm/models';
import { toast } from '../state/ui';

const compact = (n: number) => Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 }).format(n);
const month = (iso: string | null) => (iso ? new Date(iso).toLocaleDateString('en', { month: 'short', year: 'numeric' }) : null);
const EXAMPLES: Record<HubFormat, string[]> = { mlx: ['qwen3', 'qwen3 4b', 'qwen3 8b 4bit'], gguf: ['qwen3', 'llama 3.2 1b', 'gemma 3 1b'] };

/**
 * Settings → Models → Add from Hugging Face. Search, inspect one model at its
 * current commit (llm/hub.ts), then add it pinned to that commit and start its
 * verified download. Only what Andai can run can be added, and the dialog
 * says why not otherwise.
 */
export function HubModels({ open, onClose }: { open: boolean; onClose: () => void }) {
  const mlx = useEngine((s) => s.native.supported);
  const [format, setFormat] = useState<HubFormat>(mlx ? 'mlx' : 'gguf');
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<HubResult[] | null>(null);
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [picked, setPicked] = useState<string | null>(null);
  const [model, setModel] = useState<HubModel | null>(null);
  const [inspecting, setInspecting] = useState(false);
  const [inspectError, setInspectError] = useState<string | null>(null);
  const [variant, setVariant] = useState<string | null>(null);
  const [adding, setAdding] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!open) return;
    setFormat(mlx ? 'mlx' : 'gguf');
    setQuery('');
    setResults(null);
    setPicked(null);
    setModel(null);
    setTimeout(() => input.current?.focus(), 50);
  }, [open, mlx]);

  // Search as the user types, debounced; a newer query cancels the older one.
  useEffect(() => {
    if (!open) return;
    const q = query.trim();
    if (q.length < 2) {
      setResults(null);
      setSearchError(null);
      return;
    }
    const ctl = new AbortController();
    const timer = setTimeout(() => {
      setSearching(true);
      setSearchError(null);
      searchHub(q, format, ctl.signal)
        .then(setResults)
        .catch((e: unknown) => !ctl.signal.aborted && setSearchError(e instanceof Error ? e.message : String(e)))
        .finally(() => !ctl.signal.aborted && setSearching(false));
    }, 350);
    return () => {
      clearTimeout(timer);
      ctl.abort();
    };
  }, [query, format, open]);

  useEffect(() => {
    setModel(null);
    setInspectError(null);
    if (!picked) return;
    const ctl = new AbortController();
    setInspecting(true);
    inspectHub(picked, format, ctl.signal)
      .then((m) => {
        setModel(m);
        setVariant(m.gguf?.recommended ?? null);
      })
      .catch((e: unknown) => !ctl.signal.aborted && setInspectError(e instanceof Error ? e.message : String(e)))
      .finally(() => !ctl.signal.aborted && setInspecting(false));
    return () => ctl.abort();
  }, [picked, format]);

  const chosen = model?.gguf?.variants.find((v) => v.path === variant);
  const canAdd = !!model?.ok && (model.format === 'mlx' || !!chosen?.fits) && !adding;

  const add = async () => {
    if (!model || !canAdd) return;
    setAdding(true);
    try {
      const id = await addHubModel(model, chosen);
      onClose();
      toast({ tone: 'info', title: 'Model added', body: `Downloading ${model.repo}; it’s checked by sha256 before it loads.` });
      void loadModel(id);
    } catch (e) {
      setInspectError(e instanceof Error ? e.message : String(e));
    } finally {
      setAdding(false);
    }
  };

  return (
    <Modal open={open} onClose={onClose} wide label="Add a model from Hugging Face">
      <h3>Add a model from Hugging Face</h3>
      <p className="muted" style={{ margin: '4px 0 16px' }}>
        Search public models Andai can run. The one you add is pinned to its current version and checked by sha256 before it
        loads.
      </p>
      <div className="hub-bar">
        {mlx && (
          <Segmented
            label="Model format"
            value={format}
            onChange={(f) => {
              setFormat(f);
              setPicked(null);
            }}
            options={[
              { value: 'mlx', label: 'MLX · fastest on this Mac' },
              { value: 'gguf', label: 'GGUF · any computer' },
            ]}
          />
        )}
        <label className="hub-search">
          <Search size={15} />
          <input
            ref={input}
            role="searchbox"
            aria-label="Search Hugging Face"
            placeholder={format === 'mlx' ? 'Search MLX models, e.g. qwen3 4b' : 'Search GGUF models, e.g. llama 3.2 1b'}
            value={query}
            maxLength={100}
            onChange={(e) => setQuery(e.target.value)}
          />
          {searching && <Loader2 size={14} className="spin" />}
        </label>
      </div>
      <p className="faint hub-scope">
        {format === 'mlx'
          ? 'MLX models run natively on your Mac’s GPU. The native engine runs Qwen3 and Qwen3.5 models quantized for MLX.'
          : 'GGUF models run in the app on any computer (wllama), one file up to 2 GB.'}
      </p>

      <div className="hub-body">
        <section className="hub-results" aria-label="Search results">
          {searchError ? (
            <div className="kn-error">
              <X size={15} /> {searchError}
            </div>
          ) : results === null ? (
            <div className="hub-hint">
              <span className="faint">Try</span>
              {EXAMPLES[format].map((q) => (
                <button key={q} className="btn ghost sm" onClick={() => setQuery(q)}>
                  {q}
                </button>
              ))}
            </div>
          ) : results.length === 0 ? (
            <div className="faint hub-hint">No public {format.toUpperCase()} text-generation models match.</div>
          ) : (
            results.map((r) => (
              <button key={r.repo} className="hub-result" aria-pressed={picked === r.repo} onClick={() => setPicked(r.repo)}>
                <span className="hub-repo mono">
                  {picked === r.repo && <Check size={12} className="st-check" aria-hidden />} {r.repo}
                </span>
                <span className="faint hub-meta">
                  <Download size={11} /> {compact(r.downloads)} <Heart size={11} /> {compact(r.likes)}
                  {month(r.lastModified) ? ` · ${month(r.lastModified)}` : ''}
                  {r.license ? ` · ${r.license}` : ''}
                </span>
              </button>
            ))
          )}
        </section>

        <section className="hub-detail" aria-label="Model details">
          {!picked ? (
            <div className="faint hub-hint">Pick a model to see whether Andai can run it.</div>
          ) : inspecting ? (
            <div className="faint hub-hint">
              <Loader2 size={14} className="spin" /> Reading {picked}…
            </div>
          ) : inspectError && !model ? (
            <div className="kn-error">
              <X size={15} /> {inspectError}
            </div>
          ) : model ? (
            <Detail model={model} variant={variant} onVariant={setVariant} />
          ) : null}
          {inspectError && model && (
            <div className="kn-error" style={{ marginTop: 10 }}>
              <X size={15} /> {inspectError}
            </div>
          )}
        </section>
      </div>

      <div className="hub-foot">
        <button className="btn ghost" onClick={onClose}>
          Cancel
        </button>
        <button className="btn primary" disabled={!canAdd} onClick={() => void add()}>
          {adding ? <Loader2 size={14} className="spin" /> : <Download size={14} />}
          {model?.format === 'gguf' && chosen ? `Add & download ${fmtBytes(chosen.bytes)}` : model?.mlx ? `Add & download ${fmtBytes(model.mlx.bytes)}` : 'Add & download'}
        </button>
      </div>
    </Modal>
  );
}

function CheckLine({ c }: { c: Verdict }) {
  const Icon = c.level === 'ok' ? Check : c.level === 'warn' ? AlertTriangle : X;
  return (
    <li className={`hub-check ${c.level}`}>
      <Icon size={13} /> {c.text}
    </li>
  );
}

function Detail({ model, variant, onVariant }: { model: HubModel; variant: string | null; onVariant: (v: string) => void }) {
  const page = hubPageUrl(model.repo);
  return (
    <div>
      <div className="hub-title mono">{model.repo}</div>
      <div className="faint hub-meta">
        Version {model.commit.slice(0, 7)} · {compact(model.downloads)} downloads · {model.license ?? 'license not stated'}
        {page && <CopyButton text={page} label="Copy the model’s Hugging Face address" size={12} />}
      </div>
      <ul className="hub-checks" aria-label="Compatibility">
        {model.checks.map((c) => (
          <CheckLine key={c.text} c={c} />
        ))}
      </ul>
      {model.gguf && model.ok && (
        <div role="radiogroup" aria-label="File to download" className="hub-variants">
          {/* The recommended file first, so what the button downloads is in view; then smallest first. */}
          {[...model.gguf.variants].sort((a, b) => Number(b.path === model.gguf!.recommended) - Number(a.path === model.gguf!.recommended)).map((v) => (
            <button
              key={v.path}
              role="radio"
              aria-checked={variant === v.path}
              disabled={!v.fits}
              className="hub-variant"
              onClick={() => onVariant(v.path)}
              title={v.path}
            >
              {variant === v.path ? <Check size={13} className="st-check" aria-hidden /> : <span className="hub-radio" aria-hidden />}
              <span className="mono">{v.quant}</span>
              <span className="faint">{fmtBytes(v.bytes)}</span>
              {v.path === model.gguf!.recommended && <span className="pill blue">Recommended</span>}
              {!v.fits && <span className="pill">Over 2 GB</span>}
            </button>
          ))}
        </div>
      )}
      {model.mlx && (
        <div className="hub-mlx">
          <Zap size={14} /> {model.mlx.bits}-bit · {model.mlx.layers} layers · {fmtBytes(model.mlx.bytes)} · runs natively with MLX
        </div>
      )}
      <div className="hub-warn" role="note">
        <ShieldAlert size={15} />
        <span>
          <b>A third-party model.</b> Anyone can publish on Hugging Face, and Andai hasn’t reviewed this one. Its files are pinned
          to version {model.commit.slice(0, 7)} and checked before use, and these formats can’t run code, but the model can
          still give wrong, biased or unsafe answers. Check its license before relying on it.
        </span>
      </div>
    </div>
  );
}
