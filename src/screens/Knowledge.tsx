import {
  AlertTriangle,
  CloudUpload,
  Code2,
  Database,
  FileText,
  FileType2,
  FolderGit2,
  Loader2,
  MessageSquareText,
  Plus,
  RefreshCw,
  SlidersHorizontal,
  Table2,
  Trash2,
  X,
} from 'lucide-react';
import { motion } from 'motion/react';
import { useEffect, useRef, useState, type ReactNode } from 'react';
import { Bar, Modal, Pager, Segmented, Slider, fmtAgo, fmtBytes, usePaged } from '../components/ui';
import { inTauri, kbPickFiles, type KbInfo, type KbKind, type Source } from '../kb/api';
import { SAMPLES, type SampleDef } from '../kb/samples';
import { SourceDialog } from './SourceDialog';
import { UgInstall } from './UgSetup';
import { useUgInstall } from '../state/ugInstall';
import { addFiles, addSample, createKb, deleteKb, derivedKind, indexKb, removeSource, setKind, useKb } from '../state/kb';
import { toast, useUi } from '../state/ui';

export async function pickFiles(title: string): Promise<string[]> {
  try {
    return await kbPickFiles(title);
  } catch (e) {
    toast({ tone: 'error', title: 'Could not open the file picker', body: String(e) });
    return [];
  }
}

export function Knowledge() {
  const { kbs, selected, loaded, ug } = useKb();
  const installed = useUgInstall((s) => s.stage === 'done');
  const kb = kbs.find((k) => k.slug === selected) ?? null;
  const [creating, setCreating] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);

  return (
    <div className="screen">
      <div className="page kn">
        <div className="kn-main">
          {!inTauri ? (
            <Notice
              title="Desktop runtime required"
              body="Knowledge bases are built by the native ug engine. Run Andai with `bun run tauri dev` to ingest documents."
            />
          ) : ug && (!ug.found || installed) ? (
            <div className="panel pad">
              <UgInstall variant="card" />
            </div>
          ) : null}

          <KbTabs kbs={kbs} selected={selected} onNew={() => setCreating(true)} />

          <Ingest kb={kb} onNeedKb={() => setCreating(true)} />

          {kb ? (
            <Sources key={kb.slug} kb={kb} onDelete={() => setConfirmDelete(true)} />
          ) : (
            loaded && (
              <div className="panel pad empty" style={{ minHeight: 220 }}>
                <Database size={30} />
                <div style={{ color: 'var(--text)', fontWeight: 600 }}>No knowledge bases yet</div>
                <div style={{ maxWidth: 380, fontSize: 13 }}>
                  Create one, then drop documents in. ug parses them into a graph of sections and concepts, embeds them
                  locally, and Andai retrieves from it while answering.
                </div>
                <button className="btn primary" onClick={() => setCreating(true)}>
                  <Plus size={14} /> New knowledge base
                </button>
                {inTauri && <Samples title="Or try a sample, ready in a few seconds" />}
              </div>
            )
          )}
          {kb && <IndexLog kb={kb} />}
        </div>
        <aside className="kn-side">
          <RagParams />
          <Storage kb={kb} />
        </aside>
      </div>
      <CreateKbModal open={creating} onClose={() => setCreating(false)} />
      <Modal open={confirmDelete} onClose={() => setConfirmDelete(false)}>
        <h3>Delete “{kb?.name}”?</h3>
        {kb?.managed ? (
          <p className="muted" style={{ margin: '4px 0 22px' }}>
            This removes its {kb.sourceCount} copied source files and the ug graph <span className="mono">{kb.slug}</span>. Your
            original files are not touched.
          </p>
        ) : (
          <p className="muted" style={{ margin: '4px 0 22px' }}>
            This removes the ug graph <span className="mono">{kb?.slug}</span>, for Andai and for ug alike. The folder it indexes,{' '}
            <span className="mono">{kb?.root}</span>, is not touched; <span className="mono">ug gen</span> there builds it again.
          </p>
        )}
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10 }}>
          <button className="btn ghost" onClick={() => setConfirmDelete(false)}>
            Cancel
          </button>
          <button
            className="btn danger"
            onClick={() => {
              setConfirmDelete(false);
              if (kb) void deleteKb(kb.slug);
            }}
          >
            <Trash2 size={14} /> Delete
          </button>
        </div>
      </Modal>
    </div>
  );
}

function Notice({ title, body, action }: { title: string; body: string; action?: ReactNode }) {
  return (
    <div className="kn-notice">
      <AlertTriangle size={18} />
      <div>
        <div style={{ fontWeight: 600, color: 'var(--text)' }}>{title}</div>
        <div>{body}</div>
        {action && <div className="kn-notice-action">{action}</div>}
      </div>
    </div>
  );
}

function KbTabs({ kbs, selected, onNew }: { kbs: KbInfo[]; selected: string | null; onNew: () => void }) {
  const paged = usePaged(kbs, KBS_PER_PAGE, (k) => k.slug === selected);
  const grounding = useKb((s) => s.grounding);
  return (
    <div className="kn-tabs">
      {paged.items.map((k) => (
        <button
          key={k.slug}
          className="kn-tab"
          aria-pressed={k.slug === selected}
          title={k.managed ? k.slug : `ug project · ${k.root}`}
          onClick={() => useKb.setState({ selected: k.slug })}
        >
          {k.managed ? <Database size={14} /> : <FolderGit2 size={14} aria-label="ug project" />}
          {k.name}
          {k.status === 'indexing' && <Loader2 size={12} className="spin" />}
          {k.slug === grounding && (
            <span className="pill violet" title="Grounds chat answers">
              chat
            </span>
          )}
        </button>
      ))}
      <button className="kn-tab new" onClick={onNew} disabled={!inTauri}>
        <Plus size={14} /> New
      </button>
      <Pager paged={paged} label="Knowledge bases" className="kn-tabs-pager" />
    </div>
  );
}

function Ingest({ kb, onNeedKb }: { kb: KbInfo | null; onNeedKb: () => void }) {
  const dragging = useUi((s) => s.dragging);
  const busy = kb?.status === 'indexing';
  const upload = async () => {
    if (!kb) return onNeedKb();
    const paths = await pickFiles(`Add to “${kb.name}”`);
    if (paths.length) await addFiles(kb.slug, paths);
  };
  if (kb && !kb.managed) {
    // A ug project indexes the user's own folder; Andai reads it but never writes to it.
    return (
      <motion.div className="panel kn-drop" layout>
        <div className="kn-drop-icon">{busy ? <Loader2 size={30} className="spin" /> : <FolderGit2 size={30} />}</div>
        <h2 className="display">{busy ? 'Assimilating intelligence…' : 'Indexed by ug'}</h2>
        <p className="muted">
          <b>“{kb.name}”</b> is a ug project over <span className="mono selectable">{kb.root}</span>.
          <br />
          Andai searches it as is. To change what it holds, edit that folder and re-index, or run{' '}
          <span className="mono">ug gen</span> there.
        </p>
      </motion.div>
    );
  }
  return (
    <motion.div className={`panel kn-drop${dragging ? ' over' : ''}`} layout>
      <div className="kn-drop-icon">{busy ? <Loader2 size={30} className="spin" /> : <CloudUpload size={30} />}</div>
      <h2 className="display">{busy ? 'Assimilating intelligence…' : 'Ingest New Intelligence'}</h2>
      <p className="muted">
        Drag and drop PDF, Markdown, TXT, CSV or source files to expand {kb ? <b>“{kb.name}”</b> : 'the core knowledge base'}.
        <br />
        Parsed and embedded on-device by ug — nothing is uploaded.
      </p>
      <div className="kn-drop-actions">
        <button className="btn primary" style={{ height: 50, padding: '0 34px' }} onClick={() => void upload()} disabled={!inTauri}>
          Upload_Local
        </button>
      </div>
    </motion.div>
  );
}

const KIND_ICON: Record<Source['kind'], typeof FileText> = {
  PDF: FileType2,
  MD: FileText,
  TXT: FileText,
  CSV: Table2,
  CODE: Code2,
};

/** ug can hold many projects, and a repo thousands of files: both lists page. */
const KBS_PER_PAGE = 8;
const SOURCES_PER_PAGE = 50;

export function Sources({ kb, onDelete }: { kb: KbInfo; onDelete: () => void }) {
  const grounding = useKb((s) => s.grounding);
  const [viewing, setViewing] = useState<string | null>(null);
  const healthy = kb.sources.filter((s) => s.status === 'indexed').length;
  const indexing = kb.status === 'indexing';
  const offline = kb.status === 'offline';
  const sourcePage = usePaged(kb.sources, SOURCES_PER_PAGE);
  return (
    <div className="panel kn-table">
      <div className="kn-table-head">
        <span className="label blue" style={{ fontSize: 15, letterSpacing: '0.14em' }}>
          Indexed_Sources
        </span>
        <span className="pill violet">Total: {kb.sourceCount.toLocaleString()}</span>
        {offline ? (
          <span className="pill amber" title="ug (UltraGraph) isn't installed">Needs ug</span>
        ) : (
          <span className="pill blue">Healthy: {healthy}</span>
        )}
        <KindPicker kb={kb} />
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 8 }}>
          <button
            className={`btn sm ${grounding === kb.slug ? 'secondary' : 'ghost'}`}
            onClick={() => useKb.setState({ grounding: grounding === kb.slug ? null : kb.slug })}
            title="Use this knowledge base to ground chat answers"
          >
            <MessageSquareText size={13} /> {grounding === kb.slug ? 'Grounding chat' : 'Use in chat'}
          </button>
          <button className="btn ghost sm" disabled={indexing || offline || !kb.sources.length} onClick={() => void indexKb(kb.slug)}>
            <RefreshCw size={13} className={indexing ? 'spin' : ''} /> Re-index
          </button>
          <button className="btn ghost sm" onClick={onDelete} disabled={indexing} title="Delete knowledge base">
            <Trash2 size={13} />
          </button>
        </div>
      </div>
      <table>
        <thead>
          <tr>
            <th>Source name</th>
            <th>Type</th>
            <th>Token count</th>
            <th>Sync status</th>
            <th>Last updated</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {sourcePage.items.map((s) => {
            const Icon = KIND_ICON[s.kind];
            const status = indexing && s.status !== 'indexed' ? 'updating' : s.status === 'pending' && indexing ? 'updating' : s.status;
            return (
              <tr key={s.file}>
                <td>
                  <button
                    type="button"
                    className="kn-name"
                    title={`View ${s.file}: details, content and structure`}
                    aria-label={`View ${s.file}`}
                    onClick={() => setViewing(s.file)}
                  >
                    <Icon size={20} strokeWidth={1.6} color="var(--text-2)" />
                    <span className="ellipsis">{s.file}</span>
                  </button>
                </td>
                <td>
                  <span className="kn-type">{s.kind}</span>
                </td>
                <td className="mono kn-num">{s.approxTokens != null ? `~${s.approxTokens.toLocaleString()}` : fmtBytes(s.bytes)}</td>
                <td>
                  <SyncStatus status={offline ? 'offline' : indexing && s.status === 'pending' ? 'updating' : status} />
                </td>
                <td className="muted">{fmtAgo(s.status === 'indexed' ? (kb.lastIndexedAt ?? s.addedAt) : s.addedAt)}</td>
                <td style={{ textAlign: 'right' }}>
                  {kb.managed && (
                    <button
                      className="icon-btn kn-remove"
                      title="Remove source"
                      aria-label={`Remove ${s.file}`}
                      disabled={indexing}
                      onClick={() => void removeSource(kb.slug, s.file)}
                    >
                      <X size={15} />
                    </button>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {!kb.sources.length && (
        <div className="empty" style={{ padding: 34 }}>
          <div>{kb.managed ? 'No sources yet — drop files anywhere on this screen.' : 'ug indexed no files in this project.'}</div>
        </div>
      )}
      <div className="kn-table-foot">
        <Pager paged={sourcePage} label="Sources" />
      </div>
      {kb.sourceCount > kb.sources.length && (
        <div className="muted" style={{ padding: '10px 16px', fontSize: 12.5 }}>
          Showing the first {kb.sources.length.toLocaleString()} of {kb.sourceCount.toLocaleString()} files.
        </div>
      )}
      {kb.lastError && !indexing && (
        <div className="kn-error">
          <AlertTriangle size={15} /> {kb.lastError}
        </div>
      )}
      <SourceDialog kb={kb} file={viewing} onClose={() => setViewing(null)} />
    </div>
  );
}

const KIND_LABEL: Record<KbKind, string> = { document: 'Documents', code: 'Code', mixed: 'Mixed' };

/**
 * What the knowledge base holds decides which agent tools it gets (code
 * navigation only for code). Derived from the sources unless overridden.
 */
function KindPicker({ kb }: { kb: KbInfo }) {
  return (
    <label className="kn-kind" title="Decides which agent tools apply: code tools need code">
      <span className="label">Kind</span>
      <select
        aria-label="Knowledge base kind"
        value={kb.kindOverride ?? 'auto'}
        onChange={(e) => setKind(kb.slug, e.target.value === 'auto' ? null : (e.target.value as KbKind))}
      >
        <option value="auto">Auto · {KIND_LABEL[kb.kindOverride ? derivedKind(kb) : kb.kind]}</option>
        <option value="document">Documents</option>
        <option value="code">Code</option>
        <option value="mixed">Mixed</option>
      </select>
    </label>
  );
}

function SyncStatus({ status }: { status: string }) {
  const map: Record<string, [string, string]> = {
    indexed: ['Indexed', 'var(--blue)'],
    updating: ['Updating', 'var(--amber)'],
    pending: ['Pending', 'var(--text-3)'],
    failed: ['Failed', 'var(--red)'],
    offline: ['Needs ug', 'var(--amber)'],
  };
  const [label, color] = map[status] ?? ['—', 'var(--text-3)'];
  return (
    <span className="status-text" style={{ color }}>
      <span className={`dot${status === 'updating' ? ' pulse' : ''}`} />
      {label}
    </span>
  );
}

function IndexLog({ kb }: { kb: KbInfo }) {
  const lines = useKb((s) => s.logs[kb.slug]);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [lines?.length]);
  if (!lines?.length) return null;
  const live = kb.status === 'indexing';
  return (
    <div className="logblock kn-log" ref={ref}>
      <div className="head">
        <span>ug_gen.log — andai-{kb.slug}</span>
        {live ? <span className="live">LIVE</span> : <span>{new Date(lines[lines.length - 1].at).toLocaleTimeString()}</span>}
      </div>
      {lines.slice(-60).map((l, i) => (
        <div key={i} className={logClass(l.line)}>
          <span className="faint">[{new Date(l.at).toLocaleTimeString([], { hour12: false })}]</span> {l.line}
        </div>
      ))}
    </div>
  );
}

const logClass = (line: string) =>
  /fail|error|✗/i.test(line) ? 'log-err' : /warn/i.test(line) ? 'log-warn' : /✓|ready|done/i.test(line) ? 'log-ok' : 'log-info';

function RagParams() {
  const { k, maxChars } = useKb();
  return (
    <div className="panel pad kn-params">
      <div className="panel-head">
        <SlidersHorizontal size={18} color="var(--violet)" />
        <span className="label violet" style={{ fontSize: 15, letterSpacing: '0.12em' }}>
          RAG_Parameters
        </span>
      </div>

      <div className="kn-param">
        <div className="kn-param-top">
          <span>CHUNKING</span>
          <span className="mono blue-text">Structural</span>
        </div>
        <p>ug splits documents along headings, pages and symbols — each section becomes a graph node, not a fixed-size window.</p>
      </div>

      <div className="kn-param">
        <div className="kn-param-top">
          <span>CONTEXT BUDGET</span>
          <span className="mono blue-text">{maxChars.toLocaleString()} chars</span>
        </div>
        <Slider label="Context budget" min={1500} max={12000} step={500} value={maxChars} onChange={(v) => useKb.setState({ maxChars: v })} />
        <p>How much retrieved text ug assembles per question. Larger budgets cost prompt time on small models.</p>
      </div>

      <div className="kn-param">
        <div className="kn-param-top">
          <span>EMBEDDING MODEL</span>
        </div>
        <select className="field" disabled value="bge">
          <option value="bge">bge-small-en-v1.5 · 384d · local</option>
        </select>
      </div>

      <div className="kn-param">
        <div className="kn-param-top">
          <span>RETRIEVAL K-VALUE</span>
        </div>
        <Segmented
          label="Retrieval K"
          value={k}
          onChange={(v) => useKb.setState({ k: v })}
          options={[4, 8, 16, 32].map((n) => ({ value: n, label: n }))}
        />
      </div>

      <button
        className="btn secondary block kn-apply"
        onClick={() => toast({ tone: 'ok', title: 'RAG configuration applied', body: `k=${k} · budget ${maxChars.toLocaleString()} chars` })}
      >
        Apply_Config
      </button>
    </div>
  );
}

function Storage({ kb }: { kb: KbInfo | null }) {
  const { lastSearch, hits24h, kbs } = useKb();
  const total = kbs.reduce((n, k) => n + k.sizeBytes, 0);
  const cap = 1024 ** 3;
  return (
    <div className="panel pad">
      <div className="label" style={{ marginBottom: 18 }}>
        Vector_Storage_Usage
      </div>
      <div className="kn-cap">
        <span>
          Graph: {(kb?.nodes ?? 0).toLocaleString()} nodes / {(kb?.edges ?? 0).toLocaleString()} edges
        </span>
        <span className="mono blue-text">{((total / cap) * 100).toFixed(1)}%</span>
      </div>
      <Bar value={Math.max(0.01, total / cap)} />
      <div className="faint" style={{ fontSize: 12, marginTop: 8 }}>
        {fmtBytes(total)} across {kbs.length} knowledge base{kbs.length === 1 ? '' : 's'} · 1 GB soft cap
      </div>
      <div className="kn-mini">
        <div>
          <div className="label">Latency</div>
          <div className="kn-mini-v" style={{ color: 'var(--blue)' }}>
            {lastSearch ? `${lastSearch.ms}ms` : '—'}
          </div>
        </div>
        <div>
          <div className="label">Hits (session)</div>
          <div className="kn-mini-v" style={{ color: 'var(--violet)' }}>
            {hits24h}
          </div>
        </div>
      </div>
    </div>
  );
}

const SAMPLE_ICON: Record<KbKind, typeof FileText> = { document: FileText, code: Code2, mixed: Database };

/**
 * The bundled sample knowledge bases (kb/samples.ts): one click adds, indexes
 * and grounds chat in one. A sample already added shows as added.
 */
export function Samples({ title, onAdded }: { title: string; onAdded?: () => void }) {
  const kbs = useKb((s) => s.kbs);
  const [busy, setBusy] = useState<string | null>(null);
  const add = async (sample: SampleDef) => {
    setBusy(sample.id);
    const kb = await addSample(sample.id);
    setBusy(null);
    if (kb) onAdded?.();
  };
  return (
    <section className="kn-samples" aria-label="Sample knowledge bases">
      <div className="label">{title}</div>
      <div className="kn-sample-grid">
        {SAMPLES.map((sample) => {
          const Icon = SAMPLE_ICON[sample.kind];
          const added = kbs.some((k) => k.slug === sample.project);
          return (
            <div key={sample.id} className="kn-sample">
              <Icon size={17} />
              <div className="kn-sample-name">{sample.name}</div>
              <div className="kn-sample-blurb">{sample.blurb}</div>
              <button
                className="btn secondary sm"
                disabled={added || busy !== null}
                aria-label={`Add sample ${sample.name}`}
                onClick={() => void add(sample)}
              >
                {busy === sample.id ? <Loader2 size={13} className="spin" /> : <Plus size={13} />}
                {added ? 'Added' : busy === sample.id ? 'Indexing…' : 'Add sample'}
              </button>
            </div>
          );
        })}
      </div>
    </section>
  );
}

export function CreateKbModal({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [name, setName] = useState('');
  const [busy, setBusy] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  useEffect(() => {
    if (open) {
      setName('');
      setTimeout(() => input.current?.focus(), 50);
    }
  }, [open]);
  const submit = async () => {
    if (!name.trim() || busy) return;
    setBusy(true);
    const kb = await createKb(name);
    setBusy(false);
    if (kb) onClose();
  };
  return (
    <Modal open={open} onClose={onClose}>
      <h3>New knowledge base</h3>
      <p className="muted" style={{ margin: '4px 0 18px' }}>
        A named collection of documents, indexed by ug into its own graph.
      </p>
      <input
        ref={input}
        className="field"
        placeholder="e.g. Product specs, Research papers, Team handbook"
        value={name}
        onChange={(e) => setName(e.target.value)}
        onKeyDown={(e) => e.key === 'Enter' && void submit()}
        maxLength={60}
      />
      <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10, marginTop: 22 }}>
        <button className="btn ghost" onClick={onClose}>
          Cancel
        </button>
        <button className="btn primary" disabled={!name.trim() || busy} onClick={() => void submit()}>
          {busy ? <Loader2 size={14} className="spin" /> : <Plus size={14} />} Create
        </button>
      </div>
      {inTauri && <Samples title="Or start from a sample" onAdded={onClose} />}
    </Modal>
  );
}
