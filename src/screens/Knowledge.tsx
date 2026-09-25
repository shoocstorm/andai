import { open as openDialog } from '@tauri-apps/plugin-dialog';
import {
  AlertTriangle,
  CloudUpload,
  Code2,
  Database,
  FileText,
  FileType2,
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
import { useEffect, useRef, useState } from 'react';
import { Bar, Modal, Segmented, Slider, fmtAgo, fmtBytes } from '../components/ui';
import { inTauri, type KbInfo, type Source } from '../kb/api';
import { addFiles, createKb, deleteKb, indexKb, removeSource, useKb } from '../state/kb';
import { toast, useUi } from '../state/ui';

export const ACCEPT = ['pdf', 'md', 'markdown', 'mdx', 'txt', 'text', 'log', 'rst', 'csv', 'tsv', 'ts', 'tsx', 'js', 'jsx', 'mjs', 'py', 'java', 'rs'];

export async function pickFiles(title: string): Promise<string[]> {
  const picked = await openDialog({ multiple: true, title, filters: [{ name: 'Documents', extensions: ACCEPT }] });
  return Array.isArray(picked) ? picked : picked ? [picked] : [];
}

export function Knowledge() {
  const { kbs, selected, loaded, ug } = useKb();
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
              body="Knowledge bases are built by the native ug engine. Run Andai with `npm run tauri dev` to ingest documents."
            />
          ) : ug && !ug.found ? (
            <Notice
              title="ug CLI not found"
              body="Andai builds knowledge graphs with ug. Install it so `ug` is on your PATH or in ~/.local/bin, then restart Andai."
            />
          ) : null}

          <KbTabs kbs={kbs} selected={selected} onNew={() => setCreating(true)} />

          <Ingest kb={kb} onNeedKb={() => setCreating(true)} />

          {kb ? (
            <Sources kb={kb} onDelete={() => setConfirmDelete(true)} />
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
        <p className="muted" style={{ margin: '4px 0 22px' }}>
          This removes its {kb?.sources.length ?? 0} copied source files and the ug graph <span className="mono">andai-{kb?.slug}</span>.
          Your original files are not touched.
        </p>
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

function Notice({ title, body }: { title: string; body: string }) {
  return (
    <div className="kn-notice">
      <AlertTriangle size={18} />
      <div>
        <div style={{ fontWeight: 600, color: 'var(--text)' }}>{title}</div>
        <div>{body}</div>
      </div>
    </div>
  );
}

function KbTabs({ kbs, selected, onNew }: { kbs: KbInfo[]; selected: string | null; onNew: () => void }) {
  const grounding = useKb((s) => s.grounding);
  return (
    <div className="kn-tabs">
      {kbs.map((k) => (
        <button
          key={k.slug}
          className="kn-tab"
          aria-pressed={k.slug === selected}
          onClick={() => useKb.setState({ selected: k.slug })}
        >
          <Database size={14} />
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
        <button
          className="btn secondary"
          style={{ height: 50, padding: '0 30px' }}
          onClick={() => toast({ tone: 'info', title: 'S3 connector', body: 'Remote sources are on the roadmap — local files only for now.' })}
        >
          Connect_S3
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

function Sources({ kb, onDelete }: { kb: KbInfo; onDelete: () => void }) {
  const grounding = useKb((s) => s.grounding);
  const healthy = kb.sources.filter((s) => s.status === 'indexed').length;
  const indexing = kb.status === 'indexing';
  return (
    <div className="panel kn-table">
      <div className="kn-table-head">
        <span className="label blue" style={{ fontSize: 15, letterSpacing: '0.14em' }}>
          Indexed_Sources
        </span>
        <span className="pill violet">Total: {kb.sources.length}</span>
        <span className="pill blue">Healthy: {healthy}</span>
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 8 }}>
          <button
            className={`btn sm ${grounding === kb.slug ? 'secondary' : 'ghost'}`}
            onClick={() => useKb.setState({ grounding: grounding === kb.slug ? null : kb.slug })}
            title="Use this knowledge base to ground chat answers"
          >
            <MessageSquareText size={13} /> {grounding === kb.slug ? 'Grounding chat' : 'Use in chat'}
          </button>
          <button className="btn ghost sm" disabled={indexing || !kb.sources.length} onClick={() => void indexKb(kb.slug)}>
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
          {kb.sources.map((s) => {
            const Icon = KIND_ICON[s.kind];
            const status = indexing && s.status !== 'indexed' ? 'updating' : s.status === 'pending' && indexing ? 'updating' : s.status;
            return (
              <tr key={s.file}>
                <td>
                  <div className="kn-name" title={s.original}>
                    <Icon size={20} strokeWidth={1.6} color="var(--text-2)" />
                    <span className="ellipsis">{s.file}</span>
                  </div>
                </td>
                <td>
                  <span className="kn-type">{s.kind}</span>
                </td>
                <td className="mono kn-num">{s.approxTokens != null ? `~${s.approxTokens.toLocaleString()}` : fmtBytes(s.bytes)}</td>
                <td>
                  <SyncStatus status={indexing && s.status === 'pending' ? 'updating' : status} />
                </td>
                <td className="muted">{fmtAgo(s.status === 'indexed' ? (kb.lastIndexedAt ?? s.addedAt) : s.addedAt)}</td>
                <td style={{ textAlign: 'right' }}>
                  <button
                    className="icon-btn kn-remove"
                    title="Remove source"
                    disabled={indexing}
                    onClick={() => void removeSource(kb.slug, s.file)}
                  >
                    <X size={15} />
                  </button>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {!kb.sources.length && (
        <div className="empty" style={{ padding: 34 }}>
          <div>No sources yet — drop files anywhere on this screen.</div>
        </div>
      )}
      {kb.lastError && !indexing && (
        <div className="kn-error">
          <AlertTriangle size={15} /> {kb.lastError}
        </div>
      )}
    </div>
  );
}

function SyncStatus({ status }: { status: string }) {
  const map: Record<string, [string, string]> = {
    indexed: ['Indexed', 'var(--blue)'],
    updating: ['Updating', 'var(--amber)'],
    pending: ['Pending', 'var(--text-3)'],
    failed: ['Failed', 'var(--red)'],
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
    </Modal>
  );
}
