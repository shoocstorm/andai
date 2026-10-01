import { AlertTriangle, Braces, FileText, Hash, Info, ListTree, Loader2, X } from 'lucide-react';
import { useEffect, useMemo, useRef, useState, type CSSProperties, type ReactNode } from 'react';
import { CopyButton, Markdown, Modal, Segmented, fmtAgo, fmtBytes } from '../components/ui';
import type { KbInfo } from '../kb/api';
import { pdfPages, readStructure, type OutlineItem, type SourceView, type Structure } from '../kb/source';
import { viewSource } from '../state/kb';

type Tab = 'overview' | 'content' | 'structure';

const TABS: { id: Tab; label: string; icon: typeof Info }[] = [
  { id: 'overview', label: 'Overview', icon: Info },
  { id: 'content', label: 'Content', icon: FileText },
  { id: 'structure', label: 'Structure', icon: ListTree },
];

const fmtDate = (secs: number | null | undefined) =>
  secs ? new Date(secs * 1000).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' }) : '—';

/** A heading or PDF page is a Concept in ug; code keeps its own node type. */
const typeLabel = (i: OutlineItem) =>
  i.nodeType === 'Concept' ? (/^p\.\d+/.test(i.name) ? 'Page' : 'Section') : i.nodeType || 'Item';

/**
 * One source of a knowledge base: its metadata, its text, and the structure ug
 * indexed from it (outline and related files). Text is shown inert: Markdown
 * through `<Markdown>` (no images, no live links), everything else as plain text.
 */
export function SourceDialog({ kb, file, onClose }: { kb: KbInfo; file: string | null; onClose: () => void }) {
  const [shown, setShown] = useState(file);
  useEffect(() => setShown(file), [file]);
  return (
    <Modal open={!!file} onClose={onClose} wide label={shown ? `Source ${shown}` : 'Source'}>
      {shown && <SourceBody key={shown} kb={kb} file={shown} onOpen={setShown} onClose={onClose} />}
    </Modal>
  );
}

function SourceBody({ kb, file, onOpen, onClose }: { kb: KbInfo; file: string; onOpen: (file: string) => void; onClose: () => void }) {
  const [view, setView] = useState<SourceView | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>('overview');
  const [focus, setFocus] = useState<{ start: number; end: number } | null>(null);
  const listed = kb.sources.find((s) => s.file === file);

  useEffect(() => {
    let live = true;
    viewSource(kb.slug, file).then(
      (v) => live && setView(v),
      (e: Error) => live && setError(e.message),
    );
    return () => {
      live = false;
    };
  }, [kb.slug, file]);

  const structure = useMemo(() => readStructure(view?.structure), [view]);
  const source = view?.source ?? listed;
  const lines = view?.text != null ? lineCount(view.text) : structure.lines;

  const jump = (item: OutlineItem) => {
    setFocus({ start: item.start, end: item.end });
    setTab('content');
  };

  return (
    <div className="sd">
      <div className="sd-head">
        <div className="sd-glyph" aria-hidden>
          {source?.kind === 'CODE' ? <Braces size={20} /> : <FileText size={20} />}
        </div>
        <div className="sd-title">
          <div className="label violet">Source · {kb.name}</div>
          <h3 className="ellipsis selectable" title={file}>
            {file}
          </h3>
          <div className="sd-sub">
            {source && <span className="kn-type">{source.kind}</span>}
            {source && <StatusPill status={source.status} />}
            <span className="faint">
              {[
                source && fmtBytes(source.bytes),
                source?.approxTokens != null && `~${source.approxTokens.toLocaleString()} tokens`,
                lines != null && `${lines.toLocaleString()} lines`,
                structure.language,
              ]
                .filter(Boolean)
                .join(' · ')}
            </span>
          </div>
        </div>
        <button className="btn ghost sm" aria-label="Close dialog" onClick={onClose}>
          <X size={15} />
        </button>
      </div>

      <div className="sd-tabs" role="tablist" aria-label="Source views">
        {TABS.map((t) => (
          <button
            key={t.id}
            role="tab"
            id={`sd-tab-${t.id}`}
            aria-selected={tab === t.id}
            aria-controls="sd-panel"
            className="sd-tab"
            onClick={() => setTab(t.id)}
          >
            <t.icon size={14} /> {t.label}
            {t.id === 'structure' && structure.outline.length > 0 && <span className="sd-count">{structure.outline.length}</span>}
          </button>
        ))}
      </div>

      <div className="sd-panel" role="tabpanel" id="sd-panel" aria-labelledby={`sd-tab-${tab}`}>
        {error ? (
          <div className="sd-notice red" role="alert">
            <AlertTriangle size={16} /> {error}
          </div>
        ) : !view ? (
          <div className="sd-loading">
            <Loader2 size={18} className="spin" /> Reading {file}…
          </div>
        ) : tab === 'overview' ? (
          <Overview kb={kb} view={view} structure={structure} lines={lines} onOpen={onOpen} onJump={jump} />
        ) : tab === 'content' ? (
          <Content view={view} structure={structure} focus={focus} />
        ) : (
          <StructureTab kb={kb} view={view} structure={structure} onJump={jump} onOpen={onOpen} />
        )}
      </div>
    </div>
  );
}

function StatusPill({ status }: { status: string }) {
  const tone = status === 'indexed' ? 'blue' : status === 'failed' ? 'red' : 'amber';
  const label = status === 'indexed' ? 'Indexed' : status === 'failed' ? 'Failed' : 'Pending';
  return <span className={`pill ${tone}`}>{label}</span>;
}

const lineCount = (text: string) => (text ? text.replace(/\n$/, '').split('\n').length : 0);

function Overview({
  kb,
  view,
  structure,
  lines,
  onOpen,
  onJump,
}: {
  kb: KbInfo;
  view: SourceView;
  structure: Structure;
  lines: number | null;
  onOpen: (file: string) => void;
  onJump: (item: OutlineItem) => void;
}) {
  const s = view.source;
  const sections = structure.outline.length;
  const code = s.kind === 'CODE';
  return (
    <div className="sd-overview">
      <div className="sd-stats">
        <Tile label="Size" value={fmtBytes(s.bytes)} />
        <Tile label="Tokens" value={s.approxTokens != null ? `~${s.approxTokens.toLocaleString()}` : '—'} />
        <Tile label="Lines" value={lines != null ? lines.toLocaleString() : '—'} />
        <Tile label={code ? 'Symbols' : s.kind === 'PDF' ? 'Pages & sections' : 'Sections'} value={sections ? sections.toLocaleString() : '—'} />
      </div>

      <section className="sd-block" aria-label="Details">
        <h4 className="label">Details</h4>
        <dl className="sd-meta">
          <Meta k={kb.managed ? 'Stored copy' : 'File'}>
            <span className="mono selectable sd-path">{view.path}</span>
            <CopyButton text={view.path} label="Copy path" />
          </Meta>
          <Meta k={kb.managed ? 'Added' : 'Modified'}>
            {fmtDate(s.addedAt)} <span className="faint">· {fmtAgo(s.addedAt)}</span>
          </Meta>
          <Meta k="Last indexed">
            {s.status === 'indexed' ? fmtDate(kb.lastIndexedAt) : <span className="faint">Not yet</span>}
          </Meta>
          <Meta k="Language">{structure.language ?? '—'}</Meta>
          <Meta k="Classification">
            {structure.classification ?? '—'}
            {structure.isTest && <span className="pill amber sd-inline-pill">test</span>}
          </Meta>
          <Meta k="Graph project">
            <span className="mono selectable">{kb.slug}</span>
          </Meta>
          <Meta k="Node id">
            <span className="mono selectable">file:{s.file}</span>
          </Meta>
        </dl>
      </section>

      {structure.outline.length > 0 && (
        <section className="sd-block" aria-label="Outline preview">
          <h4 className="label">Starts with</h4>
          <ul className="sd-preview">
            {structure.outline.slice(0, 5).map((i) => (
              <li key={`${i.id}-${i.start}`}>
                <button type="button" onClick={() => onJump(i)}>
                  <span className="sd-kind">{typeLabel(i)}</span>
                  <span className="ellipsis">{i.name}</span>
                  <span className="sd-lines mono">L{i.start}</span>
                </button>
              </li>
            ))}
          </ul>
        </section>
      )}

      <Relations kb={kb} structure={structure} onOpen={onOpen} />
      {view.structureError && <StructureNotice text={view.structureError} />}
    </div>
  );
}

function Tile({ label, value }: { label: string; value: string }) {
  return (
    <div className="sd-tile">
      <div className="label">{label}</div>
      <div className="sd-tile-v">{value}</div>
    </div>
  );
}

function Meta({ k, children }: { k: string; children: ReactNode }) {
  return (
    <div className="sd-meta-row">
      <dt>{k}</dt>
      <dd>{children}</dd>
    </div>
  );
}

function StructureNotice({ text }: { text: string }) {
  return (
    <div className="sd-notice">
      <AlertTriangle size={15} /> {text}
    </div>
  );
}

function Content({ view, structure, focus }: { view: SourceView; structure: Structure; focus: { start: number; end: number } | null }) {
  const markdown = view.source.kind !== 'CODE' && view.text != null;
  // A jump from the outline lands on line numbers, so it opens the plain text.
  const [mode, setMode] = useState<'rendered' | 'text'>(markdown && !focus ? 'rendered' : 'text');

  if (view.text == null) {
    const pages = pdfPages(structure.outline);
    return (
      <div className="sd-content">
        <div className="sd-toolbar">
          <span className="faint sd-small">
            The text ug indexed from this PDF, page by page. Long pages may be shortened; the original file is unchanged.
          </span>
        </div>
        {pages.length ? (
          <div className="sd-pages">
            {pages.map((p, i) => (
              <article key={i} className="sd-page">
                <div className="label">{p.title}</div>
                <p className="selectable">{p.text}</p>
              </article>
            ))}
          </div>
        ) : (
          <StructureNotice text={view.structureError ?? 'ug found no text in this PDF.'} />
        )}
      </div>
    );
  }

  return (
    <div className="sd-content">
      <div className="sd-toolbar">
        {markdown && (
          <Segmented
            label="Content view"
            value={mode}
            onChange={setMode}
            options={[
              { value: 'rendered', label: 'Rendered' },
              { value: 'text', label: 'Text' },
            ]}
          />
        )}
        {view.textTruncated && <span className="pill amber">First 512 KB shown</span>}
        <span style={{ marginLeft: 'auto' }}>
          <CopyButton text={view.text} label="Copy text">
            Copy
          </CopyButton>
        </span>
      </div>
      {mode === 'rendered' ? (
        <div className="sd-doc">
          <Markdown text={view.text} />
        </div>
      ) : (
        <Lines text={view.text} focus={focus} />
      )}
    </div>
  );
}

function Lines({ text, focus }: { text: string; focus: { start: number; end: number } | null }) {
  const ref = useRef<HTMLDivElement>(null);
  const rows = useMemo(() => text.replace(/\n$/, '').split('\n'), [text]);
  useEffect(() => {
    if (!focus) return;
    ref.current?.querySelector(`[data-line="${focus.start}"]`)?.scrollIntoView?.({ block: 'center' });
  }, [focus]);
  return (
    <div className="sd-code selectable" ref={ref} role="region" aria-label="File text with line numbers">
      {rows.map((row, i) => {
        const n = i + 1;
        const hit = focus && n >= focus.start && n <= focus.end;
        return (
          <div key={n} data-line={n} className={`sd-row${hit ? ' hit' : ''}`}>
            <span className="sd-ln" aria-hidden>
              {n}
            </span>
            <span className="sd-tx">{row || ' '}</span>
          </div>
        );
      })}
    </div>
  );
}

function StructureTab({
  kb,
  view,
  structure,
  onJump,
  onOpen,
}: {
  kb: KbInfo;
  view: SourceView;
  structure: Structure;
  onJump: (item: OutlineItem) => void;
  onOpen: (file: string) => void;
}) {
  const jumpable = view.text != null;
  return (
    <div className="sd-structure">
      {view.structureError && <StructureNotice text={view.structureError} />}
      {structure.outline.length > 0 ? (
        <section className="sd-block" aria-label="Outline">
          <h4 className="label">
            Outline <span className="faint">· how ug split this file into graph nodes</span>
          </h4>
          <ol className="sd-tree">
            {structure.outline.map((i) => (
              <li key={`${i.id}-${i.start}`} style={{ '--depth': Math.min(i.depth, 6) } as CSSProperties}>
                <button
                  type="button"
                  className="sd-node"
                  disabled={!jumpable}
                  onClick={() => onJump(i)}
                  title={jumpable ? `Show lines ${i.start}–${i.end}` : undefined}
                >
                  <span className="sd-node-top">
                    <span className={`sd-kind ${i.nodeType === 'Concept' ? '' : 'code'}`}>
                      {i.nodeType === 'Concept' ? <Hash size={11} /> : <Braces size={11} />} {typeLabel(i)}
                    </span>
                    <span className="sd-node-name ellipsis">{i.name}</span>
                    <span className="sd-lines mono">
                      {i.start === i.end ? `L${i.start}` : `L${i.start}–${i.end}`}
                    </span>
                  </span>
                  {i.doc && <span className="sd-node-doc">{i.doc}</span>}
                </button>
              </li>
            ))}
          </ol>
        </section>
      ) : (
        !view.structureError && <StructureNotice text="ug found no sections or symbols in this file." />
      )}
      <Relations kb={kb} structure={structure} onOpen={onOpen} />
    </div>
  );
}

/** Files ug links to this one; those in the knowledge base open in place. */
function Relations({ kb, structure, onOpen }: { kb: KbInfo; structure: Structure; onOpen: (file: string) => void }) {
  if (!structure.related.length) return null;
  return (
    <section className="sd-block" aria-label="Related files">
      <h4 className="label">Related files</h4>
      <div className="sd-rel">
        {structure.related.map((r) => (
          <div key={r.role} className="sd-rel-row">
            <span className="sd-rel-label">{r.label}</span>
            <div className="sd-chips">
              {r.items.map((i) =>
                kb.sources.some((s) => s.file === i.file) ? (
                  <button key={i.name} type="button" className="sd-chip" onClick={() => onOpen(i.file)} title={i.why || `Open ${i.file}`}>
                    {i.name}
                  </button>
                ) : (
                  <span key={i.name} className="sd-chip static" title={i.why || undefined}>
                    {i.name}
                  </span>
                ),
              )}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
