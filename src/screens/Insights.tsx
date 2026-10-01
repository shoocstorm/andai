// Insights: what `ug analyze` says about a code knowledge base (kb/analysis.ts).
// Every value is ug's, shown as plain text; each symbol or file opens the
// source dialog, and Ask hands a question about it to the chat.
import { AlertTriangle, ClipboardCopy, Loader2, MessageSquareText } from 'lucide-react';
import { useEffect, useState, type ReactNode } from 'react';
import { col, insightsReport, parseNodeId, type Analysis, type Cell } from '../kb/analysis';
import type { AnalyzePreset, KbInfo } from '../kb/api';
import { askAbout, indexStamp, loadInsights, useKb } from '../state/kb';
import { toast } from '../state/ui';
import { SourceDialog } from './SourceDialog';

const ROWS = 6;
const HEAT_FOLDERS = 8;

const num = (v: Cell) => (typeof v === 'number' ? v : 0);
const fmt = (v: Cell) => (typeof v === 'number' ? v.toLocaleString() : (v ?? '–'));
const folderName = (f: Cell) => (typeof f === 'string' && f ? f : '(root)');

export function Insights({ kb }: { kb: KbInfo }) {
  const stamp = indexStamp(kb);
  const entry = useKb((s) => s.insights[kb.slug]);
  const [viewing, setViewing] = useState<string | null>(null);
  useEffect(() => {
    void loadInsights(kb.slug);
  }, [kb.slug, stamp]);

  const fresh = entry?.stamp === stamp ? entry : null;
  const results = fresh?.results ?? {};
  const errors = fresh?.errors ?? {};
  const loading = !fresh || fresh.loading;
  const failed = Object.values(errors);
  // Every preset failed the same way (ug too old, project gone): one notice instead of eight.
  const allFailed = !loading && failed.length > 0 && !Object.keys(results).length;

  const card = (preset: AnalyzePreset) => ({ a: results[preset], error: errors[preset] ?? null, loading });
  const ask = (q: string) => askAbout(kb.slug, q);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(insightsReport(kb.name, results));
      toast({ tone: 'ok', title: 'Report copied', body: 'Markdown, ready to paste.' });
    } catch (e) {
      toast({ tone: 'error', title: 'Could not copy', body: String(e) });
    }
  };
  const risky = new Set(
    (results.risky_symbols?.rows ?? []).map((r) => col(results.risky_symbols!, r, 'id')).filter((v): v is string => typeof v === 'string'),
  );

  return (
    <section className="ins" aria-label="Code insights">
      <div className="panel ins-head">
        <span className="label blue" style={{ fontSize: 15, letterSpacing: '0.14em' }}>
          Code_Insights
        </span>
        <span className="faint ins-sub">Whole-codebase facts from ug’s graph, computed on this Mac.</span>
        {loading && <Loader2 size={15} className="spin" aria-label="Analyzing" />}
        <button className="btn ghost sm" style={{ marginLeft: 'auto' }} disabled={loading || allFailed} onClick={() => void copy()}>
          <ClipboardCopy size={13} /> Copy report
        </button>
      </div>

      {allFailed ? (
        <div className="kn-error">
          <AlertTriangle size={15} /> {failed[0]}
        </div>
      ) : (
        <div className="ins-grid">
          <Card title="Shape" hint="What the code is written in, and how long its functions run." {...card('language_breakdown')}>
            <Shape lang={results.language_breakdown} kinds={results.file_kinds} hist={results.size_histogram} />
          </Card>
          <Card title="Start here" hint="Documented, heavily used symbols: a reading order for a newcomer." {...card('where_to_start')}>
            {(a) => <SymbolRows a={a} metric="depended_on_by" unit="users" onOpen={setViewing} ask={(n) => ask(`How does ${n} work?`)} />}
          </Card>
          <Card title="Load-bearing" hint="The most depended-upon symbols. Flagged: large and undocumented too." {...card('dependency_fanin')}>
            {(a) => (
              <SymbolRows a={a} metric="depended_on_by" unit="users" flagged={risky} onOpen={setViewing} ask={(n) => ask(`Who calls ${n}?`)} />
            )}
          </Card>
          <Card title="Biggest files" hint="Where the mass is: symbols and lines of code per file." {...card('biggest_files')}>
            {(a) => <FileRows a={a} onOpen={setViewing} ask={(f) => ask(`What would break if I changed ${f}?`)} />}
          </Card>
          <Card title="Architecture" hint="Which folders depend on which, by edges across the boundary." wide {...card('coupling_matrix')}>
            {(a) => <Heatmap a={a} />}
          </Card>
          <Card title="Untested" hint="Functions no test reaches within two hops, most used first." {...card('untested_symbols')}>
            {(a) => <SymbolRows a={a} metric="depended_on_by" unit="users" onOpen={setViewing} ask={(n) => ask(`What does ${n} do, and who calls it?`)} />}
          </Card>
          <Card title="Docs gaps" hint="Undocumented symbols that many others depend on." {...card('undocumented_hotspots')}>
            {(a) => <SymbolRows a={a} metric="depended_on_by" unit="users" onOpen={setViewing} ask={(n) => ask(`What does ${n} do?`)} />}
          </Card>
          <Card title="Long functions" hint="The longest functions outside tests." {...card('long_functions')}>
            {(a) => <SymbolRows a={a} metric="loc" unit="lines" onOpen={setViewing} ask={(n) => ask(`Walk me through ${n} step by step.`)} />}
          </Card>
          <Card title="Dead code" hint="Nothing calls or names these. Candidates, not proof: a name built at runtime is invisible." {...card('dead_code')}>
            {(a) => <SymbolRows a={a} metric="loc" unit="lines" onOpen={setViewing} ask={(n) => ask(`Is ${n} used anywhere?`)} />}
          </Card>
        </div>
      )}
      <SourceDialog kb={kb} file={viewing} onClose={() => setViewing(null)} />
    </section>
  );
}

function Card({
  title,
  hint,
  a,
  error,
  loading,
  wide,
  children,
}: {
  title: string;
  hint: string;
  a: Analysis | undefined;
  error: string | null;
  loading: boolean;
  wide?: boolean;
  children: ReactNode | ((a: Analysis) => ReactNode);
}) {
  const body = typeof children === 'function' ? (a ? children(a) : null) : children;
  return (
    <article className={`panel ins-card${wide ? ' wide' : ''}`} aria-label={title}>
      <h3>{title}</h3>
      <p className="faint ins-hint">{hint}</p>
      {error ? (
        <div className="ins-note error">
          <AlertTriangle size={13} /> {error}
        </div>
      ) : a || typeof children !== 'function' ? (
        <>
          {body}
          {a?.caveats.map((c) => (
            <div key={c} className="ins-note">
              <AlertTriangle size={13} /> {c}
            </div>
          ))}
        </>
      ) : loading ? (
        <div className="faint ins-empty">Analyzing…</div>
      ) : null}
    </article>
  );
}

function More<T>({ items, render }: { items: T[]; render: (item: T, i: number) => ReactNode }) {
  const [all, setAll] = useState(false);
  const shown = all ? items : items.slice(0, ROWS);
  return (
    <>
      <ol className="ins-rows">{shown.map(render)}</ol>
      {items.length > ROWS && (
        <button className="cc-link ins-more" onClick={() => setAll(!all)}>
          {all ? 'Show fewer' : `Show all ${items.length}`}
        </button>
      )}
    </>
  );
}

function SymbolRows({
  a,
  metric,
  unit,
  flagged,
  onOpen,
  ask,
}: {
  a: Analysis;
  metric: string;
  unit: string;
  flagged?: Set<string>;
  onOpen: (file: string) => void;
  ask: (name: string) => void;
}) {
  const rows = a.rows.flatMap((row) => {
    const id = col(a, row, 'id');
    const ref = typeof id === 'string' ? parseNodeId(id) : null;
    return ref ? [{ id: id as string, ref, value: col(a, row, metric) }] : [];
  });
  if (!rows.length) return <div className="faint ins-empty">{a.caveats.length ? 'No result.' : 'None found.'}</div>;
  return (
    <More
      items={rows}
      render={({ id, ref, value }) => (
        <li key={id} className="ins-row">
          <button type="button" className="ins-name" title={`Open ${ref.file}`} aria-label={`Open ${ref.name} in ${ref.file}`} onClick={() => onOpen(ref.file)}>
            <span className="mono ellipsis">{ref.name}</span>
            <span className="faint ellipsis ins-file">
              {ref.kind} · {ref.file}
            </span>
          </button>
          {flagged?.has(id) && <span className="pill amber" title="Large, undocumented and heavily depended upon">risky</span>}
          <span className="mono ins-metric" title={unit}>
            {fmt(value)} <span className="faint">{unit}</span>
          </span>
          <button type="button" className="icon-btn" aria-label={`Ask about ${ref.name}`} title="Ask the agent about it" onClick={() => ask(ref.name)}>
            <MessageSquareText size={14} />
          </button>
        </li>
      )}
    />
  );
}

function FileRows({ a, onOpen, ask }: { a: Analysis; onOpen: (file: string) => void; ask: (file: string) => void }) {
  const rows = a.rows.flatMap((row) => {
    const file = col(a, row, 'file');
    return typeof file === 'string' && file ? [{ file, symbols: col(a, row, 'symbols'), lines: col(a, row, 'code_lines') }] : [];
  });
  if (!rows.length) return <div className="faint ins-empty">None found.</div>;
  return (
    <More
      items={rows}
      render={({ file, symbols, lines }) => (
        <li key={file} className="ins-row">
          <button type="button" className="ins-name" aria-label={`Open ${file}`} onClick={() => onOpen(file)}>
            <span className="mono ellipsis">{file}</span>
          </button>
          <span className="mono ins-metric">
            {fmt(symbols)} <span className="faint">symbols</span> · {fmt(lines)} <span className="faint">lines</span>
          </span>
          <button type="button" className="icon-btn" aria-label={`Ask what depends on ${file}`} title="Ask what would break if it changed" onClick={() => ask(file)}>
            <MessageSquareText size={14} />
          </button>
        </li>
      )}
    />
  );
}

function Shape({ lang, kinds, hist }: { lang?: Analysis; kinds?: Analysis; hist?: Analysis }) {
  const langs = (lang?.rows ?? []).map((r) => ({ name: String(col(lang!, r, 'language') ?? '?'), symbols: col(lang!, r, 'symbols'), lines: col(lang!, r, 'code_lines') }));
  const files = (kinds?.rows ?? []).map((r) => ({ name: String(col(kinds!, r, 'language') ?? col(kinds!, r, 'extension') ?? '?'), files: col(kinds!, r, 'files') }));
  const buckets = (hist?.rows ?? []).map((r) => ({ name: String(col(hist!, r, 'bucket') ?? '').replace(/^[a-z]\.\s*/, ''), n: num(col(hist!, r, 'functions')) }));
  const top = Math.max(1, ...buckets.map((b) => b.n));
  if (!lang && !kinds && !hist) return <div className="faint ins-empty">Analyzing…</div>;
  return (
    <div className="ins-shape">
      <div className="ins-tiles">
        {langs.slice(0, 4).map((l) => (
          <div key={l.name} className="ins-tile">
            <div className="label">{l.name}</div>
            <div className="mono ins-big">{fmt(l.lines)}</div>
            <div className="faint">lines · {fmt(l.symbols)} symbols</div>
          </div>
        ))}
        {!langs.length && <div className="faint ins-empty">No code symbols indexed.</div>}
      </div>
      {!!files.length && (
        <div className="ins-kinds" aria-label="Files by language">
          {files.slice(0, 8).map((f) => (
            <span key={f.name} className="pill">
              {f.name} · {fmt(f.files)} files
            </span>
          ))}
        </div>
      )}
      {!!buckets.length && (
        <div className="ins-hist" role="list" aria-label="Function length">
          {buckets.map((b) => (
            <div key={b.name} className="ins-hist-row" role="listitem" aria-label={`${b.name} lines: ${b.n} functions`}>
              <span className="mono faint">{b.name}</span>
              <span className="ins-hist-bar">
                <span style={{ width: `${(b.n / top) * 100}%` }} />
              </span>
              <span className="mono">{b.n.toLocaleString()}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/** Folder × folder edge counts over the most connected folders; a cell's strength is its share of the busiest pair. */
function Heatmap({ a }: { a: Analysis }) {
  const edges = a.rows.map((r) => ({ from: folderName(col(a, r, 'from_folder')), to: folderName(col(a, r, 'to_folder')), n: num(col(a, r, 'edges')) }));
  if (!edges.length) return <div className="faint ins-empty">No edges cross a folder boundary.</div>;
  const weight = new Map<string, number>();
  for (const e of edges) {
    weight.set(e.from, (weight.get(e.from) ?? 0) + e.n);
    weight.set(e.to, (weight.get(e.to) ?? 0) + e.n);
  }
  const folders = [...weight].sort((x, y) => y[1] - x[1]).slice(0, HEAT_FOLDERS).map(([f]) => f);
  const cell = new Map(edges.map((e) => [`${e.from}\u0000${e.to}`, e.n]));
  const max = Math.max(1, ...edges.filter((e) => folders.includes(e.from) && folders.includes(e.to)).map((e) => e.n));
  const short = (f: string) => f.split('/').slice(-2).join('/');
  return (
    <div className="ins-heat-wrap">
      <table className="ins-heat">
        <caption className="faint">Rows depend on columns. Top {folders.length} folders by edges.</caption>
        <thead>
          <tr>
            <th />
            {folders.map((f) => (
              <th key={f} scope="col" title={f}>
                <span className="ins-heat-col">{short(f)}</span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {folders.map((from) => (
            <tr key={from}>
              <th scope="row" title={from} className="mono ellipsis">
                {short(from)}
              </th>
              {folders.map((to) => {
                const n = cell.get(`${from}\u0000${to}`) ?? 0;
                const label = from === to ? `${from}: same folder` : `${from} → ${to}: ${n} edges`;
                return (
                  <td
                    key={to}
                    aria-label={label}
                    title={label}
                    className={from === to ? 'self' : n ? 'hot' : ''}
                    style={n && from !== to ? { background: `color-mix(in srgb, var(--blue) ${Math.round(12 + (n / max) * 78)}%, transparent)` } : undefined}
                  >
                    {n && from !== to ? n : ''}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
