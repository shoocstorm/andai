import { AnimatePresence, motion } from 'motion/react';
import { Component, type CSSProperties, type ErrorInfo, type ReactNode } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Check, ChevronLeft, ChevronRight, Copy } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { toast } from '../state/ui';

/** One page of a list, as `usePaged` returns it. */
export type Paged<T> = {
  items: T[];
  page: number;
  pages: number;
  /** 1-based positions of the first and last item shown, and how many there are. */
  from: number;
  to: number;
  total: number;
  setPage: (page: number) => void;
};

/**
 * Pages through `items`, `size` at a time. With `focus`, it opens on the page
 * holding that item (the selected knowledge base, say) and moves there when
 * the focus changes; otherwise the page stays where the user put it.
 */
export function usePaged<T>(items: T[], size: number, focus?: (item: T) => boolean): Paged<T> {
  const pages = Math.max(1, Math.ceil(items.length / size));
  const at = focus ? items.findIndex(focus) : -1;
  const focusPage = at >= 0 ? Math.floor(at / size) : null;
  const [page, setPage] = useState(focusPage ?? 0);
  useEffect(() => {
    if (focusPage != null) setPage(focusPage);
  }, [focusPage]);
  const current = Math.min(page, pages - 1);
  const start = current * size;
  return {
    items: items.slice(start, start + size),
    page: current,
    pages,
    from: items.length ? start + 1 : 0,
    to: Math.min(items.length, start + size),
    total: items.length,
    setPage: (p) => setPage(Math.max(0, Math.min(pages - 1, p))),
  };
}

/** Previous / next for a `usePaged` list, with where you are. Renders nothing for a single page. */
export function Pager({ paged, label, className }: { paged: Paged<unknown>; label: string; className?: string }) {
  if (paged.pages <= 1) return null;
  const { page, pages, from, to, total, setPage } = paged;
  return (
    <nav className={`pager${className ? ` ${className}` : ''}`} aria-label={label}>
      <span className="pager-range">
        {from}–{to} of {total}
      </span>
      <button type="button" className="pager-btn" aria-label="Previous page" disabled={page === 0} onClick={() => setPage(page - 1)}>
        <ChevronLeft size={14} />
      </button>
      <span className="pager-pos mono" aria-live="polite">
        {page + 1} / {pages}
      </span>
      <button type="button" className="pager-btn" aria-label="Next page" disabled={page >= pages - 1} onClick={() => setPage(page + 1)}>
        <ChevronRight size={14} />
      </button>
    </nav>
  );
}

export function Toggle({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label: string }) {
  return (
    <button
      role="switch"
      aria-checked={checked}
      aria-label={label}
      className="toggle"
      onClick={() => onChange(!checked)}
    />
  );
}

export function Slider({
  value,
  min,
  max,
  step,
  onChange,
  label,
}: {
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  label: string;
}) {
  const fill = `${((value - min) / (max - min)) * 100}%`;
  return (
    <input
      type="range"
      className="slider"
      aria-label={label}
      min={min}
      max={max}
      step={step}
      value={value}
      style={{ '--fill': fill } as CSSProperties}
      onChange={(e) => onChange(Number(e.target.value))}
    />
  );
}

export function Segmented<T extends string | number>({
  options,
  value,
  onChange,
  label,
}: {
  options: { value: T; label: ReactNode }[];
  value: T;
  onChange: (v: T) => void;
  label: string;
}) {
  return (
    <div className="segmented" role="group" aria-label={label}>
      {options.map((o) => (
        <button key={String(o.value)} aria-pressed={o.value === value} onClick={() => onChange(o.value)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Stat({ label, value, unit, color }: { label: string; value: ReactNode; unit?: string; color?: string }) {
  return (
    <div className="panel stat">
      <div className="label">{label}</div>
      <div className="value" style={{ color }}>
        {value}
        {unit && <span className="unit">{unit}</span>}
      </div>
    </div>
  );
}

export function Bar({ value, indeterminate, color }: { value: number; indeterminate?: boolean; color?: string }) {
  return (
    <div className={`bar${indeterminate ? ' indeterminate' : ''}`}>
      <i style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%`, background: color }} />
    </div>
  );
}

const hostOf = (url?: string) => {
  try {
    return url ? new URL(url).host : '';
  } catch {
    return '';
  }
};

/**
 * Renders model output. A poisoned document can steer what the model writes
 * (prompt injection), so nothing here may reach the network or navigate
 * (AGENTS.md §9): images are never loaded, since their URL could carry chat
 * text out, and links show their real host and can only be copied. Raw HTML
 * stays off; react-markdown's default urlTransform drops `javascript:`.
 */
const CITE_MARK = /\[(\d+(?:\s*,\s*\d+)*)\](?!\()/g;
const CITE_HREF = /^#cite-(\d+)$/;

/**
 * Turns `[n]` citations (1…`max`) into links to `#cite-n`, which `Markdown`
 * renders as buttons, outside code; `[1, 2]` becomes `[1][2]`. A number past
 * `max` stays plain text. Pure.
 */
export function linkCitations(text: string, max: number): string {
  return text
    .split(/(```[\s\S]*?(?:```|$)|`[^`\n]*`)/)
    .map((part, i) =>
      i % 2
        ? part
        : part.replace(CITE_MARK, (whole, list: string) => {
            const ns = list.split(',').map((x) => Number(x.trim()));
            if (!ns.every((n) => n >= 1 && n <= max)) return whole;
            return ns.map((n) => `[\\[${n}\\]](#cite-${n})`).join('');
          }),
    )
    .join('');
}

/**
 * Model output as Markdown, made inert: no images, links copy-only. With
 * `onCite`, `[n]` citations (1…`sources`) become buttons that call it; they
 * never navigate.
 */
export function Markdown({ text, sources = 0, onCite }: { text: string; sources?: number; onCite?: (n: number) => void }) {
  return (
    <div className="md selectable">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          img: ({ src, alt }) => {
            const host = hostOf(typeof src === 'string' ? src : undefined);
            return (
              <span className="md-blocked" title="Andai never loads images from model output">
                [image blocked{alt ? `: ${alt}` : ''}
                {host && ` · ${host}`}]
              </span>
            );
          },
          a: ({ href, children }) => {
            const cite = onCite && href ? CITE_HREF.exec(href) : null;
            if (cite) {
              const n = Number(cite[1]);
              return (
                <button type="button" className="md-cite" aria-haspopup="dialog" aria-label={`Source ${n}`} onClick={() => onCite!(n)}>
                  {children}
                </button>
              );
            }
            if (!href) return <span>{children}</span>;
            const host = hostOf(href);
            return (
              <span className="md-link">
                <span>{children}</span>
                {host && <span className="md-link-host">{host}</span>}
                <button
                  type="button"
                  className="md-link-copy"
                  aria-label={`Copy link ${href}`}
                  title={href}
                  onClick={() =>
                    void navigator.clipboard
                      ?.writeText(href)
                      .then(() => toast({ tone: 'info', title: 'Link copied', body: href }))
                      .catch(() => toast({ tone: 'warn', title: 'Could not copy the link', body: href }))
                  }
                >
                  <Copy size={12} />
                </button>
              </span>
            );
          },
        }}
      >
        {onCite && sources > 0 ? linkCitations(text, sources) : text}
      </ReactMarkdown>
    </div>
  );
}

/**
 * Icon button that copies `text` (or what it returns, computed at click time)
 * and shows a check mark for a moment. For debugging aids: trace steps,
 * arguments, commands, outputs, errors.
 */
/** Icon-only by default; `children` adds a visible label (a regular button then). */
export function CopyButton({ text, label, size = 12, children }: { text: string | (() => string); label: string; size?: number; children?: ReactNode }) {
  const [done, setDone] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout>>(undefined);
  useEffect(() => () => clearTimeout(timer.current), []);
  return (
    <button
      type="button"
      className={children ? `btn secondary sm${done ? ' done' : ''}` : `copy-btn${done ? ' done' : ''}`}
      aria-label={label}
      title={label}
      onClick={(e) => {
        e.stopPropagation();
        const value = typeof text === 'function' ? text() : text;
        void navigator.clipboard
          ?.writeText(value)
          .then(() => {
            setDone(true);
            clearTimeout(timer.current);
            timer.current = setTimeout(() => setDone(false), 1400);
          })
          .catch(() => toast({ tone: 'error', title: 'Could not copy to the clipboard' }));
      }}
    >
      {done ? <Check size={size} /> : <Copy size={size} />}
      {children}
    </button>
  );
}

/**
 * A dialog over the whole window. It renders into `document.body`: opened from
 * inside a transformed element (an answer card animates with `transform`), a
 * `position: fixed` backdrop would be laid out and clipped inside that element.
 */
export function Modal({ open, onClose, children, wide, label }: { open: boolean; onClose: () => void; children: ReactNode; wide?: boolean; label?: string }) {
  return createPortal(
    <AnimatePresence>
      {open && (
        <motion.div
          className="backdrop"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.15 }}
          onMouseDown={(e) => e.target === e.currentTarget && onClose()}
          onKeyDown={(e) => e.key === 'Escape' && onClose()}
        >
          <motion.div
            className={`modal${wide ? ' wide' : ''}`}
            role="dialog"
            aria-modal="true"
            aria-label={label}
            initial={{ opacity: 0, y: 12, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 8, scale: 0.98 }}
            transition={{ duration: 0.2, ease: [0.2, 0.7, 0.2, 1] }}
          >
            {children}
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>,
    document.body,
  );
}

/**
 * Keeps a render error inside the screen it happened on. Without it React
 * unmounts the whole tree and the window goes blank, with the cause only in
 * the devtools console (seen 2026-09-27: a stored claim check the trace
 * couldn't read). App keys it by route, so moving to another screen retries.
 */
export class ScreenBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('Screen failed to render', error, info.componentStack);
  }

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;
    return (
      <div className="empty" role="alert">
        <h3>This screen couldn’t be shown</h3>
        <div className="mono selectable">{error.message}</div>
        <button className="btn sm" onClick={() => this.setState({ error: null })}>
          Try again
        </button>
      </div>
    );
  }
}

export const fmtBytes = (n: number) => {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
};

export const fmtAgo = (secs: number | null | undefined) => {
  if (!secs) return '—';
  const d = Date.now() / 1000 - secs;
  if (d < 45) return 'Just now';
  if (d < 3600) return `${Math.round(d / 60)}m ago`;
  if (d < 86400) return `${Math.round(d / 3600)}h ago`;
  return `${Math.round(d / 86400)}d ago`;
};

export const fmtTime = (ms: number) =>
  new Date(ms).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
