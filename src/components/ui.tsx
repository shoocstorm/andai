import { AnimatePresence, motion } from 'motion/react';
import type { CSSProperties, ReactNode } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Check, Copy } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { toast } from '../state/ui';

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
export function Markdown({ text }: { text: string }) {
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
        {text}
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
