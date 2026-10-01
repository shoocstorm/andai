import {
  AlertTriangle,
  ArrowRight,
  Check,
  Download,
  ExternalLink,
  FileText,
  Loader2,
  Lock,
  MessageSquareQuote,
  Network,
  RotateCcw,
  ShieldCheck,
  Sparkles,
  Terminal,
} from 'lucide-react';
import { AnimatePresence, motion } from 'motion/react';
import { useId, useState } from 'react';
import { CopyButton, Modal } from '../components/ui';
import { openUgWebsite } from '../kb/api';
import { useKb } from '../state/kb';
import { closeUgSetup, installUg, running, useUgInstall, type InstallStage } from '../state/ugInstall';
import { toast, useUi } from '../state/ui';

/* theme-literal: start */
/** UltraGraph's brand colors (its favicon), the same in both themes. */
const NODES = [
  { cx: 9, cy: 9, fill: '#22d3ee' },
  { cx: 23, cy: 9, fill: '#f472b6' },
  { cx: 23, cy: 23, fill: '#4ade80' },
  { cx: 9, cy: 23, fill: '#c084fc' },
];
/* theme-literal: end */
const SQUARE = 'M9 9L23 9L23 23L9 23Z';

/**
 * UltraGraph's mark: four linked nodes. At rest it floats, the nodes breathe
 * and a pulse of light runs along the links; installing speeds it up, and
 * once ug is ready the links light up all the way round.
 */
export function UgLogo({ size = 44, state = 'idle' }: { size?: number; state?: 'idle' | 'busy' | 'done' }) {
  const id = useId().replace(/:/g, '');
  return (
    <svg className={`ugs-logo ${state}`} width={size} height={size} viewBox="0 0 32 32" overflow="visible" aria-hidden="true">
      <defs>
        <linearGradient id={`${id}-flow`} x1="0" y1="0" x2="1" y2="1">
          {NODES.slice(0, 3).map((n, i) => (
            <stop key={n.fill} offset={i / 2} stopColor={n.fill} />
          ))}
        </linearGradient>
      </defs>
      <path className="ugs-edge" d={SQUARE} />
      <path className="ugs-flow" d={SQUARE} pathLength={100} stroke={`url(#${id}-flow)`} />
      {NODES.map((n, i) => (
        <g key={n.fill} style={{ animationDelay: `${i * 0.45}s` }} className="ugs-node">
          <circle className="ugs-halo" cx={n.cx} cy={n.cy} r={3.4} fill={n.fill} style={{ animationDelay: `${i * 0.8}s` }} />
          <circle cx={n.cx} cy={n.cy} r={3.4} fill={n.fill} />
        </g>
      ))}
    </svg>
  );
}

/** Why ug matters, in one glance: what goes in, what it builds, what you get. */
const FLOW = [
  { icon: FileText, title: 'Your files', sub: 'Docs, PDFs, code' },
  { icon: Network, title: 'Knowledge graph', sub: 'Built on-device by ug' },
  { icon: MessageSquareQuote, title: 'Cited answers', sub: 'From your own sources' },
];

function Flow() {
  return (
    <ol className="ugs-flow-row" aria-label="How Andai uses ug">
      {FLOW.map(({ icon: Icon, title, sub }, i) => (
        <li key={title}>
          {i > 0 && <ArrowRight className="ugs-flow-arrow" size={14} aria-hidden="true" />}
          <span className="ugs-flow-icon">
            <Icon size={16} />
          </span>
          <span className="ugs-flow-title">{title}</span>
          <span className="ugs-flow-sub">{sub}</span>
        </li>
      ))}
    </ol>
  );
}

const MB = (n: number) => (n / 1e6).toFixed(1);

const STEPS: { stage: InstallStage; label: string }[] = [
  { stage: 'lookup', label: 'Find the latest release' },
  { stage: 'download', label: 'Download' },
  { stage: 'verify', label: 'Verify the checksum' },
  { stage: 'install', label: 'Install to ~/.local/bin/ug' },
];
const ORDER: InstallStage[] = ['lookup', 'download', 'verify', 'install', 'check', 'done'];

function Steps() {
  const { stage, version, done, total } = useUgInstall();
  const at = ORDER.indexOf(stage);
  return (
    <ol className="ugs-steps" aria-label="Install progress">
      {STEPS.map((s, i) => {
        const state = at > i ? 'done' : at === i ? 'active' : 'todo';
        return (
          <li key={s.stage} className={`ugs-step ${state}`} aria-current={state === 'active' ? 'step' : undefined}>
            <span className="ugs-step-mark">
              {state === 'done' ? <Check size={12} strokeWidth={3} /> : state === 'active' ? <Loader2 size={13} className="spin" /> : i + 1}
            </span>
            <span className="ugs-step-body">
              <span className="ugs-step-label">
                {s.label}
                {s.stage === 'lookup' && version && <span className="mono ugs-step-meta">{version}</span>}
                {s.stage === 'download' && total > 0 && (
                  <span className="mono ugs-step-meta">
                    {MB(done)} / {MB(total)} MB
                  </span>
                )}
              </span>
              {s.stage === 'download' && state === 'active' && (
                <div
                  className="bar ugs-bar"
                  role="progressbar"
                  aria-label="Download"
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={total ? Math.round((done / total) * 100) : 0}
                >
                  <i style={{ width: `${total ? (done / total) * 100 : 0}%` }} />
                </div>
              )}
            </span>
          </li>
        );
      })}
    </ol>
  );
}

/** Installing by hand: UltraGraph's own terminal command, and its site. */
function Manual({ open: initial }: { open?: boolean }) {
  const cmd = useKb((s) => s.ug?.installCommand);
  const [open, setOpen] = useState(!!initial);
  const website = () => void openUgWebsite().catch((e) => toast({ tone: 'error', title: 'Could not open ultra-graph.web.app', body: String(e) }));
  return (
    <div className="ugs-manual">
      <button type="button" className="ugs-manual-toggle" aria-expanded={open} onClick={() => setOpen(!open)}>
        <Terminal size={13} /> Prefer the terminal?
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.18 }}
            style={{ overflow: 'hidden' }}
          >
            <div className="ugs-manual-body">
              {cmd && (
                <div className="ugs-cmd">
                  <code>{cmd}</code>
                  <CopyButton text={cmd} label="Copy the install command" size={13} />
                </div>
              )}
              <button type="button" className="btn ghost sm" onClick={website}>
                ultra-graph.web.app <ExternalLink size={12} />
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

/**
 * Offers to install ug, then shows the install through to "ready". `dialog`
 * is the launch prompt; `card` sits on the Knowledge and Settings screens.
 */
export function UgInstall({ variant, onDone }: { variant: 'dialog' | 'card'; onDone?: () => void }) {
  const canInstall = useKb((s) => s.ug?.canInstall ?? false);
  const { stage, version, error } = useUgInstall();
  const go = useUi((s) => s.go);
  const onKnowledge = useUi((s) => s.route === 'knowledge');
  const busy = running(stage);

  const status =
    stage === 'done' ? (
      <motion.div key="done" className="ugs-done" initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }}>
        <span className="ugs-done-mark">
          <Check size={22} strokeWidth={3} />
        </span>
        <div>
          <div className="ugs-done-title">UltraGraph{version ? ` ${version}` : ''} is ready</div>
          <div className="muted">Andai can now turn your files into a knowledge graph and answer from it.</div>
        </div>
        {variant === 'dialog' || !onKnowledge ? (
          <div className="ugs-actions">
            <button
              className="btn primary"
              onClick={() => {
                go('knowledge');
                onDone?.();
              }}
            >
              <Sparkles size={14} /> Build a knowledge base
            </button>
          </div>
        ) : null}
      </motion.div>
    ) : busy ? (
      <motion.div key="busy" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
        <Steps />
      </motion.div>
    ) : (
      <motion.div key="idle" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
        {stage === 'error' && (
          <div className="ugs-error" role="alert">
            <AlertTriangle size={15} />
            <span>{error}</span>
          </div>
        )}
        {canInstall ? (
          <>
            <div className="ugs-actions">
              <button className="btn primary ugs-cta" onClick={() => void installUg()}>
                {stage === 'error' ? <RotateCcw size={16} /> : <Download size={16} />}
                {stage === 'error' ? 'Try again' : 'Install UltraGraph'}
              </button>
              {variant === 'dialog' && (
                <button className="btn ghost ugs-later" onClick={onDone}>
                  Not now
                </button>
              )}
            </div>
            <ul className="ugs-trust" aria-label="About the install">
              <li>
                <Download size={12} /> About 25 MB
              </li>
              <li>
                <ShieldCheck size={12} /> SHA-256 verified
              </li>
              <li>
                <Lock size={12} /> No password · <span className="mono">~/.local/bin</span>
              </li>
            </ul>
            <Manual open={stage === 'error'} />
          </>
        ) : (
          <>
            <p className="ugs-fine">Download UltraGraph for this platform from its website, then reopen Andai.</p>
            <div className="ugs-actions">
              <button
                className="btn primary ugs-cta"
                onClick={() => void openUgWebsite().catch((e) => toast({ tone: 'error', title: 'Could not open ultra-graph.web.app', body: String(e) }))}
              >
                Get UltraGraph <ExternalLink size={14} />
              </button>
              {variant === 'dialog' && (
                <button className="btn ghost" onClick={onDone}>
                  Not now
                </button>
              )}
            </div>
          </>
        )}
      </motion.div>
    );

  return (
    <section className={`ugs ugs-${variant}`} aria-label="Install UltraGraph">
      <div className="ugs-hero">
        <UgLogo size={variant === 'dialog' ? 132 : 84} state={stage === 'done' ? 'done' : busy ? 'busy' : 'idle'} />
      </div>
      <div className="ugs-copy">
        <div className="label ugs-kicker">ug · UltraGraph</div>
        <h3 className="ugs-title">{stage === 'done' ? 'Knowledge engine connected' : busy ? 'Installing UltraGraph…' : 'Give Andai its knowledge engine'}</h3>
        {stage !== 'done' && !busy && (
          <p className="ugs-lede">
            <b>ug</b> turns your documents and code into a private knowledge graph. Andai searches it to answer from your own
            files, with sources. Nothing leaves your machine.
          </p>
        )}
      </div>
      {stage !== 'done' && !busy && <Flow />}
      <div aria-live="polite">
        <AnimatePresence mode="wait" initial={false}>
          {status}
        </AnimatePresence>
      </div>
    </section>
  );
}

/** The prompt Andai opens at launch when ug is missing (state/ugInstall.ts). */
export function UgSetupDialog() {
  const open = useUgInstall((s) => s.dialog);
  return (
    <Modal open={open} onClose={closeUgSetup} label="Install UltraGraph">
      <UgInstall variant="dialog" onDone={closeUgSetup} />
    </Modal>
  );
}
