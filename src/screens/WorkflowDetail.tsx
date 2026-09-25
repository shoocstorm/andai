import {
  ArrowLeft,
  BarChart3,
  Braces,
  Clock,
  CloudUpload,
  GitBranch,
  History,
  Maximize,
  Minus,
  Play,
  Plus,
  Puzzle,
  RotateCcw,
  Share2,
  Sparkles,
  SquareTerminal,
  UserCheck,
  X,
  Zap,
} from 'lucide-react';
import { AnimatePresence, motion } from 'motion/react';
import { useCallback, useEffect, useRef, useState, type PointerEvent as RPointerEvent } from 'react';
import { Slider } from '../components/ui';
import { FLOW_EDGES, FLOW_NODES, NODE_H, NODE_W, RUN_SCRIPT, type FlowNode, type NodeKind } from '../mock/workflows';
import { toast, useUi } from '../state/ui';

type RunState = 'idle' | 'processing' | 'pending' | 'done';

const KIND: Record<NodeKind, { icon: typeof Clock; color: string; label: string }> = {
  trigger: { icon: Clock, color: 'var(--text-2)', label: 'Trigger' },
  skill: { icon: BarChart3, color: 'var(--blue)', label: 'Atomic skill' },
  rag: { icon: Share2, color: 'var(--violet)', label: 'Knowledge search' },
  approval: { icon: UserCheck, color: 'var(--amber)', label: 'Human approval' },
  llm: { icon: Sparkles, color: 'var(--blue)', label: 'Local LLM' },
  upload: { icon: CloudUpload, color: 'var(--text-2)', label: 'S3 upload' },
};

const PALETTE = [
  { icon: Puzzle, label: 'Atomic Skills' },
  { icon: GitBranch, label: 'Logic Gates' },
  { icon: Braces, label: 'Variables' },
  { icon: Zap, label: 'Triggers' },
  { icon: History, label: 'History' },
];

type Event = { t: string; text: string; tone: 'info' | 'warn' | 'ok' };
const stamp = () => new Date().toLocaleTimeString([], { hour12: false });

export function WorkflowDetail() {
  const go = useUi((s) => s.go);
  const [nodes, setNodes] = useState<FlowNode[]>(FLOW_NODES);
  const [selected, setSelected] = useState<string | null>('n4');
  const [view, setView] = useState({ x: 120, y: 180, z: 0.85 });
  const [run, setRun] = useState<Record<string, RunState>>({});
  const [running, setRunning] = useState(false);
  const [palette, setPalette] = useState('Atomic Skills');
  const [events, setEvents] = useState<Event[]>([
    { t: '14:20:11', text: 'NODE_INIT: 0x219 successful', tone: 'info' },
    { t: '14:20:12', text: 'PAYLOAD_STREAM: Received 4.2MB', tone: 'info' },
  ]);
  const canvas = useRef<HTMLDivElement>(null);
  const approvalResolve = useRef<((ok: boolean) => void) | null>(null);
  const cancelled = useRef(false);

  const log = (text: string, tone: Event['tone'] = 'info') => setEvents((e) => [...e.slice(-40), { t: stamp(), text, tone }]);
  const node = nodes.find((n) => n.id === selected) ?? null;

  // ── simulated run ────────────────────────────────────────────────────
  const simulate = async () => {
    if (running) return;
    cancelled.current = false;
    setRunning(true);
    setRun({});
    log('RUNTIME_START: ALPHA_FLOW', 'ok');
    const order = ['n1', 'n2', ['n3', 'n4'], 'n5', 'n6'] as const;
    const step = async (id: string) => {
      const s = RUN_SCRIPT[id];
      setRun((r) => ({ ...r, [id]: id === 'n4' ? 'pending' : 'processing' }));
      if (id === 'n4') {
        setSelected('n4');
        log('AWAITING_HUMAN: Timeout in 4:59s', 'warn');
        const ok = await new Promise<boolean>((res) => (approvalResolve.current = res));
        approvalResolve.current = null;
        if (!ok) throw new Error('rejected');
        log('HUMAN_SIGNATURE: @admin_core approved', 'ok');
      } else {
        log(`NODE_EXEC: ${nodes.find((n) => n.id === id)?.hex} ${s.working}`);
        await new Promise((r) => setTimeout(r, s.ms));
      }
      if (cancelled.current) throw new Error('cancelled');
      setRun((r) => ({ ...r, [id]: 'done' }));
      log(`NODE_DONE: ${s.done}`, 'ok');
    };
    try {
      for (const o of order) {
        if (typeof o === 'string') await step(o);
        else await Promise.all(o.map(step));
      }
      log('RUNTIME_COMPLETE: 6/6 nodes nominal', 'ok');
      toast({ tone: 'ok', title: 'Workflow run complete', body: 'Simulated — digest uploaded to S3.' });
    } catch (e) {
      log(`RUNTIME_HALT: ${e instanceof Error ? e.message : e}`, 'warn');
    } finally {
      setRunning(false);
    }
  };

  useEffect(
    () => () => {
      cancelled.current = true;
      approvalResolve.current?.(false);
    },
    [],
  );

  // ── pan / zoom ───────────────────────────────────────────────────────
  const zoomAt = useCallback((factor: number, cx?: number, cy?: number) => {
    setView((v) => {
      const z = Math.min(1.6, Math.max(0.3, v.z * factor));
      const rect = canvas.current?.getBoundingClientRect();
      const px = cx ?? (rect ? rect.width / 2 : 0);
      const py = cy ?? (rect ? rect.height / 2 : 0);
      return { z, x: px - ((px - v.x) * z) / v.z, y: py - ((py - v.y) * z) / v.z };
    });
  }, []);

  useEffect(() => {
    const el = canvas.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = el.getBoundingClientRect();
      if (e.ctrlKey || e.metaKey) zoomAt(Math.exp(-e.deltaY * 0.01), e.clientX - rect.left, e.clientY - rect.top);
      else setView((v) => ({ ...v, x: v.x - e.deltaX, y: v.y - e.deltaY }));
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, [zoomAt]);

  const drag = useRef<{ kind: 'pan' | 'node'; id?: string; sx: number; sy: number; ox: number; oy: number; moved: boolean } | null>(null);

  const onPointerDown = (e: RPointerEvent, id?: string) => {
    e.stopPropagation();
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    const n = id ? nodes.find((x) => x.id === id) : null;
    drag.current = {
      kind: id ? 'node' : 'pan',
      id,
      sx: e.clientX,
      sy: e.clientY,
      ox: n ? n.x : view.x,
      oy: n ? n.y : view.y,
      moved: false,
    };
  };
  const onPointerMove = (e: RPointerEvent) => {
    const d = drag.current;
    if (!d) return;
    const dx = e.clientX - d.sx;
    const dy = e.clientY - d.sy;
    if (Math.abs(dx) + Math.abs(dy) > 3) d.moved = true;
    if (d.kind === 'pan') setView((v) => ({ ...v, x: d.ox + dx, y: d.oy + dy }));
    else setNodes((ns) => ns.map((n) => (n.id === d.id ? { ...n, x: d.ox + dx / view.z, y: d.oy + dy / view.z } : n)));
  };
  const onPointerUp = () => {
    const d = drag.current;
    drag.current = null;
    if (d?.kind === 'node' && !d.moved && d.id) setSelected(d.id);
    if (d?.kind === 'pan' && !d.moved) setSelected(null);
  };

  const fit = () => {
    const rect = canvas.current?.getBoundingClientRect();
    if (!rect) return;
    const minX = Math.min(...nodes.map((n) => n.x));
    const maxX = Math.max(...nodes.map((n) => n.x + NODE_W));
    const minY = Math.min(...nodes.map((n) => n.y));
    const maxY = Math.max(...nodes.map((n) => n.y + NODE_H));
    const z = Math.min(1.2, Math.max(0.3, Math.min((rect.width - 120) / (maxX - minX), (rect.height - 160) / (maxY - minY))));
    setView({ z, x: (rect.width - (maxX - minX) * z) / 2 - minX * z, y: (rect.height - (maxY - minY) * z) / 2 - minY * z });
  };

  const [size, setSize] = useState({ w: 900, h: 560 });
  useEffect(() => {
    const el = canvas.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setSize({ w: e.contentRect.width, h: e.contentRect.height }));
    ro.observe(el);
    // open like the reference: zoomed in on Extract Data → Human Approval, edges running off-canvas
    const t = setTimeout(() => {
      const r = el.getBoundingClientRect();
      const z = 0.85;
      setView({ z, x: r.width / 2 - 735 * z, y: r.height / 2 - 235 * z });
    }, 30);
    return () => {
      ro.disconnect();
      clearTimeout(t);
    };
  }, []);

  const byId = Object.fromEntries(nodes.map((n) => [n.id, n]));
  const activeCount = Object.values(run).filter((s) => s === 'processing' || s === 'pending').length;

  return (
    <div className="screen wd">
      <aside className="wd-left">
        <button className="wd-back" onClick={() => go('workflows')}>
          <ArrowLeft size={14} /> Library
        </button>
        <div className="wd-editor-card">
          <div className="agent-tile" style={{ width: 42, height: 42 }}>
            <SquareTerminal size={19} />
          </div>
          <div>
            <div className="agent-name" style={{ fontSize: 16 }}>
              NODE_EDITOR_V2
            </div>
            <div className="mono faint" style={{ fontSize: 11.5 }}>
              Active Session: Alpha-9
            </div>
          </div>
        </div>
        <button className="btn violet block" style={{ textTransform: 'none', letterSpacing: 0, fontFamily: 'var(--font-body)', fontSize: 14 }} onClick={() => toast({ tone: 'info', title: 'Skill builder is simulated' })}>
          <Plus size={15} /> Create New Skill
        </button>
        <div className="label" style={{ margin: '28px 0 10px' }}>
          Components
        </div>
        {PALETTE.map((p) => (
          <button key={p.label} className="wd-palette" aria-pressed={palette === p.label} onClick={() => setPalette(p.label)}>
            <p.icon size={19} />
            {p.label}
          </button>
        ))}
      </aside>

      <section
        className="wd-canvas"
        ref={canvas}
        onPointerDown={(e) => onPointerDown(e)}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        style={{ backgroundPosition: `${view.x}px ${view.y}px`, backgroundSize: `${24 * view.z}px ${24 * view.z}px` }}
      >
        <div className="wd-status panel" onPointerDown={(e) => e.stopPropagation()}>
          <div>
            <div className="label">Workflow status</div>
            <div className="wd-status-v">
              <span className={`dot${running ? ' pulse' : ''}`} style={{ color: running ? 'var(--blue)' : 'var(--text-3)' }} />
              {running ? `RUNTIME_ACTIVE: ALPHA_FLOW · ${activeCount} live` : 'RUNTIME_READY: ALPHA_FLOW'}
            </div>
          </div>
          {running ? (
            <button
              className="btn danger sm"
              onClick={() => {
                cancelled.current = true;
                approvalResolve.current?.(false);
              }}
            >
              <X size={13} /> Halt
            </button>
          ) : (
            <button className="btn primary sm" onClick={() => void simulate()}>
              <Play size={13} fill="currentColor" /> Run simulation
            </button>
          )}
        </div>

        <div className="wd-world" style={{ transform: `translate(${view.x}px, ${view.y}px) scale(${view.z})` }}>
          <svg className="wd-edges" width="4000" height="2000" style={{ left: -1000, top: -1000 }}>
            <g transform="translate(1000 1000)">
              {FLOW_EDGES.map(([a, b]) => {
                const A = byId[a];
                const B = byId[b];
                const x1 = A.x + NODE_W;
                const y1 = A.y + NODE_H / 2;
                const x2 = B.x;
                const y2 = B.y + NODE_H / 2;
                const mx = (x1 + x2) / 2;
                const hot = run[a] === 'done' && (run[b] === 'processing' || run[b] === 'pending' || run[b] === 'done');
                return (
                  <g key={a + b}>
                    <path d={`M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`} className={`wd-edge${hot ? ' hot' : ''}`} />
                    <circle cx={x1} cy={y1} r={5} className="wd-port" />
                    <circle cx={x2} cy={y2} r={5} className="wd-port" />
                  </g>
                );
              })}
            </g>
          </svg>
          {nodes.map((n) => (
            <NodeCard
              key={n.id}
              n={n}
              state={run[n.id] ?? 'idle'}
              selected={selected === n.id}
              onPointerDown={(e) => onPointerDown(e, n.id)}
            />
          ))}
        </div>

        <Minimap nodes={nodes} view={view} run={run} selected={selected} size={size} />
        <div className="wd-zoom panel" onPointerDown={(e) => e.stopPropagation()}>
          <button onClick={() => zoomAt(1 / 1.15)} aria-label="Zoom out">
            <Minus size={16} />
          </button>
          <span className="mono">{Math.round(view.z * 100)}%</span>
          <button onClick={() => zoomAt(1.15)} aria-label="Zoom in">
            <Plus size={16} />
          </button>
          <span className="wd-zoom-sep" />
          <button onClick={fit} aria-label="Fit to screen">
            <Maximize size={15} />
          </button>
        </div>
      </section>

      <AnimatePresence>
        {node && (
          <motion.aside
            className="wd-right"
            key="cfg"
            initial={{ x: 40, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: 40, opacity: 0 }}
            transition={{ duration: 0.2, ease: [0.2, 0.7, 0.2, 1] }}
          >
            <NodeConfig
              node={node}
              state={run[node.id] ?? 'idle'}
              onClose={() => setSelected(null)}
              onDecide={(ok) => approvalResolve.current?.(ok)}
              events={events}
            />
          </motion.aside>
        )}
      </AnimatePresence>
    </div>
  );
}

function NodeCard({
  n,
  state,
  selected,
  onPointerDown,
}: {
  n: FlowNode;
  state: RunState;
  selected: boolean;
  onPointerDown: (e: RPointerEvent) => void;
}) {
  const k = KIND[n.kind];
  const Icon = k.icon;
  const s = RUN_SCRIPT[n.id];
  const color = state === 'pending' || n.kind === 'approval' ? 'var(--amber)' : state === 'processing' ? 'var(--blue)' : k.color;
  const statusText =
    state === 'processing' ? 'Processing' : state === 'pending' ? 'Pending' : state === 'done' ? 'Complete' : n.kind === 'trigger' ? 'Armed' : 'Idle';
  const sub = state === 'done' ? s.done : state === 'idle' ? k.label : s.working;
  return (
    <div
      className={`wd-node ${state}${selected ? ' selected' : ''}`}
      style={{ left: n.x, top: n.y, width: NODE_W, height: NODE_H, ['--node' as string]: color }}
      onPointerDown={onPointerDown}
    >
      <div className="wd-node-top">
        <Icon size={24} color={color} />
        <span className="mono" style={{ color }}>
          {n.hex}
        </span>
      </div>
      <div className="wd-node-title">{n.title.toUpperCase()}</div>
      {state === 'processing' ? (
        <div className="bar indeterminate" style={{ margin: '10px 0 6px' }}>
          <i />
        </div>
      ) : null}
      <div className="wd-node-sub mono">{sub}</div>
      <div className="wd-node-status" style={{ color: state === 'done' ? 'var(--green)' : state === 'idle' ? 'var(--text-3)' : color }}>
        {statusText}
      </div>
    </div>
  );
}

function Minimap({
  nodes,
  view,
  run,
  selected,
  size,
}: {
  nodes: FlowNode[];
  view: { x: number; y: number; z: number };
  run: Record<string, RunState>;
  selected: string | null;
  size: { w: number; h: number };
}) {
  const s = 0.09;
  return (
    <div className="wd-minimap panel" onPointerDown={(e) => e.stopPropagation()}>
      <div style={{ position: 'relative', transform: 'translate(12px, 16px)' }}>
        {nodes.map((n) => (
          <i
            key={n.id}
            className="wd-mini-node"
            style={{
              left: n.x * s,
              top: n.y * s,
              width: NODE_W * s,
              height: NODE_H * s,
              background:
                run[n.id] === 'pending'
                  ? 'color-mix(in srgb, var(--amber) 55%, transparent)'
                  : run[n.id] === 'processing' || n.id === selected
                    ? 'var(--blue)'
                    : run[n.id] === 'done'
                      ? 'color-mix(in srgb, var(--green) 45%, transparent)'
                      : 'var(--edge)',
            }}
          />
        ))}
        <i
          className="wd-mini-view"
          style={{ left: (-view.x / view.z) * s, top: (-view.y / view.z) * s, width: (size.w / view.z) * s, height: (size.h / view.z) * s }}
        />
      </div>
    </div>
  );
}

function NodeConfig({
  node,
  state,
  onClose,
  onDecide,
  events,
}: {
  node: FlowNode;
  state: RunState;
  onClose: () => void;
  onDecide: (ok: boolean) => void;
  events: Event[];
}) {
  const k = KIND[node.kind];
  const Icon = k.icon;
  const [auditor, setAuditor] = useState('@admin_core');
  const [threshold, setThreshold] = useState(100);
  const [template, setTemplate] = useState('Workflow {{id}} requires immediate validation. Payload size: {{payload.length}}kb.');
  const [halt, setHalt] = useState(true);
  const logRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [events.length]);
  return (
    <>
      <div className="wd-cfg-head">
        <span className="label" style={{ color: 'var(--text)', fontSize: 14, letterSpacing: '0.16em' }}>
          Node configuration
        </span>
        <button className="icon-btn" onClick={onClose} aria-label="Close">
          <X size={20} color="var(--text-2)" />
        </button>
      </div>
      <div className="wd-cfg-card">
        <div className="wd-cfg-icon">
          <Icon size={22} color={k.color} />
        </div>
        <div>
          <div style={{ fontWeight: 700, letterSpacing: '0.02em' }}>{node.title.toUpperCase()}</div>
          <div className="mono" style={{ color: 'var(--amber)', fontSize: 12 }}>
            ID: {node.hex}-{node.kind.toUpperCase()}
          </div>
        </div>
      </div>

      {node.kind === 'approval' && state === 'pending' && (
        <div className="wd-cfg-pending">
          <div className="label amber" style={{ marginBottom: 10 }}>
            Awaiting your signature
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <button className="btn amber" style={{ flex: 1 }} onClick={() => onDecide(true)}>
              Approve
            </button>
            <button className="btn danger" style={{ flex: 1 }} onClick={() => onDecide(false)}>
              Reject
            </button>
          </div>
        </div>
      )}

      <div className="wd-field">
        <div className="label">Assigned auditor</div>
        <select className="field" value={auditor} onChange={(e) => setAuditor(e.target.value)}>
          <option>@admin_core</option>
          <option>@security_ops</option>
          <option>@finance_lead</option>
        </select>
      </div>
      <div className="wd-field">
        <div className="label" style={{ display: 'flex', justifyContent: 'space-between' }}>
          Approval threshold <span className="mono blue-text">{threshold}%</span>
        </div>
        <Slider label="Approval threshold" min={0} max={100} step={5} value={threshold} onChange={setThreshold} />
      </div>
      <div className="wd-field">
        <div className="label">Notification template</div>
        <textarea className="field mono" rows={3} value={template} onChange={(e) => setTemplate(e.target.value)} style={{ fontSize: 13 }} />
      </div>
      <label className="wd-check">
        <input type="checkbox" checked={halt} onChange={(e) => setHalt(e.target.checked)} />
        Halt execution until resolved
      </label>
      <button className="btn primary block" style={{ marginTop: 18 }} onClick={() => toast({ tone: 'ok', title: `${node.hex} configuration applied` })}>
        Apply changes
      </button>
      <button
        className="btn ghost block"
        style={{ marginTop: 10 }}
        onClick={() => {
          setAuditor('@admin_core');
          setThreshold(100);
          setHalt(true);
        }}
      >
        <RotateCcw size={13} /> Reset to default
      </button>

      <div className="wd-events">
        <div className="label" style={{ display: 'flex', alignItems: 'center', gap: 8, color: 'var(--text)' }}>
          <span className="dot" style={{ color: 'var(--violet)' }} /> System events
        </div>
        <div className="logblock wd-log" ref={logRef}>
          {events.map((e, i) => (
            <div key={i} className={e.tone === 'warn' ? 'log-warn' : e.tone === 'ok' ? 'log-ok' : 'log-info'}>
              <span className="log-violet">[{e.t}]</span> {e.text}
            </div>
          ))}
        </div>
      </div>
    </>
  );
}
