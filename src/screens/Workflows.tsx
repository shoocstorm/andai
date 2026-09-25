import {
  BarChart3,
  Database,
  Globe,
  Landmark,
  Mail,
  Network,
  PlusSquare,
  RefreshCcw,
  Settings2,
  Share2,
  CheckCircle2,
} from 'lucide-react';
import { AnimatePresence, motion } from 'motion/react';
import { useState } from 'react';
import { APPROVALS, TOOLS, WORKFLOWS, type Approval, type Tool } from '../mock/workflows';
import { useKb } from '../state/kb';
import { usePersona } from '../state/persona';
import { toast, useUi } from '../state/ui';

const TOOL_ICON = { globe: Globe, chart: BarChart3, mail: Mail, db: Database, sync: RefreshCcw, graph: Share2 };
const APPROVAL_ICON = { mail: Mail, bank: Landmark, db: Database };

export function Workflows() {
  const go = useUi((s) => s.go);
  const approvalOn = usePersona((s) => s.approval);
  const [pending, setPending] = useState<Approval[]>(APPROVALS);
  const [tools, setTools] = useState<Tool[]>(TOOLS);
  const kbReady = useKb((s) => s.kbs.some((k) => k.nodes > 0));

  const resolve = (a: Approval, ok: boolean) => {
    setPending((p) => p.filter((x) => x.id !== a.id));
    toast({
      tone: ok ? 'ok' : 'warn',
      title: `${a.tool} ${ok ? 'approved' : 'rejected'}`,
      body: ok ? 'Action released to the execution queue (simulated).' : 'Action cancelled and logged.',
    });
  };

  return (
    <div className="screen">
      <div className="page wf">
        <div className="wf-hero">
          <div className="page-hero">
            <h1>Workflow &amp; Tool Library</h1>
            <p>Configure atomic skills and orchestrate complex autonomous loops via the Andai substrate.</p>
          </div>
          <div className="wf-hero-actions">
            <button className="btn secondary" style={{ height: 52, padding: '0 30px' }} onClick={() => toast({ tone: 'info', title: 'Tool scaffolding is simulated', body: 'Register Tool opens the manifest editor in a future build.' })}>
              Initialize Tool
            </button>
            <button className="btn primary" style={{ height: 52, padding: '0 30px' }} onClick={() => go('workflow-detail')}>
              Deploy Workflow
            </button>
          </div>
        </div>

        <div className="wf-grid">
          <section>
            <div className="wf-approval-head">
              <span className="dot pulse" style={{ color: 'var(--amber)' }} />
              <span className="label amber" style={{ fontSize: 14, letterSpacing: '0.18em' }}>
                Man-in-the-loop approval
              </span>
              <span className="label" style={{ letterSpacing: '0.06em' }}>
                {pending.length} pending
              </span>
            </div>
            {!approvalOn && (
              <div className="wf-note">
                Approval gating is <b>off</b> in Persona — these actions would auto-execute.
              </div>
            )}
            <div className="wf-approvals">
              <AnimatePresence initial={false}>
                {pending.map((a) => {
                  const Icon = APPROVAL_ICON[a.icon];
                  return (
                    <motion.div
                      key={a.id}
                      layout
                      className={`wf-approval ${a.severity}`}
                      initial={{ opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, x: -30, transition: { duration: 0.2 } }}
                    >
                      <div className="wf-approval-top">
                        <Icon size={22} color={a.severity === 'critical' ? 'var(--amber)' : 'var(--violet)'} />
                        <span className="wf-tool mono">{a.tool}</span>
                        {a.severity === 'critical' && <span className="wf-critical mono">CRITICAL_AUTH</span>}
                      </div>
                      <p>
                        {a.body} {a.target && <span style={{ color: 'var(--blue)' }}>{a.target}</span>}
                        {a.target && '.'} {a.tail}
                      </p>
                      {a.severity === 'critical' ? (
                        <div className="wf-approval-actions">
                          <button className="btn amber" onClick={() => resolve(a, true)}>
                            Approve
                          </button>
                          <button className="btn danger" onClick={() => resolve(a, false)}>
                            Reject
                          </button>
                        </div>
                      ) : (
                        <div className="wf-approval-actions one">
                          <button className="btn secondary" onClick={() => resolve(a, true)}>
                            Review details
                          </button>
                        </div>
                      )}
                    </motion.div>
                  );
                })}
              </AnimatePresence>
              {!pending.length && (
                <div className="panel pad empty">
                  <CheckCircle2 size={26} color="var(--green)" />
                  <div>All clear — no actions awaiting approval.</div>
                </div>
              )}
            </div>
          </section>

          <section>
            <div className="section-title">
              <span className="label" style={{ color: 'var(--text)', fontSize: 14, letterSpacing: '0.22em' }}>
                Active workflows
              </span>
            </div>
            <div className="wf-active">
              {WORKFLOWS.map((w) => (
                <div key={w.id} className="panel wf-card">
                  <div className="wf-card-top">
                    {w.icon === 'network' ? (
                      <Network size={30} color="var(--blue)" />
                    ) : (
                      <RefreshCcw size={28} color="var(--violet)" />
                    )}
                    {w.status === 'nominal' && <span className="dot pulse" style={{ color: 'var(--blue)' }} />}
                  </div>
                  <h3 className="display">{w.name}</h3>
                  <p className="muted">{w.body}</p>
                  <div className="wf-card-foot">
                    <span className="mono" style={{ color: w.status === 'nominal' ? 'var(--blue)' : 'var(--text-2)' }}>
                      STATUS: {w.status.toUpperCase()}
                    </span>
                    <button className="wf-configure" onClick={() => go('workflow-detail')}>
                      CONFIGURE <Settings2 size={14} />
                    </button>
                  </div>
                </div>
              ))}
            </div>

            <div className="section-title" style={{ marginTop: 40 }}>
              <span className="label" style={{ color: 'var(--text)', fontSize: 14, letterSpacing: '0.22em' }}>
                Atomic skills &amp; tools
              </span>
            </div>
            <div className="wf-tools">
              {tools.map((t) => {
                const Icon = TOOL_ICON[t.icon];
                const off = !t.enabled;
                return (
                  <div key={t.id} className={`wf-tool-card${off ? ' off' : ''}`}>
                    <div className="wf-card-top">
                      <Icon size={26} color={off ? 'var(--text-3)' : t.accent} />
                      <span className="wf-square" style={{ background: t.enabled ? t.accent : 'var(--text-4)' }} />
                    </div>
                    <h4>
                      {t.name}
                      {t.live && (
                        <span className={`pill ${kbReady ? 'violet' : ''}`}>
                          live
                        </span>
                      )}
                    </h4>
                    <p>{t.body}</p>
                    <button
                      className="btn ghost block"
                      onClick={() => {
                        if (t.live) return go('knowledge');
                        if (off) {
                          setTools((ts) => ts.map((x) => (x.id === t.id ? { ...x, enabled: true, accent: 'var(--blue)' } : x)));
                          toast({ tone: 'ok', title: `${t.name} initialized`, body: 'Simulated connector ready.' });
                        } else toast({ tone: 'info', title: `${t.name} configuration`, body: 'Tool configuration is simulated in this build.' });
                      }}
                    >
                      {off ? 'Initialize' : t.live ? 'Open knowledge' : 'Configure'}
                    </button>
                  </div>
                );
              })}
              <button
                className="wf-register"
                onClick={() => toast({ tone: 'info', title: 'Register tool', body: 'Custom tool manifests arrive in a future build.' })}
              >
                <PlusSquare size={34} />
                <span className="label">Register tool</span>
              </button>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}

