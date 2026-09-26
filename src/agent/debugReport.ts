// One turn as plain text, for pasting into a bug report or a chat with a
// coding agent: the setup (models, settings, knowledge base), each decision
// with its top options, each tool call with its arguments and ug command, the
// sources and the answer. Pure, so it's unit-tested (debugReport.test.ts).
// Tool output is left out (a step's own copy button has it in full).

import type { KbInfo } from '../kb/api';
import type { AgentStep, Message } from '../state/chat';
import { splitThink } from '../state/chat';

export type ReportEnv = {
  chatModel: string | null;
  deciderModel: string | null;
  engine: string | null;
  ug: string | null;
  agent: { agentMode: boolean; maxSteps: number; minConfidence: number; policies: Record<string, string> };
  kb: Pick<KbInfo, 'name' | 'kind' | 'kindOverride' | 'nodes' | 'sources'> | null;
};

const pct = (p: number) => `${Math.round(p * 100)}%`;

/** A ug argv as a command a shell can run: every argument single-quoted when it needs it. */
export function shellCommand(argv: string[]): string {
  return ['ug', ...argv].map((a) => (/^[\w@%+=:,./-]+$/.test(a) ? a : `'${a.replace(/'/g, `'\\''`)}'`)).join(' ');
}

function stepLines(s: AgentStep): string[] {
  const d = s.decision;
  const head = d
    ? `${s.index + 1}. ${s.action} @ ${pct(d.confidence)} (${d.model}, ${d.slot}, ${d.ms} ms; top: ${[...d.options]
        .sort((a, b) => b.probability - a.probability)
        .slice(0, 3)
        .map((o) => `${o.id} ${pct(o.probability)}`)
        .join(', ')}${d.bounded?.length ? `; bounded: ${d.bounded.join(', ')}` : ''})`
    : `${s.index + 1}. ${s.action} (no decision scored)`;
  const out = [head];
  if (s.note) out.push(`   note: ${s.note}`);
  const c = s.call;
  if (c) {
    out.push(`   args: ${c.args ? JSON.stringify(c.args) : '—'}${c.argModel ? ` (by ${c.argModel}, ${c.argAttempts} attempt${c.argAttempts === 1 ? '' : 's'})` : ''}`);
    if (c.argsRaw) out.push(`   raw args: ${JSON.stringify(c.argsRaw)}`);
    if (c.argv) out.push(`   $ ${shellCommand(c.argv)}`);
    out.push(
      `   → ${c.status}${c.approval ? ` (${c.approval})` : ''}${c.ms != null ? ` · ${c.ms} ms` : ''}${c.outputBytes != null ? ` · ${c.outputBytes} bytes` : ''}${c.truncated ? ' · truncated' : ''}`,
    );
    if (c.observation) out.push(`   saw: ${c.observation}`);
    if (c.error) out.push(`   error: ${c.error}`);
  }
  return out;
}

export function debugReport(m: Message, question: string, env: ReportEnv): string {
  const { agent, kb } = env;
  const policies = Object.entries(agent.policies);
  const lines = [
    `## Andai turn · ${new Date(m.createdAt).toISOString()}`,
    `- models: chat ${env.chatModel ?? m.stats?.model ?? '—'} · decider ${env.deciderModel ?? '(chat model)'}${env.engine ? ` · ${env.engine}` : ''}`,
    `- agent: ${agent.agentMode ? 'on' : 'off'} · max ${agent.maxSteps} calls · min confidence ${pct(agent.minConfidence)} · policies ${policies.length ? policies.map(([k, v]) => `${k}=${v}`).join(', ') : 'defaults'}`,
    kb
      ? `- kb: “${kb.name}” · ${kb.kind}${kb.kindOverride ? ' (set by user)' : ''} · ${kb.nodes} nodes · ${kb.sources.length} files: ${kb.sources
          .slice(0, 12)
          .map((s) => s.file)
          .join(', ')}${kb.sources.length > 12 ? ', …' : ''}`
      : `- kb: ${m.kbName ?? 'none'}`,
    env.ug ? `- ${env.ug}` : '',
    '',
    '### Question',
    question,
    '',
    '### Steps',
    ...(m.steps ?? []).map((s) => `- ${s.kind}: ${s.status}${s.ms != null ? ` · ${s.ms} ms` : ''} · ${s.detail}`),
  ];
  if (m.agent?.length) lines.push('', '### Agent', ...m.agent.flatMap(stepLines));
  if (m.sources?.length) {
    lines.push('', '### Sources', ...m.sources.map((h, i) => `[${i + 1}] ${h.file}:${h.start_line}-${h.end_line}${h.name && h.name !== h.file ? ` — ${h.name}` : ''}`));
  }
  const { answer } = splitThink(m.content);
  lines.push('', '### Answer', answer.trim() || (m.stopped ? '(stopped before an answer)' : '(none)'));
  if (m.stats) {
    const s = m.stats;
    lines.push(
      '',
      `### Stats`,
      `${s.tokens} tokens · ${s.tokPerSec.toFixed(1)} tok/s · prompt ${s.promptTokens ?? '—'} / ctx ${s.nCtx} · first token ${s.firstTokenMs != null ? Math.round(s.firstTokenMs) : '—'} ms · total ${Math.round(s.totalMs)} ms`,
    );
  }
  return lines.filter((l, i, a) => !(l === '' && a[i - 1] === '')).join('\n');
}
