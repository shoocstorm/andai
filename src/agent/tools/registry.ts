// Every tool the agent can use. Registration is in code only: no plugin
// loading, no MCP, nothing the webview can add at runtime (AGENTS.md §9).

import type { KbKind } from '../../kb/api';
import type { Policy, Risk, ToolDef } from './types';
import { UG_TOOLS } from './ug';

export const TOOLS: ToolDef[] = [...UG_TOOLS];

export const toolById = (id: string) => TOOLS.find((t) => t.id === id);

/** Read-only tools run without asking by default; anything that writes or reaches the device asks. */
export const defaultPolicy = (risk: Risk): Policy => (risk === 'read' ? 'auto' : 'ask');

export const policyOf = (t: ToolDef, policies: Record<string, Policy>): Policy => policies[t.id] ?? defaultPolicy(t.risk);

/** Tools the model may be offered for a knowledge base of this kind: applicable and not switched off. */
export function available(kind: KbKind, policies: Record<string, Policy>): ToolDef[] {
  return TOOLS.filter((t) => t.kinds.includes(kind) && policyOf(t, policies) !== 'off');
}
