import type { KbKind, KbToolCall, SearchHit } from '../../kb/api';
import type { ObjectSchema } from './validate';

/** What a tool can touch. Only `read` tools exist today; the others default to Ask and can't be set to Auto casually. */
export type Risk = 'read' | 'write' | 'device';
/** Per-tool user setting: run without asking, ask each time, or never offer it to the model. */
export type Policy = 'auto' | 'ask' | 'off';

/** What a tool run gives the agent: passages for the answer, and a short line for the next decision. */
export type Evidence = { hits: SearchHit[]; summary: string };

export type ToolContext = { kind: KbKind; k: number; maxChars: number };

export type ToolDef = {
  id: string;
  title: string;
  /** One line; the option text the decision model scores (llm/decide.ts). */
  option: string;
  /** Longer, for the Tools screen. */
  description: string;
  kinds: KbKind[];
  risk: Risk;
  /** The ug command it runs, for the Tools screen. */
  command: string;
  /** Arguments the model fills in; null when the tool takes none. */
  schema: ObjectSchema | null;
  /**
   * Enum arguments Laya picks as typed choices, in the same pass as the
   * decision (llm/decide.ts `ArgChoice`); the chat model then writes the rest.
   * Without Laya the chat model writes them as usual.
   */
  choices?: { arg: string; question: string; options: { id: string; text: string }[] }[];
  /** How to fill the arguments, given what the knowledge base holds. */
  guide: (kind: KbKind) => string;
  /** Validated arguments → the call Rust re-validates and runs. */
  toCall: (args: Record<string, unknown>, ctx: ToolContext) => KbToolCall;
  /** ug's JSON → evidence. Must tolerate any shape: the output is untrusted. */
  observe: (output: unknown) => Evidence;
};
