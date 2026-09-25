import { create } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';

export type Tone = 'professional' | 'friendly' | 'creative';

export const TONES: Record<Tone, { label: string; blurb: string; clause: string }> = {
  professional: {
    label: 'Professional',
    blurb: 'Calm, precise, analytical',
    clause: 'Tone: professional. Be concise and precise; prefer short paragraphs and bullet points.',
  },
  friendly: {
    label: 'Friendly',
    blurb: 'Warm, plain-spoken, encouraging',
    clause: 'Tone: friendly. Be warm and approachable, explain simply, and avoid jargon.',
  },
  creative: {
    label: 'Creative',
    blurb: 'Vivid, exploratory, playful',
    clause: 'Tone: creative. Offer vivid framing and fresh angles while staying accurate.',
  },
};

export const DEFAULT_PROMPT =
  'You are ANDAI_01, a local-first AI agent running entirely on this computer. ' +
  'Your tone is calm, precise, and analytical. Deliver maximum information density with minimal fluff. ' +
  'When knowledge-base context is provided, ground your answer in it and cite sources as [1], [2]. ' +
  'If the context does not contain the answer, say so plainly instead of guessing.';

type PersonaState = {
  agentName: string;
  systemPrompt: string;
  tone: Tone;
  temperature: number;
  maxTokens: number;
  verbose: boolean;
  approval: boolean;
  set: (patch: Partial<Omit<PersonaState, 'set' | 'reset'>>) => void;
  reset: () => void;
};

const defaults = {
  agentName: 'ANDAI_01',
  systemPrompt: DEFAULT_PROMPT,
  tone: 'professional' as Tone,
  temperature: 0.6,
  maxTokens: 768,
  verbose: false,
  approval: true,
};

export const usePersona = create<PersonaState>()(
  persist(
    (set) => ({
      ...defaults,
      set: (patch) => set(patch),
      reset: () => set(defaults),
    }),
    { name: 'andai.persona', storage: createJSONStorage(() => localStorage) },
  ),
);
