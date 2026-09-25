import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it } from 'vitest';
import { DEFAULT_PROMPT, usePersona } from '../state/persona';
import { Persona } from './Persona';

beforeEach(() => usePersona.getState().reset());

describe('Persona', () => {
  it('edits and resets the system prompt', async () => {
    const user = userEvent.setup();
    render(<Persona />);
    const box = screen.getByDisplayValue(DEFAULT_PROMPT);
    await user.clear(box);
    await user.type(box, 'Be brief.');
    expect(usePersona.getState().systemPrompt).toBe('Be brief.');
    await user.click(screen.getByRole('button', { name: 'Reset' }));
    expect(usePersona.getState().systemPrompt).toBe(DEFAULT_PROMPT);
  });

  it('selects a tone and toggles verbose reasoning', async () => {
    const user = userEvent.setup();
    render(<Persona />);
    await user.click(screen.getByRole('button', { name: /creative/i }));
    expect(usePersona.getState().tone).toBe('creative');
    await user.click(screen.getByRole('switch', { name: /verbose reasoning/i }));
    expect(usePersona.getState().verbose).toBe(true);
  });

  it('disables auto-optimize until a model is loaded', () => {
    render(<Persona />);
    expect(screen.getByRole('button', { name: /auto-optimize/i })).toBeDisabled();
  });

  it('persists settings across sessions', async () => {
    const user = userEvent.setup();
    render(<Persona />);
    await user.click(screen.getByRole('button', { name: /friendly/i }));
    expect(JSON.parse(localStorage.getItem('andai.persona')!).state.tone).toBe('friendly');
  });
});
