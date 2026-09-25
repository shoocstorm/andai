import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';
import { MODELS } from '../llm/models';
import { useTheme } from '../state/theme';
import { Settings } from './Settings';

describe('Settings', () => {
  it('lists every catalog model with a download action', () => {
    render(<Settings />);
    for (const m of MODELS) expect(screen.getByText(m.name)).toBeInTheDocument();
    expect(screen.getAllByRole('button', { name: /download & load/i })).toHaveLength(MODELS.length);
  });

  it('switches appearance from the picker', async () => {
    const user = userEvent.setup();
    render(<Settings />);
    await user.click(screen.getByRole('radio', { name: /light/i }));
    expect(useTheme.getState().mode).toBe('light');
    expect(screen.getByRole('radio', { name: /light/i })).toHaveAttribute('aria-checked', 'true');
  });
});
