import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';
import { useUi } from '../state/ui';
import { Workflows } from './Workflows';

describe('Workflows (simulated)', () => {
  it('resolves an approval and updates the pending count', async () => {
    const user = userEvent.setup();
    render(<Workflows />);
    expect(screen.getByText('3 pending')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Approve' }));
    expect(await screen.findByText('2 pending')).toBeInTheDocument();
    expect(useUi.getState().toasts.at(-1)?.title).toMatch(/EMAIL_DISPATCH approved/);
  });

  it('opens the node editor from Deploy Workflow', async () => {
    const user = userEvent.setup();
    render(<Workflows />);
    await user.click(screen.getByRole('button', { name: /deploy workflow/i }));
    expect(useUi.getState().route).toBe('workflow-detail');
  });
});
