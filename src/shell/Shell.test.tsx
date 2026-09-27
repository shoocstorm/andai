import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it } from 'vitest';
import { useLayout } from '../state/layout';
import { initTheme, useTheme } from '../state/theme';
import { useUi } from '../state/ui';
import { Sidebar, StatusBar, TopBar } from './Shell';
import { shortcut } from '../lib/platform';

beforeEach(() => {
  useLayout.setState({ navCollapsed: false, traceOpen: true });
  useTheme.getState().setTheme('dark');
  useUi.setState({ route: 'command' });
  initTheme();
});

describe('TopBar', () => {
  it('shows core standby with no model loaded', () => {
    render(<TopBar />);
    expect(screen.getByText(/core standby/i)).toBeInTheDocument();
  });

  it('toggles appearance dark ↔ light and applies it to <html>', async () => {
    const user = userEvent.setup();
    render(<TopBar />);
    await user.click(screen.getByRole('button', { name: /appearance: dark/i }));
    expect(useTheme.getState().theme).toBe('light');
    expect(document.documentElement.dataset.theme).toBe('light');
    await user.click(screen.getByRole('button', { name: /appearance: light/i }));
    expect(document.documentElement.dataset.theme).toBe('dark');
  });
});

describe('Sidebar', () => {
  it('marks the current screen and navigates', async () => {
    const user = userEvent.setup();
    render(<Sidebar />);
    expect(screen.getByRole('button', { name: /command center/i })).toHaveAttribute('aria-current', 'page');
    await user.click(screen.getByRole('button', { name: /knowledge/i }));
    expect(useUi.getState().route).toBe('knowledge');
    expect(window.location.hash).toBe('#knowledge');
  });

  it('keeps Workflows highlighted on the workflow detail screen', () => {
    useUi.setState({ route: 'workflow-detail' });
    render(<Sidebar />);
    expect(screen.getByRole('button', { name: /workflows/i })).toHaveAttribute('aria-current', 'page');
  });
});

describe('Sidebar collapse', () => {
  it('collapses to an icon rail and expands again', async () => {
    const user = userEvent.setup();
    render(<Sidebar />);
    const nav = screen.getByRole('complementary', { name: /main navigation/i });
    await user.click(screen.getByRole('button', { name: 'Collapse sidebar' }));
    expect(nav).toHaveClass('collapsed');
    expect(useLayout.getState().navCollapsed).toBe(true);
    expect(screen.getByRole('button', { name: 'Expand sidebar' })).toHaveAttribute('aria-expanded', 'false');
    await user.click(screen.getByRole('button', { name: 'Expand sidebar' }));
    expect(nav).not.toHaveClass('collapsed');
  });

  it('keeps every destination reachable by name and tooltip when collapsed', async () => {
    const user = userEvent.setup();
    useLayout.setState({ navCollapsed: true });
    render(<Sidebar />);
    const persona = screen.getByRole('button', { name: 'Persona' });
    expect(persona).toHaveAttribute('title', `Persona (${shortcut('4')})`);
    await user.click(persona);
    expect(useUi.getState().route).toBe('persona');
  });

  it('remembers the choice across launches', () => {
    useLayout.getState().toggleNav();
    expect(JSON.parse(localStorage.getItem('andai.layout')!).state).toEqual({ navCollapsed: true, traceOpen: true });
  });
});

describe('StatusBar', () => {
  it('reports no model and knowledge off by default', () => {
    render(<StatusBar />);
    expect(screen.getByText('Model: none')).toBeInTheDocument();
    expect(screen.getByText(/Knowledge: off/)).toBeInTheDocument();
  });
});
