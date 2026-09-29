import { render } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

const reduced = vi.hoisted(() => ({ on: false }));
vi.mock('motion/react', () => ({ useReducedMotion: () => reduced.on }));

import { NeuralCore } from './NeuralCore';

describe('NeuralCore', () => {
  it('is decoration only: hidden from assistive tech', () => {
    const { container } = render(<NeuralCore />);
    expect(container.firstElementChild).toHaveAttribute('aria-hidden', 'true');
  });

  it('moves a node along each orbit, unless the user asked for reduced motion', () => {
    reduced.on = false;
    const { container, unmount } = render(<NeuralCore />);
    expect(container.querySelectorAll('.nc-node animateMotion')).toHaveLength(2);
    unmount();
    reduced.on = true;
    const still = render(<NeuralCore />);
    expect(still.container.querySelectorAll('.nc-node')).toHaveLength(2);
    expect(still.container.querySelectorAll('animateMotion')).toHaveLength(0);
  });
});
