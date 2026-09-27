import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { SAMPLES } from '../kb/samples';
import { addSample, useKb } from '../state/kb';
import { openUgWebsite } from '../kb/api';
import { GetUltraGraph, Samples } from './Knowledge';

vi.mock('../kb/api', async (original) => ({
  ...(await original<typeof import('../kb/api')>()),
  openUgWebsite: vi.fn().mockResolvedValue(undefined),
}));

vi.mock('../state/kb', async (original) => ({
  ...(await original<typeof import('../state/kb')>()),
  addSample: vi.fn().mockResolvedValue({ slug: 'x' }),
}));

beforeEach(() => {
  useKb.setState({ kbs: [], grounding: null });
  vi.mocked(addSample).mockClear();
});

describe('sample knowledge bases', () => {
  it('offers each bundled sample and adds the one picked', async () => {
    const user = userEvent.setup();
    const onAdded = vi.fn();
    render(<Samples title="Or try a sample" onAdded={onAdded} />);
    const list = within(screen.getByRole('region', { name: 'Sample knowledge bases' }));
    for (const s of SAMPLES) expect(list.getByText(s.name)).toBeInTheDocument();
    await user.click(list.getByRole('button', { name: 'Add sample Tidewater Ferries · Code' }));
    expect(addSample).toHaveBeenCalledWith('tidewater-code');
    expect(onAdded).toHaveBeenCalled();
  });

  it('shows a sample that is already added as added', () => {
    useKb.setState({ kbs: [{ slug: 'tidewater-ferries-documents', name: 'Tidewater Ferries · Documents' } as never] });
    render(<Samples title="Samples" />);
    const added = screen.getByRole('button', { name: 'Add sample Tidewater Ferries · Documents' });
    expect(added).toBeDisabled();
    expect(added).toHaveTextContent('Added');
    expect(screen.getByRole('button', { name: 'Add sample Tidewater Ferries · Code' })).toBeEnabled();
  });
});

describe('ug install prompt', () => {
  it('opens the UltraGraph website through Rust', async () => {
    const user = userEvent.setup();
    render(<GetUltraGraph />);
    await user.click(screen.getByRole('button', { name: /Get UltraGraph/ }));
    expect(openUgWebsite).toHaveBeenCalledOnce();
  });
});
