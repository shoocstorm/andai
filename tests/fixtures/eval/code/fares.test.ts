// Tests for fare calculation. Run with the booking service's test runner.

import { describe, expect, it } from 'vitest';
import { applyGroupDiscount, computeFare } from './fares';

const wednesday = new Date('2026-06-03T09:00:00Z');
const friday = new Date('2026-06-05T09:00:00Z');

describe('computeFare', () => {
  it('charges the base fare per passenger', () => {
    expect(computeFare('harlow-pellin', 2, false, wednesday)).toBe(24);
  });

  it('adds the vehicle surcharge once and the peak multiplier on Fridays', () => {
    expect(computeFare('harlow-pellin', 1, true, friday)).toBe(38.13);
  });
});

describe('applyGroupDiscount', () => {
  it('takes 15% off for parties of ten or more', () => {
    expect(applyGroupDiscount(100, 10)).toBe(85);
    expect(applyGroupDiscount(100, 9)).toBe(100);
  });
});
