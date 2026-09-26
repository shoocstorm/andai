// Refund rules. Must match docs/refund-policy.md.

/** Share of the fare refunded when a passenger cancels this many hours before departure. */
export function refundFraction(hoursBeforeDeparture: number): number {
  if (hoursBeforeDeparture > 48) return 1;
  if (hoursBeforeDeparture >= 6) return 0.5;
  return 0;
}

/** A sailing Tidewater cancelled for weather is always refunded in full. */
export function weatherRefund(fare: number): number {
  return fare;
}
