// Creating and cancelling bookings.

import { computeFare, type Route } from './fares';
import { refundFraction } from './refunds';
import { withRetry } from './retry';

export type Booking = {
  id: string;
  route: Route;
  passengers: number;
  vehicle: boolean;
  departure: Date;
  fare: number;
};

export type Gateway = {
  chargeCard: (amount: number) => Promise<string>;
  refundCard: (amount: number) => Promise<string>;
};

/** Prices the booking with computeFare and charges the card, retrying gateway timeouts. */
export async function createBooking(
  gateway: Gateway,
  route: Route,
  passengers: number,
  vehicle: boolean,
  departure: Date,
): Promise<Booking> {
  const fare = computeFare(route, passengers, vehicle, departure);
  const id = await withRetry(() => gateway.chargeCard(fare));
  return { id, route, passengers, vehicle, departure, fare };
}

/** Refunds the share refundFraction allows for the time left before departure, and returns the amount. */
export async function cancelBooking(gateway: Gateway, booking: Booking, now: Date): Promise<number> {
  const hours = (booking.departure.getTime() - now.getTime()) / 3_600_000;
  const amount = Math.round(booking.fare * refundFraction(hours) * 100) / 100;
  if (amount > 0) await withRetry(() => gateway.refundCard(amount));
  return amount;
}
