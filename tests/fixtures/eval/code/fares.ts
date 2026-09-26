// Fare calculation for the Tidewater booking service.

export type Route = 'harlow-pellin' | 'harlow-dunmere';

/** Base fare per passenger, in euros. */
export const BASE_FARES: Record<Route, number> = {
  'harlow-pellin': 12,
  'harlow-dunmere': 19.5,
};

/** Added once per booking that brings a vehicle. */
export const VEHICLE_SURCHARGE = 18.5;

/** Fridays and Sundays cost more (release 2.5.0). */
export const PEAK_MULTIPLIER = 1.25;
const PEAK_DAYS = [5, 0]; // Friday, Sunday (Date.getDay())

/** Parties of this size or larger get GROUP_DISCOUNT off the whole booking. */
export const GROUP_SIZE = 10;
export const GROUP_DISCOUNT = 0.15;

/** 15% off the whole booking for parties of GROUP_SIZE passengers or more. */
export function applyGroupDiscount(total: number, passengers: number): number {
  return passengers >= GROUP_SIZE ? total * (1 - GROUP_DISCOUNT) : total;
}

/**
 * The price of a booking: the route's base fare per passenger, plus the
 * vehicle surcharge, times the peak multiplier on peak days, then the group
 * discount. Rounded to cents.
 */
export function computeFare(route: Route, passengers: number, vehicle: boolean, date: Date): number {
  let total = BASE_FARES[route] * passengers;
  if (vehicle) total += VEHICLE_SURCHARGE;
  if (PEAK_DAYS.includes(date.getDay())) total *= PEAK_MULTIPLIER;
  total = applyGroupDiscount(total, passengers);
  return Math.round(total * 100) / 100;
}
