# Booking app release notes

## 2.5.1 (2026-05-14)

- Fixed a crash when a booking had no passengers left after a transfer.

## 2.5.0 (2026-04-20)

- Paper tickets are retired: every ticket is now a QR code in the app or in
  the confirmation email.
- Peak pricing: sailings on Fridays and Sundays cost 25% more.

## 2.4.0 (2026-03-02)

- Group bookings: parties of 10 or more passengers get a 15% discount on the
  whole booking.
- Payment calls are retried with exponential backoff when the card gateway
  times out.
