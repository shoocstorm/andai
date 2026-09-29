// Sample knowledge bases shipped with the app (src-tauri/src/samples.rs), so
// the agent can be tried before the user has documents: Tidewater Ferries, a
// fictional ferry operator, as documents, as code, and as both. They're the
// agent eval's fixtures (tests/fixtures/eval/), and every suggested question
// is one of the eval's, so it's known to be answerable from the sample.

import type { KbKind } from './api';

export type SampleDef = {
  id: string;
  /** Exactly the name Rust gives the knowledge base; how an added sample is recognized. */
  name: string;
  kind: KbKind;
  blurb: string;
  questions: string[];
};

export const SAMPLES: SampleDef[] = [
  {
    id: 'tidewater-docs',
    name: 'Tidewater Ferries · Documents',
    kind: 'document',
    blurb: 'Seven documents: handbooks, a refund policy, a maintenance manual, a passenger guide, an incident log.',
    questions: [
      'How long before departure do I need to cancel to get a full refund?',
      'Where does my dog travel on the Kestrel?',
      'Which of the ferries can take my car?',
    ],
  },
  {
    id: 'tidewater-code',
    name: 'Tidewater Ferries · Code',
    kind: 'code',
    blurb: 'The booking service in TypeScript: fares, bookings, refunds, retries.',
    questions: ['How does computeFare work out the price?', 'Which functions call computeFare?', 'How many attempts does withRetry make by default?'],
  },
  {
    id: 'tidewater-mixed',
    name: 'Tidewater Ferries · Docs + code',
    kind: 'mixed',
    blurb: 'Both together: ask how the code implements the policy.',
    questions: [
      'Does the refund code give the same share as the policy for a cancellation 10 hours before departure?',
      'Which function implements the group discount from the release notes?',
      'What does cancelBooking do to work out the refund?',
    ],
  },
];

export const sampleByName = (name: string | undefined) => SAMPLES.find((s) => s.name === name);
