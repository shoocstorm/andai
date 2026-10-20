// Answer X-ray: the claim check's verdict on every sentence of an answer, in
// place. A checked sentence is tinted by how likely its cited passage supports
// it (amber when the check flagged it) and ends in its score, which opens the
// claim dialog; a cited sentence the check has no score for, and a statement
// with no citation, are marked too. It shows what the check already scored
// (Message.support); nothing is re-run.
import { ScanEye } from 'lucide-react';
import type { CSSProperties, ReactNode } from 'react';
import type { SupportRecord } from '../agent/claims';
import type { SearchHit } from '../kb/api';
import { xrayIndex, type Verdict } from '../lib/xray';
import { ClaimBadge, ROUGH_CHECK } from './AgentTrace';

/** `Markdown`'s `sentence` renderer for an answer's claim check. */
export function xraySentence(r: SupportRecord, sources: SearchHit[]) {
  const lookup = xrayIndex(r);
  return (text: string, children: ReactNode) => (
    <XraySentence v={lookup(text)} r={r} sources={sources}>
      {children}
    </XraySentence>
  );
}

function XraySentence({ v, r, sources, children }: { v: Verdict | null; r: SupportRecord; sources: SearchHit[]; children: ReactNode }) {
  if (!v) return <>{children}</>;
  if (v.kind === 'checked') {
    return (
      <span className={`xr ${v.flagged ? 'xr-flag' : 'xr-ok'}`} style={{ '--xr-score': v.worst.score } as CSSProperties}>
        {children} <ClaimBadge x={v.worst} r={r} sources={sources} flagged={v.flagged} />
      </span>
    );
  }
  const note = v.kind === 'cited' ? 'Cited, but the claim check has no score for this sentence' : 'No citation: this sentence isn’t tied to a source';
  return (
    <span className={`xr ${v.kind === 'cited' ? 'xr-cited' : 'xr-none'}`} title={note}>
      {children}
      <span className="xr-sr">{` (${note})`}</span>
    </span>
  );
}

/** Turns the X-ray on and off under an answer; only offered when the claim check ran. */
export function XrayToggle({ on, onChange }: { on: boolean; onChange: (on: boolean) => void }) {
  return (
    <button
      type="button"
      className="xr-toggle"
      aria-pressed={on}
      title="Show how well each sentence is supported by the source it cites"
      onClick={(e) => {
        e.stopPropagation();
        onChange(!on);
      }}
    >
      <ScanEye size={12} /> X-ray
    </button>
  );
}

/** Above an X-rayed answer: what the marks mean, and how rough the check is. */
export function XrayLegend({ r }: { r: SupportRecord }) {
  return (
    <div className="xr-legend" aria-label="X-ray legend">
      <span>
        <i className="xr-key xr-ok" /> supported, by score
      </span>
      <span>
        <i className="xr-key xr-flag" /> may not be supported
      </span>
      <span>
        <i className="xr-key xr-none" /> no citation
      </span>
      <span className="faint">
        Scores are {r.model}’s claim check; click one to see the passage. {ROUGH_CHECK}
      </span>
    </div>
  );
}
