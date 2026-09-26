// Scoring for the agent eval (scripts/eval-agent.mjs, src/eval.ts). Pure
// functions over the cases file and the harness's CASE records, so the
// scorecard is unit-tested (tests/unit/eval-lib.test.ts) without a model.
import { readFileSync } from 'node:fs';

export function loadCases(path) {
  const raw = JSON.parse(readFileSync(path, 'utf8'));
  return { notFound: raw.notFound, cases: raw.cases };
}

/** A case's facts as case-insensitive regexes; `$notFound` stands for the shared "the KB doesn't say" pattern. */
export const factRegexes = (c, notFound) => c.facts.map((f) => new RegExp(f === '$notFound' ? notFound : f, 'i'));

/** The answer without Qwen3's think block, which may restate the passages. */
const answerText = (s) => s.replace(/<think>[\s\S]*?(<\/think>|$)/g, '').trim();

const mean = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null);

/** One question: what the agent did against what was expected. */
export function scoreCase(c, r, notFound) {
  const answer = answerText(r.answer ?? '');
  const first = r.steps[0]?.action ?? null;
  const facts = factRegexes(c, notFound);
  const missing = r.error ? c.facts : facts.filter((re) => !re.test(answer)).map((re) => re.source);
  const calls = r.steps.filter((s) => s.call).map((s) => s.call);
  const decisions = r.steps.filter((s) => s.decision).map((s) => s.decision);
  const errors = calls.filter((x) => x.status === 'error').length;
  const skipped = calls.filter((x) => x.status === 'skipped' || x.status === 'denied').length;
  const empty = calls.filter((x) => x.status === 'done' && !x.hits).length;
  // Cited [n] must name one of the sources the UI lists (AGENTS.md §1.7).
  const cites = [...answer.matchAll(/\[(\d+)\]/g)].map((m) => Number(m[1]));
  const grounded = r.sources.length ? cites.length > 0 && cites.every((n) => n >= 1 && n <= r.sources.length) : null;
  return {
    id: c.id,
    kb: c.kb,
    first,
    firstOk: c.first.includes(first),
    factsOk: r.error ? false : facts.length ? missing.length === 0 : null,
    missing,
    grounded,
    actions: r.steps.map((s) => s.action),
    calls: calls.length,
    wasted: errors + skipped + empty,
    wastedBy: { errors, empty, skipped },
    argsInvalid: calls.filter((x) => x.argsFallback || (x.status === 'error' && !x.args)).length,
    noSymbol: calls.filter((x) => /No symbol named/i.test(`${x.error ?? ''} ${x.observation ?? ''}`)).length,
    fallbacks: r.steps.filter((s) => s.fallback).length,
    decisions: decisions.length,
    decisionMs: decisions.map((d) => d.ms),
    decisionPromptTokens: decisions.map((d) => d.promptTokens).filter((t) => t != null),
    seconds: r.ms / 1000,
    error: r.error,
  };
}

/** The scorecard over all questions. Rates are 0–1; null when no question applies. */
export function scorecard(scored) {
  const withFacts = scored.filter((s) => s.factsOk !== null);
  const withSources = scored.filter((s) => s.grounded !== null);
  const calls = scored.reduce((n, s) => n + s.calls, 0);
  const sum = (k) => scored.reduce((n, s) => n + s[k], 0);
  return {
    questions: scored.length,
    firstActionAccuracy: scored.length ? scored.filter((s) => s.firstOk).length / scored.length : null,
    factHitRate: withFacts.length ? withFacts.filter((s) => s.factsOk).length / withFacts.length : null,
    groundedRate: withSources.length ? withSources.filter((s) => s.grounded).length / withSources.length : null,
    calls,
    wastedCalls: sum('wasted'),
    wastedPerQuestion: scored.length ? sum('wasted') / scored.length : null,
    argsInvalid: sum('argsInvalid'),
    noSymbolErrors: sum('noSymbol'),
    fallbacks: sum('fallbacks'),
    errors: scored.filter((s) => s.error).length,
    decisionsPerQuestion: mean(scored.map((s) => s.decisions)),
    secondsPerQuestion: mean(scored.map((s) => s.seconds)),
    msPerDecision: mean(scored.flatMap((s) => s.decisionMs)),
    promptTokensPerDecision: mean(scored.flatMap((s) => s.decisionPromptTokens)),
  };
}

/** Questions whose outcome differs between two reports: first action, its correctness, or the facts. */
export function diffCases(before, after) {
  const prev = new Map(before.map((s) => [s.id, s]));
  return after.flatMap((s) => {
    const p = prev.get(s.id);
    if (!p) return [{ id: s.id, change: 'new' }];
    const changes = [];
    if (p.first !== s.first) changes.push(`first ${p.first} → ${s.first}`);
    if (p.factsOk !== s.factsOk) changes.push(`facts ${p.factsOk} → ${s.factsOk}`);
    if (p.actions.join(',') !== s.actions.join(',')) changes.push(`actions ${p.actions.join('→')} ⇒ ${s.actions.join('→')}`);
    return changes.length ? [{ id: s.id, change: changes.join('; ') }] : [];
  });
}
