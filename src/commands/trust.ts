/**
 * Trader Trust scorecard.
 *
 * Surfaces Octagon's per-market scores from the `trader_trust_json` field on
 * /v1/predictions/events/{event_ticker}. Card shape: trader_dashboard_lean_v1.14.
 *
 * Two views:
 *   - kalshi trust <event-ticker>                  → table across all markets
 *   - kalshi trust <event-ticker> --market <mkt>   → single-market detail card
 *
 * Four per-market scores, each in [0, 100] and all HIGHER IS BETTER:
 *   - market_quality       (overall composite)
 *   - liquidity
 *   - move_quality
 *   - resolution_clarity
 *
 * Every score value is nullable — null means not applicable (e.g. no trades to
 * judge a move by) or suppressed for insufficient data. A null renders as "—",
 * never as 0. Prices inside the card are in cents.
 *
 * trader_trust_json is null on reports generated before this calculation
 * shipped; the handler returns a clear "no scorecard yet" error rather than
 * crashing.
 */
import { wrapSuccess, wrapError } from './json.js';
import type { CLIResponse } from './json.js';
import type { ParsedArgs } from './parse-args.js';
import { fetchOctagonEventDirect } from '../scan/octagon-events-api.js';
import { formatTable } from './scan-formatters.js';
import { theme } from '../theme.js';

/** A raw fact backing the score; shown with --verbose. */
export interface TrustEvidence {
  text: string;
  metric?: string;
  value?: unknown;
  window?: string;
  comparison?: string;
}

export interface TrustScore {
  value: number | null; // 0-100, null when not applicable or suppressed
  label: string;
  confidence: 'low' | 'medium' | 'high' | 'insufficient';
  suppressed: boolean;
  not_applicable: boolean;
  /** Pre-rendered "why" sentences, most important first. */
  drivers: string[];
  evidence?: TrustEvidence[];
  warning?: string;
}

export interface TrustMarket {
  market_ticker: string;
  title: string;
  is_primary: boolean;
  lifecycle_status?: string;
  fair_cents?: number | null;
  best_bid_cents?: number | null;
  best_ask_cents?: number | null;
  spread_cents?: number | null;
  scores: {
    market_quality: TrustScore;
    liquidity: TrustScore;
    move_quality: TrustScore;
    resolution_clarity: TrustScore;
  };
}

export interface TraderTrustCard {
  calculation_version: string;
  computed_at: string;
  event_ticker: string;
  venue?: string;
  event?: {
    event_quality?: { value: number | null; label: string; confidence: string };
    structure?: string;
    coverage?: number;
  };
  scope?: { total_markets?: number; scored_markets?: number };
  markets: TrustMarket[];
}

/** Color a 0-100 score (higher = better); null renders as a muted dash. */
function colorScore(value: number | null | undefined): string {
  if (value === null || value === undefined) return theme.muted('  —');
  const str = value.toFixed(0).padStart(3);
  if (value >= 70) return theme.success(str);
  if (value >= 40) return theme.warning(str);
  return theme.error(str);
}

/** Output shape for both table and detail views (machine-readable). */
export type TrustResult =
  | { kind: 'table'; card: TraderTrustCard; event_name: string | null }
  | { kind: 'detail'; card: TraderTrustCard; market: TrustMarket; verbose: boolean };

export async function handleTrust(args: ParsedArgs): Promise<CLIResponse<TrustResult>> {
  const eventTicker = args.positionalArgs[0]?.toUpperCase();
  if (!eventTicker) {
    return wrapError('trust', 'MISSING_EVENT', 'Usage: trust <event_ticker> [--market <market_ticker>] [--verbose]');
  }

  let event;
  try {
    event = await fetchOctagonEventDirect(eventTicker);
  } catch (err) {
    return wrapError('trust', 'OCTAGON_ERROR', err instanceof Error ? err.message : String(err));
  }
  if (!event) {
    return wrapError('trust', 'EVENT_NOT_FOUND', `No Octagon record for event ${eventTicker}.`);
  }
  if (!event.trader_trust_json) {
    return wrapError(
      'trust',
      'NO_SCORECARD',
      `No trust scorecard for ${eventTicker} yet. The Trader Trust calculation may not have run for this event — try again after the next Octagon refresh.`,
    );
  }

  let card: TraderTrustCard;
  try {
    card = JSON.parse(event.trader_trust_json) as TraderTrustCard;
  } catch (err) {
    return wrapError(
      'trust',
      'PARSE_ERROR',
      `Octagon returned malformed trader_trust_json for ${eventTicker}: ${err instanceof Error ? err.message : String(err)}`,
    );
  }
  if (!Array.isArray(card.markets) || card.markets.length === 0) {
    return wrapError('trust', 'EMPTY_SCORECARD', `Trust scorecard for ${eventTicker} has no markets.`);
  }

  // Single-market detail view
  if (args.market) {
    const wanted = args.market.toUpperCase();
    const market = card.markets.find((m) => m.market_ticker.toUpperCase() === wanted);
    if (!market) {
      return wrapError(
        'trust',
        'MARKET_NOT_IN_SCORECARD',
        `Market ${wanted} is not in the trust scorecard for ${eventTicker}. Run \`trust ${eventTicker}\` to see the available markets.`,
      );
    }
    return wrapSuccess('trust', { kind: 'detail', card, market, verbose: args.verbose });
  }

  return wrapSuccess('trust', { kind: 'table', card, event_name: event.name ?? null });
}

export function formatTrustHuman(result: TrustResult): string {
  if (result.kind === 'table') return formatTrustTable(result.card, result.event_name);
  return formatTrustDetail(result.card, result.market, result.verbose);
}

const SCORE_KEYS: Array<keyof TrustMarket['scores']> = [
  'market_quality',
  'liquidity',
  'move_quality',
  'resolution_clarity',
];

const SCORE_HEADER_LABELS: Record<keyof TrustMarket['scores'], string> = {
  market_quality: 'Quality',
  liquidity: 'Liquidity',
  move_quality: 'Move',
  resolution_clarity: 'Resol.',
};

function truncate(s: string, max: number): string {
  return s.length > max ? s.slice(0, max - 1) + '…' : s;
}

function fmtCents(v: number | null | undefined): string {
  return v === null || v === undefined ? '—' : `${Number(v.toFixed(1))}¢`;
}

function formatTrustTable(card: TraderTrustCard, eventName: string | null): string {
  const lines: string[] = [];
  const title = eventName ? ` — ${eventName}` : '';
  lines.push(`Trader Trust scorecard for ${card.event_ticker}${title}`);
  const eq = card.event?.event_quality;
  const eqStr = eq?.value != null ? `${eq.value} (${eq.label})` : '—';
  const scored = card.scope?.scored_markets ?? card.markets.length;
  const total = card.scope?.total_markets ?? card.markets.length;
  lines.push(`  Event quality ${eqStr}  ·  ${scored}/${total} markets scored`);
  lines.push(`  Calculation ${card.calculation_version}  ·  Computed ${card.computed_at.slice(0, 16).replace('T', ' ')} UTC`);
  lines.push('');

  // Sort by liquidity desc (unscored last); the most active markets surface first.
  const liq = (m: TrustMarket) => m.scores?.liquidity?.value ?? -1;
  const sorted = card.markets.slice().sort((a, b) => liq(b) - liq(a));

  const headers = ['', 'Market', 'Title', ...SCORE_KEYS.map((k) => SCORE_HEADER_LABELS[k])];
  const rows: string[][] = sorted.map((m) => [
    m.is_primary ? '*' : ' ',
    m.market_ticker,
    truncate(m.title, 30),
    ...SCORE_KEYS.map((k) => colorScore(m.scores?.[k]?.value)),
  ]);
  lines.push(formatTable(headers, rows));
  lines.push('');
  lines.push(theme.muted('  * = primary outcome.  Higher is better for every score; — = not scored (not applicable or insufficient data).'));
  lines.push(theme.muted(`  Drill into one market: trust ${card.event_ticker} --market <market_ticker> [--verbose]`));
  return lines.join('\n');
}

function formatTrustDetail(card: TraderTrustCard, market: TrustMarket, verbose: boolean): string {
  const lines: string[] = [];
  const primaryMark = market.is_primary ? ' (primary)' : '';
  lines.push(`Trader Trust — ${market.market_ticker}${primaryMark}`);
  lines.push(`  ${market.title}`);
  lines.push(`  Event ${card.event_ticker}  ·  Calculation ${card.calculation_version}  ·  Computed ${card.computed_at.slice(0, 16).replace('T', ' ')} UTC`);
  lines.push(`  Fair ${fmtCents(market.fair_cents)}  ·  Bid ${fmtCents(market.best_bid_cents)} / Ask ${fmtCents(market.best_ask_cents)}  ·  Spread ${fmtCents(market.spread_cents)}`);
  lines.push('');

  for (const key of SCORE_KEYS) {
    const score = market.scores?.[key];
    const label = SCORE_HEADER_LABELS[key].padEnd(10);
    if (!score) {
      lines.push(`  ${label}  ${colorScore(null)}      ${theme.muted('not reported')}`);
      lines.push('');
      continue;
    }
    const valueStr = score.value === null ? `${colorScore(null)}    ` : `${colorScore(score.value)}/100`;
    const why = score.value !== null ? ''
      : score.not_applicable ? ' (not applicable)'
      : score.suppressed ? ' (insufficient data)' : '';
    lines.push(`  ${label}  ${valueStr}  ${theme.muted(score.label)}${why}`);
    if (score.warning) lines.push(`      ${theme.warning(score.warning)}`);
    for (const d of score.drivers.slice(0, 3)) {
      lines.push(`      • ${d}`);
    }
    if (verbose) {
      const evidence = score.evidence ?? [];
      if (evidence.length > 0) {
        lines.push(theme.muted(`      Evidence:`));
        for (const e of evidence) lines.push(theme.muted(`        ${formatEvidence(e)}`));
      }
      lines.push(theme.muted(`      Confidence: ${score.confidence}`));
    }
    lines.push('');
  }
  return lines.join('\n');
}

function formatEvidence(e: TrustEvidence): string {
  if (!e.metric) return e.text;
  const window = e.window ? ` (${e.window})` : '';
  return `${e.metric}: ${formatEvidenceValue(e.value)}${window}`;
}

function formatEvidenceValue(v: unknown): string {
  if (v === null || v === undefined) return '—';
  if (typeof v === 'number') return v.toString();
  if (typeof v === 'string') return v;
  try { return JSON.stringify(v); } catch { return String(v); }
}
