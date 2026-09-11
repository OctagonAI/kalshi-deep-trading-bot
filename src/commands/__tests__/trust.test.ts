import { describe, test, expect, beforeEach, afterEach, mock } from 'bun:test';
import { readFileSync } from 'node:fs';
import type { ParsedArgs } from '../parse-args.js';
import { handleTrust, formatTrustHuman, type TraderTrustCard, type TrustResult } from '../trust.js';
import type { CLIResponse } from '../json.js';

function makeArgs(o: Partial<ParsedArgs>): ParsedArgs {
  return {
    subcommand: 'trust',
    positionalArgs: [],
    json: false,
    live: false, refresh: false, report: false, dryRun: false, verbose: false,
    performance: false, resolved: false, unresolved: false,
    behavioral: false, ranked: false, showCluster: false, activeOnly: false,
    cells: false, autoProbs: false,
    parseErrors: [],
    ...o,
  };
}

function makeCard(overrides?: Partial<TraderTrustCard>): TraderTrustCard {
  const score = (value: number | null) => ({
    value,
    label: value === null ? 'No trading in 7d' : value >= 70 ? 'Tradeable' : value >= 40 ? 'Thin' : 'Very thin',
    drivers: ['24h traded notional $4,090', 'Typical bar range 1.8% of price', 'All checks pass'],
    evidence: [
      { text: 'avg spread', metric: 'avg_spread_cents', value: 1.2, window: '24h' },
      { text: 'Light trading: under $2,000 in 24h' },
    ],
    confidence: 'high' as const,
    suppressed: false,
    not_applicable: value === null,
  });
  return {
    calculation_version: 'trader_dashboard_lean_v1.14',
    computed_at: '2026-06-22T15:30:00Z',
    event_ticker: 'KX-EVT',
    venue: 'kalshi',
    event: { event_quality: { value: 70, label: 'Healthy', confidence: 'high' }, structure: 'ladder', coverage: 100 },
    scope: { total_markets: 3, scored_markets: 2 },
    markets: [
      {
        market_ticker: 'KX-EVT-A',
        title: 'France',
        is_primary: true,
        lifecycle_status: 'active',
        fair_cents: 53,
        best_bid_cents: 52,
        best_ask_cents: 53,
        spread_cents: 1,
        scores: {
          market_quality: score(85),
          liquidity: score(80),
          move_quality: score(75),
          resolution_clarity: score(90),
        },
      },
      {
        market_ticker: 'KX-EVT-B',
        title: 'Brazil',
        is_primary: false,
        lifecycle_status: 'active',
        fair_cents: 21.24,
        best_bid_cents: 20,
        best_ask_cents: 22,
        spread_cents: 2,
        scores: {
          market_quality: score(55),
          liquidity: score(50),
          move_quality: score(null),
          resolution_clarity: score(70),
        },
      },
    ],
    ...overrides,
  };
}

type FetchHandler = (url: string, init?: RequestInit) => Response | Promise<Response>;
function installFetchMock(handler: FetchHandler): void {
  globalThis.fetch = mock(async (url: string | URL | Request, init?: RequestInit) => {
    const s = typeof url === 'string' ? url : url instanceof URL ? url.toString() : url.url;
    return handler(s, init);
  }) as unknown as typeof fetch;
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
}

describe('handleTrust', () => {
  let originalFetch: typeof globalThis.fetch;
  beforeEach(() => {
    process.env.OCTAGON_API_KEY = 'sk_test';
    originalFetch = globalThis.fetch;
  });
  afterEach(() => {
    globalThis.fetch = originalFetch;
    delete process.env.OCTAGON_API_KEY;
  });

  test('missing event ticker → error', async () => {
    installFetchMock(() => jsonResponse({}));
    const resp = await handleTrust(makeArgs({ positionalArgs: [] }));
    expect(resp.ok).toBe(false);
    if (resp.ok) return;
    expect(resp.error?.code).toBe('MISSING_EVENT');
  });

  test('event 404 → EVENT_NOT_FOUND', async () => {
    installFetchMock(() => new Response('{}', { status: 404 }));
    const resp = await handleTrust(makeArgs({ positionalArgs: ['KX-EVT'] }));
    expect(resp.ok).toBe(false);
    if (resp.ok) return;
    expect(resp.error?.code).toBe('EVENT_NOT_FOUND');
  });

  test('trader_trust_json null → NO_SCORECARD (graceful, not crash)', async () => {
    installFetchMock(() => jsonResponse({
      event_ticker: 'KX-EVT', name: 'Test', trader_trust_json: null,
    }));
    const resp = await handleTrust(makeArgs({ positionalArgs: ['KX-EVT'] }));
    expect(resp.ok).toBe(false);
    if (resp.ok) return;
    expect(resp.error?.code).toBe('NO_SCORECARD');
    expect(resp.error?.message).toMatch(/no trust scorecard/i);
  });

  test('malformed trader_trust_json → PARSE_ERROR', async () => {
    installFetchMock(() => jsonResponse({
      event_ticker: 'KX-EVT', name: 'Test', trader_trust_json: 'not json',
    }));
    const resp = await handleTrust(makeArgs({ positionalArgs: ['KX-EVT'] }));
    expect(resp.ok).toBe(false);
    if (resp.ok) return;
    expect(resp.error?.code).toBe('PARSE_ERROR');
  });

  test('valid event returns table result', async () => {
    const card = makeCard();
    installFetchMock(() => jsonResponse({
      event_ticker: 'KX-EVT', name: 'Test event',
      trader_trust_json: JSON.stringify(card),
    }));
    const resp = await handleTrust(makeArgs({ positionalArgs: ['KX-EVT'] }));
    expect(resp.ok).toBe(true);
    if (!resp.ok) return;
    if (resp.data.kind !== 'table') throw new Error();
    expect(resp.data.card.markets).toHaveLength(2);
    expect(resp.data.event_name).toBe('Test event');
  });

  test('--market drills into one market', async () => {
    const card = makeCard();
    installFetchMock(() => jsonResponse({
      event_ticker: 'KX-EVT', name: 'Test',
      trader_trust_json: JSON.stringify(card),
    }));
    const resp = await handleTrust(makeArgs({ positionalArgs: ['KX-EVT'], market: 'KX-EVT-A' }));
    expect(resp.ok).toBe(true);
    if (!resp.ok) return;
    if (resp.data.kind !== 'detail') throw new Error();
    expect(resp.data.market.market_ticker).toBe('KX-EVT-A');
    expect(resp.data.verbose).toBe(false);
  });

  test('--market with unknown ticker → MARKET_NOT_IN_SCORECARD', async () => {
    const card = makeCard();
    installFetchMock(() => jsonResponse({
      event_ticker: 'KX-EVT', trader_trust_json: JSON.stringify(card),
    }));
    const resp = await handleTrust(makeArgs({ positionalArgs: ['KX-EVT'], market: 'KX-EVT-Z' }));
    expect(resp.ok).toBe(false);
    if (resp.ok) return;
    expect(resp.error?.code).toBe('MARKET_NOT_IN_SCORECARD');
  });

  test('case-insensitive ticker matching for --market', async () => {
    const card = makeCard();
    installFetchMock(() => jsonResponse({
      event_ticker: 'KX-EVT', trader_trust_json: JSON.stringify(card),
    }));
    const resp = await handleTrust(makeArgs({ positionalArgs: ['kx-evt'], market: 'kx-evt-a' }));
    expect(resp.ok).toBe(true);
  });

  test('--verbose propagates into detail result', async () => {
    const card = makeCard();
    installFetchMock(() => jsonResponse({
      event_ticker: 'KX-EVT', trader_trust_json: JSON.stringify(card),
    }));
    const resp = await handleTrust(makeArgs({ positionalArgs: ['KX-EVT'], market: 'KX-EVT-A', verbose: true }));
    expect(resp.ok).toBe(true);
    if (!resp.ok || resp.data.kind !== 'detail') throw new Error();
    expect(resp.data.verbose).toBe(true);
  });
});

describe('formatTrustHuman', () => {
  test('table view contains event roll-up, header, both markets, and legend', () => {
    const card = makeCard();
    const result: TrustResult = { kind: 'table', card, event_name: 'Test event' };
    const out = formatTrustHuman(result);
    expect(out).toContain('Trader Trust scorecard for KX-EVT');
    expect(out).toContain('Test event');
    // Roll-up comes from event.event_quality + scope, not a rollup object
    expect(out).toContain('Event quality 70 (Healthy)');
    expect(out).toContain('2/3 markets scored');
    expect(out).toContain('trader_dashboard_lean_v1.14');
    expect(out).toContain('KX-EVT-A');
    expect(out).toContain('KX-EVT-B');
    expect(out).toContain('France');
    expect(out).toContain('Brazil');
    // is_primary mark
    expect(out).toContain('*');
    expect(out).toMatch(/Higher is better/i);
  });

  test('table sorted by market quality desc', () => {
    const card = makeCard();
    // Make B have higher quality than A
    card.markets[0].scores.market_quality.value = 30;
    card.markets[1].scores.market_quality.value = 90;
    const out = formatTrustHuman({ kind: 'table', card, event_name: null });
    const aIdx = out.indexOf('KX-EVT-A');
    const bIdx = out.indexOf('KX-EVT-B');
    expect(bIdx).toBeGreaterThan(0);
    expect(bIdx).toBeLessThan(aIdx);
  });

  test('table omits the Liquidity column (it duplicates Quality)', () => {
    const out = formatTrustHuman({ kind: 'table', card: makeCard(), event_name: null });
    expect(out).toContain('Quality');
    expect(out).not.toContain('Liquidity');
  });

  test('a null score renders as em dash, never as zero', () => {
    const card = makeCard();
    const out = formatTrustHuman({ kind: 'detail', card, market: card.markets[1], verbose: false });
    expect(out).toContain('—');
    expect(out).toContain('not applicable');
    expect(out).not.toMatch(/Move.*\b0\/100/);
  });

  test('detail view shows each score with label, quote context and top drivers', () => {
    const card = makeCard();
    const out = formatTrustHuman({ kind: 'detail', card, market: card.markets[0], verbose: false });
    expect(out).toContain('KX-EVT-A');
    expect(out).toContain('(primary)');
    // Each of the four score keys appears
    expect(out).toContain('Quality');
    expect(out).toContain('Liquidity');
    expect(out).toContain('Move');
    expect(out).toContain('Resol');
    // Quote context from the card, in cents
    expect(out).toContain('Fair 53¢');
    expect(out).toContain('Spread 1¢');
    // Drivers are pre-rendered strings
    expect(out).toContain('24h traded notional $4,090');
    // Evidence is NOT shown without --verbose
    expect(out).not.toContain('Evidence:');
  });

  test('fractional cents keep one decimal', () => {
    const card = makeCard();
    const out = formatTrustHuman({ kind: 'detail', card, market: card.markets[1], verbose: false });
    expect(out).toContain('Fair 21.2¢');
  });

  test('detail view with --verbose surfaces evidence + confidence', () => {
    const card = makeCard();
    const out = formatTrustHuman({ kind: 'detail', card, market: card.markets[0], verbose: true });
    expect(out).toContain('Evidence:');
    expect(out).toContain('avg_spread_cents: 1.2 (24h)');
    // Evidence without a metric falls back to its text
    expect(out).toContain('Light trading: under $2,000 in 24h');
    expect(out).toContain('Confidence: high');
  });
});

describe('real v1.14 payload (KXNVDAA-28JANHEAD)', () => {
  const card = JSON.parse(
    readFileSync(new URL('./fixtures/trader-trust-v1.14.json', import.meta.url), 'utf8'),
  ) as TraderTrustCard;

  test('table view renders without throwing', () => {
    const out = formatTrustHuman({ kind: 'table', card, event_name: null });
    expect(out).toContain('Event quality 58 (Weak)');
    expect(out).toContain('6/6 markets scored');
    expect(out).toContain('KXNVDAA-28JANHEAD-56000');
  });

  test('detail view of a market with a not-applicable move score', () => {
    const market = card.markets.find((m) => m.market_ticker === 'KXNVDAA-28JANHEAD-46000')!;
    const out = formatTrustHuman({ kind: 'detail', card, market, verbose: true });
    expect(out).toContain('No trading in 7d');
    expect(out).toContain('(not applicable)');
    expect(out).toContain('Fair 96.1¢');
    expect(out).toContain('Light recent trading');
  });
});
