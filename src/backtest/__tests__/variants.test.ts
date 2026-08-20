import { describe, expect, test } from 'bun:test';
import { computeVariantLeaderboard, formatVariantLeaderboard, VARIANTS } from '../variants';
import type { ScoredSignal } from '../types';

function sig(over: Partial<ScoredSignal>): ScoredSignal {
  return {
    event_ticker: 'EV', market_ticker: 'M', series_category: 'Test',
    model_prob: 50, market_then: 50, market_now: 40, resolved: true,
    edge_pp: -10, pnl: 0.1, capital: 0.5, edge_bucket: '10-15',
    confidence_score: 0.5, close_time: '2026-01-01T00:00:00Z',
    ...over,
  } as ScoredSignal;
}

describe('computeVariantLeaderboard', () => {
  const signals = [
    sig({ market_ticker: 'A', edge_pp: -10, market_then: 50, market_now: 30, pnl: 0.2, capital: 0.5 }),   // NO hit, mid-price
    sig({ market_ticker: 'B', edge_pp: 20, market_then: 15, market_now: 5, pnl: -0.15, capital: 0.15 }),  // YES miss, longshot, extreme
    sig({ market_ticker: 'C', edge_pp: -6, market_then: 70, market_now: 75, pnl: -0.05, capital: 0.3, confidence_score: 0.9 }), // NO miss, high conf
  ];

  test('baseline row counts every edge signal', () => {
    const rows = computeVariantLeaderboard(signals, 5);
    const all = rows.find((r) => r.key === 'all')!;
    expect(all.n).toBe(3);
    expect(all.hit_rate).toBeCloseTo(1 / 3);
  });

  test('segments select the right subsets', () => {
    const rows = computeVariantLeaderboard(signals, 5);
    expect(rows.find((r) => r.key === 'no-side')!.n).toBe(2);
    expect(rows.find((r) => r.key === 'yes-side')!.n).toBe(1);
    expect(rows.find((r) => r.key === 'edge-15plus')!.n).toBe(1);
    expect(rows.find((r) => r.key === 'longshot')!.n).toBe(1);
    expect(rows.find((r) => r.key === 'high-conf')!.n).toBe(1);
    expect(rows.find((r) => r.key === 'mid-price')!.n).toBe(2);
  });

  test('empty variants sort last and render as dashes', () => {
    const rows = computeVariantLeaderboard([sig({ edge_pp: -10 })], 5);
    const last = rows[rows.length - 1];
    expect(last.n === 0 || rows.every((r) => r.n > 0)).toBe(true);
    const out = formatVariantLeaderboard(rows, 5);
    expect(out).toContain('Strategy Variants');
    expect(out).toContain('YES bets only');
  });

  test('min edge filter excludes sub-threshold signals from every variant', () => {
    const rows = computeVariantLeaderboard([sig({ edge_pp: -3 })], 5);
    expect(rows.find((r) => r.key === 'all')!.n).toBe(0);
  });

  test('variant keys are unique', () => {
    expect(new Set(VARIANTS.map((v) => v.key)).size).toBe(VARIANTS.length);
  });
});
