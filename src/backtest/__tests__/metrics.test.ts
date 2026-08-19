
// ─── Risk-adjusted additions (Phase-1 #2) ───────────────────────────────────
import { describe as d3, expect as e3, test as t3 } from 'bun:test';
import { computeEquityRisk, alphaVsAlwaysNoPp } from '../metrics';
import type { ScoredSignal as SS3 } from '../types';

function sig(over: Partial<SS3>): SS3 {
  return {
    event_ticker: 'EV', market_ticker: 'M', series_category: 'Test',
    model_prob: 50, market_then: 50, market_now: 50, resolved: true,
    edge_pp: -10, pnl: 0, capital: 0.5, edge_bucket: '10-15',
    confidence_score: 0, close_time: '2026-01-01T00:00:00Z',
    ...over,
  } as SS3;
}

d3('computeEquityRisk', () => {
  t3('flat winners have zero drawdown and infinite risk-adjusted return', () => {
    const s = [sig({ pnl: 0.1, close_time: '2026-01-01' }), sig({ pnl: 0.1, close_time: '2026-01-02' })];
    const r = computeEquityRisk(s, 1);
    e3(r.max_drawdown_pct).toBe(0);
    e3(r.risk_adjusted_return).toBe(Infinity);
  });

  t3('peak-to-trough drop is measured against capital', () => {
    // +0.4 → -0.6 → +0.4: peak 0.4, trough -0.2 → DD 0.6 on capital 2
    const s = [
      sig({ pnl: 0.4, close_time: '2026-01-01' }),
      sig({ pnl: -0.6, close_time: '2026-01-02' }),
      sig({ pnl: 0.4, close_time: '2026-01-03' }),
    ];
    const r = computeEquityRisk(s, 2);
    e3(r.max_drawdown_pct).toBeCloseTo(0.3);
    e3(r.risk_adjusted_return).toBeCloseTo((0.2 / 2) / 0.3);
  });

  t3('ordering follows close_time, not array order', () => {
    const s = [
      sig({ pnl: 0.4, close_time: '2026-01-03' }),
      sig({ pnl: -0.6, close_time: '2026-01-02' }),
      sig({ pnl: 0.4, close_time: '2026-01-01' }),
    ];
    // Chronological: +0.4, -0.6, +0.4 → same DD as above
    const r = computeEquityRisk(s, 2);
    e3(r.max_drawdown_pct).toBeCloseTo(0.3);
  });

  t3('empty input is zeroed', () => {
    e3(computeEquityRisk([], 0)).toEqual({ max_drawdown_pct: 0, risk_adjusted_return: 0 });
  });
});

d3('alphaVsAlwaysNoPp', () => {
  t3('model matching always-NO has zero alpha', () => {
    // Model bet NO (edge<0) with the same capital/pnl as the NO benchmark.
    const s = [sig({ edge_pp: -10, market_then: 60, market_now: 40, pnl: 0.2, capital: 0.4 })];
    e3(alphaVsAlwaysNoPp(s)).toBeCloseTo(0);
  });

  t3('model betting YES when NO wins has negative alpha', () => {
    // YES bet: entry 60, now 40 → pnl -0.2 on 0.6 capital (ROI -33%).
    // Always-NO on same row: pnl +0.2 on 0.4 capital (ROI +50%).
    const s = [sig({ edge_pp: 10, market_then: 60, market_now: 40, pnl: -0.2, capital: 0.6 })];
    e3(alphaVsAlwaysNoPp(s)).toBeCloseTo(-83.33, 1);
  });
});
