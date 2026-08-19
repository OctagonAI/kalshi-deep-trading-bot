import { describe, expect, test } from 'bun:test';
import { Database } from 'bun:sqlite';
import { migrate } from '../schema';
import { computeRealizedPnl, recordSettlement, getSettlements, summarizeSettlements } from '../settlements';
import type { KalshiSettlement } from '../settlements';

function freshDb() {
  const db = new Database(':memory:');
  migrate(db);
  return db;
}

const BASE: KalshiSettlement = {
  ticker: 'KXTEST-26-T50',
  event_ticker: 'KXTEST-26',
  market_result: 'no',
  yes_count_fp: '0.00',
  no_count_fp: '10.00',
  yes_total_cost_dollars: '0.000000',
  no_total_cost_dollars: '6.000000',
  revenue: 10,
  fee_cost: '0.35',
  settled_time: '2026-08-01T05:00:00Z',
};

describe('computeRealizedPnl', () => {
  test('revenue minus cost minus fees', () => {
    expect(computeRealizedPnl(BASE)).toBeCloseTo(10 - 6 - 0.35);
  });

  test('losing side: zero revenue leaves negative pnl', () => {
    expect(computeRealizedPnl({ ...BASE, revenue: 0, market_result: 'yes' })).toBeCloseTo(-6.35);
  });
});

describe('recordSettlement', () => {
  test('inserts once, idempotent on re-sync', () => {
    const db = freshDb();
    expect(recordSettlement(db, BASE)).toBe(true);
    expect(recordSettlement(db, BASE)).toBe(false);
    expect(getSettlements(db)).toHaveLength(1);
  });

  test('joins the model view at entry from edge_history', () => {
    const db = freshDb();
    db.prepare(
      `INSERT INTO edge_history (ticker, event_ticker, timestamp, model_prob, market_prob, edge, cache_hit, cache_miss)
       VALUES (?, ?, ?, ?, ?, ?, 0, 0)`,
    ).run('KXTEST-26-T50', 'KXTEST-26', Math.floor(new Date('2026-07-25').getTime() / 1000), 0.3, 0.6, -0.3);
    recordSettlement(db, BASE);
    const [row] = getSettlements(db);
    expect(row.model_prob_entry).toBeCloseTo(0.3);
    expect(row.edge_entry).toBeCloseTo(-0.3);
  });

  test('edge rows AFTER settlement are not used as entry context', () => {
    const db = freshDb();
    db.prepare(
      `INSERT INTO edge_history (ticker, event_ticker, timestamp, model_prob, market_prob, edge, cache_hit, cache_miss)
       VALUES (?, ?, ?, ?, ?, ?, 0, 0)`,
    ).run('KXTEST-26-T50', 'KXTEST-26', Math.floor(new Date('2026-08-15').getTime() / 1000), 0.9, 0.5, 0.4);
    recordSettlement(db, BASE);
    const [row] = getSettlements(db);
    expect(row.model_prob_entry).toBeNull();
  });
});

describe('summarizeSettlements', () => {
  test('aggregates pnl, win/loss, and model-side accuracy', () => {
    const db = freshDb();
    db.prepare(
      `INSERT INTO edge_history (ticker, event_ticker, timestamp, model_prob, market_prob, edge, cache_hit, cache_miss)
       VALUES (?, ?, ?, ?, ?, ?, 0, 0)`,
    ).run('KXTEST-26-T50', 'KXTEST-26', Math.floor(new Date('2026-07-25').getTime() / 1000), 0.3, 0.6, -0.3);
    recordSettlement(db, BASE); // model leaned NO (edge<0), result no → model side won
    recordSettlement(db, {
      ...BASE,
      ticker: 'KXTEST-26-T60',
      revenue: 0,
      market_result: 'yes',
      settled_time: '2026-08-02T05:00:00Z',
    });
    const s = summarizeSettlements(db);
    expect(s.count).toBe(2);
    expect(s.wins).toBe(1);
    expect(s.losses).toBe(1);
    expect(s.total_realized_pnl).toBeCloseTo((10 - 6 - 0.35) + (0 - 6 - 0.35));
    expect(s.with_model_view).toBe(1);
    expect(s.model_side_wins).toBe(1);
  });
});
