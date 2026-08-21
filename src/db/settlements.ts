/**
 * Settlement ledger — realized P&L captured from Kalshi /portfolio/settlements.
 *
 * This is the ground-truth side of the learning loop: every settled position
 * is recorded with the model's view at entry (joined from edge_history), so
 * calibration and reflection can compare what the model believed with what
 * actually happened. Rows are immutable; syncs are idempotent.
 */
import type { Database } from 'bun:sqlite';
import { computeBrier } from '../eval/brier.js';

export interface SettlementRow {
  ticker: string;
  event_ticker: string;
  market_result: string;
  yes_count_fp: number;
  no_count_fp: number;
  yes_total_cost: number;
  no_total_cost: number;
  revenue: number;
  fee_cost: number;
  realized_pnl: number;
  settled_time: string;
  model_prob_entry: number | null;
  market_prob_entry: number | null;
  edge_entry: number | null;
}

/** Raw settlement object from Kalshi's API (fixed-point dollar strings). */
export interface KalshiSettlement {
  ticker: string;
  event_ticker?: string;
  market_result: string;
  yes_count_fp?: string | number;
  no_count_fp?: string | number;
  yes_total_cost_dollars?: string | number;
  no_total_cost_dollars?: string | number;
  revenue?: string | number;
  value?: string | number;
  fee_cost?: string | number;
  settled_time: string;
  [key: string]: unknown;
}

function num(v: string | number | undefined | null): number {
  if (v === undefined || v === null) return 0;
  const n = typeof v === 'number' ? v : parseFloat(v);
  return Number.isFinite(n) ? n : 0;
}

/**
 * Realized P&L for one settlement: what the position paid out minus what it
 * cost, minus fees. `revenue` is the settlement payout; costs are the sum
 * paid for both sides (a spread holds both YES and NO).
 */
export function computeRealizedPnl(s: KalshiSettlement): number {
  const revenue = num(s.revenue) || num(s.value);
  const cost = num(s.yes_total_cost_dollars) + num(s.no_total_cost_dollars);
  return revenue - cost - num(s.fee_cost);
}

/**
 * Insert a settlement, joining the model's entry view from edge_history
 * (latest edge row for the ticker at or before the settlement time).
 * Returns true when the row is new, false when it was already recorded.
 */
export function recordSettlement(db: Database, s: KalshiSettlement): boolean {
  const settledEpoch = Math.floor(new Date(s.settled_time).getTime() / 1000);
  // "Entry view" means the model's belief when the position was OPENED. When
  // a locally-tracked position exists, bound the edge lookup by its
  // opened_at; otherwise fall back to the settlement time (edges recorded
  // between entry and settlement would otherwise masquerade as entry views).
  const localPosition = db
    .prepare(`SELECT opened_at FROM positions WHERE ticker = ? AND opened_at IS NOT NULL ORDER BY opened_at ASC LIMIT 1`)
    .get(s.ticker) as { opened_at: number } | undefined;
  const entryBound = localPosition?.opened_at ?? settledEpoch;
  const entry = db
    .prepare(
      `SELECT model_prob, market_prob, edge FROM edge_history
       WHERE ticker = ? AND cache_miss = 0 AND timestamp <= ?
       ORDER BY timestamp DESC LIMIT 1`,
    )
    .get(s.ticker, entryBound) as { model_prob: number; market_prob: number; edge: number } | undefined;

  const result = db
    .prepare(
      `INSERT OR IGNORE INTO settlements (
         ticker, event_ticker, market_result,
         yes_count_fp, no_count_fp, yes_total_cost, no_total_cost,
         revenue, fee_cost, realized_pnl, settled_time,
         model_prob_entry, market_prob_entry, edge_entry,
         raw_json, synced_at
       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    )
    .run(
      s.ticker,
      s.event_ticker ?? '',
      s.market_result,
      num(s.yes_count_fp),
      num(s.no_count_fp),
      num(s.yes_total_cost_dollars),
      num(s.no_total_cost_dollars),
      num(s.revenue) || num(s.value),
      num(s.fee_cost),
      computeRealizedPnl(s),
      s.settled_time,
      entry?.model_prob ?? null,
      entry?.market_prob ?? null,
      entry?.edge ?? null,
      JSON.stringify(s),
      Math.floor(Date.now() / 1000),
    );

  // Materialize the Brier score for calibration when we had a model view at
  // entry. Uses the entry-time-bounded probability captured above — never the
  // latest edge row, which would be hindsight. Only binary outcomes score:
  // a voided/scalar result has no 0/1 truth, and writing outcome=0 for it
  // would fabricate a (possibly huge) Brier penalty.
  const resultLower = s.market_result.toLowerCase();
  if (result.changes > 0 && entry && (resultLower === 'yes' || resultLower === 'no')) {
    const outcome = resultLower === 'yes' ? 1 : 0;
    const category = (s.event_ticker ?? s.ticker).split('-')[0] || 'unknown';
    db.prepare(
      `INSERT INTO brier_scores (ticker, event_ticker, category, model_prob, actual_outcome, brier_score, settled_at)
       VALUES (?, ?, ?, ?, ?, ?, ?)`,
    ).run(
      s.ticker,
      s.event_ticker ?? '',
      category,
      entry.model_prob,
      outcome,
      computeBrier(entry.model_prob, outcome as 0 | 1),
      settledEpoch,
    );
  }
  return result.changes > 0;
}

export function getSettlements(db: Database, limit = 50): SettlementRow[] {
  return db
    .prepare(
      `SELECT ticker, event_ticker, market_result, yes_count_fp, no_count_fp,
              yes_total_cost, no_total_cost, revenue, fee_cost, realized_pnl,
              settled_time, model_prob_entry, market_prob_entry, edge_entry
       FROM settlements ORDER BY settled_time DESC LIMIT ?`,
    )
    .all(limit) as SettlementRow[];
}

export interface SettlementSummary {
  count: number;
  total_realized_pnl: number;
  total_fees: number;
  wins: number;
  losses: number;
  /** Settlements where we had a model view at entry. */
  with_model_view: number;
  /** Of those, how often the model's side (edge sign) won. */
  model_side_wins: number;
}

export function summarizeSettlements(db: Database): SettlementSummary {
  // SQL aggregation over the full table — no row cap.
  const agg = db
    .prepare(
      `SELECT COUNT(*) AS count,
              COALESCE(SUM(realized_pnl), 0) AS total_realized_pnl,
              COALESCE(SUM(fee_cost), 0) AS total_fees,
              COALESCE(SUM(realized_pnl > 0), 0) AS wins,
              COALESCE(SUM(realized_pnl < 0), 0) AS losses,
              COALESCE(SUM(edge_entry IS NOT NULL AND edge_entry != 0 AND LOWER(market_result) IN ('yes', 'no')), 0) AS with_model_view,
              COALESCE(SUM(
                edge_entry IS NOT NULL AND edge_entry != 0
                AND LOWER(market_result) IN ('yes', 'no')
                AND ((edge_entry > 0) = (LOWER(market_result) = 'yes'))
              ), 0) AS model_side_wins
       FROM settlements`,
    )
    .get() as SettlementSummary;
  return agg;
}
