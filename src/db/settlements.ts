/**
 * Settlement ledger — realized P&L captured from Kalshi /portfolio/settlements.
 *
 * This is the ground-truth side of the learning loop: every settled position
 * is recorded with the model's view at entry (joined from edge_history), so
 * calibration and reflection can compare what the model believed with what
 * actually happened. Rows are immutable; syncs are idempotent.
 */
import type { Database } from 'bun:sqlite';

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
  const entry = db
    .prepare(
      `SELECT model_prob, market_prob, edge FROM edge_history
       WHERE ticker = ? AND cache_miss = 0 AND timestamp <= ?
       ORDER BY timestamp DESC LIMIT 1`,
    )
    .get(s.ticker, settledEpoch) as { model_prob: number; market_prob: number; edge: number } | undefined;

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
  const rows = getSettlements(db, 10_000);
  const summary: SettlementSummary = {
    count: rows.length,
    total_realized_pnl: 0,
    total_fees: 0,
    wins: 0,
    losses: 0,
    with_model_view: 0,
    model_side_wins: 0,
  };
  for (const r of rows) {
    summary.total_realized_pnl += r.realized_pnl;
    summary.total_fees += r.fee_cost;
    if (r.realized_pnl > 0) summary.wins++;
    else if (r.realized_pnl < 0) summary.losses++;
    if (r.edge_entry !== null && r.edge_entry !== 0) {
      summary.with_model_view++;
      // Positive edge = model leaned YES; result 'yes' means the model side won.
      const modelSaidYes = r.edge_entry > 0;
      const yesWon = r.market_result.toLowerCase() === 'yes';
      if (modelSaidYes === yesWon) summary.model_side_wins++;
    }
  }
  return summary;
}
