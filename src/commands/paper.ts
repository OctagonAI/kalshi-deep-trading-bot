/**
 * Paper-trading ledger (AI-Trader pattern): forward-test the model's calls
 * with the exact entry semantics of /buy but no exchange order. Positions
 * settle against real Kalshi results and are scored with the same flat-bet
 * definitions as the backtest, so backtest → paper → live is one comparable
 * chain. Every paper entry auto-files a hypothesis (source 'paper').
 */
import type { Database } from 'bun:sqlite';
import { callKalshiApi, KalshiApiError } from '../tools/kalshi/api.js';
import { getLatestEdge } from '../db/edge.js';
import { addHypothesis } from '../db/hypotheses.js';

export interface PaperPosition {
  id: number;
  ticker: string;
  event_ticker: string;
  action: 'buy' | 'sell';
  side: 'yes' | 'no';
  count: number;
  entry_price: number;
  model_prob: number | null;
  market_prob: number | null;
  edge: number | null;
  status: 'open' | 'settled' | 'closed';
  outcome: string | null;
  realized_pnl: number | null;
  opened_at: number;
  settled_at: number | null;
}

/** The settlement side that profits: buy yes / sell no ⇒ yes, else no. */
export function winningSide(action: 'buy' | 'sell', side: 'yes' | 'no'): 'yes' | 'no' {
  return (action === 'buy') === (side === 'yes') ? 'yes' : 'no';
}

/** Dollars at risk: a buyer stakes the price, a seller stakes the complement. */
export function capitalAtRisk(p: { action: 'buy' | 'sell'; count: number; entry_price: number }): number {
  const perContract = p.action === 'buy' ? p.entry_price : 100 - p.entry_price;
  return (perContract / 100) * p.count;
}

/**
 * Flat-bet P&L in dollars. entry_price is quoted on the position's own side.
 * Buyer at p: win +(100-p), lose -p. Seller at p keeps the premium on a win
 * (+p) and pays out the complement on a loss (-(100-p)) — short economics
 * are the mirror of the long side, not the same formula.
 */
export function settlePnl(p: { action: 'buy' | 'sell'; side: 'yes' | 'no'; count: number; entry_price: number }, result: 'yes' | 'no'): number {
  const won = winningSide(p.action, p.side) === result;
  const winPerContract = p.action === 'buy' ? 100 - p.entry_price : p.entry_price;
  const lossPerContract = p.action === 'buy' ? p.entry_price : 100 - p.entry_price;
  return won ? (winPerContract / 100) * p.count : -(lossPerContract / 100) * p.count;
}

export function openPaperPosition(
  db: Database,
  o: { ticker: string; action: 'buy' | 'sell'; side: 'yes' | 'no'; count: number; priceCents: number; eventTicker?: string },
): number {
  const edge = getLatestEdge(db, o.ticker);
  const result = db
    .prepare(
      `INSERT INTO paper_positions (ticker, event_ticker, action, side, count, entry_price, model_prob, market_prob, edge, opened_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    )
    .run(
      o.ticker,
      o.eventTicker ?? '',
      o.action,
      o.side,
      o.count,
      o.priceCents,
      edge && !edge.cache_miss ? edge.model_prob : null,
      edge && !edge.cache_miss ? edge.market_prob : null,
      edge && !edge.cache_miss ? edge.edge : null,
      Math.floor(Date.now() / 1000),
    );
  const id = Number(result.lastInsertRowid);
  try {
    const predicted = winningSide(o.action, o.side);
    addHypothesis(db, {
      claim: `PAPER ${o.action.toUpperCase()} ${o.side.toUpperCase()} x${o.count} on ${o.ticker} at ${o.priceCents}¢ settles ${predicted.toUpperCase()}`,
      ticker: o.ticker,
      predictedSide: predicted,
      source: 'paper',
    });
  } catch { /* registry is additive */ }
  return id;
}

/** Fetch a market's settlement result; 'yes' | 'no' | null when unresolved. */
export type MarketResultFetcher = (ticker: string) => Promise<'yes' | 'no' | null>;

async function fetchMarketResult(ticker: string): Promise<'yes' | 'no' | null> {
  try {
    const res = (await callKalshiApi('GET', `/markets/${ticker}`)) as { market?: { result?: string } };
    const result = (res.market ?? (res as { result?: string })).result?.toLowerCase();
    return result === 'yes' || result === 'no' ? result : null;
  } catch (err) {
    if (err instanceof KalshiApiError && err.statusCode === 404) return null;
    return null; // transient — retry next pass
  }
}

/**
 * Settle open paper positions whose markets have resolved on Kalshi.
 * Returns how many settled this pass. Network failures leave positions open.
 * `fetchResult` is injectable for tests.
 */
export async function settlePaperPositions(db: Database, fetchResult: MarketResultFetcher = fetchMarketResult): Promise<number> {
  const open = db
    .prepare(`SELECT * FROM paper_positions WHERE status = 'open'`)
    .all() as PaperPosition[];
  let settled = 0;
  const byTicker = new Map<string, PaperPosition[]>();
  for (const p of open) {
    const arr = byTicker.get(p.ticker) ?? [];
    arr.push(p);
    byTicker.set(p.ticker, arr);
  }
  for (const [ticker, positions] of byTicker) {
    const result = await fetchResult(ticker);
    if (result === null) continue;
    for (const p of positions) {
      const pnl = settlePnl(p, result);
      db.prepare(
        `UPDATE paper_positions SET status = 'settled', outcome = ?, realized_pnl = ?, settled_at = ? WHERE id = ? AND status = 'open'`,
      ).run(result, pnl, Math.floor(Date.now() / 1000), p.id);
      settled++;
    }
  }
  return settled;
}

export interface PaperSummary {
  open: number;
  settled: number;
  open_capital: number;
  realized_pnl: number;
  capital_settled: number;
  roi: number;
  wins: number;
  losses: number;
}

export function paperSummary(db: Database): PaperSummary {
  const agg = db
    .prepare(
      `SELECT
         COALESCE(SUM(status = 'open'), 0) AS open,
         COALESCE(SUM(status = 'settled'), 0) AS settled,
         COALESCE(SUM(CASE WHEN status = 'open' THEN (CASE WHEN action = 'buy' THEN entry_price ELSE 100 - entry_price END) / 100.0 * count END), 0) AS open_capital,
         COALESCE(SUM(CASE WHEN status = 'settled' THEN realized_pnl END), 0) AS realized_pnl,
         COALESCE(SUM(CASE WHEN status = 'settled' THEN (CASE WHEN action = 'buy' THEN entry_price ELSE 100 - entry_price END) / 100.0 * count END), 0) AS capital_settled,
         COALESCE(SUM(status = 'settled' AND realized_pnl > 0), 0) AS wins,
         COALESCE(SUM(status = 'settled' AND realized_pnl < 0), 0) AS losses
       FROM paper_positions`,
    )
    .get() as Omit<PaperSummary, 'roi'>;
  return { ...agg, roi: agg.capital_settled > 0 ? agg.realized_pnl / agg.capital_settled : 0 };
}

export function listPaperPositions(db: Database, limit = 20): PaperPosition[] {
  return db
    .prepare(`SELECT * FROM paper_positions ORDER BY (status = 'open') DESC, opened_at DESC LIMIT ?`)
    .all(limit) as PaperPosition[];
}

export function formatPaperHuman(rows: PaperPosition[], summary: PaperSummary, justSettled: number): string {
  const lines: string[] = [];
  lines.push('**Paper Trading Ledger**');
  lines.push('');
  if (justSettled > 0) lines.push(`${justSettled} position${justSettled === 1 ? '' : 's'} settled this pass.`);
  lines.push(
    `${summary.open} open ($${summary.open_capital.toFixed(2)} at risk) · ${summary.settled} settled · ` +
    `realized ${summary.realized_pnl >= 0 ? '+' : ''}$${summary.realized_pnl.toFixed(2)} (${(summary.roi * 100).toFixed(1)}% ROI) · ${summary.wins}W/${summary.losses}L`,
  );
  if (rows.length === 0) {
    lines.push('');
    lines.push('No paper positions. Open one: /paper buy <ticker> <count> [price] [yes|no]');
    return lines.join('\n');
  }
  lines.push('');
  for (const p of rows) {
    const badge = p.status === 'open' ? '○' : (p.realized_pnl ?? 0) >= 0 ? '✓' : '✗';
    const pnl = p.status === 'settled' ? `  ${p.realized_pnl! >= 0 ? '+' : ''}$${p.realized_pnl!.toFixed(2)} (${p.outcome?.toUpperCase()})` : '';
    const model = p.model_prob !== null ? `  model=${(p.model_prob * 100).toFixed(0)}%` : '';
    lines.push(`  ${badge} #${p.id} ${p.action.toUpperCase()} ${p.side.toUpperCase()} x${p.count} ${p.ticker} @ ${p.entry_price}¢${model}${pnl}`);
  }
  return lines.join('\n');
}
