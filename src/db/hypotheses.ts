/**
 * Hypothesis registry (Vibe-Trading pattern, market-native):
 * falsifiable claims with a lifecycle. Claims bound to a market ticker and a
 * predicted side are resolved automatically when the market settles;
 * thematic claims are resolved by hand. Trade execution auto-files a
 * hypothesis per position so live claims mirror live risk.
 */
import type { Database } from 'bun:sqlite';
import { categoryOf } from '../eval/calibration.js';

export type HypothesisStatus = 'open' | 'confirmed' | 'refuted' | 'expired' | 'retired';

export interface HypothesisRow {
  id: number;
  claim: string;
  category: string;
  ticker: string | null;
  event_ticker: string | null;
  predicted_side: 'yes' | 'no' | null;
  status: HypothesisStatus;
  evidence: string | null;
  source: string;
  created_at: number;
  resolved_at: number | null;
}

export function addHypothesis(
  db: Database,
  h: {
    claim: string;
    ticker?: string;
    eventTicker?: string;
    predictedSide?: 'yes' | 'no';
    source?: string;
  },
): number {
  const category = h.eventTicker || h.ticker ? categoryOf(h.eventTicker ?? '', h.ticker ?? '') : 'unknown';
  const result = db
    .prepare(
      `INSERT INTO hypotheses (claim, category, ticker, event_ticker, predicted_side, source, created_at)
       VALUES (?, ?, ?, ?, ?, ?, ?)`,
    )
    .run(
      h.claim,
      category,
      h.ticker ?? null,
      h.eventTicker ?? null,
      h.predictedSide ?? null,
      h.source ?? 'manual',
      Math.floor(Date.now() / 1000),
    );
  return Number(result.lastInsertRowid);
}

export function resolveHypothesis(
  db: Database,
  id: number,
  status: 'confirmed' | 'refuted' | 'expired' | 'retired',
  evidence?: string,
): boolean {
  const result = db
    .prepare(
      `UPDATE hypotheses SET status = ?, evidence = COALESCE(?, evidence), resolved_at = ?
       WHERE id = ? AND status = 'open'`,
    )
    .run(status, evidence ?? null, Math.floor(Date.now() / 1000), id);
  return result.changes > 0;
}

/**
 * Auto-resolve open market-bound hypotheses against a settlement result.
 * Called by settlement sync for each newly-recorded settlement.
 */
export function resolveHypothesesForSettlement(
  db: Database,
  ticker: string,
  marketResult: string,
  realizedPnl: number,
): number {
  const won = marketResult.toLowerCase();
  // Only binary outcomes can confirm/refute a yes/no prediction. Voided,
  // scalar, or empty results leave hypotheses open for manual review.
  if (won !== 'yes' && won !== 'no') return 0;
  const open = db
    .prepare(`SELECT id, predicted_side FROM hypotheses WHERE ticker = ? AND status = 'open' AND predicted_side IS NOT NULL`)
    .all(ticker) as Array<{ id: number; predicted_side: 'yes' | 'no' }>;
  let resolved = 0;
  for (const h of open) {
    const confirmed = h.predicted_side === won;
    const evidence = `Settled ${won.toUpperCase()}; realized ${realizedPnl >= 0 ? '+' : ''}$${realizedPnl.toFixed(2)}`;
    if (resolveHypothesis(db, h.id, confirmed ? 'confirmed' : 'refuted', evidence)) resolved++;
  }
  return resolved;
}

export function listHypotheses(db: Database, status?: HypothesisStatus, limit = 30): HypothesisRow[] {
  if (status) {
    return db
      .prepare(`SELECT * FROM hypotheses WHERE status = ? ORDER BY created_at DESC LIMIT ?`)
      .all(status, limit) as HypothesisRow[];
  }
  return db
    .prepare(`SELECT * FROM hypotheses ORDER BY (status = 'open') DESC, created_at DESC LIMIT ?`)
    .all(limit) as HypothesisRow[];
}

export interface HypothesisScoreboard {
  open: number;
  confirmed: number;
  refuted: number;
  other: number;
  hit_rate: number | null;
}

export function scoreboard(db: Database): HypothesisScoreboard {
  const rows = db
    .prepare(`SELECT status, COUNT(*) n FROM hypotheses GROUP BY status`)
    .all() as Array<{ status: HypothesisStatus; n: number }>;
  const by = new Map(rows.map((r) => [r.status, r.n]));
  const confirmed = by.get('confirmed') ?? 0;
  const refuted = by.get('refuted') ?? 0;
  return {
    open: by.get('open') ?? 0,
    confirmed,
    refuted,
    other: (by.get('expired') ?? 0) + (by.get('retired') ?? 0),
    hit_rate: confirmed + refuted > 0 ? confirmed / (confirmed + refuted) : null,
  };
}

export function formatHypothesesHuman(rows: HypothesisRow[], board: HypothesisScoreboard): string {
  const lines: string[] = [];
  lines.push('**Hypothesis Registry**');
  lines.push('');
  const hr = board.hit_rate !== null ? ` · resolved hit rate ${(board.hit_rate * 100).toFixed(0)}%` : '';
  lines.push(`${board.open} open · ${board.confirmed} confirmed · ${board.refuted} refuted${board.other ? ` · ${board.other} expired/retired` : ''}${hr}`);
  if (rows.length === 0) {
    lines.push('');
    lines.push('No hypotheses yet. Add one with: /hypothesis add "claim" [--ticker KX... --side yes|no]');
    lines.push('Executed trades auto-file market-bound hypotheses.');
    return lines.join('\n');
  }
  lines.push('');
  for (const h of rows) {
    const badge = h.status === 'open' ? '○' : h.status === 'confirmed' ? '✓' : h.status === 'refuted' ? '✗' : '–';
    const bind = h.ticker ? ` [${h.ticker}${h.predicted_side ? ` → ${h.predicted_side.toUpperCase()}` : ''}]` : '';
    lines.push(`  ${badge} #${h.id} ${h.claim}${bind}`);
    if (h.evidence) lines.push(`       ${h.evidence}`);
  }
  return lines.join('\n');
}
