/**
 * Auto-settlement sync: pull resolved positions from Kalshi's authoritative
 * /portfolio/settlements ledger into the local settlements table, and close
 * any matching local position rows. Idempotent — safe to call on every
 * portfolio view or review pass.
 */
import type { Database } from 'bun:sqlite';
import { callKalshiApi } from './api.js';
import { recordSettlement, type KalshiSettlement } from '../../db/settlements.js';
import { resolveHypothesesForSettlement } from '../../db/hypotheses.js';
import { computeRealizedPnl } from '../../db/settlements.js';

const PAGE_LIMIT = 100;
const MAX_PAGES = 20;

export interface SettleSyncResult {
  fetched: number;
  new_settlements: number;
  positions_closed: number;
  /** False when the page cap was hit with more history remaining. */
  complete: boolean;
}

export async function syncSettlements(db: Database): Promise<SettleSyncResult> {
  let cursor: string | undefined;
  let fetched = 0;
  let added = 0;
  let closed = 0;
  let complete = true;

  for (let page = 0; ; page++) {
    if (page >= MAX_PAGES) {
      // Cap hit with a cursor still live: report the sync as incomplete
      // rather than silently truncating history.
      complete = false;
      break;
    }
    const res = (await callKalshiApi('GET', '/portfolio/settlements', {
      params: { limit: PAGE_LIMIT, ...(cursor ? { cursor } : {}) },
    })) as { settlements?: KalshiSettlement[]; cursor?: string };

    const batch = res.settlements ?? [];
    fetched += batch.length;
    let newInBatch = 0;
    for (const s of batch) {
      // Position close-out and hypothesis resolution are idempotent and must
      // run even for already-recorded settlements: a replay (fresh DB row
      // from an earlier sync, position opened later) still needs closing.
      closed += closeLocalPosition(db, s);
      resolveHypothesesForSettlement(db, s.ticker, s.market_result, computeRealizedPnl(s));
      if (recordSettlement(db, s)) newInBatch++;
    }
    added += newInBatch;

    cursor = res.cursor || undefined;
    // Settlements come newest-first; once a full page is already known we've
    // caught up to history recorded by a previous sync.
    if (!cursor || batch.length < PAGE_LIMIT || (batch.length > 0 && newInBatch === 0)) break;
  }

  return { fetched, new_settlements: added, positions_closed: closed, complete };
}

/** Close a locally-tracked open position that just settled. */
function closeLocalPosition(db: Database, s: KalshiSettlement): number {
  const settledEpoch = Math.floor(new Date(s.settled_time).getTime() / 1000);
  const result = db
    .prepare(
      `UPDATE positions SET status = 'settled', closed_at = ?
       WHERE ticker = ? AND status = 'open'`,
    )
    .run(settledEpoch, s.ticker);
  return result.changes;
}
