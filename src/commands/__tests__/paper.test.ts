import { describe, expect, test } from 'bun:test';
import { Database } from 'bun:sqlite';
import { migrate } from '../../db/schema';
import { openPaperPosition, settlePaperPositions, paperSummary, settlePnl, winningSide, listPaperPositions } from '../paper';

function freshDb() {
  const db = new Database(':memory:');
  migrate(db);
  return db;
}

describe('paper P&L math', () => {
  test('winning side follows the quadrant', () => {
    expect(winningSide('buy', 'yes')).toBe('yes');
    expect(winningSide('sell', 'yes')).toBe('no');
    expect(winningSide('buy', 'no')).toBe('no');
    expect(winningSide('sell', 'no')).toBe('yes');
  });

  test('settlePnl pays (100-entry) on wins, -entry on losses', () => {
    const p = { action: 'buy' as const, side: 'yes' as const, count: 10, entry_price: 40 };
    expect(settlePnl(p, 'yes')).toBeCloseTo(6);
    expect(settlePnl(p, 'no')).toBeCloseTo(-4);
  });
});

describe('paper ledger lifecycle', () => {
  test('open → settle → summary, and a hypothesis is auto-filed', async () => {
    const db = freshDb();
    openPaperPosition(db, { ticker: 'KX-T', action: 'buy', side: 'yes', count: 10, priceCents: 40 });
    expect(paperSummary(db).open).toBe(1);
    const hyp = db.prepare(`SELECT claim, predicted_side, source FROM hypotheses WHERE ticker = 'KX-T'`).get() as { claim: string; predicted_side: string; source: string };
    expect(hyp.source).toBe('paper');
    expect(hyp.predicted_side).toBe('yes');

    const settled = await settlePaperPositions(db, async () => 'yes');
    expect(settled).toBe(1);
    const s = paperSummary(db);
    expect(s.settled).toBe(1);
    expect(s.realized_pnl).toBeCloseTo(6);
    expect(s.roi).toBeCloseTo(6 / 4);
    expect(listPaperPositions(db)[0].outcome).toBe('yes');
  });

  test('unresolved markets stay open', async () => {
    const db = freshDb();
    openPaperPosition(db, { ticker: 'KX-T', action: 'buy', side: 'no', count: 1, priceCents: 60 });
    expect(await settlePaperPositions(db, async () => null)).toBe(0);
    expect(paperSummary(db).open).toBe(1);
  });
});

// ─── Short economics (review round) ─────────────────────────────────────────
import { describe as ds, expect as es, test as ts2 } from 'bun:test';
import { capitalAtRisk } from '../paper';

ds('paper short economics', () => {
  ts2('seller keeps premium on win, pays complement on loss', () => {
    // Sell 10 YES @ 60¢: win (settles NO) → +$6.00; lose (settles YES) → -$4.00
    const p = { action: 'sell' as const, side: 'yes' as const, count: 10, entry_price: 60 };
    expect(settlePnl(p, 'no')).toBeCloseTo(6);
    expect(settlePnl(p, 'yes')).toBeCloseTo(-4);
  });

  ts2('capital at risk mirrors: buyer stakes price, seller stakes complement', () => {
    expect(capitalAtRisk({ action: 'buy', count: 10, entry_price: 60 })).toBeCloseTo(6);
    expect(capitalAtRisk({ action: 'sell', count: 10, entry_price: 60 })).toBeCloseTo(4);
  });

  ts2('settled sell flows into summary with short capital', async () => {
    const db = freshDb();
    openPaperPosition(db, { ticker: 'KX-S', action: 'sell', side: 'yes', count: 10, priceCents: 60 });
    await settlePaperPositions(db, async () => 'no');
    const s = paperSummary(db);
    expect(s.realized_pnl).toBeCloseTo(6);
    expect(s.capital_settled).toBeCloseTo(4);
    expect(s.roi).toBeCloseTo(1.5);
  });
});
