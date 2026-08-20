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
