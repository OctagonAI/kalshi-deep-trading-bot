import { describe, expect, test } from 'bun:test';
import { Database } from 'bun:sqlite';
import { migrate } from '../schema';
import { addHypothesis, resolveHypothesis, resolveHypothesesForSettlement, listHypotheses, scoreboard } from '../hypotheses';

function freshDb() {
  const db = new Database(':memory:');
  migrate(db);
  return db;
}

describe('hypothesis registry', () => {
  test('add + list + manual resolve lifecycle', () => {
    const db = freshDb();
    const id = addHypothesis(db, { claim: 'Fed cuts twice before December' });
    expect(listHypotheses(db, 'open')).toHaveLength(1);
    expect(resolveHypothesis(db, id, 'refuted', 'only one cut')).toBe(true);
    // Already resolved: second resolve is a no-op
    expect(resolveHypothesis(db, id, 'confirmed')).toBe(false);
    const board = scoreboard(db);
    expect(board.refuted).toBe(1);
    expect(board.hit_rate).toBe(0);
  });

  test('market-bound hypotheses auto-resolve from settlement result', () => {
    const db = freshDb();
    const yes = addHypothesis(db, { claim: 'A settles YES', ticker: 'KXA-1-T', predictedSide: 'yes', source: 'trade' });
    const no = addHypothesis(db, { claim: 'A settles NO', ticker: 'KXA-1-T', predictedSide: 'no' });
    const untouched = addHypothesis(db, { claim: 'other market', ticker: 'KXB-1-T', predictedSide: 'yes' });
    const thematic = addHypothesis(db, { claim: 'no binding' });

    const resolved = resolveHypothesesForSettlement(db, 'KXA-1-T', 'yes', 12.5);
    expect(resolved).toBe(2);

    const rows = new Map(listHypotheses(db).map((h) => [h.id, h]));
    expect(rows.get(yes)!.status).toBe('confirmed');
    expect(rows.get(yes)!.evidence).toContain('+$12.50');
    expect(rows.get(no)!.status).toBe('refuted');
    expect(rows.get(untouched)!.status).toBe('open');
    expect(rows.get(thematic)!.status).toBe('open');
    expect(scoreboard(db).hit_rate).toBeCloseTo(0.5);
  });

  test('category derives from binding', () => {
    const db = freshDb();
    addHypothesis(db, { claim: 'x', ticker: 'KXFED-26SEP-T1' });
    expect(listHypotheses(db)[0].category).toBe('KXFED');
  });
});

// ─── Non-binary results & claim-after-flags (review round) ──────────────────
import { describe as d7, expect as e7, test as t7 } from 'bun:test';

d7('non-binary settlement results', () => {
  t7('void/scalar results leave market-bound hypotheses open', () => {
    const db = freshDb();
    const id = addHypothesis(db, { claim: 'x', ticker: 'KXV-1-T', predictedSide: 'yes' });
    e7(resolveHypothesesForSettlement(db, 'KXV-1-T', 'void', 0)).toBe(0);
    e7(resolveHypothesesForSettlement(db, 'KXV-1-T', '', 0)).toBe(0);
    e7(listHypotheses(db).find((h) => h.id === id)!.status).toBe('open');
  });

  t7('unbound claims store category "unknown"', () => {
    const db = freshDb();
    addHypothesis(db, { claim: 'thematic claim' });
    e7(listHypotheses(db)[0].category).toBe('unknown');
  });
});
