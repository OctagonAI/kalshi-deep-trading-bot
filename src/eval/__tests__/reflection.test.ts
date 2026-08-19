import { describe, expect, test } from 'bun:test';
import { Database } from 'bun:sqlite';
import { migrate } from '../../db/schema';
import { generateMissingLessons, getLessonsForCategory, getPendingReflections, fallbackLesson, formatLessonsForContext } from '../reflection';

function freshDb() {
  const db = new Database(':memory:');
  migrate(db);
  return db;
}

function seedSettlement(db: Database, over: Record<string, unknown> = {}) {
  const row = {
    ticker: 'KXFED-26SEP-T1', event_ticker: 'KXFED-26SEP', market_result: 'no',
    realized_pnl: -4.2, settled_time: '2026-08-10T00:00:00Z',
    model_prob_entry: 0.8, market_prob_entry: 0.6, edge_entry: 0.2,
    ...over,
  };
  db.prepare(
    `INSERT INTO settlements (ticker, event_ticker, market_result, revenue, realized_pnl, settled_time, model_prob_entry, market_prob_entry, edge_entry, synced_at)
     VALUES ($ticker, $event_ticker, $market_result, 0, $realized_pnl, $settled_time, $model_prob_entry, $market_prob_entry, $edge_entry, 0)`,
  ).run(Object.fromEntries(Object.entries(row).map(([k, v]) => ['$' + k, v])) as any);
}

describe('reflection loop', () => {
  test('pending reflections are settlements with a model view and no lesson', () => {
    const db = freshDb();
    seedSettlement(db);
    seedSettlement(db, { ticker: 'KXA-1-T', event_ticker: 'KXA-1', model_prob_entry: null, market_prob_entry: null, edge_entry: null, settled_time: '2026-08-11T00:00:00Z' });
    expect(getPendingReflections(db)).toHaveLength(1);
  });

  test('llm lesson is stored and injected by category', async () => {
    const db = freshDb();
    seedSettlement(db);
    const result = await generateMissingLessons(db, { llm: async () => 'Fed cut markets: model overweighted hawkish speeches; anchor to dot plot instead.' });
    expect(result.generated).toBe(1);
    expect(result.source).toBe('llm');
    const lessons = getLessonsForCategory(db, 'KXFED');
    expect(lessons).toHaveLength(1);
    expect(lessons[0].lesson).toContain('dot plot');
    expect(formatLessonsForContext(lessons)).toContain('Lessons from settled positions');
    // Idempotent: second run generates nothing
    expect((await generateMissingLessons(db, { llm: async () => 'x' })).generated).toBe(0);
  });

  test('llm failure falls back to deterministic lesson', async () => {
    const db = freshDb();
    seedSettlement(db);
    const result = await generateMissingLessons(db, { llm: async () => { throw new Error('rate limit'); } });
    expect(result.generated).toBe(1);
    expect(result.source).toBe('fallback');
    const [l] = getLessonsForCategory(db, 'KXFED');
    expect(l.lesson).toContain('overconfident on YES');
    expect(l.lesson).toContain('80pp');
  });

  test('fallbackLesson tolerates near-misses without prescribing corrections', () => {
    const lesson = fallbackLesson({
      ticker: 'KXX-1-T', event_ticker: 'KXX-1', settled_time: '2026-08-01T00:00:00Z',
      model_prob_entry: 0.9, market_prob_entry: null, edge_entry: null,
      market_result: 'yes', realized_pnl: 3, drivers_json: null,
    });
    expect(lesson).toContain('within tolerance');
  });
});
