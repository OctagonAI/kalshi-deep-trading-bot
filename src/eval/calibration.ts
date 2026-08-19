/**
 * Calibration report — how good the model's probabilities actually are,
 * measured against realized settlements (ground truth captured by
 * syncSettlements, with entry-time-bounded model views).
 *
 * Two views:
 *  - Per-category skill: Brier(model) vs Brier(market-at-entry). Skill > 0
 *    means the model out-forecast the market price in that category.
 *  - Calibration buckets: mean predicted probability vs realized YES
 *    frequency per decile — the classic reliability diagram, in a table.
 */
import type { Database } from 'bun:sqlite';

export interface CategoryCalibration {
  category: string;
  n: number;
  brier_model: number;
  brier_market: number;
  /** 1 - brier_model / brier_market. Positive = model beats market. */
  skill: number;
  realized_pnl: number;
}

export interface CalibrationBucket {
  label: string;
  n: number;
  mean_predicted: number;
  realized_yes_rate: number;
}

export interface CalibrationReport {
  n_scored: number;
  n_unscored: number;
  brier_model: number;
  brier_market: number;
  skill: number;
  categories: CategoryCalibration[];
  buckets: CalibrationBucket[];
}

interface ScoredSettlement {
  ticker: string;
  event_ticker: string;
  model_prob_entry: number;
  market_prob_entry: number | null;
  market_result: string;
  realized_pnl: number;
}

/** Series prefix of an event ticker (KXPRESNOMD-28 → KXPRESNOMD). */
export function categoryOf(eventTicker: string, ticker: string): string {
  const base = eventTicker || ticker;
  return base.split('-')[0] || 'unknown';
}

export function computeCalibration(db: Database): CalibrationReport {
  const rows = db
    .prepare(
      `SELECT ticker, event_ticker, model_prob_entry, market_prob_entry, market_result, realized_pnl
       FROM settlements`,
    )
    .all() as Array<ScoredSettlement & { model_prob_entry: number | null }>;

  const scored = rows.filter((r): r is ScoredSettlement => r.model_prob_entry !== null);
  const unscored = rows.length - scored.length;

  let brierModelSum = 0;
  let brierMarketSum = 0;
  let marketN = 0;
  const byCategory = new Map<string, { n: number; bm: number; bmkt: number; mktN: number; pnl: number }>();
  const bucketDefs = Array.from({ length: 10 }, (_, i) => ({ lo: i / 10, hi: (i + 1) / 10 }));
  const buckets = bucketDefs.map(() => ({ n: 0, predictedSum: 0, yes: 0 }));

  for (const r of scored) {
    const outcome = r.market_result.toLowerCase() === 'yes' ? 1 : 0;
    const bm = (r.model_prob_entry - outcome) ** 2;
    brierModelSum += bm;

    let bmkt = 0;
    const hasMarket = r.market_prob_entry !== null;
    if (hasMarket) {
      bmkt = (r.market_prob_entry! - outcome) ** 2;
      brierMarketSum += bmkt;
      marketN++;
    }

    const cat = categoryOf(r.event_ticker, r.ticker);
    const c = byCategory.get(cat) ?? { n: 0, bm: 0, bmkt: 0, mktN: 0, pnl: 0 };
    c.n++;
    c.bm += bm;
    if (hasMarket) { c.bmkt += bmkt; c.mktN++; }
    c.pnl += r.realized_pnl;
    byCategory.set(cat, c);

    const idx = Math.min(9, Math.floor(r.model_prob_entry * 10));
    buckets[idx].n++;
    buckets[idx].predictedSum += r.model_prob_entry;
    buckets[idx].yes += outcome;
  }

  const brierModel = scored.length > 0 ? brierModelSum / scored.length : 0;
  const brierMarket = marketN > 0 ? brierMarketSum / marketN : 0;

  return {
    n_scored: scored.length,
    n_unscored: unscored,
    brier_model: brierModel,
    brier_market: brierMarket,
    skill: brierMarket > 0 ? 1 - brierModel / brierMarket : 0,
    categories: [...byCategory.entries()]
      .map(([category, c]) => ({
        category,
        n: c.n,
        brier_model: c.n > 0 ? c.bm / c.n : 0,
        brier_market: c.mktN > 0 ? c.bmkt / c.mktN : 0,
        skill: c.mktN > 0 && c.bmkt > 0 ? 1 - (c.bm / c.n) / (c.bmkt / c.mktN) : 0,
        realized_pnl: c.pnl,
      }))
      .sort((a, b) => b.n - a.n),
    buckets: bucketDefs.map((d, i) => ({
      label: `${Math.round(d.lo * 100)}-${Math.round(d.hi * 100)}%`,
      n: buckets[i].n,
      mean_predicted: buckets[i].n > 0 ? buckets[i].predictedSum / buckets[i].n : 0,
      realized_yes_rate: buckets[i].n > 0 ? buckets[i].yes / buckets[i].n : 0,
    })),
  };
}

export function formatCalibrationHuman(report: CalibrationReport): string {
  const lines: string[] = [];
  lines.push('**Model Calibration (realized settlements)**');
  lines.push('');
  if (report.n_scored === 0) {
    lines.push(`No scored settlements yet (${report.n_unscored} settlements lack a model view at entry).`);
    lines.push('Calibration accumulates automatically as positions settle — trade with /analyze coverage to build it.');
    return lines.join('\n');
  }
  lines.push(`Scored settlements: ${report.n_scored}${report.n_unscored ? ` (+${report.n_unscored} without a model view)` : ''}`);
  lines.push(`Brier — model ${report.brier_model.toFixed(3)} vs market-at-entry ${report.brier_market.toFixed(3)}  →  skill ${report.skill >= 0 ? '+' : ''}${(report.skill * 100).toFixed(1)}%`);
  lines.push('');
  lines.push('By category:');
  lines.push('  Category          n    Brier(model)  Brier(mkt)  Skill     Realized P&L');
  for (const c of report.categories) {
    lines.push(
      `  ${c.category.padEnd(16)}${String(c.n).padStart(3)}    ${c.brier_model.toFixed(3).padStart(8)}      ${c.brier_market.toFixed(3).padStart(6)}   ${((c.skill >= 0 ? '+' : '') + (c.skill * 100).toFixed(1) + '%').padStart(7)}   ${(c.realized_pnl >= 0 ? '+' : '') + '$' + c.realized_pnl.toFixed(2)}`,
    );
  }
  const active = report.buckets.filter((b) => b.n > 0);
  if (active.length > 0) {
    lines.push('');
    lines.push('Reliability (predicted vs realized YES rate):');
    lines.push('  Bucket     n    Predicted   Realized');
    for (const b of active) {
      lines.push(`  ${b.label.padEnd(9)}${String(b.n).padStart(3)}    ${(b.mean_predicted * 100).toFixed(0).padStart(6)}%    ${(b.realized_yes_rate * 100).toFixed(0).padStart(6)}%`);
    }
  }
  return lines.join('\n');
}
