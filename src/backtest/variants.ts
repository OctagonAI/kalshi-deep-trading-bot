/**
 * Strategy-variant leaderboard (AI-Trader pattern): one signal pipeline,
 * many lenses. Each variant is a named filter over the backtest's scored
 * signals; every variant is scored with the same definitions (hit rate,
 * capital-weighted ROI, alpha vs always-NO on the same rows, max drawdown,
 * ROI/maxDD), so the leaderboard is an apples-to-apples comparison of the
 * segmentations people otherwise run by hand.
 */
import type { ScoredSignal } from './types.js';
import { alphaVsAlwaysNoPp, computeEquityRisk } from './metrics.js';

export interface VariantDef {
  key: string;
  label: string;
  filter: (s: ScoredSignal) => boolean;
}

/**
 * Built-in variants. "all" is the baseline row; the rest segment by the
 * dimensions the backtest keeps showing as behaviorally distinct: side,
 * edge magnitude, entry price band, confidence, volume.
 */
export const VARIANTS: VariantDef[] = [
  { key: 'all', label: 'All edge signals', filter: () => true },
  { key: 'no-side', label: 'NO bets only', filter: (s) => s.edge_pp < 0 },
  { key: 'yes-side', label: 'YES bets only', filter: (s) => s.edge_pp > 0 },
  { key: 'edge-5-15', label: 'Moderate edge 5-15pp', filter: (s) => Math.abs(s.edge_pp) >= 5 && Math.abs(s.edge_pp) < 15 },
  { key: 'edge-15plus', label: 'Extreme edge ≥15pp', filter: (s) => Math.abs(s.edge_pp) >= 15 },
  { key: 'mid-price', label: 'Entry 20-80¢ (no tails)', filter: (s) => s.market_then >= 20 && s.market_then <= 80 },
  { key: 'longshot', label: 'Entry <20¢ longshots', filter: (s) => s.market_then < 20 },
  { key: 'high-conf', label: 'Confidence ≥0.7', filter: (s) => (s.confidence_score ?? 0) >= 0.7 },
  { key: 'resolved-only', label: 'Resolved leg only', filter: (s) => s.resolved },
];

export interface VariantRow {
  key: string;
  label: string;
  n: number;
  hit_rate: number;
  roi: number;
  alpha_pp: number;
  max_drawdown_pct: number;
  risk_adjusted: number;
  total_capital: number;
}

export function computeVariantLeaderboard(
  signals: ScoredSignal[],
  minEdgePp: number,
  variants: VariantDef[] = VARIANTS,
): VariantRow[] {
  const edgeSignals = signals.filter((s) => s.edge_pp !== 0 && Math.abs(s.edge_pp) >= minEdgePp);
  const rows: VariantRow[] = [];
  for (const v of variants) {
    const subset = edgeSignals.filter(v.filter);
    if (subset.length === 0) {
      rows.push({ key: v.key, label: v.label, n: 0, hit_rate: 0, roi: 0, alpha_pp: 0, max_drawdown_pct: 0, risk_adjusted: 0, total_capital: 0 });
      continue;
    }
    const hits = subset.filter((s) => (s.edge_pp > 0 ? s.market_now > s.market_then : s.market_now < s.market_then)).length;
    const pnl = subset.reduce((sum, s) => sum + s.pnl, 0);
    const capital = subset.reduce((sum, s) => sum + s.capital, 0);
    const roi = capital > 0 ? pnl / capital : 0;
    const risk = computeEquityRisk(subset, capital);
    rows.push({
      key: v.key,
      label: v.label,
      n: subset.length,
      hit_rate: hits / subset.length,
      roi,
      alpha_pp: alphaVsAlwaysNoPp(subset),
      max_drawdown_pct: risk.max_drawdown_pct,
      risk_adjusted: risk.risk_adjusted_return,
      total_capital: capital,
    });
  }
  // Leaderboard order: risk-adjusted return, empty rows last. Infinity
  // (no drawdown, positive ROI) sorts first by construction.
  return rows.sort((a, b) => {
    if (a.n === 0 && b.n === 0) return 0;
    if (a.n === 0) return 1;
    if (b.n === 0) return -1;
    return b.risk_adjusted - a.risk_adjusted;
  });
}

export function formatVariantLeaderboard(rows: VariantRow[], minEdgePp: number): string {
  const lines: string[] = [];
  lines.push(`**Strategy Variants** — same signals, segmented (min edge ${minEdgePp}pp; ranked by ROI/maxDD)`);
  lines.push('');
  lines.push('  Variant                     n    Hit%     ROI    Alpha   MaxDD   ROI/DD');
  for (const r of rows) {
    if (r.n === 0) {
      lines.push(`  ${r.label.padEnd(26)}  0       —       —       —       —       —`);
      continue;
    }
    const ra = Number.isFinite(r.risk_adjusted) ? r.risk_adjusted.toFixed(2) : '∞';
    lines.push(
      `  ${r.label.padEnd(26)}${String(r.n).padStart(3)}  ${(r.hit_rate * 100).toFixed(0).padStart(4)}%  ${((r.roi >= 0 ? '+' : '') + (r.roi * 100).toFixed(1) + '%').padStart(7)}  ${((r.alpha_pp >= 0 ? '+' : '') + r.alpha_pp.toFixed(1) + 'pp').padStart(7)}  ${((r.max_drawdown_pct * 100).toFixed(1) + '%').padStart(6)}  ${ra.padStart(7)}`,
    );
  }
  lines.push('');
  lines.push('One pipeline, many lenses: every row is scored with the backtest definitions on the same signal set.');
  return lines.join('\n');
}
