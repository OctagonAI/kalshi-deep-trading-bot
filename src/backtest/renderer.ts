import type { BacktestResult, ScoredSignal } from './types.js';
import { writeFileSync } from 'fs';

export interface FormatOpts {
  minEdge?: number;          // 0-1 scale, default 0.005 (0.5pp)
}

/** Format a 0-1 ROI as a signed percentage string. */
function fmtRoi(roi: number): string {
  return `${roi >= 0 ? '+' : ''}${(roi * 100).toFixed(1)}%`;
}

/** Format a percentage-point delta with sign. */
function fmtPp(pp: number): string {
  return `${pp >= 0 ? '+' : ''}${pp.toFixed(1)}pp`;
}

/**
 * Format complete backtest result for terminal display.
 */
export function formatBacktestHuman(result: BacktestResult, opts?: FormatOpts): string {
  const minEdgePp = ((opts?.minEdge ?? 0.005) * 100).toFixed(1);
  const now = new Date();
  const from = new Date(now.getTime() - result.days * 24 * 60 * 60 * 1000);
  const fromStr = from.toISOString().slice(5, 10).replace('-', '/');
  const toStr = now.toISOString().slice(5, 10).replace('-', '/');

  const lines: string[] = [];
  lines.push(`Octagon Backtest — ${result.days}-day lookback (${fromStr} – ${toStr})`);
  lines.push(`Universe: ${result.universe_description}`);
  let feeHeader: string;
  switch (result.fee_model) {
    case 'none':
      feeHeader = 'none — output is GROSS (pre-fee)';
      break;
    case 'taker':
      feeHeader = 'taker (entries charged Kalshi taker fee = 0.07·p·(1−p))';
      break;
    case 'maker':
      feeHeader = 'maker (free-entry execution assumption — net P&L equals gross)';
      break;
  }
  lines.push(`Fee model: ${feeHeader}`);
  lines.push('══════════════════════════════════════════════════════════');
  lines.push('');

  if (result.subscription_notice) {
    lines.push(`  ${result.subscription_notice}`);
    lines.push('');
    // Still show unresolved signals if any
    const unresolvedSignals = result.signals.filter(s => !s.resolved);
    if (unresolvedSignals.length > 0) {
      lines.push(formatUnresolvedTable(unresolvedSignals, minEdgePp));
    }
    return lines.join('\n');
  }

  if (result.signals.length === 0) {
    lines.push('No data available. Try a longer lookback (--days 60) or broader filter.');
    return lines.join('\n');
  }

  // Unified scorecard
  lines.push(`  Events         ${result.events_scored}`);
  lines.push(`  Markets        ${result.markets_resolved + result.markets_unresolved}   (${result.markets_resolved} resolved, ${result.markets_unresolved} unresolved)`);
  lines.push('');
  // Brier scores and Skill Score are hidden for now (keep values in result for JSON/CSV consumers).
  // lines.push(`  Brier (Octagon)   ${result.brier_octagon.toFixed(3)}`);
  // lines.push(`  Brier (Market)    ${result.brier_market.toFixed(3)}`);
  // lines.push(`  Skill Score       ${result.skill_score >= 0 ? '+' : ''}${(result.skill_score * 100).toFixed(1)}%  [95% CI: ${(result.skill_ci[0] * 100).toFixed(1)}% to ${(result.skill_ci[1] * 100).toFixed(1)}%]`);
  // lines.push('');
  lines.push(`  Edge signals      ${result.edge_signals}   (min edge: ${minEdgePp}pp)`);
  if (result.edge_signals > 0) {
    // Resolved settles at 0/100 — realized. Unresolved is marked to the
    // current Kalshi price — paper P&L that can reverse. Splitting them
    // makes it visible when one leg is carrying a weak other.
    const r = result.resolved_metrics;
    const u = result.unresolved_metrics;
    if (r.edge_signals > 0) {
      lines.push('');
      lines.push('  RESOLVED (realized P&L)');
      lines.push(`    Hit rate        ${(r.edge_hit_rate * 100).toFixed(1)}%  [95% CI: ${(r.hit_rate_ci[0] * 100).toFixed(1)}% to ${(r.hit_rate_ci[1] * 100).toFixed(1)}%, event-clustered]   n=${r.edge_signals}`);
      lines.push(`    Flat-bet P&L    ${fmtRoi(r.flat_bet_roi)} ROI  (${r.flat_bet_pnl >= 0 ? '+' : ''}$${r.flat_bet_pnl.toFixed(2)} on $${r.total_capital.toFixed(2)} capital)`);
    }
    if (u.edge_signals > 0) {
      lines.push('');
      lines.push('  UNRESOLVED (mark-to-market — paper P&L)');
      lines.push(`    Directional drift ${(u.edge_hit_rate * 100).toFixed(1)}%  [95% CI: ${(u.hit_rate_ci[0] * 100).toFixed(1)}% to ${(u.hit_rate_ci[1] * 100).toFixed(1)}%, event-clustered]   n=${u.edge_signals}`);
      lines.push(`    M2M P&L         ${fmtRoi(u.flat_bet_roi)} ROI  (${u.flat_bet_pnl >= 0 ? '+' : ''}$${u.flat_bet_pnl.toFixed(2)} on $${u.total_capital.toFixed(2)} capital)`);
    }
    if (r.edge_signals > 0 && u.edge_signals > 0) {
      lines.push('');
      lines.push('  COMBINED (both legs blended — interpret with care)');
      lines.push(`    Hit rate        ${(result.edge_hit_rate * 100).toFixed(1)}%  [95% CI: ${(result.hit_rate_ci[0] * 100).toFixed(1)}% to ${(result.hit_rate_ci[1] * 100).toFixed(1)}%, event-clustered]`);
      lines.push(`    Flat-bet P&L    ${fmtRoi(result.flat_bet_roi)} ROI  (${result.flat_bet_pnl >= 0 ? '+' : ''}$${result.flat_bet_pnl.toFixed(2)} on $${result.total_capital.toFixed(2)} capital)`);
    }
    // Fee drag — show only when --fees is on so existing output is unchanged.
    if (result.fee_model !== 'none' && result.flat_bet_pnl !== result.flat_bet_pnl_net) {
      const feeDrag = result.flat_bet_pnl - result.flat_bet_pnl_net;
      lines.push('');
      lines.push(`  Fees applied (${result.fee_model})`);
      lines.push(`    Gross P&L       ${result.flat_bet_pnl >= 0 ? '+' : ''}$${result.flat_bet_pnl.toFixed(2)} (${fmtRoi(result.flat_bet_roi)} ROI)`);
      lines.push(`    Fee drag        -$${feeDrag.toFixed(2)}`);
      lines.push(`    Net P&L         ${result.flat_bet_pnl_net >= 0 ? '+' : ''}$${result.flat_bet_pnl_net.toFixed(2)} (${fmtRoi(result.flat_bet_roi_net)} ROI)`);
    } else if (r.edge_signals === 0 && u.edge_signals === 0) {
      // No edge signals on either leg — fall back to the old single-line view.
      lines.push(`  Hit rate          ${(result.edge_hit_rate * 100).toFixed(1)}%  [95% CI: ${(result.hit_rate_ci[0] * 100).toFixed(1)}% to ${(result.hit_rate_ci[1] * 100).toFixed(1)}%]`);
      lines.push(`  Flat-bet P&L      ${result.flat_bet_pnl >= 0 ? '+' : ''}$${result.flat_bet_pnl.toFixed(2)} (ROI: ${fmtRoi(result.flat_bet_roi)})`);
    }
  }

  // ─── Risk-adjusted view ───────────────────────────────────────────────
  // ROI alone hides interim losses and the structural NO tilt. Max drawdown
  // is the worst peak-to-trough drop of the flat-bet equity curve; alpha is
  // the model's ROI minus always-NO ROI on the same rows it bet on.
  if (result.edge_signals > 0) {
    lines.push('');
    lines.push('  RISK-ADJUSTED');
    lines.push(`    Max drawdown    ${(result.max_drawdown_pct * 100).toFixed(1)}% of deployed capital`);
    const rar = result.risk_adjusted_return;
    lines.push(`    ROI / max DD    ${Number.isFinite(rar) ? rar.toFixed(2) : '∞ (no drawdown)'}`);
    lines.push(`    Alpha vs NO     ${result.alpha_vs_always_no_pp >= 0 ? '+' : ''}${result.alpha_vs_always_no_pp.toFixed(1)}pp (same rows, always-NO benchmark)`);
  }

  // ─── Zero-skill baselines ─────────────────────────────────────────────
  // The headline ROI / hit rate can look strong purely from the universe's
  // structural NO tilt (multi-outcome events resolve mostly NO). These two
  // baselines run the same post-filter universe under zero-skill strategies
  // so the user can see whether the model adds anything.
  const b = result.baselines;
  if (result.signals.length > 0) {
    lines.push('');
    lines.push('  Zero-skill baselines (same universe, no model):');
    lines.push(`    Always-NO ROI     ${fmtRoi(b.always_no_roi)}   hit rate ${(b.always_no_hit_rate * 100).toFixed(1)}%`);
    lines.push(`    Always-YES ROI    ${fmtRoi(b.always_yes_roi)}   hit rate ${(b.always_yes_hit_rate * 100).toFixed(1)}%`);
    lines.push(`    Within-band skill ${fmtPp(b.within_band_skill_pp)}   (model NO-ROI minus always-NO ROI, capital-weighted across entry-price bands)`);
    // Per-band breakdown when at least one band has model bets
    if (b.within_band_breakdown.some((r) => r.n_model > 0)) {
      lines.push('');
      lines.push('    Per-band skill breakdown:');
      lines.push(`      ${'Band'.padEnd(8)}  ${'Model NO ROI'.padStart(13)}  ${'Always-NO ROI'.padStart(14)}  ${'Delta'.padStart(9)}  ${'n_model'.padStart(7)}  ${'n_total'.padStart(7)}`);
      for (const row of b.within_band_breakdown) {
        if (row.n_universe === 0) continue;
        const delta = `${row.skill_delta_pp >= 0 ? '+' : ''}${row.skill_delta_pp.toFixed(1)}pp`;
        lines.push(`      ${row.band.padEnd(8)}  ${fmtRoi(row.model_no_roi).padStart(13)}  ${fmtRoi(row.always_no_roi).padStart(14)}  ${delta.padStart(9)}  ${String(row.n_model).padStart(7)}  ${String(row.n_universe).padStart(7)}`);
      }
    }
  }

  // Coverage cost of the strict (no lifetime-volume look-ahead) volume gate.
  if (result.signals_dropped_no_volume > 0) {
    lines.push('');
    lines.push(`  Signals dropped: ${result.signals_dropped_no_volume} (no per-contract volume in Octagon snapshot; lifetime-volume fallback removed to avoid look-ahead bias)`);
  }

  // Resolved detail table
  const resolved = result.signals.filter(s => s.resolved);
  if (resolved.length > 0) {
    lines.push('');
    lines.push(formatResolvedTable(resolved));
  }

  // Unresolved detail table
  const unresolved = result.signals.filter(s => !s.resolved);
  if (unresolved.length > 0) {
    lines.push('');
    lines.push(formatUnresolvedTable(unresolved, minEdgePp));
  }

  return lines.join('\n');
}

function formatResolvedTable(signals: ScoredSignal[]): string {
  const lines: string[] = [];
  lines.push(`RESOLVED (${signals.length} markets — scored against Kalshi settlement)`);
  lines.push('─────────────────────────────────────────────────────────');

  const header = '  ' + [
    'Ticker'.padEnd(30),
    'Model'.padStart(6),
    'Mkt Then'.padStart(9),
    'Outcome'.padStart(10),
    'Edge'.padStart(7),
    'Bkt'.padStart(7),
    'P&L'.padStart(8),
    'ROI'.padStart(8),
  ].join('  ');
  lines.push(header);

  // Sort by |P&L| descending
  const sorted = [...signals].sort((a, b) => Math.abs(b.pnl) - Math.abs(a.pnl));
  for (const s of sorted.slice(0, 20)) {
    const outcome = s.market_now === 100 ? 'YES 100%' : 'NO  0%';
    const roi = s.capital > 0 ? (s.pnl / s.capital) * 100 : 0;
    const roiStr = s.capital > 0
      ? `${roi >= 0 ? '+' : ''}${roi.toFixed(1)}%`
      : '—';
    const row = '  ' + [
      s.market_ticker.padEnd(30),
      `${s.model_prob.toFixed(0)}%`.padStart(6),
      `${s.market_then.toFixed(0)}%`.padStart(9),
      outcome.padStart(10),
      `${s.edge_pp >= 0 ? '+' : ''}${s.edge_pp.toFixed(0)}pp`.padStart(7),
      s.edge_bucket.padStart(7),
      `${s.pnl >= 0 ? '+' : ''}$${s.pnl.toFixed(2)}`.padStart(8),
      roiStr.padStart(8),
    ].join('  ');
    lines.push(row);
  }
  if (sorted.length > 20) {
    lines.push(`  ... and ${sorted.length - 20} more`);
  }

  return lines.join('\n');
}

function formatUnresolvedTable(signals: ScoredSignal[], minEdgePp: string): string {
  const lines: string[] = [];
  lines.push(`UNRESOLVED (${signals.length} markets — mark-to-market vs Kalshi trading price)`);
  lines.push('────────────────────────────────────────────────────────────────');

  const header = '  ' + [
    'Ticker'.padEnd(30),
    'Model'.padStart(6),
    'Mkt Then'.padStart(9),
    'Now'.padStart(6),
    'Edge'.padStart(7),
    'Bkt'.padStart(7),
    'M2M'.padStart(8),
    'ROI'.padStart(8),
  ].join('  ');
  lines.push(header);

  // Sort by |edge| descending
  const sorted = [...signals].sort((a, b) => Math.abs(b.edge_pp) - Math.abs(a.edge_pp));
  for (const s of sorted.slice(0, 20)) {
    const roi = s.capital > 0 ? (s.pnl / s.capital) * 100 : 0;
    const roiStr = s.capital > 0
      ? `${roi >= 0 ? '+' : ''}${roi.toFixed(1)}%`
      : '—';
    const row = '  ' + [
      s.market_ticker.padEnd(30),
      `${s.model_prob.toFixed(0)}%`.padStart(6),
      `${s.market_then.toFixed(0)}%`.padStart(9),
      `${s.market_now.toFixed(0)}%`.padStart(6),
      `${s.edge_pp >= 0 ? '+' : ''}${s.edge_pp.toFixed(0)}pp`.padStart(7),
      s.edge_bucket.padStart(7),
      `${s.pnl >= 0 ? '+' : ''}$${s.pnl.toFixed(2)}`.padStart(8),
      roiStr.padStart(8),
    ].join('  ');
    lines.push(row);
  }
  if (sorted.length > 20) {
    lines.push(`  ... and ${sorted.length - 20} more`);
  }

  return lines.join('\n');
}

/** Escape a CSV cell: wrap in quotes if it contains comma, quote, or newline. */
function csvEscape(val: string | number): string {
  const s = String(val);
  if (s.includes(',') || s.includes('"') || s.includes('\n') || s.includes('\r')) {
    return '"' + s.replace(/"/g, '""') + '"';
  }
  return s;
}

/**
 * Export per-market detail to CSV.
 */
export function exportCSV(result: BacktestResult, path: string): void {
  const rows: string[] = [];
  rows.push('type,ticker,event_ticker,series_category,edge_bucket,model_prob,market_then,market_now,edge_pp,pnl,capital,resolved,close_time');

  for (const s of result.signals) {
    rows.push([
      s.resolved ? 'resolved' : 'unresolved',
      csvEscape(s.market_ticker),
      csvEscape(s.event_ticker),
      csvEscape(s.series_category),
      csvEscape(s.edge_bucket),
      s.model_prob.toFixed(1),
      s.market_then.toFixed(1),
      s.market_now.toFixed(1),
      s.edge_pp.toFixed(1),
      s.pnl.toFixed(4),
      s.capital.toFixed(4),
      s.resolved ? '1' : '0',
      csvEscape(s.close_time),
    ].join(','));
  }

  writeFileSync(path, rows.join('\n') + '\n');
}
