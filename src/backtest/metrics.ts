import type { ScoredSignal, BacktestResult, LegMetrics } from './types.js';

/**
 * Skill score: how much better Octagon is vs the market as a forecaster.
 * Positive = model beats market. Negative = market is better.
 */
export function computeSkillScore(brierOctagon: number, brierMarket: number): number {
  if (brierMarket === 0) return 0;
  return 1 - (brierOctagon / brierMarket);
}

/**
 * Bootstrap confidence interval for a statistic.
 * Resamples `data` with replacement `iterations` times, computes `statFn` on each sample.
 * Returns [lower, upper] at the given confidence level (default 95%).
 */
export function bootstrapCI(
  data: number[],
  statFn: (sample: number[]) => number,
  iterations = 10_000,
  alpha = 0.05,
): [number, number] {
  if (data.length === 0) return [0, 0];
  if (!Number.isFinite(iterations) || !Number.isInteger(iterations) || iterations <= 0) {
    throw new Error(`bootstrapCI: iterations must be a finite integer > 0, got ${iterations}`);
  }
  if (!Number.isFinite(alpha) || alpha <= 0 || alpha >= 1) {
    throw new Error(`bootstrapCI: alpha must be a finite number in (0, 1), got ${alpha}`);
  }

  const stats: number[] = [];
  for (let i = 0; i < iterations; i++) {
    const sample: number[] = [];
    for (let j = 0; j < data.length; j++) {
      sample.push(data[Math.floor(Math.random() * data.length)]);
    }
    stats.push(statFn(sample));
  }
  stats.sort((a, b) => a - b);

  if (stats.length === 0) return [0, 0];
  const lo = Math.min(Math.max(0, Math.floor((alpha / 2) * stats.length)), stats.length - 1);
  const hi = Math.min(Math.max(0, Math.floor((1 - alpha / 2) * stats.length)), stats.length - 1);
  return [stats[lo], stats[hi]];
}

/**
 * Cluster bootstrap — resamples GROUPS with replacement, not individual rows.
 *
 * Use when the unit of risk is the group, not the row. In our case Kalshi
 * events are multi-outcome: a single "Will the Fed cut N times?" event has
 * a ladder of NO contracts that all settle together. If the model bets NO
 * on five rungs and the Fed cuts once, all five settle NO simultaneously
 * — that's *one* underlying outcome contributing five rows.
 *
 * Row-level bootstrap (`bootstrapCI` above) treats those rows as independent
 * → CI width shrinks with √N where N is the row count. Real effective N is
 * closer to the event count, so the honest interval is roughly 2× wider.
 *
 * This function takes `groups` (each group = a contiguous block of indices
 * referring to per-row data carried in closures by `statFn`), draws
 * `groups.length` groups with replacement per iteration, concatenates the
 * indices, and applies `statFn` to the pooled sample.
 */
export function clusterBootstrapCI(
  groups: number[][],
  statFn: (sampleIndices: number[]) => number,
  iterations = 10_000,
  alpha = 0.05,
): [number, number] {
  if (groups.length === 0) return [0, 0];
  if (!Number.isFinite(iterations) || !Number.isInteger(iterations) || iterations <= 0) {
    throw new Error(`clusterBootstrapCI: iterations must be a finite integer > 0, got ${iterations}`);
  }
  if (!Number.isFinite(alpha) || alpha <= 0 || alpha >= 1) {
    throw new Error(`clusterBootstrapCI: alpha must be a finite number in (0, 1), got ${alpha}`);
  }
  const stats: number[] = [];
  for (let i = 0; i < iterations; i++) {
    const pooled: number[] = [];
    for (let j = 0; j < groups.length; j++) {
      const g = groups[Math.floor(Math.random() * groups.length)];
      pooled.push(...g);
    }
    if (pooled.length === 0) { stats.push(0); continue; }
    stats.push(statFn(pooled));
  }
  stats.sort((a, b) => a - b);
  const lo = Math.min(Math.max(0, Math.floor((alpha / 2) * stats.length)), stats.length - 1);
  const hi = Math.min(Math.max(0, Math.floor((1 - alpha / 2) * stats.length)), stats.length - 1);
  return [stats[lo], stats[hi]];
}

/**
 * Build event-clustered groups: indices for each signal grouped by event_ticker.
 * Used to feed clusterBootstrapCI when the underlying signals are correlated
 * within an event (multi-outcome ladders, mutually-exclusive option sets).
 */
function groupIndicesByEvent<T extends { event_ticker: string }>(items: T[]): number[][] {
  const byEvent = new Map<string, number[]>();
  items.forEach((item, idx) => {
    const arr = byEvent.get(item.event_ticker) ?? [];
    arr.push(idx);
    byEvent.set(item.event_ticker, arr);
  });
  return [...byEvent.values()];
}

/**
 * Compute Brier score: ((forecast/100) - (outcome/100))²
 * Both forecast and outcome are on 0-100 scale.
 */
function brier(forecast: number, outcome: number): number {
  return ((forecast / 100) - (outcome / 100)) ** 2;
}

/** Entry-price bands used by the within-band skill calculation. */
const PRICE_BANDS: Array<{ label: string; lo: number; hi: number }> = [
  { label: '5-20¢',   lo: 5,  hi: 20 },
  { label: '20-40¢',  lo: 20, hi: 40 },
  { label: '40-60¢',  lo: 40, hi: 60 },
  { label: '60-80¢',  lo: 60, hi: 80 },
  { label: '80-95¢',  lo: 80, hi: 95 },
];

/**
 * Compute zero-skill baselines on the same post-filter universe as the model.
 *
 * Why: Kalshi events are multi-outcome — most resolved contracts settle NO
 * because each event has one YES outcome and many NOs. A model that
 * consistently picks NO will hit ~75% by structure alone. The "always-NO"
 * baseline strips that structural tilt out so we can see whether the model
 * has any selection skill beyond the universe's bias.
 *
 * Within-band skill: model NO-bet ROI minus always-NO ROI, computed inside
 * entry-price buckets and capital-weighted across buckets. This also
 * controls for the entry-price mix — a model that only bets cheap
 * longshots will look great vs. an always-NO baseline run over the full
 * universe, but mediocre once we compare within the same price band.
 */

/**
 * Equity-curve risk on the edge signals, ordered by close_time (entry-order
 * proxy). Max drawdown is the largest peak-to-trough drop of cumulative
 * flat-bet P&L, expressed as a fraction of total capital deployed so it is
 * comparable to the ROI number (AI-Trader-style peak tracking). The
 * risk-adjusted return divides ROI by max drawdown (Calmar-like): it punishes
 * strategies whose headline ROI rides through deep interim losses.
 */
export function computeEquityRisk(edgeSignals: ScoredSignal[], totalCapital: number): {
  max_drawdown_pct: number;
  risk_adjusted_return: number;
} {
  if (edgeSignals.length === 0 || totalCapital <= 0) {
    return { max_drawdown_pct: 0, risk_adjusted_return: 0 };
  }
  // Aggregate P&L per close_time first: signals settling at the same moment
  // have no defined order, and walking them individually would make the
  // drawdown depend on input order.
  const pnlByCloseTime = new Map<string, number>();
  for (const sgl of edgeSignals) {
    const key = String(sgl.close_time ?? '');
    pnlByCloseTime.set(key, (pnlByCloseTime.get(key) ?? 0) + sgl.pnl);
  }
  const orderedTimes = [...pnlByCloseTime.keys()].sort((a, b) => a.localeCompare(b));
  let equity = 0;
  let peak = 0;
  let maxDdAbs = 0;
  for (const t of orderedTimes) {
    equity += pnlByCloseTime.get(t)!;
    peak = Math.max(peak, equity);
    maxDdAbs = Math.max(maxDdAbs, peak - equity);
  }
  const maxDdPct = maxDdAbs / totalCapital;
  const roi = edgeSignals.reduce((sum, sgl) => sum + sgl.pnl, 0) / totalCapital;
  const riskAdjusted = maxDdPct > 0 ? roi / maxDdPct : (roi > 0 ? Infinity : 0);
  return { max_drawdown_pct: maxDdPct, risk_adjusted_return: riskAdjusted };
}

/**
 * Always-NO ROI restricted to the SAME edge-signal rows the model bet on —
 * the apples-to-apples benchmark for the headline alpha. (The universe-wide
 * always-NO baseline in computeBaselines answers a different question:
 * structural tilt of the whole post-filter universe.)
 */
export function alphaVsAlwaysNoPp(edgeSignals: ScoredSignal[]): number {
  if (edgeSignals.length === 0) return 0;
  let noPnl = 0;
  let noCap = 0;
  let modelPnl = 0;
  let modelCap = 0;
  for (const s of edgeSignals) {
    noCap += (100 - s.market_then) / 100;
    noPnl += ((100 - s.market_now) - (100 - s.market_then)) / 100;
    modelPnl += s.pnl;
    modelCap += s.capital;
  }
  const modelRoi = modelCap > 0 ? modelPnl / modelCap : 0;
  const noRoi = noCap > 0 ? noPnl / noCap : 0;
  return (modelRoi - noRoi) * 100;
}

function computeBaselines(signals: ScoredSignal[]): BacktestResult['baselines'] {
  // Universe-wide always-NO / always-YES on the same post-filter rows.
  const noPnl = (s: ScoredSignal): { pnl: number; capital: number; hit: boolean } => {
    const capital = (100 - s.market_then) / 100;
    const settlement = 100 - s.market_now;
    const pnl = (settlement - (100 - s.market_then)) / 100;
    return { pnl, capital, hit: s.market_now < s.market_then };
  };
  const yesPnl = (s: ScoredSignal): { pnl: number; capital: number; hit: boolean } => {
    const capital = s.market_then / 100;
    const pnl = (s.market_now - s.market_then) / 100;
    return { pnl, capital, hit: s.market_now > s.market_then };
  };

  let noP = 0, noC = 0, noHits = 0;
  let yesP = 0, yesC = 0, yesHits = 0;
  for (const s of signals) {
    const n = noPnl(s); noP += n.pnl; noC += n.capital; if (n.hit) noHits++;
    const y = yesPnl(s); yesP += y.pnl; yesC += y.capital; if (y.hit) yesHits++;
  }
  const alwaysNoRoi = noC > 0 ? noP / noC : 0;
  const alwaysYesRoi = yesC > 0 ? yesP / yesC : 0;
  const alwaysNoHitRate = signals.length > 0 ? noHits / signals.length : 0;
  const alwaysYesHitRate = signals.length > 0 ? yesHits / signals.length : 0;

  // Within-band: bucket by entry price, compute model-NO-bet ROI minus
  // always-NO ROI per band, then capital-weight the deltas across bands.
  // The model's NO bets are the meaningful comparable population (the
  // structural NO tilt is the dominant source of "skill" in the universe).
  const breakdown: BacktestResult['baselines']['within_band_breakdown'] = [];
  let weightedDeltaNumer = 0;
  let weightedDeltaDenom = 0;
  for (const band of PRICE_BANDS) {
    const inBand = signals.filter((s) => s.market_then >= band.lo && s.market_then < band.hi);
    if (inBand.length === 0) continue;

    let bandModelPnl = 0, bandModelCap = 0, bandModelN = 0;
    let bandNoPnl = 0, bandNoCap = 0;
    for (const s of inBand) {
      const n = noPnl(s); bandNoPnl += n.pnl; bandNoCap += n.capital;
      // Model "bet" here = signal where model picked NO (edge_pp < 0) and
      // capital is the NO-side capital we already stored in s.capital.
      if (s.edge_pp < 0) {
        bandModelPnl += s.pnl;
        bandModelCap += s.capital;
        bandModelN++;
      }
    }
    const modelRoi = bandModelCap > 0 ? bandModelPnl / bandModelCap : 0;
    const baselineRoi = bandNoCap > 0 ? bandNoPnl / bandNoCap : 0;
    const deltaPp = (modelRoi - baselineRoi) * 100;
    breakdown.push({
      band: band.label,
      model_no_roi: modelRoi,
      always_no_roi: baselineRoi,
      skill_delta_pp: deltaPp,
      n_model: bandModelN,
      n_universe: inBand.length,
    });
    weightedDeltaNumer += deltaPp * bandModelCap;
    weightedDeltaDenom += bandModelCap;
  }
  const withinBandSkillPp = weightedDeltaDenom > 0 ? weightedDeltaNumer / weightedDeltaDenom : 0;

  return {
    always_no_roi: alwaysNoRoi,
    always_no_hit_rate: alwaysNoHitRate,
    always_yes_roi: alwaysYesRoi,
    always_yes_hit_rate: alwaysYesHitRate,
    within_band_skill_pp: withinBandSkillPp,
    within_band_breakdown: breakdown,
  };
}

/**
 * Compute the scorecard for one leg (resolved-only or unresolved-only).
 * Same hit-rate and capital-weighted ROI definitions as the blended
 * computation, just scoped to the subset.
 */
function computeLegMetrics(signals: ScoredSignal[], minEdgePp: number): LegMetrics {
  const edgeSignals = signals.filter((s) => s.edge_pp !== 0 && Math.abs(s.edge_pp) >= minEdgePp);
  const edgeCount = edgeSignals.length;
  const hits = edgeSignals.filter((s) =>
    s.edge_pp > 0 ? s.market_now > s.market_then : s.market_now < s.market_then,
  );
  const hitRate = edgeCount > 0 ? hits.length / edgeCount : 0;
  const hitRateData = edgeSignals.map((s) =>
    s.edge_pp > 0 ? (s.market_now > s.market_then ? 1 : 0) : (s.market_now < s.market_then ? 1 : 0),
  );
  const legEventGroups = groupIndicesByEvent(edgeSignals);
  const hitRateCI: [number, number] = edgeCount > 0
    ? clusterBootstrapCI(legEventGroups, (sample) => {
        let sum = 0;
        for (const idx of sample) sum += hitRateData[idx];
        return sample.length > 0 ? sum / sample.length : 0;
      })
    : [0, 0];
  const pnl = edgeSignals.reduce((sum, s) => sum + s.pnl, 0);
  const totalCapital = edgeSignals.reduce((sum, s) => sum + s.capital, 0);
  const roi = totalCapital > 0 ? pnl / totalCapital : 0;
  return {
    edge_signals: edgeCount,
    edge_hit_rate: hitRate,
    hit_rate_ci: hitRateCI,
    flat_bet_pnl: pnl,
    flat_bet_roi: roi,
    total_capital: totalCapital,
  };
}

const EMPTY_LEG: LegMetrics = {
  edge_signals: 0,
  edge_hit_rate: 0,
  hit_rate_ci: [0, 0],
  flat_bet_pnl: 0,
  flat_bet_roi: 0,
  total_capital: 0,
};

const EMPTY_BASELINES: BacktestResult['baselines'] = {
  always_no_roi: 0,
  always_no_hit_rate: 0,
  always_yes_roi: 0,
  always_yes_hit_rate: 0,
  within_band_skill_pp: 0,
  within_band_breakdown: [],
};

/**
 * Compute all backtest metrics from a unified list of scored signals.
 */
export function computeMetrics(signals: ScoredSignal[], minEdgePp = 0.5): Omit<BacktestResult, 'subscription_notice' | 'signals_dropped_no_volume' | 'universe_source' | 'universe_size' | 'universe_description' | 'fee_model' | 'flat_bet_pnl_net' | 'flat_bet_roi_net'> {
  const n = signals.length;
  if (n === 0) {
    return {
      verdict: { summary: 'No markets with Octagon coverage found.', significant: false, profitable: false, beats_baseline: false },
      max_drawdown_pct: 0,
      risk_adjusted_return: 0,
      alpha_vs_always_no_pp: 0,
      days: 0,
      events_scored: 0,
      markets_resolved: 0,
      markets_unresolved: 0,
      brier_octagon: 0,
      brier_market: 0,
      skill_score: 0,
      skill_ci: [0, 0],
      edge_signals: 0,
      edge_hit_rate: 0,
      hit_rate_ci: [0, 0],
      flat_bet_pnl: 0,
      flat_bet_roi: 0,
      total_capital: 0,
      signals: [],
      baselines: EMPTY_BASELINES,
      resolved_metrics: EMPTY_LEG,
      unresolved_metrics: EMPTY_LEG,
    };
  }

  // Brier scores — model vs market, both compared to outcome (market_now)
  const brierOctagonScores = signals.map(s => brier(s.model_prob, s.market_now));
  const brierMarketScores = signals.map(s => brier(s.market_then, s.market_now));
  const brierOctagon = brierOctagonScores.reduce((a, b) => a + b, 0) / n;
  const brierMarket = brierMarketScores.reduce((a, b) => a + b, 0) / n;

  // Skill score with EVENT-CLUSTERED bootstrap CI. Why clustered: multi-
  // outcome events (Fed-cut ladders, election option sets, price strikes)
  // settle as a block — N contracts from one event aren't N independent
  // observations. Row-level bootstrap shrinks the CI with sqrt(N rows)
  // when the right denominator is sqrt(N events).
  const skillScore = computeSkillScore(brierOctagon, brierMarket);
  const eventGroups = groupIndicesByEvent(signals);
  const skillCI = clusterBootstrapCI(eventGroups, (sample) => {
    let sumOctagon = 0;
    let sumMarket = 0;
    for (const idx of sample) {
      sumOctagon += brierOctagonScores[idx];
      sumMarket += brierMarketScores[idx];
    }
    const avgOctagon = sumOctagon / sample.length;
    const avgMarket = sumMarket / sample.length;
    return avgMarket === 0 ? 0 : 1 - (avgOctagon / avgMarket);
  });

  // Edge signals: where |edge| >= minEdgePp AND edge is non-zero
  const edgeSignals = signals.filter(s => s.edge_pp !== 0 && Math.abs(s.edge_pp) >= minEdgePp);
  const edgeCount = edgeSignals.length;

  // Hit rate: did the market move in the direction the model predicted?
  const hits = edgeSignals.filter(s => {
    // Model said YES (edge > 0): hit if market_now > market_then
    // Model said NO (edge < 0): hit if market_now < market_then
    if (s.edge_pp > 0) return s.market_now > s.market_then;
    return s.market_now < s.market_then;
  });
  const hitRate = edgeCount > 0 ? hits.length / edgeCount : 0;

  // Event-clustered hit rate CI on the EDGE signals only.
  const hitRateData = edgeSignals.map(s => {
    if (s.edge_pp > 0) return s.market_now > s.market_then ? 1 : 0;
    return s.market_now < s.market_then ? 1 : 0;
  });
  const edgeEventGroups = groupIndicesByEvent(edgeSignals);
  const hitRateCI = clusterBootstrapCI(edgeEventGroups, (sample) => {
    let sum = 0;
    for (const idx of sample) sum += hitRateData[idx];
    return sample.length > 0 ? sum / sample.length : 0;
  });

  // P&L and capital-weighted ROI (matches Supabase methodology):
  //   ROI = sum(pnl) / sum(capital) across edge signals.
  const pnl = edgeSignals.reduce((sum, s) => sum + s.pnl, 0);
  const totalCapital = edgeSignals.reduce((sum, s) => sum + s.capital, 0);
  const roi = totalCapital > 0 ? pnl / totalCapital : 0;

  // Counts
  const uniqueEvents = new Set(signals.map(s => s.event_ticker));
  const resolved = signals.filter(s => s.resolved).length;
  const unresolved = signals.filter(s => !s.resolved).length;

  // Risk-adjusted view: max drawdown of the flat-bet equity curve and alpha
  // over always-NO on the same rows. ROI alone hides both (a +0.9% ROI with
  // a -18% interim drawdown and negative alpha is not a strategy).
  const equityRisk = computeEquityRisk(edgeSignals, totalCapital);
  const alphaPp = alphaVsAlwaysNoPp(edgeSignals);

  // Verdict
  const significant = skillCI[0] > 0;
  const profitable = pnl > 0;
  const beatsBaseline = alphaPp > 0;
  const riskLine = `alpha vs always-NO ${alphaPp >= 0 ? '+' : ''}${alphaPp.toFixed(1)}pp; max DD ${(equityRisk.max_drawdown_pct * 100).toFixed(1)}%`;
  let summary: string;
  if (skillScore > 0.05 && significant && profitable && beatsBaseline) {
    summary = `Model has edge (Skill +${(skillScore * 100).toFixed(1)}% [CI: +${(skillCI[0] * 100).toFixed(1)}%, +${(skillCI[1] * 100).toFixed(1)}%]; ROI +${(roi * 100).toFixed(1)}%; ${riskLine})`;
  } else if (skillScore > 0.05 && significant && profitable && !beatsBaseline) {
    summary = `ROI is baseline tilt, not selection skill (ROI +${(roi * 100).toFixed(1)}% but ${riskLine})`;
  } else if (skillScore > 0 && !significant) {
    summary = `Inconclusive — need more data (Skill +${(skillScore * 100).toFixed(1)}%, CI includes zero; ${riskLine})`;
  } else {
    summary = `No edge detected (Skill ${(skillScore * 100).toFixed(1)}%; ${riskLine})`;
  }

  return {
    verdict: { summary, significant, profitable, beats_baseline: beatsBaseline },
    max_drawdown_pct: equityRisk.max_drawdown_pct,
    risk_adjusted_return: equityRisk.risk_adjusted_return,
    alpha_vs_always_no_pp: alphaPp,
    days: 0, // filled by caller
    events_scored: uniqueEvents.size,
    markets_resolved: resolved,
    markets_unresolved: unresolved,
    brier_octagon: brierOctagon,
    brier_market: brierMarket,
    skill_score: skillScore,
    skill_ci: skillCI,
    edge_signals: edgeCount,
    edge_hit_rate: hitRate,
    hit_rate_ci: hitRateCI,
    flat_bet_pnl: pnl,
    flat_bet_roi: roi,
    total_capital: totalCapital,
    signals,
    baselines: computeBaselines(signals),
    resolved_metrics: computeLegMetrics(signals.filter((s) => s.resolved), minEdgePp),
    unresolved_metrics: computeLegMetrics(signals.filter((s) => !s.resolved), minEdgePp),
  };
}
