/**
 * Adversarial bear-check (TradingAgents pattern): when the model claims an
 * extreme edge, a skeptic pass argues AGAINST the trade before anyone acts
 * on it. Backtest evidence motivates this: 20pp+ edges hit only 32% with
 * -18.6% ROI — exactly the signals that look most attractive are the ones
 * most likely to be model error (stale data, misread resolution rules,
 * fat-tail news already priced in).
 *
 * The skeptic runs on the fast model and returns a structured verdict.
 * It is advisory: it annotates the analysis, it does not block the mandate
 * path (hard caps do that). No LLM key → the check reports itself skipped.
 */
import { z } from 'zod';
import { getBotSetting } from '../utils/bot-config.js';

export const DEFAULT_BEAR_CHECK_MIN_EDGE_PP = 15;

export interface BearCheckInput {
  ticker: string;
  title: string;
  modelProb: number;   // 0-1
  marketProb: number;  // 0-1
  edgePp: number;      // signed, model - market, in points
  keyDrivers: string[];
  lessons?: string[];
}

export interface BearCheckResult {
  verdict: 'confirms' | 'cautions' | 'rejects';
  argument: string;
}

export type BearCheckLlm = (prompt: string, schema: z.ZodTypeAny) => Promise<unknown>;

export function bearCheckThresholdPp(): number {
  const v = getBotSetting('risk.bear_check_min_edge_pp');
  return typeof v === 'number' && Number.isFinite(v) && v > 0 ? v : DEFAULT_BEAR_CHECK_MIN_EDGE_PP;
}

export function needsBearCheck(edgePp: number | null): boolean {
  return edgePp !== null && Math.abs(edgePp) >= bearCheckThresholdPp();
}

export const BEAR_CHECK_SCHEMA = z.object({
  verdict: z.enum(['confirms', 'cautions', 'rejects'])
    .describe('rejects = the edge is probably model error; cautions = real risks unpriced by the model; confirms = the skeptic case is weak'),
  argument: z.string().describe('The strongest 2-3 sentence case AGAINST the trade (or why no strong case exists)'),
});

export function buildBearPrompt(input: BearCheckInput): string {
  const side = input.edgePp > 0 ? 'YES' : 'NO';
  return [
    `You are the designated skeptic. A model claims a ${Math.abs(input.edgePp).toFixed(0)}pp edge on a prediction market — historically, edges this large are usually model error, not market error. Argue AGAINST taking the trade.`,
    '',
    `Market: ${input.title} (${input.ticker})`,
    `Model: ${(input.modelProb * 100).toFixed(0)}% YES · Market price: ${(input.marketProb * 100).toFixed(0)}% YES → model wants ${side}`,
    input.keyDrivers.length ? `Model's stated drivers:\n${input.keyDrivers.map((d) => `- ${d}`).join('\n')}` : '',
    input.lessons?.length ? `Past lessons in this series:\n${input.lessons.map((l) => `- ${l}`).join('\n')}` : '',
    '',
    'Consider: is the model working from stale data? Could the resolution rules read differently than the model assumes? Is the "edge" actually the market pricing information the model lacks? Return your verdict and the strongest case against.',
  ].filter(Boolean).join('\n');
}

/**
 * Run the skeptic. Returns null when no LLM is available or the call fails —
 * callers render "bear check unavailable" rather than false confidence.
 */
export async function runBearCheck(
  input: BearCheckInput,
  llm: BearCheckLlm | undefined,
): Promise<BearCheckResult | null> {
  if (!llm) return null;
  try {
    const raw = await llm(buildBearPrompt(input), BEAR_CHECK_SCHEMA);
    const parsed = BEAR_CHECK_SCHEMA.safeParse(raw);
    return parsed.success ? parsed.data : null;
  } catch {
    return null;
  }
}

export function formatBearCheck(result: BearCheckResult | null, edgePp: number): string {
  const header = `  ⚔ BEAR CHECK (|edge| ${Math.abs(edgePp).toFixed(0)}pp ≥ ${bearCheckThresholdPp()}pp — extreme edges hit 32% historically)`;
  if (!result) {
    return `${header}\n    Skeptic unavailable (no LLM provider) — treat this edge with suspicion by default.`;
  }
  const badge = result.verdict === 'rejects' ? '✗ REJECTS' : result.verdict === 'cautions' ? '⚠ CAUTIONS' : '✓ survives';
  return `${header}\n    ${badge}: ${result.argument}`;
}
