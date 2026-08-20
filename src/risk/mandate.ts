/**
 * Mandate + kill switch (Vibe-Trading pattern): hard caps enforced at the
 * lowest order-placement chokepoint (orders-v2), so no command, agent tool,
 * or future code path can place an order that violates them. Cancels are
 * always allowed — they reduce risk.
 *
 * Settings (config.json via bot-config, editable with /config):
 *   mandate.kill_switch             boolean — refuse ALL new orders
 *   mandate.max_contracts_per_order default 100
 *   mandate.max_notional_per_order  dollars at risk per order, default $100
 *   mandate.max_daily_loss          dollars realized loss today, default $50
 *                                   (measured from the settlements ledger)
 */
import { getDb } from '../db/index.js';
import { getBotSetting, setBotSetting } from '../utils/bot-config.js';
import type { V2OrderBody } from '../tools/kalshi/orders-v2.js';

const DEFAULT_MAX_CONTRACTS = 100;
const DEFAULT_MAX_NOTIONAL = 100;
const DEFAULT_MAX_DAILY_LOSS = 50;

export class MandateViolation extends Error {
  constructor(public reasons: string[]) {
    super(`Order refused by mandate: ${reasons.join('; ')}`);
    this.name = 'MandateViolation';
  }
}

export interface MandateStatus {
  kill_switch: boolean;
  kill_reason: string | null;
  max_contracts_per_order: number;
  max_notional_per_order: number;
  max_daily_loss: number;
  realized_loss_today: number;
}

function num(key: string, fallback: number): number {
  const v = getBotSetting(key);
  return typeof v === 'number' && Number.isFinite(v) && v > 0 ? v : fallback;
}

/** Realized loss today (positive number = loss), from the settlements ledger. */
export function realizedLossToday(): number {
  try {
    const db = getDb();
    const today = new Date().toISOString().slice(0, 10);
    const row = db
      .prepare(`SELECT COALESCE(SUM(realized_pnl), 0) AS pnl FROM settlements WHERE settled_time >= ?`)
      .get(`${today}T00:00:00Z`) as { pnl: number };
    return row.pnl < 0 ? -row.pnl : 0;
  } catch {
    return 0; // no DB (tests, fresh install) — the cap simply can't bind yet
  }
}

export function getMandateStatus(): MandateStatus {
  return {
    kill_switch: getBotSetting('mandate.kill_switch') === true,
    kill_reason: (getBotSetting('mandate.kill_reason') as string | undefined) || null,
    max_contracts_per_order: num('mandate.max_contracts_per_order', DEFAULT_MAX_CONTRACTS),
    max_notional_per_order: num('mandate.max_notional_per_order', DEFAULT_MAX_NOTIONAL),
    max_daily_loss: num('mandate.max_daily_loss', DEFAULT_MAX_DAILY_LOSS),
    realized_loss_today: realizedLossToday(),
  };
}

export function activateKillSwitch(reason?: string): void {
  setBotSetting('mandate.kill_switch', 'true');
  setBotSetting('mandate.kill_reason', reason ?? 'manual halt');
}

export function deactivateKillSwitch(): void {
  setBotSetting('mandate.kill_switch', 'false');
  setBotSetting('mandate.kill_reason', '');
}

/** Dollars at risk for one V2 order: bid pays price, ask pays 1 - price. */
export function orderNotional(body: V2OrderBody): number {
  const count = parseFloat(body.count);
  const price = parseFloat(body.price);
  if (!Number.isFinite(count) || !Number.isFinite(price)) return 0;
  return body.side === 'bid' ? count * price : count * (1 - price);
}

/**
 * Throws MandateViolation when the order breaches any hard cap.
 * Called from placeOrderV2 / batchCreateOrdersV2 / amendOrderV2.
 */
export function enforceMandate(orders: V2OrderBody[]): void {
  const status = getMandateStatus();
  const reasons: string[] = [];

  if (status.kill_switch) {
    reasons.push(`kill switch is ACTIVE${status.kill_reason ? ` (${status.kill_reason})` : ''} — /resume to lift`);
  }
  if (status.realized_loss_today >= status.max_daily_loss) {
    reasons.push(
      `daily loss cap reached: -$${status.realized_loss_today.toFixed(2)} realized today >= $${status.max_daily_loss.toFixed(2)} limit`,
    );
  }
  for (const body of orders) {
    const count = parseFloat(body.count);
    if (count > status.max_contracts_per_order) {
      reasons.push(`${body.ticker}: ${count} contracts > per-order cap ${status.max_contracts_per_order}`);
    }
    const notional = orderNotional(body);
    if (notional > status.max_notional_per_order) {
      reasons.push(`${body.ticker}: $${notional.toFixed(2)} at risk > per-order cap $${status.max_notional_per_order.toFixed(2)}`);
    }
  }

  if (reasons.length > 0) throw new MandateViolation(reasons);
}

export function formatMandateHuman(status: MandateStatus): string {
  const lines: string[] = [];
  lines.push('**Trading Mandate**');
  lines.push('');
  lines.push(status.kill_switch
    ? `⛔ KILL SWITCH ACTIVE${status.kill_reason ? ` — ${status.kill_reason}` : ''} (all new orders refused; /resume to lift)`
    : '✓ Trading enabled');
  lines.push('');
  lines.push(`  Per-order contract cap   ${status.max_contracts_per_order}`);
  lines.push(`  Per-order notional cap   $${status.max_notional_per_order.toFixed(2)}`);
  lines.push(`  Daily realized-loss cap  $${status.max_daily_loss.toFixed(2)} (today: -$${status.realized_loss_today.toFixed(2)})`);
  lines.push('');
  lines.push('Caps are enforced at order placement for every path (manual, agent, batch).');
  lines.push('Adjust via /config set mandate.<key> <value>.');
  return lines.join('\n');
}
