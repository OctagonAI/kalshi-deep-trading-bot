/**
 * Kalshi V2 order endpoints (/portfolio/events/orders).
 *
 * The legacy /portfolio/orders endpoints now return
 * 410 Gone (code: deprecated_v1_order_endpoint). V2 quotes everything from
 * the YES side of the single book: `bid` buys YES, `ask` sells YES. Buying
 * NO at p is expressed as selling YES at 1 - p, and vice versa. Counts and
 * prices are fixed-point strings ("10.00", "0.5600").
 *
 * Docs: https://docs.kalshi.com/api-reference/orders/create-order-v2
 */
import { callKalshiApi, type KalshiApiResponse } from './api.js';

export type V2Side = 'bid' | 'ask';
export type V2TimeInForce = 'fill_or_kill' | 'good_till_canceled' | 'immediate_or_cancel';

export interface V2OrderBody {
  ticker: string;
  side: V2Side;
  count: string;
  price: string;
  time_in_force: V2TimeInForce;
  self_trade_prevention_type: 'taker_at_cross' | 'maker';
  client_order_id?: string;
  post_only?: boolean;
  reduce_only?: boolean;
  expiration_time?: number;
}

/** Map v1 action/side semantics onto the V2 YES-side book. */
export function toV2Side(action: 'buy' | 'sell', side: 'yes' | 'no'): V2Side {
  // buy yes = bid · sell yes = ask · buy no = sell yes = ask · sell no = bid
  return (action === 'buy') === (side === 'yes') ? 'bid' : 'ask';
}

/** Map a v1 cents price onto the V2 YES-side dollar price. */
export function toV2Price(priceCents: number, side: 'yes' | 'no'): string {
  const yesCents = side === 'yes' ? priceCents : 100 - priceCents;
  return (yesCents / 100).toFixed(4);
}

export function toV2Count(count: number): string {
  return count.toFixed(2);
}

export interface V1OrderIntent {
  ticker: string;
  action: 'buy' | 'sell';
  side: 'yes' | 'no';
  count: number;
  /** Price in cents on the chosen side (1-99). Omit for market orders. */
  priceCents?: number;
  timeInForce?: V2TimeInForce;
  clientOrderId?: string;
  expirationTs?: number;
}

/**
 * Build a V2 order body from v1-style intent. Market orders (no priceCents)
 * are emulated as immediate-or-cancel at the worst acceptable price, which
 * matches v1 market-order semantics on a 1-99 cent book.
 */
export function buildV2Order(intent: V1OrderIntent): V2OrderBody {
  const v2side = toV2Side(intent.action, intent.side);
  const isMarket = intent.priceCents === undefined;
  const price = isMarket
    ? (v2side === 'bid' ? '0.9900' : '0.0100')
    : toV2Price(intent.priceCents!, intent.side);
  const body: V2OrderBody = {
    ticker: intent.ticker,
    side: v2side,
    count: toV2Count(intent.count),
    price,
    time_in_force: intent.timeInForce ?? (isMarket ? 'immediate_or_cancel' : 'good_till_canceled'),
    self_trade_prevention_type: 'taker_at_cross',
  };
  if (intent.clientOrderId) body.client_order_id = intent.clientOrderId;
  if (intent.expirationTs !== undefined) body.expiration_time = intent.expirationTs;
  return body;
}

export async function placeOrderV2(body: V2OrderBody): Promise<KalshiApiResponse> {
  return callKalshiApi('POST', '/portfolio/events/orders', { body: body as unknown as Record<string, unknown> });
}

export async function cancelOrderV2(orderId: string): Promise<KalshiApiResponse> {
  return callKalshiApi('DELETE', `/portfolio/events/orders/${encodeURIComponent(orderId)}`);
}

export async function batchCancelOrdersV2(orderIds: string[]): Promise<KalshiApiResponse> {
  return callKalshiApi('DELETE', '/portfolio/events/orders/batched', {
    body: { orders: orderIds.map((order_id) => ({ order_id })) },
  });
}

export async function batchCreateOrdersV2(orders: V2OrderBody[]): Promise<KalshiApiResponse> {
  return callKalshiApi('POST', '/portfolio/events/orders/batched', {
    body: { orders: orders as unknown as Record<string, unknown>[] },
  });
}

export interface V2AmendIntent {
  orderId: string;
  ticker: string;
  action: 'buy' | 'sell';
  side: 'yes' | 'no';
  /** Updated total fillable count (not a delta). */
  count: number;
  priceCents: number;
}

export async function amendOrderV2(intent: V2AmendIntent): Promise<KalshiApiResponse> {
  return callKalshiApi('POST', `/portfolio/events/orders/${encodeURIComponent(intent.orderId)}/amend`, {
    body: {
      ticker: intent.ticker,
      side: toV2Side(intent.action, intent.side),
      count: toV2Count(intent.count),
      price: toV2Price(intent.priceCents, intent.side),
    },
  });
}
