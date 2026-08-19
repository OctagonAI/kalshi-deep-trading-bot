import { DynamicStructuredTool } from '@langchain/core/tools';
import { z } from 'zod';
import { formatToolResult } from '../types.js';
import {
  amendOrderV2,
  batchCancelOrdersV2,
  batchCreateOrdersV2,
  buildV2Order,
  cancelOrderV2,
  placeOrderV2,
  toV2Count,
  toV2Side,
  type V2OrderBody,
} from './orders-v2.js';

// All order endpoints use the Kalshi V2 book (/portfolio/events/orders):
// prices are quoted from the YES side only, so `yes_price` inputs map to the
// V2 dollar price directly and action/side pick bid (buy YES) or ask (sell YES).

function yesCentsToV2Price(yesPrice: number): string {
  return (yesPrice / 100).toFixed(4);
}

function v2BodyFromYesCents(o: {
  ticker: string;
  action: 'buy' | 'sell';
  side: 'yes' | 'no';
  type: 'limit' | 'market';
  count: number;
  yes_price?: number;
  expiration_ts?: number;
  client_order_id?: string;
}): V2OrderBody {
  const side = toV2Side(o.action, o.side);
  const isMarket = o.type === 'market' || o.yes_price === undefined;
  const body: V2OrderBody = {
    ticker: o.ticker,
    side,
    count: toV2Count(o.count),
    // Market orders: immediate-or-cancel at the worst acceptable YES price.
    price: isMarket ? (side === 'bid' ? '0.9900' : '0.0100') : yesCentsToV2Price(o.yes_price!),
    time_in_force: isMarket ? 'immediate_or_cancel' : 'good_till_canceled',
    self_trade_prevention_type: 'taker_at_cross',
  };
  if (o.client_order_id) body.client_order_id = o.client_order_id;
  if (o.expiration_ts !== undefined) body.expiration_time = o.expiration_ts;
  return body;
}

export const placeOrder = new DynamicStructuredTool({
  name: 'place_order',
  description: 'Place a new order on a Kalshi market.',
  schema: z.object({
    ticker: z.string().describe('Market ticker'),
    action: z.enum(['buy', 'sell']).describe('Buy or sell'),
    side: z.enum(['yes', 'no']).describe('Yes or No side'),
    type: z.enum(['limit', 'market']).describe('Order type'),
    count: z.number().int().positive().describe('Number of contracts'),
    yes_price: z.number().int().min(1).max(99).optional().describe('Price in cents (1-99), always quoted on the YES side, for limit orders'),
    expiration_ts: z.number().optional().describe('Order expiration Unix timestamp'),
    client_order_id: z.string().optional().describe('Optional client-provided order ID'),
  }),
  func: async (input) => {
    const data = await placeOrderV2(v2BodyFromYesCents(input));
    return formatToolResult(data);
  },
});

export const amendOrder = new DynamicStructuredTool({
  name: 'amend_order',
  description: 'Amend an existing resting order. Count is the updated total fillable quantity, not a delta.',
  schema: z.object({
    order_id: z.string().describe('Order ID to amend'),
    ticker: z.string().describe('Market ticker of the order'),
    action: z.enum(['buy', 'sell']).describe('Original order action'),
    side: z.enum(['yes', 'no']).describe('Original order side'),
    count: z.number().int().positive().describe('New total contract count'),
    yes_price: z.number().int().min(1).max(99).describe('New price in cents, quoted on the YES side'),
  }),
  func: async (input) => {
    const data = await amendOrderV2({
      orderId: input.order_id,
      ticker: input.ticker,
      action: input.action,
      side: input.side,
      count: input.count,
      // Input is YES-side cents; express it on the order's own side so the
      // V2 conversion lands back on the same YES price.
      priceCents: input.side === 'yes' ? input.yes_price : 100 - input.yes_price,
    });
    return formatToolResult(data);
  },
});

export const cancelOrder = new DynamicStructuredTool({
  name: 'cancel_order',
  description: 'Cancel an existing resting order.',
  schema: z.object({
    order_id: z.string().describe('Order ID to cancel'),
  }),
  func: async (input) => {
    const data = await cancelOrderV2(input.order_id);
    return formatToolResult(data);
  },
});

export const cancelOrders = new DynamicStructuredTool({
  name: 'cancel_orders',
  description: 'Cancel multiple resting orders in batch.',
  schema: z.object({
    order_ids: z.array(z.string()).describe('List of order IDs to cancel'),
  }),
  func: async (input) => {
    const data = await batchCancelOrdersV2(input.order_ids);
    return formatToolResult(data);
  },
});

export const placeBatchOrders = new DynamicStructuredTool({
  name: 'place_batch_orders',
  description: 'Place multiple orders in a single batch request.',
  schema: z.object({
    orders: z
      .array(
        z.object({
          ticker: z.string(),
          action: z.enum(['buy', 'sell']),
          side: z.enum(['yes', 'no']),
          type: z.enum(['limit', 'market']),
          count: z.number().int().positive(),
          yes_price: z.number().int().min(1).max(99).optional(),
        })
      )
      .describe('List of orders to place'),
  }),
  func: async (input) => {
    const data = await batchCreateOrdersV2(input.orders.map((o) => v2BodyFromYesCents(o)));
    return formatToolResult(data);
  },
});
