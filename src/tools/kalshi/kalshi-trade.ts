import { DynamicStructuredTool, StructuredToolInterface } from '@langchain/core/tools';
import type { RunnableConfig } from '@langchain/core/runnables';
import { AIMessage, ToolCall } from '@langchain/core/messages';
import { z } from 'zod';
import { callLlm } from '../../model/llm.js';
import { formatToolResult } from '../types.js';
import { getCurrentDate } from '../../agent/prompts.js';
import { placeOrder, amendOrder, cancelOrder, cancelOrders, placeBatchOrders } from './trading.js';

const TRADING_TOOLS: StructuredToolInterface[] = [
  placeOrder,
  amendOrder,
  cancelOrder,
  cancelOrders,
  placeBatchOrders,
];

const TRADING_TOOL_MAP = new Map(TRADING_TOOLS.map((t) => [t.name, t]));

export const KALSHI_TRADE_DESCRIPTION = `
Execute trading actions on Kalshi prediction markets. Routes natural language trade instructions to the correct API endpoints.

## When to Use

- Placing buy or sell orders (limit or market)
- Amending resting orders (price or quantity)
- Canceling one or more resting orders
- Placing batch orders

## IMPORTANT

- NEVER call this tool without explicit user confirmation of trade details
- Always confirm: ticker, side (yes/no), action (buy/sell), count — and price for limit orders (market orders need no price)
- Prices are in cents: $0.56 = 56 cents
`.trim();

function buildTradingRouterPrompt(): string {
  return `You are a Kalshi trading execution assistant.
Current date: ${getCurrentDate()}

Given a trading instruction, call the appropriate trading tool.

## Key Facts
- Prices are in cents (1-99), always quoted on the YES side: $0.56 → yes_price: 56
- side: "yes" or "no" (the contract type)
- action: "buy" or "sell"
- type: "limit" (use yes_price) or "market" (no price needed)
- To amend, pass order_id plus the order's ticker, action, and side; count is the new total quantity
- To cancel by order_id, use cancel_order
- For multiple cancels, use cancel_orders with order_ids array

Execute the trading action now.`;
}

const KalshiTradeInputSchema = z.object({
  action: z.string().describe('Natural language description of the trade to execute'),
});

export function createKalshiTrade(model: string): DynamicStructuredTool {
  return new DynamicStructuredTool({
    name: 'kalshi_trade',
    description: KALSHI_TRADE_DESCRIPTION,
    schema: KalshiTradeInputSchema,
    func: async (input, _runManager, config?: RunnableConfig) => {
      const onProgress = config?.metadata?.onProgress as ((msg: string) => void) | undefined;

      onProgress?.('Preparing trade...');
      const { response } = await callLlm(input.action, {
        model,
        systemPrompt: buildTradingRouterPrompt(),
        tools: TRADING_TOOLS,
      });
      const aiMessage = response as AIMessage;

      const toolCalls = aiMessage.tool_calls as ToolCall[];
      if (!toolCalls || toolCalls.length === 0) {
        return formatToolResult({ error: 'Could not parse trade instruction into a valid order' });
      }

      const results = [];
      for (const tc of toolCalls) {
        onProgress?.(`Executing ${tc.name.replace(/_/g, ' ')}...`);
        try {
          const tool = TRADING_TOOL_MAP.get(tc.name);
          if (!tool) throw new Error(`Tool '${tc.name}' not found`);
          const rawResult = await tool.invoke(tc.args);
          const result = typeof rawResult === 'string' ? rawResult : JSON.stringify(rawResult);
          const parsed = JSON.parse(result);
          results.push({ tool: tc.name, args: tc.args, data: parsed.data, error: null });
        } catch (error) {
          results.push({
            tool: tc.name,
            args: tc.args,
            data: null,
            error: error instanceof Error ? error.message : String(error),
          });
        }
      }

      return formatToolResult({ results });
    },
  });
}
