import { afterEach, beforeEach, describe, expect, test } from 'bun:test';
import { tmpdir } from 'os';
import { join } from 'path';
import { rmSync } from 'fs';
import { enforceMandate, orderNotional, MandateViolation } from '../mandate';
import type { V2OrderBody } from '../../tools/kalshi/orders-v2';

// bot-config reads/writes config.json under the app dir; isolate via env if
// supported — otherwise these tests only exercise cap math + defaults, which
// don't touch the kill switch file state.

function order(over: Partial<V2OrderBody> = {}): V2OrderBody {
  return {
    ticker: 'KX-T', side: 'bid', count: '1.00', price: '0.5000',
    time_in_force: 'good_till_canceled', self_trade_prevention_type: 'taker_at_cross',
    ...over,
  };
}

describe('orderNotional', () => {
  test('bid risks price × count', () => {
    expect(orderNotional(order({ count: '10.00', price: '0.4000' }))).toBeCloseTo(4);
  });
  test('ask risks (1 - price) × count', () => {
    expect(orderNotional(order({ side: 'ask', count: '10.00', price: '0.4000' }))).toBeCloseTo(6);
  });
});

describe('enforceMandate caps', () => {
  test('small order passes default caps', () => {
    expect(() => enforceMandate([order()])).not.toThrow();
  });

  test('contract cap violation throws with the ticker named', () => {
    expect(() => enforceMandate([order({ count: '500.00' })])).toThrow(MandateViolation);
    try {
      enforceMandate([order({ count: '500.00' })]);
    } catch (e) {
      expect((e as MandateViolation).reasons.join(' ')).toContain('KX-T');
      expect((e as MandateViolation).reasons.join(' ')).toContain('per-order cap');
    }
  });

  test('notional cap violation throws', () => {
    // 200 contracts @ 0.99 = $198 > default $100 cap (also breaches count cap)
    expect(() => enforceMandate([order({ count: '200.00', price: '0.9900' })])).toThrow(/at risk/);
  });

  test('batch: one bad order fails the whole batch', () => {
    expect(() => enforceMandate([order(), order({ count: '500.00', ticker: 'KX-BAD' })])).toThrow(/KX-BAD/);
  });
});
