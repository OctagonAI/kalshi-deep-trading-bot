import { describe, expect, test } from 'bun:test';
import { buildV2Order, toV2Count, toV2Price, toV2Side } from '../orders-v2';

describe('toV2Side quadrant mapping', () => {
  test('buy yes → bid', () => expect(toV2Side('buy', 'yes')).toBe('bid'));
  test('sell yes → ask', () => expect(toV2Side('sell', 'yes')).toBe('ask'));
  test('buy no → ask (sell YES)', () => expect(toV2Side('buy', 'no')).toBe('ask'));
  test('sell no → bid (buy YES)', () => expect(toV2Side('sell', 'no')).toBe('bid'));
});

describe('toV2Price', () => {
  test('yes side passes through', () => expect(toV2Price(56, 'yes')).toBe('0.5600'));
  test('no side complements to YES', () => expect(toV2Price(56, 'no')).toBe('0.4400'));
  test('1 cent boundary', () => expect(toV2Price(1, 'yes')).toBe('0.0100'));
  test('99 cent boundary on no side', () => expect(toV2Price(99, 'no')).toBe('0.0100'));
});

describe('toV2Count', () => {
  test('integer count is fixed-point', () => expect(toV2Count(10)).toBe('10.00'));
});

describe('buildV2Order', () => {
  test('limit buy yes', () => {
    const body = buildV2Order({ ticker: 'KX-TEST', action: 'buy', side: 'yes', count: 3, priceCents: 42 });
    expect(body).toMatchObject({
      ticker: 'KX-TEST',
      side: 'bid',
      count: '3.00',
      price: '0.4200',
      time_in_force: 'good_till_canceled',
      self_trade_prevention_type: 'taker_at_cross',
    });
  });

  test('limit buy no expresses YES-side price', () => {
    const body = buildV2Order({ ticker: 'KX-TEST', action: 'buy', side: 'no', count: 1, priceCents: 30 });
    expect(body.side).toBe('ask');
    expect(body.price).toBe('0.7000');
  });

  test('market order becomes IOC at worst acceptable price', () => {
    const buy = buildV2Order({ ticker: 'KX-TEST', action: 'buy', side: 'yes', count: 1 });
    expect(buy.time_in_force).toBe('immediate_or_cancel');
    expect(buy.price).toBe('0.9900');
    const sell = buildV2Order({ ticker: 'KX-TEST', action: 'sell', side: 'yes', count: 1 });
    expect(sell.price).toBe('0.0100');
  });

  test('market buy no is IOC ask at floor price', () => {
    const body = buildV2Order({ ticker: 'KX-TEST', action: 'buy', side: 'no', count: 2 });
    expect(body.side).toBe('ask');
    expect(body.price).toBe('0.0100');
    expect(body.time_in_force).toBe('immediate_or_cancel');
  });

  test('optional fields pass through', () => {
    const body = buildV2Order({
      ticker: 'KX-TEST', action: 'buy', side: 'yes', count: 1, priceCents: 50,
      clientOrderId: 'abc', expirationTs: 1234,
    });
    expect(body.client_order_id).toBe('abc');
    expect(body.expiration_time).toBe(1234);
  });
});

// ─── Input validation (review round) ────────────────────────────────────────
import { describe as dv, expect as ev, test as tv } from 'bun:test';
import { v2BodyFromYesCents } from '../trading';

dv('v2BodyFromYesCents validation', () => {
  tv('limit order without yes_price is rejected, not silently market', () => {
    ev(() => v2BodyFromYesCents({ ticker: 'KX-T', action: 'buy', side: 'yes', type: 'limit', count: 1 }))
      .toThrow(/requires yes_price/);
  });

  tv('market order with expiration_ts is rejected (IOC cannot expire)', () => {
    ev(() => v2BodyFromYesCents({ ticker: 'KX-T', action: 'buy', side: 'yes', type: 'market', count: 1, expiration_ts: 123 }))
      .toThrow(/cannot take expiration_ts/);
  });

  tv('valid market order becomes IOC at worst price', () => {
    const body = v2BodyFromYesCents({ ticker: 'KX-T', action: 'buy', side: 'yes', type: 'market', count: 2 });
    ev(body.time_in_force).toBe('immediate_or_cancel');
    ev(body.price).toBe('0.9900');
    ev(body.count).toBe('2.00');
  });

  tv('valid limit NO order converts YES-cents onto the V2 YES book', () => {
    const body = v2BodyFromYesCents({ ticker: 'KX-T', action: 'buy', side: 'no', type: 'limit', count: 1, yes_price: 30 });
    ev(body.side).toBe('ask');   // buy NO = sell YES
    ev(body.price).toBe('0.3000'); // yes_price is already YES-side
  });
});
