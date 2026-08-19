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
