import { describe, expect, test } from 'bun:test';
import { buildBearPrompt, needsBearCheck, runBearCheck, formatBearCheck, BEAR_CHECK_SCHEMA } from '../bear-check';

const INPUT = {
  ticker: 'KX-T', title: 'Test market', modelProb: 0.9, marketProb: 0.6,
  edgePp: 30, keyDrivers: ['driver one'], lessons: ['past lesson'],
};

describe('needsBearCheck', () => {
  test('fires at |edge| >= threshold, both signs', () => {
    expect(needsBearCheck(15)).toBe(true);
    expect(needsBearCheck(-20)).toBe(true);
    expect(needsBearCheck(14.9)).toBe(false);
    expect(needsBearCheck(null)).toBe(false);
  });
});

describe('runBearCheck', () => {
  test('valid structured verdict passes through', async () => {
    const res = await runBearCheck(INPUT, async () => ({ verdict: 'rejects', argument: 'Stale polling data.' }));
    expect(res).toEqual({ verdict: 'rejects', argument: 'Stale polling data.' });
  });

  test('malformed output and LLM failure both yield null', async () => {
    expect(await runBearCheck(INPUT, async () => ({ nonsense: true }))).toBeNull();
    expect(await runBearCheck(INPUT, async () => { throw new Error('down'); })).toBeNull();
    expect(await runBearCheck(INPUT, undefined)).toBeNull();
  });
});

describe('formatting and prompt', () => {
  test('prompt contains the adversarial frame and inputs', () => {
    const p = buildBearPrompt(INPUT);
    expect(p).toContain('Argue AGAINST');
    expect(p).toContain('30pp');
    expect(p).toContain('driver one');
    expect(p).toContain('past lesson');
  });

  test('null result renders suspicion-by-default', () => {
    expect(formatBearCheck(null, 30)).toContain('treat this edge with suspicion');
  });

  test('rejects verdict renders prominently', () => {
    expect(formatBearCheck({ verdict: 'rejects', argument: 'x' }, -22)).toContain('✗ REJECTS');
  });
});
