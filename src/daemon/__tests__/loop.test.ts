import { afterEach, beforeEach, describe, expect, test } from 'bun:test';
import { Database } from 'bun:sqlite';
import { migrate } from '../../db/schema';
import { runDaemonCycle, formatCycleSummary } from '../loop';

const realFetch = globalThis.fetch;

beforeEach(() => {
  // Every network call fails: the cycle must still complete all steps.
  globalThis.fetch = (async () => { throw new Error('network down'); }) as unknown as typeof fetch;
});

afterEach(() => {
  globalThis.fetch = realFetch;
});

describe('runDaemonCycle', () => {
  test('completes all steps fail-soft when the network is down', async () => {
    const db = new Database(':memory:');
    migrate(db);
    const cycle = await runDaemonCycle(db);
    expect(cycle.steps).toHaveLength(5);
    expect(cycle.steps.map((s) => s.step)).toEqual(['index', 'prefetch', 'settlements', 'paper', 'reflection']);
    // DB-only steps succeed even offline; network steps report failure
    const settlements = cycle.steps.find((s) => s.step === 'settlements')!;
    expect(settlements.ok).toBe(false);
    const reflection = cycle.steps.find((s) => s.step === 'reflection')!;
    expect(reflection.ok).toBe(true); // nothing pending, no network needed
  }, 30_000);

  test('summary renders one line with per-step status', async () => {
    const db = new Database(':memory:');
    migrate(db);
    const cycle = await runDaemonCycle(db);
    const line = formatCycleSummary(cycle);
    expect(line).toContain('[daemon');
    expect(line).toContain('settlements');
    expect(line.split('\n')).toHaveLength(1);
  }, 30_000);
});
