/**
 * Maintenance daemon: keeps every cache warm and every ledger current so
 * interactive commands never pay the refresh cost on the critical path.
 *
 * One cycle runs, in order:
 *   1. event index refresh          (search/browse stay instant)
 *   2. Octagon events prefetch      (analyze/report serve from prefetch)
 *   3. settlements sync             (realized P&L + Brier + hypotheses current)
 *   4. paper-position settlement    (forward-test ledger current)
 *   5. reflection                   (new settlements become lessons)
 *
 * Every step is fail-soft: an error is reported in the cycle summary and the
 * next step still runs — a dead Octagon API must not stop Kalshi settlement
 * capture, and vice versa.
 */
import type { Database } from 'bun:sqlite';
import { ensureIndex, getRefreshPromise } from '../tools/kalshi/search-index.js';
import { prefetchOctagonEvents } from '../scan/octagon-prefetch.js';
import { syncSettlements } from '../tools/kalshi/settle.js';
import { settlePaperPositions } from '../commands/paper.js';
import { generateMissingLessons, type LessonLlm } from '../eval/reflection.js';

export interface DaemonStepResult {
  step: string;
  ok: boolean;
  detail: string;
}

export interface DaemonCycleResult {
  started_at: string;
  duration_ms: number;
  steps: DaemonStepResult[];
}

async function runStep(step: string, fn: () => Promise<string>): Promise<DaemonStepResult> {
  try {
    return { step, ok: true, detail: await fn() };
  } catch (err) {
    return { step, ok: false, detail: err instanceof Error ? err.message.slice(0, 140) : String(err) };
  }
}

export async function runDaemonCycle(db: Database, opts?: { reflectionLlm?: LessonLlm }): Promise<DaemonCycleResult> {
  const t0 = Date.now();
  const steps: DaemonStepResult[] = [];

  steps.push(await runStep('index', async () => {
    await ensureIndex();
    // ensureIndex is fire-and-forget on staleness; await any refresh it kicked
    // off so the cycle summary reflects reality.
    const refresh = getRefreshPromise();
    if (refresh) {
      await refresh;
      return 'refreshed';
    }
    return 'fresh';
  }));

  steps.push(await runStep('prefetch', async () => {
    const r = await prefetchOctagonEvents(db);
    return `${r.inserted} inserted, ${r.skipped} skipped`;
  }));

  steps.push(await runStep('settlements', async () => {
    const r = await syncSettlements(db);
    return `${r.new_settlements} new of ${r.fetched} fetched${r.positions_closed ? `, ${r.positions_closed} positions closed` : ''}${r.complete ? '' : ' (INCOMPLETE — page cap)'}`;
  }));

  steps.push(await runStep('paper', async () => {
    const settled = await settlePaperPositions(db);
    return settled > 0 ? `${settled} settled` : 'nothing to settle';
  }));

  steps.push(await runStep('reflection', async () => {
    const r = await generateMissingLessons(db, { llm: opts?.reflectionLlm });
    return r.generated > 0 ? `${r.generated} lessons (${r.source})` : 'nothing pending';
  }));

  return {
    started_at: new Date(t0).toISOString(),
    duration_ms: Date.now() - t0,
    steps,
  };
}

export function formatCycleSummary(cycle: DaemonCycleResult): string {
  const parts = cycle.steps.map((s) => `${s.ok ? '✓' : '✗'} ${s.step}: ${s.detail}`);
  return `[daemon ${cycle.started_at.slice(11, 19)}] ${parts.join(' · ')} (${(cycle.duration_ms / 1000).toFixed(1)}s)`;
}
