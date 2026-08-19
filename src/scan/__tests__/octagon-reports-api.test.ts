import { afterEach, beforeEach, describe, expect, test } from 'bun:test';
import {
  fetchLatestReportMarkdown,
  fetchReportRunStatus,
  fetchReportVersions,
  generateReportAndWait,
  OctagonReportsApiError,
  triggerReportGeneration,
} from '../octagon-reports-api';

const realFetch = globalThis.fetch;
let calls: Array<{ url: string; method: string }> = [];
let responder: (url: string, init?: RequestInit) => Response | Promise<Response>;

function json(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
}

beforeEach(() => {
  calls = [];
  process.env.OCTAGON_API_KEY = 'test-key';
  globalThis.fetch = (async (input: any, init?: RequestInit) => {
    const url = String(input);
    calls.push({ url, method: init?.method ?? 'GET' });
    return responder(url, init);
  }) as typeof fetch;
});

afterEach(() => {
  globalThis.fetch = realFetch;
});

describe('fetchReportVersions', () => {
  test('lists versions without a body by default', async () => {
    responder = (url) => {
      expect(url).toContain('/predictions/reports/kalshi/KXTEST-26');
      expect(url).not.toContain('version=');
      return json(200, { event_ticker: 'KXTEST-26', venue: 'kalshi', requested_url: null, versions: [{ run_id: 'r1' }], markdown_report: null, run_id: null });
    };
    const res = await fetchReportVersions('KXTEST-26');
    expect(res.versions).toHaveLength(1);
    expect(res.markdown_report).toBeNull();
  });

  test('version=latest requests the markdown body', async () => {
    responder = (url) => {
      expect(url).toContain('?version=latest');
      return json(200, { event_ticker: 'KXTEST-26', venue: 'kalshi', requested_url: null, versions: [{ run_id: 'r1' }], markdown_report: '# Report', run_id: 'r1' });
    };
    const res = await fetchLatestReportMarkdown('KXTEST-26');
    expect(res.markdown).toBe('# Report');
    expect(res.runId).toBe('r1');
  });

  test('404 surfaces the error envelope code', async () => {
    responder = () => json(404, { error: { code: 'not_found', message: 'no such event' } });
    await expect(fetchReportVersions('KXNOPE')).rejects.toThrow(/404 \(not_found\): no such event/);
  });

  test('retries 503 then succeeds', async () => {
    let n = 0;
    responder = () => (++n === 1 ? json(503, { error: { code: 'service_unavailable', message: 'down' } })
      : json(200, { event_ticker: 'KXTEST-26', venue: 'kalshi', requested_url: null, versions: [], markdown_report: null, run_id: null }));
    const res = await fetchReportVersions('KXTEST-26');
    expect(res.versions).toEqual([]);
    expect(n).toBe(2);
  });
});

describe('triggerReportGeneration', () => {
  test('POSTs and returns the accepted run', async () => {
    responder = (url, init) => {
      expect(init?.method).toBe('POST');
      return json(202, { run_id: 'run-1', status: 'processing', event_ticker: 'KXTEST-26', venue: 'kalshi' });
    };
    const res = await triggerReportGeneration('KXTEST-26');
    expect(res.run_id).toBe('run-1');
  });

  test('does not retry 503 (credit safety)', async () => {
    let n = 0;
    responder = () => { n++; return json(503, { error: { code: 'service_unavailable', message: 'down' } }); };
    await expect(triggerReportGeneration('KXTEST-26')).rejects.toThrow(OctagonReportsApiError);
    expect(n).toBe(1);
  });

  test('409 not-open surfaces cleanly', async () => {
    responder = () => json(409, { error: { code: 'kalshi_market_not_open', message: 'market expired' } });
    await expect(triggerReportGeneration('KXOLD')).rejects.toThrow(/kalshi_market_not_open/);
  });
});

describe('generateReportAndWait', () => {
  test('polls status then fetches the pinned version', async () => {
    let statusCalls = 0;
    responder = (url, init) => {
      if (init?.method === 'POST') return json(202, { run_id: 'run-9', status: 'processing', event_ticker: 'KXTEST-26', venue: 'kalshi' });
      if (url.includes('/status/run-9')) {
        statusCalls++;
        return json(200, { run_id: 'run-9', status: statusCalls < 2 ? 'processing' : 'completed', venue: 'kalshi', event_ticker: 'KXTEST-26', requested_url: null });
      }
      expect(url).toContain('?version=run-9');
      return json(200, { event_ticker: 'KXTEST-26', venue: 'kalshi', requested_url: null, versions: [{ run_id: 'run-9' }], markdown_report: '# Fresh', run_id: 'run-9' });
    };
    const res = await generateReportAndWait('KXTEST-26', { pollIntervalMs: 5 });
    expect(res.markdown).toBe('# Fresh');
    expect(res.runId).toBe('run-9');
    expect(statusCalls).toBe(2);
  });

  test('failed run throws with refund note', async () => {
    responder = (url, init) => {
      if (init?.method === 'POST') return json(202, { run_id: 'run-x', status: 'processing', event_ticker: 'KXTEST-26', venue: 'kalshi' });
      return json(200, { run_id: 'run-x', status: 'failed', venue: 'kalshi', event_ticker: 'KXTEST-26', requested_url: null });
    };
    await expect(generateReportAndWait('KXTEST-26', { pollIntervalMs: 5 })).rejects.toThrow(/failed .*refunded/i);
  });
});

describe('fetchReportRunStatus', () => {
  test('returns run status', async () => {
    responder = () => json(200, { run_id: 'r', status: 'processing', venue: 'kalshi', event_ticker: null, requested_url: null });
    const res = await fetchReportRunStatus('r');
    expect(res.status).toBe('processing');
  });
});
