import { afterEach, beforeEach, describe, expect, test } from 'bun:test';
import { askOctagonAgent, handleOctagonChat, octagonConversationLength, resetOctagonConversation } from '../octagon-chat';

const realFetch = globalThis.fetch;
let requests: Array<Record<string, unknown>> = [];
let responder: (body: Record<string, unknown>) => Response;

function json(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status });
}

beforeEach(() => {
  requests = [];
  process.env.OCTAGON_API_KEY = 'test-key';
  resetOctagonConversation();
  globalThis.fetch = (async (_url: any, init?: RequestInit) => {
    const body = JSON.parse(String(init?.body ?? '{}')) as Record<string, unknown>;
    requests.push(body);
    return responder(body);
  }) as typeof fetch;
});

afterEach(() => {
  globalThis.fetch = realFetch;
  resetOctagonConversation();
});

describe('handleOctagonChat', () => {
  test('no args prints usage', () => {
    const res = handleOctagonChat([]);
    expect('followUp' in res).toBe(false);
    expect(res.output).toContain('Usage: /octagon');
  });

  test('reset clears the conversation', async () => {
    responder = () => json(200, { id: 'resp-1', output_text: 'answer' });
    await askOctagonAgent('first question');
    expect(octagonConversationLength()).toBe(2);
    const res = handleOctagonChat(['reset']);
    expect(res.output).toContain('reset');
    expect(octagonConversationLength()).toBe(0);
  });

  test('question returns a followUp', () => {
    const res = handleOctagonChat(['find', 'fed', 'markets']);
    expect('followUp' in res).toBe(true);
  });
});

describe('askOctagonAgent multi-turn', () => {
  test('first turn sends the bare question', async () => {
    responder = () => json(200, { id: 'resp-1', output_text: 'first answer' });
    await askOctagonAgent('q1');
    expect(requests[0].input).toBe('q1');
    expect(requests[0].model).toBe('octagon-prediction-markets-agent');
  });

  test('second turn replays the transcript', async () => {
    responder = () => json(200, { id: 'resp-1', output_text: 'first answer' });
    await askOctagonAgent('q1');
    responder = () => json(200, { id: 'resp-2', output_text: 'second answer' });
    await askOctagonAgent('q2');
    const replay = String(requests[1].input);
    expect(replay).toContain('Conversation so far');
    expect(replay).toContain('User: q1');
    expect(replay).toContain('Agent: first answer');
    expect(replay).toContain('User: q2');
  });

  test('transcript is bounded', async () => {
    responder = () => json(200, { output_text: 'a'.repeat(10) });
    for (let i = 0; i < 30; i++) await askOctagonAgent(`q${i}`);
    expect(octagonConversationLength()).toBeLessThanOrEqual(24);
  });

  test('API error surfaces the envelope message', async () => {
    responder = () => json(429, { error: { message: 'Not enough credits', code: 'insufficient_credits' } });
    await expect(askOctagonAgent('q')).rejects.toThrow(/Not enough credits \(insufficient_credits\)/);
  });
});
