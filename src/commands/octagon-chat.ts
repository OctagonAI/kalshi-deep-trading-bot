/**
 * /octagon — conversational access to the Octagon Prediction Markets Agent.
 *
 * The base `octagon-prediction-markets-agent` model is a multi-turn
 * conversational agent: discovery, edge screening, similar markets, model
 * history, and full reports, all in natural language.
 *
 * Multi-turn context is carried by replaying the local transcript into each
 * request. The server's /responses shape includes `previous_response_id`, but
 * live testing (Aug 2026) showed chained requests answering from unrelated
 * context — replay is the dependable route until server-side chaining lands.
 *
 * The transcript is persisted to ~/.kalshi-bot/octagon-conversation.json so
 * one-shot CLI invocations (`kalshi octagon ...`) keep context across
 * processes, not just within a TUI session. Requests are serialized per
 * process, and `/octagon reset` bumps a generation counter so an in-flight
 * request from before the reset cannot write stale turns back.
 *
 * Billing (per Octagon docs): discovery queries 1 credit, fresh reports 3,
 * cached reports and conversational follow-ups free.
 */
import { existsSync, readFileSync, writeFileSync, mkdirSync } from 'fs';
import { dirname } from 'path';
import { extractTextFromResponse } from '../scan/invoker.js';
import { appPath } from '../utils/paths.js';

const TIMEOUT_MS = 600_000;
/** Number of user↔agent exchanges kept and replayed. */
const MAX_TRANSCRIPT_TURNS = 12;
/** Keep replayed assistant turns bounded so requests stay small. */
const MAX_TURN_CHARS = 4_000;

function transcriptPath(): string {
  // Env override keeps tests and parallel setups isolated from the real file.
  return process.env.OCTAGON_TRANSCRIPT_PATH ?? appPath('octagon-conversation.json');
}

interface ConversationTurn {
  role: 'user' | 'assistant';
  text: string;
}

let transcript: ConversationTurn[] = [];
let loaded = false;
/** Bumped on reset; in-flight requests from an older generation discard their writes. */
let generation = 0;
/** Serializes agent requests so replay input and transcript writes stay ordered. */
let queue: Promise<unknown> = Promise.resolve();

function loadTranscript(): void {
  if (loaded) return;
  loaded = true;
  try {
    if (existsSync(transcriptPath())) {
      const parsed = JSON.parse(readFileSync(transcriptPath(), 'utf-8')) as { turns?: ConversationTurn[] };
      if (Array.isArray(parsed.turns)) {
        transcript = parsed.turns.filter(
          (t) => (t.role === 'user' || t.role === 'assistant') && typeof t.text === 'string',
        );
      }
    }
  } catch {
    transcript = [];
  }
}

function saveTranscript(): void {
  try {
    mkdirSync(dirname(transcriptPath()), { recursive: true });
    writeFileSync(transcriptPath(), JSON.stringify({ turns: transcript }));
  } catch {
    // persistence is best-effort; in-memory conversation still works
  }
}

export function resetOctagonConversation(): void {
  loaded = true;
  generation++;
  transcript = [];
  saveTranscript();
}

export function octagonConversationLength(): number {
  loadTranscript();
  return transcript.length;
}

function buildReplayInput(question: string): string {
  if (transcript.length === 0) return question;
  // Each exchange is two entries (user + agent) — slice entries accordingly.
  const history = transcript
    .slice(-MAX_TRANSCRIPT_TURNS * 2)
    .map((t) => `${t.role === 'user' ? 'User' : 'Agent'}: ${t.text.slice(0, MAX_TURN_CHARS)}`)
    .join('\n\n');
  return `Conversation so far:\n\n${history}\n\nUser: ${question}`;
}

async function sendToAgent(question: string, startGeneration: number): Promise<string> {
  const apiKey = process.env.OCTAGON_API_KEY;
  if (!apiKey) throw new Error('OCTAGON_API_KEY not set. Get one at https://app.octagonai.co');
  const baseUrl = process.env.OCTAGON_BASE_URL ?? 'https://api.octagonai.co/v1';

  loadTranscript();

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
  let resp: Response;
  try {
    resp = await fetch(`${baseUrl}/responses`, {
      method: 'POST',
      headers: { Authorization: `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: 'octagon-prediction-markets-agent', input: buildReplayInput(question) }),
      signal: controller.signal,
    });
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError') {
      throw new Error(`Octagon agent timed out after ${Math.round(TIMEOUT_MS / 1000)}s.`);
    }
    throw err;
  } finally {
    clearTimeout(timer);
  }

  if (!resp.ok) {
    const body = await resp.text().catch(() => '');
    let message = body.slice(0, 300);
    try {
      const parsed = JSON.parse(body) as { error?: { message?: string; code?: string } };
      if (parsed.error?.message) message = `${parsed.error.message}${parsed.error.code ? ` (${parsed.error.code})` : ''}`;
    } catch { /* keep raw slice */ }
    throw new Error(`Octagon agent error: ${resp.status} ${resp.statusText} — ${message}`);
  }

  const data = (await resp.json()) as Record<string, unknown>;
  const text = extractTextFromResponse(data);

  // A reset that happened while this request was in flight wins: the answer
  // is still returned, but it must not repopulate the cleared conversation.
  if (generation === startGeneration) {
    transcript.push({ role: 'user', text: question });
    transcript.push({ role: 'assistant', text });
    if (transcript.length > MAX_TRANSCRIPT_TURNS * 2) {
      transcript = transcript.slice(-MAX_TRANSCRIPT_TURNS * 2);
    }
    saveTranscript();
  }
  return text;
}

/**
 * Send one question to the conversational agent, keeping multi-turn context.
 * Requests are serialized so concurrent questions replay in a stable order.
 */
export function askOctagonAgent(question: string): Promise<string> {
  // Generation is captured when the question is asked: a reset issued after
  // the ask but before the serialized request runs still wins.
  const startGeneration = generation;
  const run = queue.then(() => sendToAgent(question, startGeneration));
  // Keep the chain alive even when a request fails.
  queue = run.catch(() => undefined);
  return run;
}

export interface OctagonChatResult {
  /** Immediate line to show; the answer arrives via followUp(). */
  output: string;
  followUp: () => Promise<string>;
}

/** Entry point for the /octagon slash command and CLI subcommand. */
export function handleOctagonChat(args: string[]): OctagonChatResult | { output: string } {
  const sub = args[0]?.toLowerCase();
  if (sub === 'reset' || sub === 'new') {
    const turns = octagonConversationLength() / 2;
    resetOctagonConversation();
    return { output: turns > 0 ? `Conversation reset (${turns} turns discarded).` : 'Conversation reset.' };
  }
  const question = args.join(' ').trim();
  if (!question) {
    return {
      output: [
        'Usage: /octagon <question>   (multi-turn — context persists across turns and CLI invocations)',
        '       /octagon reset        start a new conversation',
        '',
        'Examples:',
        '  /octagon Where does the model disagree most with market prices in Politics?',
        '  /octagon Which of those has the best expected return?',
        '  /octagon Find markets like "Fed cuts rates twice this year"',
      ].join('\n'),
    };
  }
  return {
    output: 'Asking the Octagon agent...',
    followUp: () => askOctagonAgent(question),
  };
}
