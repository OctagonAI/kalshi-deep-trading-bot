import { describe, expect, test } from 'bun:test';
import { tokenizeCommand } from '../index';
import { formatPeersHuman } from '../peers';
import type { PeersResult } from '../peers';

describe('tokenizeCommand', () => {
  test('plain whitespace split unchanged', () => {
    expect(tokenizeCommand('basket build --size 5')).toEqual(['basket', 'build', '--size', '5']);
  });

  test('double-quoted value stays one token', () => {
    expect(tokenizeCommand('basket build --theme "Bitcoin Breakout" --size 5')).toEqual([
      'basket', 'build', '--theme', 'Bitcoin Breakout', '--size', '5',
    ]);
  });

  test('single-quoted value stays one token', () => {
    expect(tokenizeCommand("themes show 'Elon Musk / Tesla / SpaceX'")).toEqual([
      'themes', 'show', 'Elon Musk / Tesla / SpaceX',
    ]);
  });

  test('empty quoted string yields empty token', () => {
    expect(tokenizeCommand('themes show ""')).toEqual(['themes', 'show', '']);
  });

  test('unterminated quote degrades without losing text', () => {
    expect(tokenizeCommand('themes show "Bitcoin')).toEqual(['themes', 'show', 'Bitcoin']);
  });
});

describe('formatPeersHuman with null cluster', () => {
  test('unassigned market gets a friendly message instead of a crash', () => {
    const result: PeersResult = {
      kind: 'peers',
      data: { market_ticker: 'KX-UNASSIGNED', kind: 'thematic', cluster: null, data: [] },
    };
    const text = formatPeersHuman(result);
    expect(text).toContain('KX-UNASSIGNED');
    expect(text).toContain('not assigned');
  });
});

// ─── Attached quoted flag values (review round) ─────────────────────────────
import { describe as dtk, expect as etk, test as ttk } from 'bun:test';
import { tokenizeCommand as tok } from '../index';

dtk('tokenizeCommand attached quotes', () => {
  ttk('--flag="multi word" stays one token with the full value', () => {
    etk(tok('basket build --theme="Bitcoin Breakout" --size 5')).toEqual([
      'basket', 'build', '--theme=Bitcoin Breakout', '--size', '5',
    ]);
  });

  ttk("single-quoted attached value also stays whole", () => {
    etk(tok("show --name='Elon Musk / Tesla'")).toEqual(['show', '--name=Elon Musk / Tesla']);
  });

  ttk('mixed bare and quoted segments concatenate', () => {
    etk(tok('pre"mid dle"post')).toEqual(['premid dlepost']);
  });
});
