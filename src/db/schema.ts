import type { Database } from 'bun:sqlite';

/**
 * Run all CREATE TABLE IF NOT EXISTS statements and indexes.
 * Safe to call on every startup — idempotent.
 */
export function migrate(db: Database): void {
  db.exec(`
    CREATE TABLE IF NOT EXISTS themes (
      theme_id        TEXT PRIMARY KEY,
      name            TEXT NOT NULL,
      filter_query    TEXT,
      tickers         TEXT,
      last_resolved_at INTEGER
    );

    CREATE TABLE IF NOT EXISTS events (
      ticker      TEXT PRIMARY KEY,
      category    TEXT,
      expiry      INTEGER,
      vol_24h     REAL,
      theme_id    TEXT REFERENCES themes,
      active      INTEGER DEFAULT 1,
      updated_at  INTEGER
    );

    CREATE TABLE IF NOT EXISTS edge_history (
      id                INTEGER PRIMARY KEY AUTOINCREMENT,
      ticker            TEXT NOT NULL,
      event_ticker      TEXT NOT NULL,
      timestamp         INTEGER NOT NULL,
      model_prob        REAL NOT NULL,
      market_prob       REAL NOT NULL,
      edge              REAL NOT NULL,
      octagon_report_id TEXT,
      drivers_json      TEXT,
      sources_json      TEXT,
      catalysts_json    TEXT,
      cache_hit         INTEGER,
      cache_miss        INTEGER DEFAULT 0,
      confidence        TEXT,
      UNIQUE(ticker, timestamp)
    );

    CREATE INDEX IF NOT EXISTS idx_edge_ticker_ts
      ON edge_history(ticker, timestamp DESC);

    CREATE INDEX IF NOT EXISTS idx_edge_confidence
      ON edge_history(confidence, timestamp DESC);

    CREATE TABLE IF NOT EXISTS octagon_reports (
      report_id                TEXT PRIMARY KEY,
      ticker                   TEXT NOT NULL,
      event_ticker             TEXT NOT NULL,
      model_prob               REAL NOT NULL,
      market_prob              REAL,
      mispricing_signal        TEXT,
      drivers_json             TEXT,
      catalysts_json           TEXT,
      sources_json             TEXT,
      resolution_history_json  TEXT,
      contract_snapshot_json   TEXT,
      raw_response             TEXT,
      model_accuracy           REAL,
      variant_used             TEXT,
      fetched_at               INTEGER NOT NULL,
      expires_at               INTEGER NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_octagon_ticker
      ON octagon_reports(ticker, fetched_at DESC);

    CREATE TABLE IF NOT EXISTS positions (
      position_id  TEXT PRIMARY KEY,
      ticker       TEXT NOT NULL,
      event_ticker TEXT NOT NULL,
      direction    TEXT NOT NULL,
      size         REAL NOT NULL,
      entry_price  REAL NOT NULL,
      entry_edge   REAL,
      entry_kelly  REAL,
      current_pnl  REAL DEFAULT 0,
      status       TEXT DEFAULT 'open',
      opened_at    INTEGER,
      closed_at    INTEGER
    );

    CREATE TABLE IF NOT EXISTS trades (
      trade_id     TEXT PRIMARY KEY,
      position_id  TEXT REFERENCES positions,
      order_id     TEXT,
      ticker       TEXT NOT NULL,
      action       TEXT NOT NULL,
      side         TEXT NOT NULL,
      size         REAL NOT NULL,
      price        REAL NOT NULL,
      fill_status  TEXT,
      kalshi_response TEXT,
      created_at   INTEGER
    );

    CREATE TABLE IF NOT EXISTS risk_snapshots (
      id                  INTEGER PRIMARY KEY AUTOINCREMENT,
      timestamp           INTEGER NOT NULL,
      cash_balance        REAL,
      portfolio_value     REAL,
      open_exposure       REAL,
      available_bankroll  REAL,
      daily_pnl           REAL,
      drawdown_current    REAL,
      drawdown_max        REAL,
      correlation_max     REAL,
      positions_count     INTEGER,
      circuit_breaker_on  INTEGER DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS alerts (
      alert_id    TEXT PRIMARY KEY,
      ticker      TEXT,
      alert_type  TEXT NOT NULL,
      edge        REAL,
      message     TEXT NOT NULL,
      channels    TEXT,
      status      TEXT DEFAULT 'pending',
      created_at  INTEGER
    );

    CREATE TABLE IF NOT EXISTS settlements (
      -- Realized outcomes from Kalshi /portfolio/settlements: the ground
      -- truth the learning loop (calibration, reflection) is built on.
      -- One row per (ticker, settled_time) — INSERT OR IGNORE keeps syncs
      -- idempotent.
      ticker            TEXT NOT NULL,
      event_ticker      TEXT NOT NULL,
      market_result     TEXT NOT NULL,
      yes_count_fp      REAL NOT NULL DEFAULT 0,
      no_count_fp       REAL NOT NULL DEFAULT 0,
      yes_total_cost    REAL NOT NULL DEFAULT 0,
      no_total_cost     REAL NOT NULL DEFAULT 0,
      revenue           REAL NOT NULL DEFAULT 0,
      fee_cost          REAL NOT NULL DEFAULT 0,
      realized_pnl      REAL NOT NULL DEFAULT 0,
      settled_time      TEXT NOT NULL,
      -- Model context at entry, joined from edge_history (nullable: we may
      -- not have scored the market before entering it).
      model_prob_entry  REAL,
      market_prob_entry REAL,
      edge_entry        REAL,
      raw_json          TEXT,
      synced_at         INTEGER NOT NULL,
      PRIMARY KEY (ticker, settled_time)
    );

    CREATE TABLE IF NOT EXISTS paper_positions (
      -- Forward-test ledger: same entry semantics as live /buy, no exchange
      -- order. Settled against real market results, scored with the same
      -- flat-bet definitions as the backtest so paper vs live vs backtest
      -- numbers are directly comparable.
      id            INTEGER PRIMARY KEY AUTOINCREMENT,
      ticker        TEXT NOT NULL,
      event_ticker  TEXT NOT NULL DEFAULT '',
      action        TEXT NOT NULL CHECK (action IN ('buy', 'sell')),
      side          TEXT NOT NULL CHECK (side IN ('yes', 'no')),
      count         REAL NOT NULL,
      entry_price   REAL NOT NULL,          -- cents, on the chosen side
      model_prob    REAL,                   -- model view at entry, 0-1
      market_prob   REAL,
      edge          REAL,
      status        TEXT NOT NULL DEFAULT 'open'
                    CHECK (status IN ('open', 'settled', 'closed')),
      outcome       TEXT,                   -- 'yes' | 'no' when settled
      realized_pnl  REAL,
      opened_at     INTEGER NOT NULL,
      settled_at    INTEGER
    );

    CREATE INDEX IF NOT EXISTS idx_paper_status
      ON paper_positions(status, opened_at DESC);

    CREATE TABLE IF NOT EXISTS hypotheses (
      -- Falsifiable-claim registry (Vibe-Trading pattern). Market-bound
      -- hypotheses (ticker + predicted_side) are auto-resolved by
      -- settlement sync; thematic ones resolve manually. Every executed
      -- trade auto-files one, so the book of live claims is always the
      -- book of live risk.
      id             INTEGER PRIMARY KEY AUTOINCREMENT,
      claim          TEXT NOT NULL,
      category       TEXT NOT NULL DEFAULT '',
      ticker         TEXT,
      event_ticker   TEXT,
      predicted_side TEXT CHECK (predicted_side IN ('yes', 'no') OR predicted_side IS NULL),
      status         TEXT NOT NULL DEFAULT 'open'
                     CHECK (status IN ('open', 'confirmed', 'refuted', 'expired', 'retired')),
      evidence       TEXT,
      source         TEXT NOT NULL DEFAULT 'manual',
      created_at     INTEGER NOT NULL,
      resolved_at    INTEGER
    );

    CREATE INDEX IF NOT EXISTS idx_hypotheses_status
      ON hypotheses(status, created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_hypotheses_ticker
      ON hypotheses(ticker) WHERE ticker IS NOT NULL;

    CREATE TABLE IF NOT EXISTS lessons (
      -- Reflection loop: one terse lesson per settled position with a model
      -- view, generated from the gap between what the model believed at
      -- entry and what actually happened. Injected into /analyze context
      -- for markets in the same category.
      id            INTEGER PRIMARY KEY AUTOINCREMENT,
      ticker        TEXT NOT NULL,
      event_ticker  TEXT NOT NULL,
      category      TEXT NOT NULL,
      settled_time  TEXT NOT NULL,
      model_prob    REAL NOT NULL,
      market_prob   REAL,
      outcome       INTEGER NOT NULL,
      realized_pnl  REAL NOT NULL,
      lesson        TEXT NOT NULL,
      source        TEXT NOT NULL DEFAULT 'llm',
      created_at    INTEGER NOT NULL,
      UNIQUE (ticker, settled_time)
    );

    CREATE INDEX IF NOT EXISTS idx_lessons_category
      ON lessons(category, created_at DESC);

    CREATE TABLE IF NOT EXISTS brier_scores (
      id              INTEGER PRIMARY KEY AUTOINCREMENT,
      ticker          TEXT NOT NULL,
      event_ticker    TEXT NOT NULL,
      category        TEXT NOT NULL,
      model_prob      REAL NOT NULL,
      actual_outcome  INTEGER NOT NULL,
      brier_score     REAL NOT NULL,
      settled_at      INTEGER NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_brier_category
      ON brier_scores(category, settled_at DESC);

    CREATE TABLE IF NOT EXISTS event_index (
      event_ticker   TEXT PRIMARY KEY,
      series_ticker  TEXT,
      title          TEXT NOT NULL,
      category       TEXT,
      strike_date    TEXT,
      sub_title      TEXT,
      tags           TEXT,
      markets_json   TEXT,
      indexed_at     INTEGER NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_event_index_title
      ON event_index(title);
    CREATE INDEX IF NOT EXISTS idx_event_index_series
      ON event_index(series_ticker);
    CREATE INDEX IF NOT EXISTS idx_event_index_category
      ON event_index(category);

    CREATE TABLE IF NOT EXISTS event_index_meta (
      key   TEXT PRIMARY KEY,
      value TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS octagon_history (
      id                  INTEGER PRIMARY KEY AUTOINCREMENT,
      history_id          INTEGER NOT NULL,
      event_ticker        TEXT NOT NULL,
      captured_at         TEXT NOT NULL,
      model_probability   REAL NOT NULL,
      market_probability  REAL NOT NULL,
      edge_pp             REAL,
      confidence_score    REAL,
      series_category     TEXT,
      close_time          TEXT,
      name                TEXT,
      outcome_probabilities_json TEXT,
      UNIQUE(event_ticker, history_id)
    );

    CREATE INDEX IF NOT EXISTS idx_history_event
      ON octagon_history(event_ticker, captured_at);

    CREATE TABLE IF NOT EXISTS editorial_themes (
      name           TEXT PRIMARY KEY,
      description    TEXT,
      search_volume  INTEGER,
      created_at     INTEGER NOT NULL,
      updated_at     INTEGER NOT NULL
    );

    CREATE TABLE IF NOT EXISTS editorial_theme_series (
      theme_name     TEXT NOT NULL,
      series_ticker  TEXT NOT NULL,
      PRIMARY KEY (theme_name, series_ticker),
      FOREIGN KEY (theme_name) REFERENCES editorial_themes(name) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_ets_series
      ON editorial_theme_series(series_ticker);
  `);

  // Schema migrations for columns added after initial release
  const edgeCols = db.query(`PRAGMA table_info(edge_history)`).all() as Array<{ name: string }>;
  if (!edgeCols.some((c) => c.name === 'cache_miss')) {
    db.exec(`ALTER TABLE edge_history ADD COLUMN cache_miss INTEGER DEFAULT 0`);
  }

  const reportCols = db.query(`PRAGMA table_info(octagon_reports)`).all() as Array<{ name: string }>;
  if (!reportCols.some((c) => c.name === 'raw_response')) {
    db.exec(`ALTER TABLE octagon_reports ADD COLUMN raw_response TEXT`);
  }

  const eventIndexCols = db.query(`PRAGMA table_info(event_index)`).all() as Array<{ name: string }>;
  if (!eventIndexCols.some((c) => c.name === 'tags')) {
    db.exec(`ALTER TABLE event_index ADD COLUMN tags TEXT`);
    // Force re-index so tags get populated on next ensureIndex() call
    db.exec(`DELETE FROM event_index_meta WHERE key = 'last_refresh'`);
  }

  if (!reportCols.some((c) => c.name === 'has_history')) {
    db.exec(`ALTER TABLE octagon_reports ADD COLUMN has_history INTEGER DEFAULT 0`);
  }
  if (!reportCols.some((c) => c.name === 'mutually_exclusive')) {
    db.exec(`ALTER TABLE octagon_reports ADD COLUMN mutually_exclusive INTEGER DEFAULT 0`);
  }
  if (!reportCols.some((c) => c.name === 'series_category')) {
    db.exec(`ALTER TABLE octagon_reports ADD COLUMN series_category TEXT`);
  }
  if (!reportCols.some((c) => c.name === 'confidence_score')) {
    db.exec(`ALTER TABLE octagon_reports ADD COLUMN confidence_score REAL`);
  }
  if (!reportCols.some((c) => c.name === 'outcome_probabilities_json')) {
    db.exec(`ALTER TABLE octagon_reports ADD COLUMN outcome_probabilities_json TEXT`);
  }
  if (!reportCols.some((c) => c.name === 'close_time')) {
    db.exec(`ALTER TABLE octagon_reports ADD COLUMN close_time TEXT`);
  }
  // Upstream model-run timestamp (Octagon `analysis_last_updated`). Distinct
  // from `fetched_at` (when we pulled the report into our local cache).
  if (!reportCols.some((c) => c.name === 'analysis_last_updated')) {
    db.exec(`ALTER TABLE octagon_reports ADD COLUMN analysis_last_updated TEXT`);
  }

  const historyCols = db.query(`PRAGMA table_info(octagon_history)`).all() as Array<{ name: string }>;
  if (!historyCols.some((c) => c.name === 'outcome_probabilities_json')) {
    db.exec(`ALTER TABLE octagon_history ADD COLUMN outcome_probabilities_json TEXT`);
  }
}
