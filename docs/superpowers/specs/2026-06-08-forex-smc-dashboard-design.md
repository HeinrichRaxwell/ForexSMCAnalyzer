# Forex SMC Dashboard Design

## Goal

Build a local read-only operational dashboard for the Forex SMC Analyzer so the owner can monitor live scanner behavior, all accepted and shadow signals, trade management actions, model learning, calibration, and forward-test readiness from one place.

This dashboard must not promise profit. It must expose evidence: what the bot saw, why a signal passed or failed, how confidence buckets perform, and whether learning/retraining is healthy.

## Scope

Phase 1 is a local dashboard that reads existing project files and refreshes periodically. It does not place trades, change strategy settings, edit `.env`, send Telegram messages, or connect to MT5 for live execution. It is an observation layer only.

Primary data sources:

- `data/sent_signals.json`
- `data/shadow_signals.json`
- `data/labeled_setups.csv`
- `data/shadow_labeled_setups.csv`
- `data/calibration_report.json`
- `data/learning_status.json`
- `data/backtest_simulation_results.csv`
- `data/real_tick_backtest_*.csv`
- `.env`
- `models/smc_xgb_classifier.joblib`
- `models/smc_lgb_classifier.joblib`

## Recommended Stack

Use Streamlit for Phase 1 because this project is Python-first and the dashboard mostly reads CSV/JSON artifacts. Streamlit gives a usable dashboard quickly without a separate backend/frontend split.

Required additions:

- `streamlit>=1.35.0` in `requirements.txt`
- `src/dashboard_data.py` for data loading, flattening, metrics, and safety handling
- `src/dashboard.py` for the UI
- `tests/test_dashboard_data.py` for behavior tests

## Information Architecture

### 1. Command Center

Purpose: show whether the bot is configured and learning correctly.

Cards:

- Symbol and broker suffix inferred from config/data
- `ML_ACCEPT_THRESHOLD`
- `ML_TRAINING_MAX_SETUPS`
- `ML_RETRAIN_THRESHOLD`
- `MT5_EXECUTE_TRADES`
- Real labeled row count
- Shadow labeled row count
- Last retrain time from `learning_status.json`
- XGB/LGB model timestamps
- Calibration sample count and winrate

Warnings:

- Model timestamp older than newest labeled/shadow data
- Missing Telegram config
- Missing calibration report
- Empty shadow data
- `MT5_EXECUTE_TRADES=True` visible as live execution risk

### 2. Live Signal Monitor

Purpose: inspect every signal the bot has seen, not only signals above threshold.

Tables:

- Accepted/live signals from `sent_signals.json`
- Shadow signals from `shadow_signals.json`
- Combined signal table with source, strategy, timeframe, direction, confidence, leg, entry, SL, TP, status, result, created/latest/resolved timestamps

Filters:

- Source: accepted, shadow, all
- Strategy: FVG, OB, BPR, IC, SND, Pivot, Swapzone, Breaker, BPR
- Timeframe: M15, M30, H1, H4, D1
- Direction: BUY/BULL, SELL/BEAR
- Confidence bucket
- Status: open, resolved, filtered, executed
- Result: TP, SL, expired, unknown

Charts:

- Signal count by strategy/timeframe
- Confidence distribution
- Shadow signals below threshold that later hit TP/SL

### 3. Trade Manager

Purpose: explain what the active trade manager will do or already did.

Content:

- Ticket IDs from `sent_signals.json`
- Option A/B lot mapping
- Entry, SL, TP1, TP2, TP3
- Outcome recorded flags
- CHoCH emergency status rules
- BEP and trailing rules summary

Special CHoCH note:

- M15/M30 emergency close requires two closed opposite candles or HTF confirmation
- H1/H4/D1 emergency close can use one closed opposite candle
- Candle in progress is ignored

### 4. AI Learning

Purpose: show whether the bot is actually learning from wins/losses.

Metrics:

- `labeled_setups.csv` real samples
- `shadow_labeled_setups.csv` resolved shadow samples
- Training cap from `ML_TRAINING_MAX_SETUPS`
- New trades since last retrain
- Last retrain time
- Sample source mix: real vs shadow
- Shadow sample weight if present
- Data freshness: latest labeled time vs model file timestamp

Charts:

- Outcomes over time
- Real vs shadow contribution
- Win/loss counts by strategy/timeframe

### 5. Confidence Calibration

Purpose: verify whether model confidence is meaningful.

Use `data/calibration_report.json` when present.

Views:

- Overall sample count, winrate, expectancy, profit factor, drawdown, loss streak
- Threshold table for 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, etc.
- Bucket table by confidence range
- Recommended threshold notes based on calibration, not hype

### 6. Strategy Performance

Purpose: compare which setup types and timeframes are helping or hurting.

Views:

- Strategy winrate table
- Timeframe winrate table
- Strategy x timeframe matrix
- Average R, profit factor, max consecutive losses
- Best and worst conditions

Known strategy labels:

- FVG
- OB
- BPR
- IC
- SND
- Pivot
- Swapzone
- Breaker

### 7. Backtest vs Forward Test

Purpose: avoid being fooled by backtest-only performance.

Views:

- Backtest simulation summary from `data/backtest_simulation_results.csv`
- Real-tick backtest summaries from `data/real_tick_backtest_*.csv`
- Forward-test labeled outcomes from `labeled_setups.csv` and shadow outcomes
- Gap analysis: if forward-test winrate or drawdown diverges from backtest

## UX Principles

- Quiet operational layout, not a marketing page.
- Dense but readable tables.
- Essential values visible without hover.
- Mobile should be usable, but desktop is primary for monitoring.
- Use restrained color roles: green for profit/healthy, red for loss/risk, amber for warning, neutral for inactive/missing.
- Every warning should include the file or config behind it.

## Safety

Phase 1 dashboard is read-only.

It must not:

- call `order_send`
- modify `.env`
- retrain models
- delete data
- write to MT5
- send Telegram alerts

It may read local files and display stale/missing data warnings.

## Acceptance Criteria

- Dashboard starts with `streamlit run src/dashboard.py`.
- It loads with missing optional files without crashing.
- It shows command-center cards for config, data, models, and learning status.
- It shows accepted signals and shadow signals in filterable tables.
- It shows confidence calibration when `calibration_report.json` exists.
- It shows model freshness warnings when data files are newer than model files.
- It includes CHoCH cutloss rules visibly in Trade Manager.
- It has tests for data loading and flattening behavior.

## Open Decisions

- Phase 1 will not connect directly to MT5. Live positions can be added later if needed.
- Phase 1 will use Streamlit. A FastAPI/React dashboard can be Phase 2 after the useful metrics are proven.
