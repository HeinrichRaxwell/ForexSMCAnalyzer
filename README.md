# Forex SMC Analyzer

Forex SMC Analyzer is a Windows-focused XAUUSD analysis and MetaTrader 5
automation project. It combines Smart Money Concepts (SMC) structure detection,
multi-timeframe context, model scoring, and MT5 order management.

The repository is designed to start in monitoring mode. `MT5_EXECUTE_TRADES`
defaults to `False`, so scanning, Telegram alerts, WatchZone monitoring, and
shadow tracking can run without submitting an order.

> Historical backtests and forward tests are evidence, not a profit guarantee
> or trading recommendation. Automated trading can lose all deposited capital.

## Contents

- [What the system does](#what-the-system-does)
- [Entry paths and policy](#entry-paths-and-policy)
- [Requirements and installation](#requirements-and-installation)
- [Configuration](#configuration)
- [Run modes](#run-modes)
- [Risk controls](#risk-controls)
- [Performance evidence](#performance-evidence)
- [Reports and daily publication](#reports-and-daily-publication)
- [Project layout](#project-layout)
- [Troubleshooting](#troubleshooting)

## What the system does

The scanner evaluates completed candles on the enabled analysis timeframes,
detects SMC structures, scores eligible setups, and records both executed and
shadow outcomes. The primary components are:

- SMC detectors for FVG, order blocks, BPR, imbalance, swing structure,
  BOS/CHoCH, pivots, and related context.
- Multi-timeframe indicators including FLoOP, KNN directional context, volume
  profile, RSI(8), and Stochastic.
- Model scoring and probability calibration. The confidence threshold selects
  candidates; the legacy entry-quality gate is telemetry-only by default and
  does not restore a hidden execution threshold.
- Standard Limit pending orders and WatchZone market entries, each with its own
  policy and execution checks.
- MT5 comments, Telegram delivery journaling, shadow tracking, closed-trade
  labeling, and model retraining inputs.

```text
Completed MT5 candles
        |
        v
SMC detection + multi-timeframe context
        |
        v
Model score + configured strategy policy
        |
        +--> Shadow/monitor record when not eligible or execution is disabled
        |
        +--> Standard Limit pending order
        |
        +--> WatchZone hit -> rejection/range/market-safety checks -> market order
        |
        v
MT5 order result, trade management, closed-trade feedback, public reports
```

## Entry Paths and Policy

### Standard Limit

Standard Limit places a pending buy or sell limit at the selected structure's
entry level. It is the passive-entry path and uses the planned entry, SL, TP,
spread adjustment, exposure checks, and strategy policy before a request is
sent to MT5.

The current policy blocks:

| Scope | Rule | Reason |
| --- | --- | --- |
| All timeframes | `BB` | Weak real-tick replay win rate on M15 and M30. |
| H4 | `OB` | Small, negative forward sample; requires replay before reconsideration. |

Other Standard Limit strategies remain monitored rather than automatically
promoted. A policy block for WatchZone does not block the same strategy on the
Standard Limit path.

### WatchZone

A WatchZone is registered after a scan and checked against live price between
full scans. A hit is not itself an order: the selected A/B leg must meet the
configured confidence threshold, rejection confirmation, valid market-entry
range, strategy policy, and market safety checks.

Instant WatchZone execution is currently limited to these restricted
candidates:

| Timeframe | Strategy | Status |
| --- | --- | --- |
| M30 | OB | Candidate |
| H1 | OB | Candidate |

All other WatchZone strategies/timeframes are blocked from instant entry by the
current configuration. A candidate needs at least 30 resolved forward trades
and a profit factor of at least 1.30 before it can be promoted. High win rate
alone is not sufficient.

The policy is configured with timeframe-and-strategy rules:

```env
MT5_WATCH_ZONE_STRATEGY_ALLOWLIST=M30:OB,H1:OB
MT5_STANDARD_LIMIT_STRATEGY_BLOCKLIST=*:BB,H4:OB
```

`*` matches every timeframe. The executor enforces these rules again as a
defense-in-depth check, including account-sync and recovery paths.

## Requirements and Installation

### Requirements

- Windows 10 or 11
- Python 3.10 or later
- MetaTrader 5 installed and logged in to an account that exposes XAUUSD
- MT5 terminal kept running in the same Windows user session as Python
- Optional Telegram bot token and chat ID for notifications

### Installation

```powershell
git clone https://github.com/HeinrichRaxwell/ForexSMCAnalyzer.git
cd ForexSMCAnalyzer
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set the MT5 terminal's automated-trading permission before enabling execution.
Keep `.env` private: it can contain the MT5 password and Telegram bot token and
is intentionally excluded from Git.

## Configuration

Use `.env.example` as the complete, versioned configuration reference. Start
with the following minimum values in `.env`:

```env
MT5_LOGIN=YOUR_MT5_LOGIN
MT5_PASSWORD=YOUR_MT5_PASSWORD
MT5_SERVER=YOUR_MT5_SERVER
TELEGRAM_BOT_TOKEN=YOUR_TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID=YOUR_TELEGRAM_CHAT_ID

# Safe default: the scanner monitors but cannot send MT5 orders.
MT5_EXECUTE_TRADES=False
```

Important controls:

| Setting | Current template value | Effect |
| --- | --- | --- |
| `MT5_EXECUTE_TRADES` | `False` | Enables or disables MT5 order submission. |
| `MT5_REQUIRE_ROLLOUT_READY` | `False` | Requires a passing rollout gate when enabled. |
| `MT5_ALLOWED_TIMEFRAMES` | `M30,H1,H4` | Limits order execution to the listed timeframes. |
| `MT5_MAX_CONCURRENT_TRADES` | `12` | Caps positions plus pending orders for this magic number. |
| `MT5_MAX_SAME_DIRECTION_TRADES` | `6` | Caps combined same-direction exposure, allowing at most six buy and six sell orders. |
| `MT5_DAILY_GOVERNOR_ENABLED` | `False` | Enables daily target/loss and consecutive-loss controls. |
| `MT5_ENFORCE_ENTRY_GATE` | `False` | Enables the legacy quality gate; leave off to keep it observer-only. |
| `MT5_ENFORCE_SPREAD_FILTER` | `True` | Rejects entries when live spread is too large relative to risk. |
| `ML_ACCEPT_THRESHOLD` | `0.50` | Candidate confidence threshold for scan alerts/orders. |

Lot sizing and Daily Governor values are intentionally not changed by the
strategy policy. Review them independently for the account and broker.

## Run Modes

### Validate the checkout

```powershell
python -m pytest -q
python -m src.scanner_worker --help
python -m src.rollout_status --profile paper
```

`rollout_status` is an offline preflight command. It reports whether the
configured rollout criteria pass; it does not enable trading.

### Monitor without MT5 execution

Keep `MT5_EXECUTE_TRADES=False`, then run:

```powershell
python -m src.scanner_worker --symbol XAUUSD --loop --interval 5 --threshold 0.50
```

For fast monitoring of previously detected setups between full scans:

```powershell
python -m src.scanner_worker --symbol XAUUSD --loop --interval 5 --threshold 0.50 --realtime-reaction --tick-interval 1.0 --min-reaction-move 0.10
```

`--symbol` accepts a broker symbol, comma-separated symbols, `all`, or
`marketwatch`. Use the exact symbol visible in MT5 Market Watch when the broker
uses a suffix such as `XAUUSDm`.

### Enable execution only after preflight

Before changing `MT5_EXECUTE_TRADES=True`, confirm MT5 connectivity, broker
symbol availability, spread behavior, policy settings, lot sizing, and rollout
status. Read every console skip reason during monitoring; an order is only
considered submitted after MT5 returns a successful execution result.

### Dashboard and analysis tools

```powershell
streamlit run src/dashboard.py
python -m src.main --symbol XAUUSD --timeframe M30
python -m src.model_trainer
```

The dashboard reads local runtime state. It is not a substitute for checking
MT5 order results or the public forward-test evidence.

## Risk Controls

The live execution layer includes these independent protections:

- Account-wide magic-number lock and same-direction exposure cap.
- Pending-order, cluster, and same-timeframe proximity limits.
- Spread-aware pending entries and server-side SL/TP levels.
- Optional Daily Governor for daily loss, target, runner, and consecutive-loss
  limits.
- WatchZone rejection and range checks before a market order.
- Market-order safety based on completed-bar volume, RSI(8), Stochastic, and
  higher-timeframe oscillator opposition.
- Closed-candle processing for market-safety decisions; the running candle is
  excluded when marked as in progress.

The volume/oscillator market-safety gate applies to market entries, not passive
Standard Limit orders. Do not add a filter to pending orders without replaying
its effect on fills, expectancy, and drawdown.

## Performance Evidence

All public artifacts are under [`reports/`](reports/README.md). They are
generated from local MT5 exports and contain no credentials or account state.

Current closed-trade forward-test aggregate:

| Entry type | Closed trades | Win rate | Net PnL | Profit factor |
| --- | ---: | ---: | ---: | ---: |
| Standard Limit | 115 | 46.96% | -212.44 USD | 0.72 |
| WatchZone | 309 | 52.10% | -232.08 USD | 0.84 |

### Performance Breakdown by Strategy (since May 2026)

| Strategy | Entry Type | Trades | Wins | Losses | Win Rate | Net PnL (USD) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **OB** (Order Block) | Standard Limit | 27 | 15 | 12 | 55.56% | +126.17 |
| **OB** (Order Block) | WatchZone | 45 | 21 | 24 | 46.67% | -126.84 |
| **FVG** (Fair Value Gap) | Standard Limit | 52 | 27 | 25 | 51.92% | -248.32 |
| **FVG** (Fair Value Gap) | WatchZone | 114 | 57 | 57 | 50.00% | -202.02 |
| **IC** (Institutional Candle) | Standard Limit | 34 | 10 | 24 | 29.41% | -90.69 |
| **IC** (Institutional Candle) | WatchZone | 89 | 49 | 40 | 55.06% | +16.56 |
| **Swapzone** | WatchZone | 29 | 18 | 11 | 62.07% | +183.45 |
| **SND** (Supply & Demand) | Standard Limit | 2 | 2 | 0 | 100.00% | +0.40 |
| **SND** (Supply & Demand) | WatchZone | 11 | 8 | 3 | 72.73% | +34.75 |
| **BPR** (Balanced Price Range) | WatchZone | 2 | 2 | 0 | 100.00% | +9.84 |
| **Pivot** | WatchZone | 14 | 6 | 8 | 42.86% | -53.57 |
| **Breaker** | WatchZone | 5 | 0 | 5 | 0.00% | -94.25 |

### Performance Breakdown by Timeframe (since May 2026)

| Timeframe | Entry Type | Trades | Wins | Losses | Win Rate | Net PnL (USD) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **M15** | WatchZone | 27 | 14 | 13 | 51.85% | +8.09 |
| **M30** | Standard Limit | 55 | 23 | 32 | 41.82% | +3.67 |
| **M30** | WatchZone | 161 | 81 | 80 | 50.31% | -171.53 |
| **H1** | Standard Limit | 46 | 25 | 21 | 54.35% | +12.48 |
| **H1** | WatchZone | 74 | 36 | 38 | 48.65% | -177.75 |
| **H4** | Standard Limit | 14 | 6 | 8 | 42.86% | -228.59 |
| **H4** | WatchZone | 38 | 26 | 12 | 68.42% | +113.55 |
| **D1** | WatchZone | 9 | 4 | 5 | 44.44% | -4.44 |

The WatchZone aggregate is negative despite a win rate above 50%, which is why
the instant-entry policy is restrictive. The numbers are historical results,
not a prediction.

Standard Limit real-tick evidence is available in
[`standard_limit_real_tick_may2026.csv`](reports/standard_limit_real_tick_may2026.csv).
It replays MT5 bid/ask ticks for XAUUSDm from 1 May through 7 June 2026 with a
0.50 threshold, weighted A/B sizing, one concurrent structure, and 100%
required tick-day coverage per reported row. It is a short historical window,
not a full reproduction of the current WatchZone event path.

Downloadable evidence:

- [`forward_test_trades.csv`](reports/forward_test_trades.csv): closed MT5
  trade-level export with raw MT5 comments, prices, PnL, and matched planned
  levels when available.
- [`forward_test_summary.csv`](reports/forward_test_summary.csv): win rate,
  gross profit/loss, profit factor, average win/loss, payoff ratio, and net PnL
  by entry type, timeframe, and strategy.
- [`forward_exit_summary.csv`](reports/forward_exit_summary.csv): `Soft TP`,
  server TP, SL, and other exits by strategy.
- [`strategy_policy_report.csv`](reports/strategy_policy_report.csv): current
  evidence-based deployment status for each reported group.
- [`forward_test_report.xlsx`](reports/forward_test_report.xlsx): workbook
  containing the public evidence sheets.
- [`telegram_delivery_events.csv`](reports/telegram_delivery_events.csv):
  secret-free Telegram delivery journal after local journaling is enabled.

Small samples, missing signal matches, and `Unknown` historical strategies are
retained in the reports rather than being silently treated as positive results.
WatchZone historical rows are actual forward-test trades, not a reconstructed
real-tick backtest: exact replay also needs zone registration, first tick hit,
and fresh M1/M5 rejection data for each event.

## Reports and Daily Publication

Regenerate public reports from local MT5 exports:

```powershell
python scripts/update_public_reports.py
```

The publication helper stages only public report artifacts and documentation:

```powershell
.\scripts\publish_daily_reports.ps1
```

Do not add `.env`, `data/` runtime state, raw tick caches, or `scratch/` source
exports to a public commit.

## Project Layout

```text
src/
  scanner_worker.py       Scanner loop and WatchZone processing
  execution.py            MT5 order submission and active trade management
  live_trade_policy.py    Entry-path/timeframe/strategy policy enforcement
  price_watch_zones.py    WatchZone registration and hit handling
  smc_detector.py         SMC structure detection
  entry_quality_gate.py   Observer telemetry and oscillator context
  live_risk_governor.py   Daily risk controls
  real_tick_backtester.py MT5 bid/ask real-tick replay
  rollout_status.py       Offline rollout preflight
  dashboard.py            Streamlit dashboard
  model_trainer.py        Model training workflow
  telegram_bot.py         Telegram notifications and delivery journal
reports/                  Public performance evidence
tests/                    Regression and unit tests
.env.example              Versioned configuration template
```

## Troubleshooting

| Symptom | Checks |
| --- | --- |
| MT5 cannot connect | Keep the terminal open and logged in under the same Windows user; verify the login, password, server, and terminal installation. |
| No order is sent | Confirm `MT5_EXECUTE_TRADES`, timeframe, strategy policy, spread, exposure limits, Daily Governor, and the console skip reason. |
| A WatchZone does not enter | Check its timeframe/strategy against the allowlist, confidence, rejection/range state, and market-safety result. |
| Telegram is silent | Verify token/chat ID, start the chat with the bot, and inspect the runtime log or delivery journal. |
| Symbol is not found | Use the exact Market Watch symbol, including any broker suffix. |
| A backtest looks strong but live differs | Check tick coverage, costs, concurrent-structure limit, the sample period, and whether the replay includes the same event path. |

## Security and Limitations

- Never commit `.env`, credentials, account identifiers, or raw local exports.
- Confirm MT5 execution retcodes; a non-null API response alone is not proof of
  a filled order.
- Treat model output as a ranking signal, not certainty.
- Run a new or changed policy in monitoring/forward-test conditions before
  enabling execution.
- This repository contains no claim of future profitability.
