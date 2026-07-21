# Forex SMC Analyzer

Forex SMC Analyzer is a Windows-focused XAUUSD analysis and MetaTrader 5
automation project. It combines Smart Money Concepts (SMC) structure detection,
multi-timeframe context, model scoring, WatchZone monitoring, and MT5 order management.

The repository is designed to start in monitoring mode. `MT5_EXECUTE_TRADES`
defaults to `False`, so scanning, Telegram alerts, WatchZone monitoring, and
shadow tracking can run without submitting an order.

> Historical backtests and forward tests are evidence, not a profit guarantee
> or trading recommendation. Automated trading can lose all deposited capital.

## Contents

- [What the system does](#what-the-system-does)
- [Entry paths and policy](#entry-paths-and-policy)
- [100% Real-Tick Backtest Evidence](#100-real-tick-backtest-evidence)
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

- SMC detectors for FVG, order blocks (OB), BPR, imbalance (IC), swing structure,
  BOS/CHoCH, pivots, and related context. Breaker Blocks (BB) are deactivated.
- Multi-timeframe indicators including FLoOP, KNN directional context, volume
  profile, RSI(8), and Stochastic.
- Model scoring and probability calibration via XGBoost.
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

The current policy blocklist configuration:

| Scope | Rule | Reason |
| --- | --- | --- |
| All timeframes | `*:BB` | Breaker Blocks deactivated (weak real-tick win rate: 23.44%). |

H4 Order Block (`H4 OB`) is fully enabled for Standard Limit entries.

### WatchZone

A WatchZone is registered after a scan and checked against live price between
full scans. A hit is not itself an order: the selected A/B leg must meet the
configured confidence threshold, lower-timeframe rejection confirmation (M1/M5/M15),
valid market-entry range, strategy policy, and market safety checks.

Instant WatchZone execution is currently enabled for high-winrate Order Block candidates:

| Timeframe | Strategy | Status | Real-Tick Winrate |
| --- | --- | --- | --- |
| M30 | OB | Active | 79.53% |
| H1 | OB | Active | 85.71% |
| H4 | OB | Active | 100.00% |

The policy is configured with timeframe-and-strategy rules in `.env`:

```env
MT5_WATCH_ZONE_STRATEGY_ALLOWLIST=M30:OB,H1:OB,H4:OB
MT5_STANDARD_LIMIT_STRATEGY_BLOCKLIST=*:BB
```

`*` matches every timeframe. The executor enforces these rules as a
defense-in-depth check, including account-sync and recovery paths.

---

## 100% Real-Tick Backtest Evidence

**Evaluation Period:** January 2, 2026 – June 7, 2026 (133 Days, 100% Real Ticks Coverage)  
**Symbol:** XAUUSDm | **Starting Capital:** $100.00 | **Max Concurrent Trades:** 3  
**Winrate Formula:** $\text{Winrate (\%)} = \frac{\text{Wins}}{\text{Wins} + \text{Losses}} \times 100$ *(Unfilled/Missed setups excluded)*

### Verified Performance Matrix

| Timeframe | Strategy / Mode | Total Resolved | Wins | Losses | Missed | Winrate (%) | Max Drawdown (%) | Final Balance ($100 Start) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **H4** | **OB (WatchZone)** | 12 | 12 | 0 | 25 | **100.00%** | **0.00%** | **$1,006.84** |
| **H1** | **OB (WatchZone)** | 70 | 60 | 10 | 121 | **85.71%** | **18.73%** | **$1,931.44** |
| **H4** | **COMBINED (WatchZone)** | 81 | 67 | 14 | 180 | **82.72%** | **9.95%** | **$2,540.09** |
| **H1** | **OB (Limit Order)** | 172 | 140 | 32 | 256 | **81.40%** | **17.93%** | **$4,719.31** |
| **H1** | **IC (Limit Order)** | 488 | 395 | 93 | 488 | **80.94%** | **13.41%** | **$3,912.79** |
| **M30** | **IC (Limit Order)** | 997 | 806 | 191 | 949 | **80.84%** | **6.55%** | **$6,958.83** |
| **H1** | **COMBINED (WatchZone)** | 424 | 340 | 84 | 665 | **80.19%** | **18.99%** | **$5,883.89** |
| **M30** | **OB (WatchZone)** | 127 | 101 | 26 | 212 | **79.53%** | **13.36%** | **$2,128.13** |
| **M30** | **COMBINED (WatchZone)** | 948 | 713 | 235 | 1,285 | **75.21%** | **14.50%** | **$8,088.71** |
| **H1** | **BPR (Limit Order)** | 43 | 31 | 12 | 227 | **72.09%** | 28.42% | **$641.10** |
| **H1** | **FVG (Limit Order)** | 190 | 136 | 54 | 771 | **71.58%** | 14.60% | **$2,895.37** |
| **H4** | **OB (Limit Order)** | 42 | 30 | 12 | 69 | **71.43%** | 7.84% | **$1,724.28** |
| **H1** | **COMBINED (Limit Order)** | 1,268 | 874 | 394 | 2,106 | **68.93%** | 11.46% | **$15,005.00** |
| **H4** | **FVG (Limit Order)** | 64 | 44 | 20 | 230 | **68.75%** | 24.30% | **$1,930.47** |
| **M30** | **COMBINED (Limit Order)** | 2,695 | 1,818 | 877 | 4,294 | **67.46%** | 5.95% | **$22,122.99** |
| **M30** | **FVG (Limit Order)** | 486 | 320 | 166 | 1,485 | **65.84%** | 28.29% | **$4,061.42** |
| **H4** | **COMBINED (Limit Order)** | 313 | 201 | 112 | 591 | **64.22%** | 29.52% | **$7,681.87** |
| **M30** | **BPR (Limit Order)** | 130 | 76 | 54 | 466 | **58.46%** | 32.92% | **$1,029.67** |
| **H4** | **Swapzone (Limit Order)** | 86 | 44 | 42 | 90 | **51.16%** | 25.69% | **$2,788.76** |
| **H1** | **Swapzone (Limit Order)** | 374 | 185 | 189 | 301 | **49.47%** | 31.02% | **$3,691.53** |
| **M30** | **Swapzone (Limit Order)** | 833 | 410 | 423 | 576 | **49.22%** | 28.88% | **$5,269.32** |
| **H4** | **BPR (Limit Order)** | 24 | 10 | 14 | 72 | **41.67%** | 42.12% | **$530.33** |
| **M30** | **BB (Limit Order - Disabled)**| 64 | 15 | 49 | 28 | **23.44%** | 68.12% | *$18.42 (Blown)* |

---

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
with the following values in `.env`:

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
| `MT5_MAX_CONCURRENT_TRADES` | `6` | Caps total active positions plus pending orders across symbols. |
| `MT5_MAX_SAME_DIRECTION_TRADES` | `3` | Caps same-direction exposure (max 3 buys and 3 sells). |
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
python -m src.scanner_worker --symbol XAUUSD --loop --interval 1 --threshold 0.50
```

For fast monitoring of previously detected setups between full scans:

```powershell
python -m src.scanner_worker --symbol XAUUSD --loop --interval 1 --threshold 0.50 --realtime-reaction --tick-interval 0.1 --min-reaction-move 0.10
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

## Performance Evidence

All public artifacts are under [`reports/`](reports/README.md). They are
generated from local MT5 exports and contain no credentials or account state.

Downloadable evidence:

- [`real_tick_backtest_with_watchzone.csv`](data/real_tick_backtest_with_watchzone.csv): 100% real tick backtest results for WatchZone & Standard Limit modes.
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
