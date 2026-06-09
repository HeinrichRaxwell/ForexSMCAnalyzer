# Phase 7 Real Tick Backtest Plan - 2026-06-07

## Goal

Build and run a one-year XAUUSD backtest for all available strategies using MT5 bid/ask ticks instead of candle-only path assumptions.

## Scope

Phase 7 will add:

- tick downloader/cache from MT5 `copy_ticks_range`,
- daily compressed tick cache under `data/ticks/`,
- bid/ask tick replay engine for pending entry, SL, and TP,
- one-year backtest runner for:
  - `FVG`,
  - `OB`,
  - `BB`,
  - `Swapzone`,
  - `BPR`,
  - `IC`,
  - `COMBINED`,
- threshold `0.50`,
- capitals `$50` and `$100`,
- max concurrent `3` by default,
- equal and weighted sizing.

## Definition of 100% Real Tick

For this project, a result may be called real-tick only if:

- entry fill is decided by tick bid/ask,
- SL and TP are decided by tick bid/ask,
- every tested date range has tick data available from MT5,
- missing tick days are reported and not silently treated as wins,
- the output report states the tick coverage.

It still does not mean broker-perfect execution unless spread, commission, swap, slippage, requote, and margin checks are included.

## Files

Planned files:

- `src/tick_data.py`: MT5 tick download/cache helpers.
- `src/tick_backtester.py`: bid/ask tick replay simulation.
- `src/real_tick_backtester.py`: CLI runner for 1-year matrix.
- `tests/test_tick_data.py`: downloader/cache unit tests with fake MT5 module.
- `tests/test_tick_backtester.py`: tick replay fill/SL/TP tests.
- `docs/REAL_TICK_BACKTEST_1Y_2026-06-07.md`: final result doc.

## Execution Plan

1. Add unit-tested tick cache helpers.
2. Add unit-tested tick replay simulator.
3. Add CLI runner that can download one-year candle data and tick data.
4. Run a small smoke test.
5. Run the one-year matrix if MT5 terminal and broker tick history are available.
6. Document exact result and coverage.

## Known Risks

- MT5/broker may not provide a full one-year tick history.
- One year of XAUUSD ticks can be large and slow to download.
- The current model was trained on candle-derived features; tick execution improves path accuracy but not necessarily feature generation.
- The old backtester did not include SND/Pivot in the matrix; Phase 7 keeps the current matrix strategies first, then can extend if needed.
