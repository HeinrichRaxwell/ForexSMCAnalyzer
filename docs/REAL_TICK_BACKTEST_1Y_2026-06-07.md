# Phase 7 Real Tick Backtest Result - 2026-06-07

## Scope

Phase 7 adds and verifies a MetaTrader 5 real-tick backtest path for XAUUSD:

- daily MT5 tick downloader/cache under `data/ticks/<symbol>/YYYY-MM-DD.csv.gz`,
- bid/ask tick replay for pending entry, SL, and TP,
- CSV matrix runner for H4 strategy results,
- tick coverage reporting so incomplete tick history is not silently treated as valid.

Historical strategy matrix tested:

- FVG
- OB
- BB
- Swapzone
- BPR
- IC
- COMBINED

Parameters used:

- symbol: `XAUUSD`, resolved by MT5 to `XAUUSDm`
- timeframe: `H4`
- threshold: `0.50`
- capitals: `$50`, `$100`
- max concurrent setups: `3`
- sizing: `equal`, `weighted`
- contract size: `100.0`

## Files Created Or Updated

- `src/tick_data.py`
- `src/tick_backtester.py`
- `src/real_tick_backtester.py`
- `tests/test_tick_data.py`
- `tests/test_tick_backtester.py`
- `tests/test_real_tick_backtester.py`
- `data/real_tick_backtest_1y_h4.csv`
- `data/real_tick_backtest_verified_window_h4_trimmed.csv`

## Important Fixes During Phase 7

- Added handling for corrupted partial `.csv.gz` tick cache files.
- A corrupted cache file was found:
  - `data/ticks/XAUUSDm/2026-01-21.csv.gz`
- The downloader now re-downloads corrupt cached days instead of crashing.
- The cache range loader now marks corrupt cached days as missing instead of treating them as valid.
- Candle data returned by MT5 is trimmed to the requested `[start, end)` range before coverage is calculated.

## One-Year Result

Command target:

```powershell
python -m src.real_tick_backtester --symbol XAUUSD --days 365 --timeframes H4 --thresholds 0.50 --capitals 50,100 --max-concurrent 3 --sizing equal,weighted --download-ticks --output data\real_tick_backtest_1y_h4.csv
```

Result file:

- `data/real_tick_backtest_1y_h4.csv`

Coverage:

| Requested calendar days | Candle days required | Candle days with ticks | Missing candle days | Coverage | Complete real tick? |
|---:|---:|---:|---:|---:|:---:|
| 366 | 311 | 133 | 178 | 42.765273% | NO |

This means the 365-day run is not a 100% real-tick-complete backtest. MT5/broker tick history for this terminal did not provide usable tick data for many candle days from 2025-06-08 through 2025-12-31.

Top one-year H4 rows, but partial tick coverage:

| Strategy | Sizing | Capital | Resolved | W/L | Winrate | Final Balance | Max DD | Blown |
|---|---|---:|---:|---:|---:|---:|---:|:---:|
| COMBINED | weighted | 100 | 128 | 100/28 | 78.1250% | 5614.72 | 37.21% | No |
| COMBINED | weighted | 50 | 128 | 100/28 | 78.1250% | 5564.72 | 39.29% | No |
| COMBINED | equal | 100 | 128 | 100/28 | 78.1250% | 3971.24 | 34.64% | No |
| COMBINED | equal | 50 | 128 | 100/28 | 78.1250% | 3921.24 | 37.27% | No |
| FVG | weighted | 100 | 56 | 43/13 | 76.7857% | 3251.00 | 27.01% | No |
| FVG | weighted | 50 | 56 | 43/13 | 76.7857% | 3201.00 | 32.77% | No |

Because coverage is incomplete, these rows are useful as a partial replay only, not a clean 1-year real-tick proof.

## Tick-Complete Window Result

Command target:

```powershell
python -m src.real_tick_backtester --symbol XAUUSD --start 2026-01-02 --end 2026-06-07 --timeframes H4 --thresholds 0.50 --capitals 50,100 --max-concurrent 3 --sizing equal,weighted --output data\real_tick_backtest_verified_window_h4_trimmed.csv
```

Result file:

- `data/real_tick_backtest_verified_window_h4_trimmed.csv`

Coverage:

| Requested calendar days | Candle days required | Candle days with ticks | Missing candle days | Coverage | Complete real tick? |
|---:|---:|---:|---:|---:|:---:|
| 157 | 132 | 132 | 0 | 100.0% | YES |

Top tick-complete H4 rows:

| Strategy | Sizing | Capital | Setups | Resolved | W/L | Winrate | Final Balance | Max DD | Blown |
|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| COMBINED | weighted | 100 | 394 | 127 | 99/28 | 77.9528% | 5542.76 | 40.28% | No |
| COMBINED | weighted | 50 | 394 | 127 | 99/28 | 77.9528% | 5492.76 | 42.73% | No |
| COMBINED | equal | 100 | 394 | 127 | 99/28 | 77.9528% | 3939.80 | 36.25% | No |
| COMBINED | equal | 50 | 394 | 127 | 99/28 | 77.9528% | 3889.80 | 39.14% | No |
| FVG | weighted | 100 | 262 | 56 | 43/13 | 76.7857% | 3251.00 | 27.01% | No |
| FVG | weighted | 50 | 262 | 56 | 43/13 | 76.7857% | 3201.00 | 32.77% | No |
| FVG | equal | 100 | 262 | 56 | 43/13 | 76.7857% | 2247.14 | 22.19% | No |
| FVG | equal | 50 | 262 | 56 | 43/13 | 76.7857% | 2197.14 | 28.01% | No |

## Interpretation

The strategy performed strongly in the tick-complete H4 window, especially COMBINED and FVG, but this is not a guarantee of future profit.

Important limitations:

- The one-year run is not 100% tick-complete because broker/MT5 tick history is incomplete for the first part of the requested year.
- The tick-complete proof currently covers 2026-01-02 to 2026-06-07 on H4.
- Entry, TP, and SL are replayed by bid/ask ticks.
- Commission, swap, slippage, requotes, broker execution delay, and margin validation are not yet included.
- SND/pivot/rejection market-order live-entry logic is not yet part of this historical matrix. This matrix tests the existing `generate_all_setups()` historical strategies.

## Verification Run

Fresh verification after Phase 7 implementation:

```powershell
python -m pytest tests\test_tick_data.py tests\test_tick_backtester.py tests\test_real_tick_backtester.py -q
```

Result:

```text
13 passed
```

Additional focused checks also passed earlier:

```text
tests\test_tick_data.py: 6 passed
tests\test_real_tick_backtester.py: 6 passed
```

