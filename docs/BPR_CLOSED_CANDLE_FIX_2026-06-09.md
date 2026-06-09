# BPR Closed-Candle Fix - 2026-06-09

## Issue

Live MT5 OHLC fetches include the newest bar at position `0`, which can still be forming. The BPR detector can create a Balanced Price Range on the latest row when overlapping opposite FVGs appear there. In live scanning this made BPR look detected before the candle had closed.

## Root Cause

`scanner_worker.apply_smc_detectors()` previously sent the full MT5 frame into every strategy detector. Because the latest MT5 candle was still open, all detector output that depended on the final row could be early, including BPR, FVG, OB/Breaker, Swapzone, IC, SND, Pivot Rejection, and LTF rejection confirmation.

## Fix

- Added `drop_latest_forming_candle()` in `src/scanner_worker.py`.
- Added `closed_only=True` support to `apply_smc_detectors()`.
- Live scanner now runs every SMC detector on closed candles only.
- Raw D1 data is still preserved for daily pivot calculation, because classic daily pivots need today's D1 row as the date anchor while using the previous day's High/Low/Close.

## Verification

- Added tests proving a BPR formed only on the latest live bar is ignored until a new candle opens.
- Target strategy tests: `42 passed`.
- Scanner execution/shadow tests: `17 passed`.
- Compile check passed for `src/scanner_worker.py`, `src/smc_detector.py`, `src/main.py`, and `src/rejection_detector.py`.
- Full suite: `214 passed, 35 warnings`.

## Operational Note

This fixes premature live detection from forming candles. It does not guarantee profit and does not change the underlying SMC/Fibonacci formulas.
