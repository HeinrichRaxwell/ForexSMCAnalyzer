# Loss Duplicate Feedback Fix - 2026-06-09

## What Happened

The dashboard/training data looked like it had many losses today, but most of the loss rows were duplicates from one already-closed trade:

- Signal: `M30_BPR_DUAL_BEAR_4328.399_4329.428_2026-06-09_04:30:00`
- Ticket: `3245218014`
- Result: `cut_loss_early`
- PnL: `-0.088R`
- Net profit: `-10,199 IDR`
- Confidence: `0.4217` on the 0.5 layer

Before cleanup:

- Today rows in `data/labeled_setups.csv`: `101`
- Today losses shown: `96`
- Unique today setup keys: `18`
- Duplicate today rows: `83`
- The same M30 BPR loss appeared `84` times.

After cleanup:

- Total real labeled rows: `959`
- Today rows: `18`
- Today wins: `5`
- Today losses: `13`
- Today duplicate rows: `0`

Backup before cleanup:

- `data/labeled_setups_before_dedupe_20260609_141027.csv`

## Root Cause

`process_mt5_history_feedback()` intentionally re-checks recorded trades so it can backfill missing CSV rows. The duplicate guard used a feedback key built from raw string values.

In real data, the registry had values like `4328.3994999999995`, while the CSV had `4328.3995`. Those represent the same entry, but raw string comparison treated them as different, so the same outcome was appended again every scanner cycle.

The registry `resolved_at` value was also refreshed on repeated processing, creating unnecessary registry writes for an already-resolved outcome.

## Fix

- Normalized feedback dedupe keys in `src/inference.py`:
  - time as stripped text
  - timeframe/direction/entry/SL/TP as rounded numeric strings
- Preserved existing `resolved_at` once an outcome is already resolved.
- Added regression test:
  - `test_process_mt5_history_feedback_does_not_duplicate_recorded_dual_option_in_csv`

## Loss Pattern After Deduplication

The remaining unique losses today still need review, but they are no longer inflated:

- `13` unique losses, `5` unique wins.
- `8/13` losses were against structure trend or FLoOP trend.
- `10/13` losses had KNN opposing probability greater than signal-side probability.
- `7/13` were full-loss style (`<= -0.5R`).
- `6/13` were early cut-loss style (`> -0.5R`).

Operationally, several losing tickets had confidence below `0.50`, which is consistent with the earlier scanner run that used a lower threshold before the scanner was restarted at `0.50`.

## Verification

- Duplicate regression test: passed.
- Target ML/inference/calibration/rollout tests: `34 passed`.
- `src.rollout_status --threshold 0.50`: `READY`.
- Cleaned calibration report regenerated.
- Clean retrain run kept the Champion model active because the Challenger underperformed.
