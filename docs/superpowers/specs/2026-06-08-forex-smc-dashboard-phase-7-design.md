# Forex SMC Dashboard Phase 7 Design - 2026-06-08

## Objective

Phase 7 adds execution-path diagnostics for live scanner decisions. It answers why a setup became a live ticket, stayed as shadow monitoring, or was accepted without a ticket. The immediate trigger was a Daily Pivot rejection that appeared on chart but did not market order.

This phase does not promise guaranteed profit, perfect entry, or next-trade profit. It makes the execution decision auditable.

## Scope

Phase 7 covers:

1. Model freshness explanation for stale `smc_xgb_classifier.joblib` and `smc_lgb_classifier.joblib` warnings.
2. Pivot/key-level/rejection diagnostics in the dashboard.
3. Correct source labeling for low-confidence registry rows so they are not counted as accepted live signals.
4. Scanner hardening so a low-confidence registry record can be promoted to live execution if the same active setup later passes threshold.
5. Phase documentation for recovery after session interruption.

## Data Flow

Signals can come from two stores:

- `data/sent_signals.json`: accepted live signals and low-confidence registry rows.
- `data/shadow_signals.json`: below-threshold shadow signals with virtual TP/SL tracking.

Phase 7 treats `is_low_confidence=True` rows in `sent_signals.json` as `shadow_registry`, not accepted live trades. The dashboard combines accepted, shadow registry, and shadow records in one diagnostic view.

## Diagnostic Decisions

The dashboard uses these decision labels:

- `live_ticket`: a ticket id is stored.
- `shadow_monitoring`: confidence is below accept threshold, so no MT5 order should be sent.
- `accepted_no_ticket`: confidence passed threshold but no ticket is stored; check `.env`, MT5 logs, or execution skip messages.
- `needs_live_promotion_review`: shadow/low-confidence row now meets threshold; scanner should promote on the next live pass if the setup still exists.
- `registry_missing_confidence`: registry row lacks a confidence value.
- `below_threshold_registry`: registry row remains below threshold.

## Key-Level Context

Rows are tagged with key-level context when evidence exists:

- `pivot`: strategy is Pivot or nearest pivot distance is very small.
- `volume_poc`: nearest POC distance is very small.
- `support_resistance_flip`: strategy is Swapzone.
- `supply_demand`: strategy is SND or supply/demand.

## Safety Rules

The dashboard remains read-only:

- no MT5 order calls,
- no retrain button,
- no Telegram send,
- no `.env` mutation,
- no subprocess execution from the Streamlit UI.

Scanner promotion is still gated by live scanner conditions:

- confidence must pass threshold,
- setup must still be active,
- market order still requires rejection confirmation and price inside entry zone,
- execution still depends on `MT5_EXECUTE_TRADES` and MT5 order checks.

## Verification

Required checks:

```powershell
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m pytest tests\test_scanner_entry_decisions.py -q
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m pytest tests\test_dashboard_data.py -q
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m compileall -q src tests
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m pytest -q
```
