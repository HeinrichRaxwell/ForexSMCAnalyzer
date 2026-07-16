# Public Performance Evidence

This directory contains files intended for download and independent review.

- `standard_limit_real_tick_may2026.csv`: standard-limit replay with MT5 bid/ask ticks. Each row includes its tick-coverage fields.
- `forward_test_trades.csv`: closed trades from the local MT5 forward-test export, including raw MT5 entry/exit comments, open/close time, entry, exit, PnL, and planned SL/TP only when matched to a saved source signal.
- `forward_test_summary.csv`: forward-test aggregates by entry type, timeframe, and strategy.
- `forward_test_report.xlsx`: the same evidence in an Excel workbook.
- `telegram_delivery_events.csv`: a secret-free local delivery journal for future Telegram alerts. It is populated only after local journaling is enabled.
- `report_metadata.json`: generation timestamp, source paths, and source-match counts.

The WatchZone rows are **actual forward-test trades**, not a reconstructed real-tick backtest. The repository does not yet contain an event-replay engine that reproduces WatchZone registration, first tick hit, and fresh M1/M5 rejection confirmation historically.

Run this daily on the machine that has refreshed MT5 exports:

```powershell
.\scripts\publish_daily_reports.ps1
```

The script stages and pushes only the public report artifacts, README, and report tooling. It does not publish `.env`, account state, scanner state, raw tick cache, or raw scratch data.
