# PROGRESS - ML Confidence Predictive & Expectancy Repair

> Catatan status hidup. Kalau sesi ke-reset, baca file ini sebelum lanjut.

Branch: `ml-confidence-predictive`
Spec: `docs/superpowers/specs/2026-06-09-ml-confidence-predictive-design.md`
Plan: `docs/superpowers/plans/2026-06-09-ml-confidence-predictive.md`
Tanggal mulai: 2026-06-09
Update terakhir: 2026-06-10

---

## Konteks

Bot ML trading XAUUSD memakai SMC + FLoOP volume + ensemble XGB/LGBM + feedback loop.
User ingin confidence benar-benar predictive, belajar dari win/loss, dan tidak berbasis angka leakage.

Temuan awal:
1. Calibration report lama leakage karena model di-score pada data training.
2. Walk-forward OOF sebelumnya membuat angka lebih jujur, tetapi confidence belum predictive.
3. Spread/slippage belum dimodelkan, sehingga expectancy lama terlalu optimistis.

Keputusan yang tetap berlaku:
1. Cost modeling masuk dengan default `ML_SPREAD_USD=0.30`, `ML_SLIPPAGE_USD=0.0`.
2. Calibration memakai isotonic regression dari walk-forward OOF.
3. Hanya `PIVOT_REJECTION` yang logic entry-nya diubah menjadi reaction-based routing.
4. Strategi lain tidak diubah rule deteksi/entry-nya; hanya ditambah cost/features/calibration.
5. Hasil harus dilaporkan apa adanya. Tidak ada klaim profit bila rollout gate belum lolos.

---

## Status Task

### Workstream 1: Cost Modeling
- selesai Task 1 - `compute_cost_r`
  - `src/labeler.py` membaca spread/slippage dari env dan menghitung cost dalam R.
  - Test: `tests/test_cost_modeling.py`.
- selesai Task 2 - `compute_pnl_relative`
  - Win = RR - cost_R, loss = -1.0 - cost_R, risk 0 aman.
  - Test: `tests/test_cost_modeling.py`.
- selesai Task 3 - tulis `pnl_relative` cost-adjusted
  - Implementasi lewat titik fitur terpusat di `label_smc_setups`.
  - Test: `tests/test_labeler_pnl_integration.py`.

### Workstream 2: Feature Engineering
- selesai Task 4 - modul `src/setup_features.py`
  - Fungsi: `rr_ratio`, `atr_percentile`, `body_to_range_ratio`,
    `dist_to_recent_swing_norm`, `htf_trend_aligned`, `confluence_score`.
  - Test: `tests/test_setup_features.py`.
- selesai Task 5 - integrasi fitur ke labeler
  - Historical setups mendapat `rr_ratio`, `atr_percentile`, `body_to_range_ratio`,
    `dist_to_recent_swing`, `htf_trend_aligned`, `confluence_score`.
  - `src/inference.py` dan `src/shadow_tracker.py` juga mengisi default/derived untuk
    legacy live feedback/shadow payload agar kolom baru tidak NaN.

### Workstream 3: PIVOT_REJECTION Reaction Routing
- selesai Task 6 - `src/reaction_router.py`
  - `classify_reaction`, `reaction_strength`, konstanta `ORDER_*` dan `STATE_*`.
- selesai Task 7 - `compute_levels`
  - MARKET/LIMIT/STOP menghasilkan entry, SL, dan TP konsisten.
- selesai Task 8 - pivot memakai reaction router
  - Pivot setup menulis `order_type` dan `reaction_strength`.
  - Non-pivot setup mendapat default `order_type=ORDER_MARKET`, `reaction_strength=0.0`.

### Workstream 4: Isotonic Confidence Calibration
- selesai Task 9 - `src/calibrator.py`
  - `fit_calibrator`, `apply_calibrator`, `save_calibrator`, `load_calibrator`.
  - Missing/invalid calibrator fallback identity.
- selesai Task 10 - fit calibrator dari walk-forward OOF
  - `src.calibration_report` default `mode=walk_forward`.
  - Menyimpan `models/confidence_calibrator.joblib`.
- selesai Task 11 - apply calibrator di inference
  - `predict_setup_probability()` menerapkan calibrator bila ada.
  - `_build_inference_matrix()` mengisi fitur model yang hilang dengan `0.0`.
  - `src.model_trainer` diperbaiki agar champion/challenger gate meng-align schema fitur
    champion lama; feature migration tidak lagi otomatis menjadi `REJECTED_GATE_ERROR`.

### Verifikasi Akhir
- selesai Task 12 - relabel, retrain, rescore, rollout check
  - Backup dibuat: `data/labeled_setups_precostfeat_backup.csv`.
  - `python -m src.labeler` menulis ulang `data/labeled_setups.csv`.
  - `python -m src.model_trainer` accepted challenger dan mem-promote model baru.
  - `python -m src.calibration_report` menulis report walk-forward dan calibrator.
  - `python -m src.rollout_status --threshold 0.50` masih `BLOCKED`.

---

## Bukti Verifikasi 2026-06-10

Commands:
- `python -m pytest tests\test_reaction_router.py tests\test_labeler_pnl_integration.py tests\test_setup_features.py tests\test_calibrator.py tests\test_calibration_report_calibrator.py tests\test_calibration_report.py tests\test_inference_calibration.py tests\test_inference.py::test_predict_setup_probability -q`
  - Result: `44 passed, 9 warnings`
- `python -m pytest tests\test_inference.py tests\test_shadow_tracker.py tests\test_model_trainer.py tests\test_reaction_router.py tests\test_labeler_pnl_integration.py tests\test_setup_features.py tests\test_calibrator.py tests\test_calibration_report_calibrator.py tests\test_calibration_report.py tests\test_inference_calibration.py -q`
  - Result: `78 passed, 33 warnings`
- `python -m pytest -q`
  - Result: `260 passed, 35 warnings`
- `python -m py_compile src\inference.py src\model_trainer.py src\calibration_report.py src\labeler.py src\calibrator.py src\reaction_router.py`
  - Result: exit 0
- `python -m src.model_trainer`
  - Result: challenger accepted; model files overwritten.
- `python -m src.calibration_report`
  - Result: `scoring_mode=walk_forward`; calibrator saved.
- `python -m src.rollout_status --threshold 0.50`
  - Result: `BLOCKED`.

Artifact checks:
- `data/labeled_setups.csv`: 5000 rows, no NaN in the 8 new feature/routing columns.
- `data/shadow_labeled_setups.csv`: 337 rows, no NaN in the 8 new feature/routing columns.
- `models/smc_xgb_classifier.joblib` feature schema includes:
  `rr_ratio`, `atr_percentile`, `body_to_range_ratio`, `dist_to_recent_swing`,
  `htf_trend_aligned`, `confluence_score`, `order_type`, `reaction_strength`.
- `models/confidence_calibrator.joblib` exists.

Final walk-forward report:
- Overall: sample `5337`, winrate `36.87%`, expectancy `-0.19R`, PF `0.72`,
  max drawdown `1202.56R`.
- Source `real`: sample `5000`, winrate `34.98%`, expectancy `-0.24R`, PF `0.66`.
- Source `shadow`: sample `337`, winrate `64.99%`, expectancy `+0.57R`, PF `2.64`.
- Threshold `0.50`: sample `727`, winrate `46.49%`, expectancy `-0.02R`, PF `0.97`,
  max drawdown `64.13R`.
- Threshold `0.60`: sample `302`, winrate `49.67%`, expectancy `+0.06R`, PF `1.12`,
  max drawdown `15.10R`.
- Threshold `0.70`: sample `80`, winrate `55.00%`, expectancy `+0.14R`, PF `1.31`,
  max drawdown `13.57R`.
- Recommendation reason: `no_threshold_met_all_rules`.

Rollout status:
- Status: `BLOCKED`
- Live execution: `True`
- Failure reasons at threshold `0.50`: expectancy below gate and max drawdown above gate.
- Do not claim ready for real/live rollout until rollout gate passes. Scanner process was active
  during this work; if live code needs the new feedback/shadow schema guards, restart scanner so it
  loads the patched modules.

---

## Cara Melanjutkan

1. If live scanner is still running from before this patch, stop/restart it deliberately so
   `src.inference` and `src.shadow_tracker` schema guards are loaded.
2. Investigate why real-source historical setups remain negative while shadow is positive.
3. Consider raising live threshold only after a gate that accounts for drawdown passes; threshold
   `0.60` and `0.70` show positive expectancy but still fail drawdown rules.
4. Keep using walk-forward reports for performance claims.

---

## Update Entry Gate 2026-06-10

Tambahan setelah review strategi live:

- Added `src.entry_quality_gate` for spread-aware and oscillator-aware live entry filtering.
- Exness/XAUUSD 3-digit spread is normalized as broker points to price:
  - `300 points * 0.001 point = 0.300 USD`.
  - Live gate reads MT5 `symbol_info.point`, `digits`, `tick.ask`, and `tick.bid`.
- FVG and BPR remain live-eligible core strategies.
  - They are not disabled.
  - Live entry gate requires at least `0.60` effective confidence by default, even if scanner threshold is still `0.50`.
- Entry gate blocks trades when:
  - RR is below `ML_ENTRY_MIN_RR` default `1.20`,
  - confidence is below the effective strategy/entry-gate threshold.
- Spread is not a skip reason.
  - `spread_points`, `spread_price`, and `spread_r` stay as diagnostic metadata.
  - Pending order execution adjusts the broker entry price with live spread:
    - BUY pending: `broker_entry_price = raw_entry_price + live_spread`.
    - SELL pending: `broker_entry_price = raw_entry_price - live_spread`.
    - Example with raw entry `4000.000` and spread `0.260`: BUY sends `4000.260`, SELL sends `3999.740`.
- RSI 8 and Stoch RSI (9,3,3) are not absolute buy/sell blockers.
  - If oscillator is extreme against the trade but HTF/context supports the setup, required confidence rises to `ML_OSCILLATOR_EXTREME_HTF_CONFIDENCE` default `0.70`.
  - If oscillator is extreme without HTF support, required confidence rises to `ML_OSCILLATOR_EXTREME_UNSUPPORTED_CONFIDENCE` default `0.80`.
  - If confidence passes the raised requirement, the setup can still execute and its reason is recorded in `entry_gate`.
- High-confidence candidates blocked by the gate are routed to shadow tracking with `filtered_reason`.
  - Dual entries are evaluated per leg.
  - If one dual leg passes, it can still execute while the blocked leg is shadow-tracked.
- Recovery/re-place of inactive tickets also passes through the same live entry gate before re-ordering.

This does not prove guaranteed profit. It reduces low-quality live entries and records filtered outcomes so future learning can compare accepted vs rejected setups.

Verification added:
- `python -m pytest tests\test_entry_quality_gate.py tests\test_scanner_entry_decisions.py tests\test_scanner_market_orders.py tests\test_scanner_shadow_signals.py tests\test_shadow_tracker.py -q`
  - Result: `57 passed`
- `python -m py_compile src\entry_quality_gate.py src\scanner_worker.py src\shadow_tracker.py src\execution.py`
  - Result: exit 0
- `python -m pytest -q`
  - Result: `277 passed, 35 warnings`
- `python -m src.rollout_status --threshold 0.60`
  - Result: `BLOCKED`
  - Metrics: sample `302`, winrate `49.67%`, expectancy `0.06R`, max DD `15.10R`.
  - Fail reasons: expectancy below rollout gate and max drawdown above rollout gate.

Additional RED/GREEN evidence for spread-adjusted pending orders:
- Before implementation, the pending-order tests failed because both BUY and SELL sent raw `4000.000`.
- After implementation:
  - `python -m pytest tests\test_entry_quality_gate.py tests\test_scanner_market_orders.py::test_pending_buy_order_adds_live_spread_to_entry_price tests\test_scanner_market_orders.py::test_pending_sell_order_subtracts_live_spread_from_entry_price -q`
  - Result: `13 passed`

Honest forward-test status:
- Code-level checks pass, but rollout remains `BLOCKED` at threshold `0.60`.
- Current real/shadow combined gate metrics are not a "masterpiece" yet: winrate is near breakeven, expectancy is only slightly positive, and drawdown is still too high.
- This patch fixes the mechanical entry logic requested for spread and oscillator context; it does not guarantee profit or near-constant wins.

---

## Update Profit Lock Ladder 2026-06-10

Tambahan setelah dua trade terakhir sempat floating besar lalu kembali kena BEP:

- `manage_active_trades()` now has an XAUUSD pip-ladder profit lock on top of the existing BEP and structural trailing logic.
- XAUUSD pip scale follows the project convention:
  - `1 pip = 0.100`.
  - `100 pips = 10.000` price.
  - Example: `4000.000 -> 4010.000` is `+100 pips`.
- Profit lock defaults:
  - `MT5_PROFIT_LOCK_ENABLED=True`.
  - `MT5_PROFIT_LOCK_STEP_PIPS=100`.
  - `MT5_PROFIT_LOCK_GAP_PIPS=100`.
- Ladder behavior:
  - Floating `+100 pips` locks around BEP plus spread buffer.
  - Floating `+200 pips` locks `+100 pips`.
  - Floating `+300 pips` locks `+200 pips`.
  - Floating `+400 pips` locks `+300 pips`.
  - General rule: at each full 100-pip step, SL locks `step_profit - 100 pips`.
- BUY example:
  - Entry `4000.000`, bid reaches `4035.000` (`+350 pips`).
  - Full step is `+300 pips`, so locked profit is `+200 pips`.
  - SL moves to `4020.000` if it improves the current SL and remains below market.
- SELL example:
  - Entry `4000.000`, ask reaches `3965.000` (`+350 pips`).
  - Full step is `+300 pips`, so locked profit is `+200 pips`.
  - SL moves to `3980.000` if it improves the current SL and remains above market.
- The manager chooses the strongest valid SL candidate from:
  - existing 1R BEP rule,
  - new pip-ladder profit lock,
  - structural swing trailing.
- SL never intentionally moves backward:
  - BUY SL must move up and stay below bid.
  - SELL SL must move down and stay above ask.

Verification added:
- RED before implementation:
  - `python -m pytest tests\test_active_trade_management.py::test_manage_active_trades_buy_locks_200_pips_after_350_pips_profit tests\test_active_trade_management.py::test_manage_active_trades_sell_locks_200_pips_after_350_pips_profit -q`
  - Result: failed because BUY only moved to `4000.200` and SELL only moved to `3999.800`.
- GREEN after implementation:
  - `python -m pytest tests\test_active_trade_management.py::test_manage_active_trades_buy_locks_200_pips_after_350_pips_profit tests\test_active_trade_management.py::test_manage_active_trades_sell_locks_200_pips_after_350_pips_profit tests\test_active_trade_management.py -q`
  - Result: `7 passed`
- `python -m py_compile src\execution.py`
  - Result: exit 0
- `python -m pytest -q`
  - Result: `279 passed, 35 warnings`

Operational note:
- If the scanner/trade manager process is already running, restart it deliberately before expecting this new profit-lock ladder to manage live positions.

---

## Update Realtime Reaction Watcher 2026-06-10

Tambahan supaya entry tidak telat menunggu full scanner interval:

- Added `src.realtime_reaction_watcher`.
- Closed-candle scanner remains the source of truth for SMC/FVG/BPR structure.
  - Full scan still uses closed candles so FVG/BPR/structure is not created from forming candle noise.
  - Realtime watcher only watches accepted/registered setups that already exist in `sent_signals.json`.
- New realtime behavior:
  - Between full scanner cycles, the worker reads live MT5 bid/ask ticks.
  - If an accepted setup has no ticket yet and price enters its entry zone, the watcher checks immediate tick reaction.
  - BUY requires bid to turn upward inside the zone.
  - SELL requires ask to turn downward inside the zone.
  - If reaction is confirmed, the watcher can trigger `execute_market_order_for_setup()` immediately instead of waiting for the next full scan.
- Dual setup guard:
  - A dual signal can only execute one leg per realtime tick pass.
  - This prevents Option A and Option B from both becoming market orders from the same reaction tick.
- Active trade management:
  - During realtime waiting windows, `manage_active_trades(symbol, magic, {})` is called so profit-lock and SL protection can react faster between full scans.
- New CLI flags:
  - `--realtime-reaction`
  - `--tick-interval 1.0`
  - `--min-reaction-move 0.10`
- Example live loop:
  - `python -m src.scanner_worker --symbol XAUUSD --threshold 0.60 --loop --interval 5 --realtime-reaction --tick-interval 1.0 --min-reaction-move 0.10`

Important limitations:
- This is not "detect every new FVG/BPR every tick".
- It is "detect structure on closed candles, then watch live ticks for precise reaction entry".
- It improves timing and reduces missed reactions, but it does not guarantee profit or next-trade win.

Verification added:
- RED before implementation:
  - `python -m pytest tests\test_realtime_reaction_watcher.py -q`
  - Result: failed because `src.realtime_reaction_watcher` did not exist.
- Additional RED:
  - dual realtime pass initially executed both legs in one tick; fixed to one leg per signal per pass.
  - scanner helper initially did not exist; added `run_realtime_reaction_cycle()`.
- GREEN:
  - `python -m pytest tests\test_realtime_reaction_watcher.py tests\test_scanner_entry_decisions.py tests\test_scanner_market_orders.py -q`
  - Result: `31 passed`
- `python -m py_compile src\realtime_reaction_watcher.py src\scanner_worker.py`
  - Result: exit 0
- `python -m pytest -q`
  - Result: `287 passed, 35 warnings`

---

## Update Live Execution Governor 2026-06-10

Tambahan setelah user minta logika live dibuat lebih agresif. Catatan 2026-06-11:
daily profit/loss cap kemudian dimatikan atas instruksi user, jadi bagian daily governor
di bawah ini sekarang bersifat historical note, bukan behavior live execution saat ini.

- Scanner lama dihentikan sebelum perubahan live-risk dilanjutkan.
- `evaluate_live_entry_gate()` sekarang hard gate, bukan observer-only:
  - setup yang gagal entry gate tidak lagi lanjut execute,
  - blocked setup tetap masuk shadow tracking agar hasil reject vs accept bisa dipelajari.
- Live threshold dinaikkan otomatis saat `MT5_EXECUTE_TRADES=True`:
  - `ML_ACCEPT_THRESHOLD=0.60`,
  - `ML_LIVE_MIN_THRESHOLD=0.60`,
  - `ML_ENTRY_BASE_THRESHOLD=0.60`,
  - `ML_ENTRY_BUY_CONFIDENCE_BONUS=0.05`.
- Strategy policy live:
  - default blocklist: `Pivot,SND,Swapzone`,
  - `PIVOT_REJECTION` dinormalisasi terpisah dari legacy `Pivot`,
  - labeler baru menulis pivot-reaction sebagai `PIVOT_REJECTION`.
- Max concurrent trade guard diaktifkan lagi:
  - `MT5_MAX_CONCURRENT_TRADES=3`,
  - posisi dan pending order difilter berdasarkan `magic` secara lokal,
  - tidak lagi memakai unsupported `magic=` keyword pada `positions_get()` / `orders_get()`.
- Daily cap behavior updated 2026-06-11:
  - `MT5_DAILY_GOVERNOR_ENABLED=False`,
  - execution no longer blocks new pending/market orders after daily profit target,
  - execution no longer blocks new pending/market orders after daily loss or loss streak,
  - `src.live_risk_governor` remains only as a daily pip summary/monitor helper.
- Scanner singleton lock ditambahkan:
  - lock file: `data/scanner_<symbol>_<magic>.lock`,
  - mencegah dua scanner worker live berjalan bersamaan untuk symbol+magic yang sama.

Verification added:
- RED before implementation:
  - live entry gate tests failed because blocked entry gate was only observed, not enforced.
  - scanner lock test failed because no singleton lock existed.
  - max concurrent/daily governor tests failed because execution did not block those paths.
  - MT5 keyword regression tests failed because `magic=` was passed to `positions_get()` / `orders_get()`.
  - daily-governor fail-closed test was superseded on 2026-06-11 because daily caps no longer block execution.
  - pivot report normalization test failed because setup_type 2 with `strategy=Pivot` stayed under `Pivot`.
- GREEN targeted checks:
  - `python -m pytest tests\test_live_risk_governor.py tests\test_scanner_market_orders.py::test_active_trade_count_filters_magic_without_unsupported_mt5_kwarg tests\test_scanner_market_orders.py::test_prune_pending_orders_filters_magic_without_unsupported_mt5_kwarg tests\test_active_trade_management.py::test_manage_active_trades_filters_magic_without_unsupported_mt5_kwarg -q`
    - Result: `9 passed`
  - `python -m pytest tests\test_scanner_market_orders.py::test_pending_order_blocks_when_daily_governor_cannot_evaluate tests\test_scanner_market_orders.py::test_pending_order_blocks_after_daily_runner_target tests\test_live_risk_governor.py -q`
    - Result: `8 passed`
  - `python -m pytest tests\test_calibration_report.py::test_build_calibration_report_normalizes_pivot_rejection_strategy_name tests\test_calibration_report.py::test_build_calibration_report_groups_by_threshold_bucket_timeframe_and_source tests\test_calibration_report.py::test_build_calibration_report_uses_setup_type_strategy_fallback_when_strategy_missing tests\test_live_trade_policy.py -q`
    - Result: `7 passed`
- Operational verification:
  - `python -m pytest -q`
    - Result: `309 passed, 35 warnings`
  - `python -m py_compile src\scanner_worker.py src\execution.py src\live_trade_policy.py src\live_risk_governor.py src\entry_quality_gate.py src\realtime_reaction_watcher.py src\calibration_report.py src\labeler.py`
    - Result: exit 0
  - `python -m src.calibration_report`
    - Result: `scoring_mode=walk_forward`; report and calibrator regenerated.
    - Overall metrics: sample `5411`, winrate `37.31%`, expectancy `-0.18R`, max DD `1193.51R`.
  - `python -m src.rollout_status --threshold 0.60`
    - Result: `BLOCKED`
    - Metrics: sample `281`, winrate `46.98%`, expectancy `-0.03R`, max DD `32.52R`.
  - `Get-CimInstance Win32_Process -Filter "name = 'python.exe'"`
    - Result after verification commands completed: no active `python.exe` process detected.

Honest status:
- The live mechanics are safer and more selective than before.
- This is still not proven profitable enough for unrestricted live rollout because rollout remains `BLOCKED`.
- Daily 100-300 pips is only a performance ambition/monitoring reference, not a live execution stop cap and not a guaranteed outcome.

---

## Update No Daily Cap 2026-06-11

Per instruksi user, bot tidak lagi memakai daily max profit/loss sebagai penghenti entry.

Changed:
- `src.execution` no longer calls daily risk governor before pending or market execution.
- `.env` now sets `MT5_DAILY_GOVERNOR_ENABLED=False`.
- `src.live_risk_governor.evaluate_daily_risk()` no longer returns blocked decisions for:
  - runner target reached,
  - daily max loss reached,
  - daily consecutive loss streak reached.
- Existing guard yang masih aktif:
  - max concurrent trade,
  - hard entry quality gate,
  - live threshold minimum,
  - strategy blocklist,
  - scanner singleton lock,
  - broker duplicate order/position checks.

Verification:
- RED before implementation:
  - daily runner target test failed because entry was still blocked at `+305` pips.
  - daily loss/loss-streak test failed because `evaluate_daily_risk()` still returned blocked.
  - daily history unavailable test failed because execution still blocked on governor failure.
- GREEN after implementation:
  - `python -m pytest tests\test_live_risk_governor.py::test_daily_risk_keeps_trading_after_runner_target tests\test_live_risk_governor.py::test_daily_risk_keeps_trading_after_max_loss_or_loss_streak tests\test_scanner_market_orders.py::test_pending_order_continues_after_daily_runner_target tests\test_scanner_market_orders.py::test_pending_order_continues_when_daily_history_is_unavailable -q`
    - Result: `4 passed`
  - `python -m pytest tests\test_live_risk_governor.py tests\test_scanner_market_orders.py tests\test_active_trade_management.py -q`
    - Result: `24 passed`
  - `python -m py_compile src\execution.py src\live_risk_governor.py src\scanner_worker.py`
    - Result: exit 0
  - `python -m pytest -q`
    - Result: `309 passed, 35 warnings`
  - `python -m src.rollout_status --threshold 0.60`
    - Result: `BLOCKED`
    - Metrics: sample `281`, winrate `46.98%`, expectancy `-0.03R`, max DD `32.52R`.
  - `Get-CimInstance Win32_Process -Filter "name = 'python.exe'"`
    - Result: no active `python.exe` process detected after checks completed.

Honest status:
- No daily profit/loss cap remains in the live execution path.
- This increases continuity/aggressiveness, but it also removes an account-protection stop.
- Rollout metrics still do not prove the bot is profitable enough to call performance "masterpiece".
