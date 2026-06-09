# PROGRESS — ML Confidence Predictive & Expectancy Repair

> **Tujuan dokumen ini:** Catatan status hidup. Kalau memori/sesi ke-reset, BACA FILE INI DULU
> sebelum lanjut. Berisi apa yang sudah selesai, apa yang belum, keputusan penting, dan cara melanjutkan.

**Branch:** `ml-confidence-predictive`
**Spec:** `docs/superpowers/specs/2026-06-09-ml-confidence-predictive-design.md`
**Plan:** `docs/superpowers/plans/2026-06-09-ml-confidence-predictive.md`
**Tanggal mulai:** 2026-06-09

---

## KONTEKS BESAR (kenapa kerjaan ini ada)

Bot ML trading XAUUSD (SMC + FLoOP volume + ensemble XGB/LGBM + feedback loop).
User mau forward test 1 bulan di demo $100, lalu akun real bulan depan kalau profit.

**Temuan audit awal (KRITIS):**
1. Calibration report lama **bocor (data leakage)** — model di-score pakai data yang sama
   dengan data training. Winrate keliatan 94–100% padahal PALSU.
2. Sudah diperbaiki di sesi sebelumnya: **walk-forward OOF scoring** (commit `00fd52a`).
   Angka jujur: winrate 42–48%, dan **confidence TIDAK predictive** — di threshold 0.70–0.80
   expectancy malah NEGATIF (−0.03R). Source `real` (trade nyata): PF 0.97 (rugi pelan).
3. **Spread/slippage tidak dimodelkan** — label & expectancy optimistis.

**Visi user:** confidence harus nyambung sama winrate asli; bot belajar dari menang/kalah;
makin lama makin pintar. Shadow = paper-trade sinyal confidence rendah buat ngumpulin data.

---

## KEPUTUSAN PENTING (jangan dilanggar)

1. **Cost modeling masuk.** Default `ML_SPREAD_USD=0.30`, `ML_SLIPPAGE_USD=0.0` (env-configurable).
2. **Calibration: Isotonic regression**, dilatih dari walk-forward OOF (out-of-sample, tidak bocor).
3. **PIVOT_REJECTION di-rework** jadi reaction-based order routing (market/limit/stop).
   Ini SATU-SATUNYA strategi yang logikanya diubah — atas permintaan user.
4. **SEMUA strategi lain UTUH** (FVG, OB, BPR, Swap, Indecision Candle, Supply/Demand).
   Workstream cost + fitur + kalibrasi HANYA menambah kolom/pengukuran, TIDAK mengubah
   deteksi atau entry/SL/TP strategi-strategi ini. User sudah konfirmasi 2x.
5. **Jujur soal hasil.** Cost modeling SENGAJA menurunkan expectancy yang terlihat.
   Kalau setelah semua ini edge tetap tipis/negatif → lapor apa adanya, jangan ditutupi.
   Tidak ada janji "pasti profit" — forward test demo adalah hakimnya.

---

## STATUS TASK

Legend: ✅ selesai & committed | 🔄 sedang dikerjakan | ⬜ belum

### Workstream 1: Cost Modeling
- ✅ **Task 1 — `compute_cost_r` helper** (commit `5f5187e`)
  - File: `src/labeler.py` (+ `_read_float_env`, `get_spread_usd`, `get_slippage_usd`)
  - `cost_R = (spread_usd + slippage_usd) / abs(risk_pips)`; risk 0 → return 0.0 (no div-by-zero).
  - Test: `tests/test_cost_modeling.py` (5 test, hijau).
- ✅ **Task 2 — `compute_pnl_relative`** (commit `5f5187e`)
  - File: `src/labeler.py`. Win → `RR − cost_R`; Loss → `−1.0 − cost_R`; risk 0 → 0.0.
  - Test: `tests/test_cost_modeling.py` (total 8 test, hijau).
- ✅ **Task 3 — pnl_relative cost-adjusted di labeler** (commit terbaru)
  - PENTING: ternyata `pnl_relative` dihitung di SATU tempat terpusat (dulu line ~752-763),
    BUKAN tersebar di tiap blok. Jadi cukup ganti blok terpusat itu pakai `compute_pnl_relative`.
    Lebih bersih dari rencana awal & strategi makin aman (tidak menyentuh tiap blok append).
  - Test: `tests/test_labeler_pnl_integration.py` — pakai DATA HISTORIS ASLI
    (`data/historical_xauusdm_15.csv`) supaya benar-benar menghasilkan setup (bukan SKIP).
    4 test hijau: produces_setups, has_pnl_relative_column, win<raw_rr, loss<−1.

### Workstream 2: Feature Engineering
- 🔄 **Task 4 — modul `src/setup_features.py`** (SEDANG DIKERJAKAN)
  - 6 fungsi fitur murni + testable: `rr_ratio`, `atr_percentile`, `body_to_range_ratio`,
    `dist_to_recent_swing_norm`, `htf_trend_aligned`, `confluence_score`.
  - Test: `tests/test_setup_features.py`.
- ⬜ **Task 5 — integrasi 6 fitur ke labeler** (tiap blok setup, default jelas, tanpa NaN)

### Workstream 3: Rework PIVOT_REJECTION → Reaction-Based Order Routing
- ⬜ **Task 6 — modul `src/reaction_router.py`** (`classify_reaction`, konstanta ORDER_*/STATE_*, `reaction_strength`)
- ⬜ **Task 7 — `compute_levels` per order_type** (market/limit/stop → entry/SL/TP)
- ⬜ **Task 8 — pakai reaction_router di blok pivot** (+ fitur `order_type`, `reaction_strength`; default di blok non-pivot)

### Workstream 4: Confidence Calibration (Isotonic)
- ⬜ **Task 9 — modul `src/calibrator.py`** (`fit_calibrator`/`apply_calibrator`/`save`/`load`; fallback identity)
- ⬜ **Task 10 — fit calibrator dari OOF** (`fit_calibrator_from_scored` di `calibration_report.py`; simpan `models/confidence_calibrator.joblib`)
- ⬜ **Task 11 — apply calibrator di inference** (`apply_confidence_calibration` di `src/inference.py`, cache, fallback identity)

### Verifikasi Akhir
- ⬜ **Task 12 — re-label + re-train + re-score walk-forward + bandingkan**
  - Backup `data/labeled_setups.csv` dulu.
  - Re-label historis → re-train (`python -m src.model_trainer`) → re-score (`python -m src.calibration_report`).
  - Bandingkan expectancy per threshold (target: ≥1 threshold >0 SETELAH cost), calibration curve
    (avg_confidence vs winrate aktual makin sejajar), PF per source.
  - Lapor apa adanya sebelum forward test demo $100.

---

## CARA MELANJUTKAN (kalau memori reset)

1. `git log --oneline -8` — cek commit terakhir, cocokkan dengan STATUS TASK di atas.
2. `git status` — pastikan branch `ml-confidence-predictive`.
3. Baca task berikutnya yang ⬜/🔄 di plan: `docs/superpowers/plans/2026-06-09-ml-confidence-predictive.md`.
4. Ikuti TDD tiap task: tulis test → run (RED) → implement → run (GREEN) → commit.
5. Update file PROGRESS ini setiap selesai 1 task (ubah ⬜→✅ + nomor commit).
6. Jalankan dari root repo. Test pakai venv: `python -m pytest <path> -v`.

---

## CATATAN TEKNIS

- Environment: Windows, Python 3.10.11, venv di `.venv/`. pytest 9.0.3.
- `.env` berisi kredensial (MT5/Telegram) — SUDAH di-gitignore, JANGAN di-commit.
- Git memunculkan warning "LF will be replaced by CRLF" — normal di Windows, abaikan.
- Labeler memanggil KNN + volume clusters → integration test agak lambat (~2-3 detik), wajar.
- Setelah semua task: gunakan `superpowers:finishing-a-development-branch` untuk menutup branch.
