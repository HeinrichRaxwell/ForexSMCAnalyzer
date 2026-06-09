# ML Trading Bot — Confidence Predictive & Expectancy Repair

**Tanggal:** 2026-06-09
**Status:** Approved (user: "gas aja langsung")
**Instrumen:** XAUUSDm (MT5)

## Konteks & Masalah

Walk-forward OOF scoring (sudah terpasang sebagai baseline jujur) mengungkap:

- Winrate jujur 42–48% (bukan 94–100% seperti laporan in-sample yang bocor).
- **Confidence tidak predictive**: pada threshold 0.70–0.80 expectancy justru NEGATIF (−0.03R). Confidence tinggi tidak berarti peluang menang tinggi.
- Source `real` (trade dieksekusi nyata): PF 0.97, expectancy −0.02R — rugi pelan.
- `PIVOT_REJECTION`: winrate 23%, PF 0.61 — rugi konsisten.
- **Spread/slippage tidak dimodelkan** di `simulate_trade` — TP/SL diasumsikan kena di harga persis, sehingga label & expectancy optimistis.

**Tujuan:** confidence jadi predictive, expectancy naik, feedback loop benar-benar bikin model makin pintar.

## Prinsip Urutan (Wajib)

Urutan mempengaruhi hasil:
1. **Cost modeling** → mengubah `pnl_relative` (R aktual)
2. **Feature engineering** → hubungan fitur ke outcome
3. **Rework PIVOT_REJECTION** → menghasilkan setup + fitur baru
4. **Confidence calibration** → numpang di atas label & OOF yang jujur
5. **Re-score walk-forward** → ukur hasil sebenarnya

Mengerjakan kalibrasi sebelum cost/fitur = mengalibrasi ke angka yang akan berubah.

---

## Workstream 1: Cost Modeling

**File:** `src/labeler.py` (`simulate_trade`, pemanggilan label)

Di simulasi first-touch, spread tidak mengubah mana yang kena duluan (TP vs SL tetap level harga). Yang berubah adalah `pnl_relative` (R aktual), yang dipakai menghitung expectancy & PF.

- `cost_R = (spread_usd + slippage_usd) / risk_pips` (risk_pips = |entry − sl| dalam USD)
- Win: `pnl_relative = base_win_R − cost_R`
- Loss: `pnl_relative = base_loss_R − cost_R` (loss sedikit lebih dalam)
- Default `ML_SPREAD_USD=0.30` (env-configurable), `ML_SLIPPAGE_USD=0.0` default.
- Label biner (0/1) tetap dari first-touch; cost hanya menyesuaikan R.

**Efek:** winrate biner sama, expectancy turun ke angka jujur.

**TDD:** `cost_R` benar untuk risk_pips besar/kecil; win & loss R berkurang sebesar cost_R; risk_pips=0 tidak membagi nol.

---

## Workstream 2: Feature Engineering

**File:** `src/labeler.py` (blok pembentukan fitur), `src/model_trainer.py` (NON_FEATURE_COLUMNS bila perlu)

6 fitur dengan hipotesis eksplisit kenapa membedakan TP vs SL:

| Fitur | Hipotesis |
|---|---|
| `rr_ratio` = (tp−entry)/(entry−sl) | RR eksplisit; kemungkinan paling predictive, belum ada |
| `atr_percentile` | Regime volatilitas (ATR sekarang vs trailing window) |
| `confluence_score` | Jumlah elemen SMC numpuk di entry (FVG+OB+pivot) |
| `body_to_range_ratio` | Konviksi/momentum candle sinyal |
| `dist_to_recent_swing` (norm ATR) | Ruang menuju target |
| `htf_trend_aligned` | Searah trend HTF; dipastikan ada di semua setup |

Fitur ditulis ke setiap blok setup (FVG/OB/BPR/Swap/Pivot) secara konsisten. Fitur yang tidak relevan untuk satu setup diberi nilai default yang jelas (0.0), bukan NaN diam-diam.

**TDD:** tiap fitur dihitung benar pada kasus contoh; tidak ada NaN bocor; kolom konsisten di semua setup.

---

## Workstream 3: Rework PIVOT_REJECTION → Reaction-Based Order Routing

**File:** `src/labeler.py` (blok pivot), reuse `src/rejection_detector.py`, fitur baru `order_type` + `reaction_strength`

State machine membaca reaksi market di Key Level / Support / Resistance, lalu memilih order type:

- **CONFIRMED_REJECTION** (touch + wick ≥50% + candle close balik arah) → **MARKET order** (entry harga market saat itu)
- **APPROACHING** (harga belum sampai level) → **PENDING limit** di level
- **BREAKOUT_MOMENTUM** (nembus level + momentum, retest trigger) → **STOP order** di atas/bawah trigger

Rule-based state machine (interpretable, testable). Setup tidak dibuang — dipinterin. Model + kalibrasi yang menentukan reaksi mana yang benar-benar jalan.

Tambah fitur:
- `order_type` (0=market, 1=limit, 2=stop) — kategorikal
- `reaction_strength` (rasio wick penolakan / range candle di level)

Entry/SL/TP disesuaikan per order_type:
- MARKET: entry = harga konfirmasi, SL di luar wick, TP ke target terdekat
- LIMIT: entry = level, SL di luar level, TP ke target
- STOP: entry = trigger breakout, SL di sisi berlawanan, TP momentum

**TDD:** klasifikasi reaksi → order_type benar untuk tiap kondisi; entry/SL/TP konsisten dengan order_type; `reaction_strength` terhitung benar.

---

## Workstream 4: Confidence Calibration (Isotonic)

**File:** baru `src/calibrator.py`, integrasi `src/calibration_report.py` & `src/inference.py`

- Fit `IsotonicRegression` memetakan `raw_ensemble_prob → winrate empiris`.
- **Dilatih dari data walk-forward OOF** (sudah out-of-sample → tidak bocor).
- Simpan calibrator ke `models/confidence_calibrator.joblib`.
- Inference: `calibrated_conf = calibrator.predict(raw_prob)`. Confidence 0.80 → ~80% winrate historis.
- Threshold selection (`recommend_threshold`) memakai confidence terkalibrasi.
- Fallback: jika calibrator tidak ada / data < minimum, pakai raw prob + warning.

**TDD:** kalibrasi monotonik non-turun; pemetaan masuk akal pada titik contoh; fallback aman saat data kurang.

---

## Verifikasi Akhir

- Re-run walk-forward calibration report (`scoring_mode=walk_forward`).
- Bandingkan sebelum/sesudah: expectancy per threshold, PF per source, calibration curve (confidence vs winrate aktual).
- Target kelayakan: confidence terkalibrasi monotonik terhadap winrate aktual; setidaknya satu threshold dengan expectancy > 0 setelah cost.
- Semua unit test hijau.

## Catatan Risiko

- Cost modeling akan **menurunkan** expectancy yang terlihat — ini disengaja (jujur), bukan regresi.
- Jika setelah semua ini edge tetap tipis/negatif, itu temuan valid: strategi dasar perlu revisi, bukan ditutupi angka. Akan dilaporkan apa adanya sebelum forward test demo $100.
