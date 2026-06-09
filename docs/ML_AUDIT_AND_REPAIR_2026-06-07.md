# ML Audit And Repair - 2026-06-07

Dokumen ini mencatat pekerjaan audit dan repair yang sudah dilakukan pada project
`forex-smc-analyzer` sebelum sesi konteks ini dipotong. Isinya sengaja faktual:
angka di bawah berasal dari file project dan hasil command terakhir, bukan klaim profit.

## Status Singkat

- Machine Learning yang dipakai bukan dummy:
  - Model XGBoost aktif di `models/smc_xgb_classifier.joblib`.
  - Model LightGBM aktif di `models/smc_lgb_classifier.joblib`.
  - Inference memakai `predict_setup_probability()` untuk menghasilkan probabilitas setup.
  - Scanner memakai probabilitas ML untuk memfilter sinyal.
  - Feedback loop membaca history MT5 dan memasukkan hasil trade yang sudah close ke dataset.
- MT5 pernah berhasil initialize saat audit.
- `.env` terdeteksi mengaktifkan real execution dengan `MT5_EXECUTE_TRADES=True`.
- Karena real execution aktif, scanner/live loop tidak boleh dijalankan otomatis dari agent tanpa approval eksplisit.

## Bug Yang Ditemukan

### 1. CSV feedback bisa korup karena urutan kolom tidak disejajarkan

Masalah:

- `update_feedback_data()` append row baru ke `data/labeled_setups.csv`.
- Row baru dapat memiliki urutan kolom berbeda dari header CSV lama.
- Akibatnya nilai bisa bergeser kolom, termasuk label menjadi non-binary.

Dampak:

- Dataset training bisa tercemar.
- Model bisa belajar dari label yang salah.
- Confidence model bisa terlihat valid padahal input training rusak.

Perbaikan:

- `src/inference.py`
  - `update_feedback_data()` sekarang align row baru mengikuti header CSV yang sudah ada.
  - Field yang belum ada dibuat kosong sesuai struktur existing CSV.
  - Ditambahkan test khusus untuk memastikan urutan kolom tetap benar.

### 2. Feedback trade yang sudah recorded di `sent_signals.json` bisa tidak masuk CSV

Masalah:

- Ada trade MT5 yang sudah closed dan tercatat di `sent_signals.json`.
- Jika sebelumnya gagal append ke CSV atau row korup dibuang, feedback itu tidak otomatis dibackfill.

Dampak:

- Bot kehilangan data kemenangan/kekalahan real.
- Learning loop terlihat jalan, tapi sebagian real trade tidak dipakai untuk training.

Perbaikan:

- `src/inference.py`
  - Ditambahkan `_feedback_row_key()`.
  - Ditambahkan `_load_existing_feedback_keys()`.
  - `process_mt5_history_feedback()` sekarang bisa backfill trade closed yang sudah recorded tapi belum ada di CSV.
  - Backfill idempotent: rerun tidak menambah duplicate row.
  - Ditambahkan test untuk single ticket dan dual-option ticket.

### 3. Window training 1000 row memakai urutan CSV, bukan timestamp

Masalah:

- `model_trainer.py` membatasi dataset ke 1000 row terakhir.
- Sebelumnya ini raw CSV order.
- Labeler append data per timeframe, jadi row terakhir CSV belum tentu data paling baru secara waktu.

Dampak:

- Timeframe kecil seperti M15/M30 bisa ikut terbuang walaupun datanya lebih baru.
- Dataset training bias ke timeframe tertentu.

Perbaikan:

- `src/model_trainer.py`
  - Window 1000 row sekarang disortir berdasarkan timestamp dulu.
  - Jika timestamp tidak tersedia, fallback tetap memakai row order.
  - Ditambahkan test agar M15/M30 terbaru tetap masuk window.

## Repair Data Yang Sudah Dilakukan

- Menghapus 4 row korup dengan label non-binary dari `data/labeled_setups.csv`.
- Menjalankan feedback loop untuk re-add feedback real yang valid.
- Feedback loop sempat menambah real closed-trade rows dari MT5 history.
- Rerun terakhir `process_mt5_history_feedback()` menghasilkan `feedback_count=0`, artinya tidak ada feedback baru yang belum masuk pada saat itu.

## Status Dataset Terakhir

File: `data/labeled_setups.csv`

- Total row: `1000`
- Invalid label: `0`
- Label `0.0`: `642`
- Label `1.0`: `358`
- Timeframe distribution:
  - M15 / `15`: `3`
  - M30 / `30`: `2`
  - H1 / `60`: `415`
  - H4 / `240`: `487`
  - D1 / `1440`: `93`
- Time min: `2025-12-10 08:00:00`
- Time max: `2026-06-05 20:01:12`

Catatan penting:

- Dataset valid secara struktur label.
- Tetapi M15/M30 masih sangat kecil. Ini tidak ideal untuk confidence intraday kecil.
- Window 1000 terlalu sempit kalau semua timeframe dicampur dan H1/H4 mendominasi.

## Status Learning

File: `data/learning_status.json`

```json
{
  "new_trades_since_last_train": 0,
  "last_train_time": "2026-06-07 10:16:06"
}
```

Artinya:

- Pada status terakhir, tidak ada trade baru yang menunggu retrain.
- Retrain terakhir terjadi pada `2026-06-07 10:16:06`.

## Status Model Terakhir

File model:

- `models/smc_xgb_classifier.joblib`
  - Last write: `2026-06-06 05:55:26`
- `models/smc_lgb_classifier.joblib`
  - Last write: `2026-06-06 05:55:26`

Saat retrain setelah repair:

- Challenger model dilatih ulang.
- Champion-vs-challenger gate membandingkan model baru vs model lama.
- Challenger ditolak karena akurasi test lebih rendah dari champion.
- File model tidak dioverwrite.

Ini behavior yang benar:

- Bot tidak asal mengganti model hanya karena ada data baru.
- Model lama tetap aktif jika model baru lebih buruk di validasi.

## Test Yang Sudah Lulus

Command terakhir:

```powershell
& 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer\.venv\Scripts\python.exe' -m pytest -q
```

Hasil:

```text
83 passed, 32 warnings
```

Test yang ditambahkan/terkait:

- `tests/test_inference.py`
  - `test_update_feedback_data_aligns_to_existing_csv_column_order`
  - `test_process_mt5_history_feedback_backfills_recorded_trade_missing_from_csv`
  - `test_process_mt5_history_feedback_backfills_recorded_dual_option_missing_from_csv`
- `tests/test_model_trainer.py`
  - `test_window_limit_keeps_latest_setups_by_timestamp_not_csv_order`

## Jawaban Teknis Soal "Sudah Perfect?"

Tidak. Bot tidak boleh disebut perfect.

Yang benar:

- ML sudah real dan feedback loop sudah jauh lebih aman dibanding sebelumnya.
- Dataset sudah tidak korup pada cek terakhir.
- Retrain punya gate agar model baru tidak otomatis menggantikan champion jika lebih buruk.
- Sistem lebih siap diuji di market dibanding sebelum repair.

Yang belum bisa dijamin:

- Next trade profit.
- Konsisten profit tanpa periode drawdown.
- Confidence 50% pasti menghasilkan edge.
- Akurasi backtest sama dengan performa live.

Market-ready yang realistis berarti:

- Real execution aktif hanya kalau risk management jelas.
- Semua trade dan sinyal filtered tercatat.
- Performa diukur pakai winrate, expectancy, profit factor, drawdown, dan calibration.
- Model naik kelas hanya kalau data real membuktikan performanya lebih baik.

## Risiko Yang Masih Ada

- Distribusi timeframe tidak seimbang. H1/H4 mendominasi.
- M15/M30 terlalu sedikit untuk dipercaya sebagai sample kuat.
- Confidence belum dikalibrasi per bucket probabilitas.
- Sinyal di bawah threshold belum otomatis dipantau sebagai shadow sample pada saat audit awal. Status ini sudah berubah setelah Phase 1; lihat `docs/ML_PHASE_1_SHADOW_TRACKING_2026-06-07.md`.
- Dataset window `1000` valid untuk mencegah regime lama mendominasi, tapi terlalu kecil untuk multi-timeframe jika tidak balanced.
- Accuracy saja belum cukup. Trading harus dinilai dengan expectancy dan drawdown.

## Prinsip Operasional Aman

- Jangan menaikkan lot karena confidence tinggi tanpa batas risiko.
- Jangan auto-promote model hanya karena satu atau dua trade menang.
- Jangan mencampur data virtual/shadow dengan real trade dalam bobot yang sama.
- Jangan menganggap threshold `0.50` artinya peluang profit nyata 50% sebelum calibration diuji.
- Jangan menjalankan live scanner dari agent tanpa approval eksplisit karena real execution aktif.
