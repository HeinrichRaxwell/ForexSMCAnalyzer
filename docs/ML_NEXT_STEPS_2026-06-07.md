# ML Next Steps - 2026-06-07

Dokumen ini adalah rencana lanjutan untuk membuat feedback loop lebih siap tempur.
Tujuannya bukan menjanjikan profit pasti, tapi membuat bot belajar dari real market
dengan kontrol risiko, data yang bersih, dan evaluasi yang bisa dipercaya.

## Target Utama

Bot perlu bisa:

- Execute hanya sinyal yang lolos threshold live.
- Tetap memantau sinyal yang tidak lolos threshold sebagai shadow signal.
- Menilai apakah shadow signal akhirnya TP atau SL secara virtual.
- Memakai hasil real trade dan shadow trade untuk memperbaiki model dengan bobot berbeda.
- Mengukur apakah confidence model benar-benar sesuai winrate nyata.

## Kenapa Shadow Tracking Penting

Saat threshold live `0.50`, sinyal dengan confidence `0.00` sampai `<0.50` tidak dieksekusi.
Saat ini, sinyal seperti itu bisa hilang sebagai data belajar.

Padahal sinyal filtered tetap berharga:

- Jika banyak sinyal `0.40-0.49` ternyata TP, model mungkin terlalu pesimis.
- Jika banyak sinyal `0.50-0.60` ternyata SL, threshold 50% mungkin terlalu longgar.
- Jika bucket tertentu menang di timeframe tertentu, threshold harus adaptif per kondisi, bukan satu angka global.

## Pendekatan Yang Direkomendasikan

### Approach A - Recommended: Shadow dataset terpisah

Buat file baru:

- `data/shadow_signals.json`
- `data/shadow_labeled_setups.csv`

Flow:

1. Scanner menemukan setup.
2. Jika confidence >= live threshold, setup tetap masuk jalur eksekusi normal.
3. Jika confidence < live threshold, setup disimpan sebagai shadow signal. Default Phase 1 adalah mulai dari `0.00`; bisa dinaikkan lewat `ML_SHADOW_MIN_CONFIDENCE` kalau terlalu banyak noise.
4. Worker/feedback process mengecek candle berikutnya untuk menentukan apakah virtual TP atau SL tersentuh dulu.
5. Hasilnya masuk `shadow_labeled_setups.csv`, bukan langsung dicampur penuh ke real dataset.

Kelebihan:

- Aman untuk live account karena low-confidence signal tidak dieksekusi.
- Tetap belajar dari sinyal yang tadinya ditolak.
- Real data dan virtual data tidak tercampur sembarangan.

Kekurangan:

- Butuh logic monitor virtual outcome.
- Butuh sample weighting agar virtual data tidak terlalu dominan.

### Approach B - Campur langsung ke labeled_setups.csv dengan sample_source

Tambahkan kolom:

- `sample_source`: `real`, `historical`, `shadow`
- `sample_weight`: angka bobot training

Kelebihan:

- Satu dataset lebih simpel.
- Trainer langsung bisa pakai semua sample.

Kekurangan:

- Lebih rawan salah bobot.
- Kalau bug, virtual label bisa mencemari real feedback.

### Approach C - Real trade only

Tetap hanya belajar dari trade yang benar-benar dieksekusi.

Kelebihan:

- Data paling nyata.
- Risiko label virtual lebih kecil.

Kekurangan:

- Learning lambat.
- Bot tidak belajar dari sinyal yang difilter.
- Threshold 50% sulit dikalibrasi karena lower bucket tidak punya outcome.

Rekomendasi:

- Pakai Approach A dulu.
- Setelah stabil dan test lulus, trainer bisa dibuat source-aware untuk memakai shadow data dengan bobot rendah.

## Desain Shadow Tracking

### Config yang disarankan

Tambahkan config/env:

- `ML_ACCEPT_THRESHOLD=0.50`
- `ML_SHADOW_MIN_CONFIDENCE=0.00`
- `ML_SHADOW_SAMPLE_WEIGHT=0.35`
- `ML_MAX_TRAINING_ROWS=5000`

Makna:

- Confidence >= `0.50`: boleh masuk jalur live execution jika semua filter lain lolos.
- Confidence `0.00-0.49`: tidak dieksekusi, tapi dipantau.
- Jika nanti terlalu banyak noise, set `ML_SHADOW_MIN_CONFIDENCE` ke angka lebih tinggi seperti `0.10` atau `0.30`.

### Data yang harus disimpan untuk shadow signal

Minimal fields:

- `signal_id`
- `time`
- `symbol`
- `timeframe`
- `direction`
- `entry_price`
- `sl_price`
- `tp_price`
- `confidence`
- `features`
- `status`: `open`, `resolved`, `expired`
- `result`: kosong, `tp`, `sl`, `expired`
- `label`: kosong, `1`, `0`
- `created_at`
- `resolved_at`
- `source`: `shadow`

### Outcome virtual

Untuk BUY:

- TP jika high candle masa depan menyentuh `tp_price`.
- SL jika low candle masa depan menyentuh `sl_price`.

Untuk SELL:

- TP jika low candle masa depan menyentuh `tp_price`.
- SL jika high candle masa depan menyentuh `sl_price`.

Jika TP dan SL tersentuh dalam candle yang sama:

- Conservative rule: anggap SL dulu kecuali ada data tick yang membuktikan urutan sebaliknya.

Jika tidak tersentuh sampai batas waktu:

- Tandai `expired`.
- Jangan dipakai sebagai label menang/kalah sampai ada aturan exit yang jelas.

## Dataset Memory: 1000, 5000, atau 10000?

Jawaban praktis:

- `1000` terlalu kecil untuk bot multi-timeframe saat H1/H4 mendominasi.
- `5000` adalah target awal yang lebih masuk akal.
- `10000+` boleh nanti, tapi harus balanced dan diuji agar tidak membawa regime lama terlalu berat.

Rekomendasi sekarang:

- Naikkan dari `1000` ke `5000`.
- Jangan pakai raw latest 5000 saja.
- Pakai balanced recency window:
  - Simpan data terbaru per timeframe.
  - Pastikan M15/M30 tidak hilang.
  - Batasi dominasi H1/H4.

Contoh alokasi awal untuk max 5000:

- M15: sampai 1000 sample
- M30: sampai 1000 sample
- H1: sampai 1200 sample
- H4: sampai 1200 sample
- D1: sampai 600 sample

Kalau satu timeframe belum punya cukup sample, sisa kuota boleh diisi timeframe lain, tapi tetap dengan cap agar tidak satu timeframe mendominasi.

## Cara Confidence Harus Dinaikkan

Jangan menaikkan confidence satu sinyal secara manual sampai 100% hanya karena beberapa win.

Yang benar:

- Model retrain dari dataset outcome.
- Confidence naik kalau pola fitur yang mirip terbukti sering menang dalam banyak sample.
- Kenaikan harus lewat validation gate.
- Confidence 100% hampir tidak sehat di market karena berarti model terlalu yakin; itu sering tanda overfit.

Yang perlu dibuat:

- Calibration report per bucket:
  - `0.30-0.40`
  - `0.40-0.50`
  - `0.50-0.60`
  - `0.60-0.70`
  - `0.70-0.80`
  - `0.80-0.90`
  - `0.90-1.00`
- Untuk tiap bucket tampilkan:
  - jumlah sample
  - actual winrate
  - average R
  - profit factor
  - max drawdown

Kalau bucket `0.40-0.50` punya actual winrate dan expectancy bagus, threshold bisa dipertimbangkan turun atau model dikalibrasi ulang.
Kalau bucket `0.50-0.60` jelek, threshold 50% harus dinaikkan.

## Metrics Yang Harus Jadi Patokan

Accuracy tidak cukup untuk trading.

Tambahkan report:

- Winrate
- Average win R
- Average loss R
- Expectancy R
- Profit factor
- Max drawdown
- Consecutive losses
- Trade count
- Calibration by confidence bucket
- Performance per timeframe
- Performance per setup type
- Performance per session/hour

Rumus penting:

```text
expectancy_R = (winrate * avg_win_R) - ((1 - winrate) * avg_loss_R)
```

Bot bisa winrate di bawah 50% tapi profit jika avg win jauh lebih besar dari avg loss.
Bot juga bisa winrate tinggi tapi rugi jika loss jauh lebih besar dari win.

## Urutan Implementasi Disarankan

### Phase 1 - Safety and observability

- Tambahkan config threshold dan shadow threshold.
- Tambahkan file store untuk shadow signals.
- Tambahkan test bahwa sinyal `0.00-0.49` masuk shadow, bukan execution.
- Tambahkan test bahwa sinyal >= 0.50 tetap masuk jalur normal.

Status implementasi:

- Selesai di Phase 1.
- Shadow signal store dibuat di `data/shadow_signals.json`.
- Default live threshold sekarang `0.50` lewat `ML_ACCEPT_THRESHOLD`/fallback CLI.
- Default shadow minimum sekarang `0.00`, jadi sinyal di bawah 30% juga ikut dipantau.
- Phase 1 belum menentukan virtual TP/SL; itu masuk Phase 2.

### Phase 2 - Shadow outcome resolver

- Tambahkan resolver yang membaca historical candles setelah signal time.
- Label TP/SL/expired secara virtual.
- Simpan hasil ke `data/shadow_labeled_setups.csv`.
- Test BUY/SELL TP dulu, SL dulu, same-candle conservative SL, dan expired.

### Phase 3 - Source-aware training

- Trainer membaca real dataset plus shadow dataset.
- Real trade bobot `1.0`.
- Historical/backtest bobot sesuai logic existing.
- Shadow bobot rendah, awal `0.35`.
- Champion-vs-challenger gate tetap wajib.

### Phase 4 - Calibration and performance report

- Tambahkan report CLI/script untuk bucket confidence.
- Report harus bisa jalan tanpa live trading.
- Gunakan report untuk memutuskan threshold, bukan feeling.

### Phase 5 - Paper/live rollout

- Jalankan shadow + paper observation dulu.
- Jika metrics stabil, baru live dengan lot kecil.
- Naikkan size hanya jika drawdown dan expectancy real sudah terbukti.

## Acceptance Criteria

Sebelum disebut siap live lebih serius:

- Semua test lulus.
- Dataset label valid.
- Feedback loop idempotent.
- Shadow signal tidak pernah dieksekusi.
- Shadow outcome resolver tidak membuat duplicate.
- Trainer tidak membiarkan shadow data mengalahkan real trade secara bobot.
- Report menunjukkan minimal:
  - per-bucket confidence
  - per-timeframe performance
  - expectancy
  - drawdown
- Threshold live dipilih berdasarkan report.

## Jawaban Langsung Untuk Kondisi Sekarang

- Threshold `0.50` boleh dipakai sebagai awal konservatif, tapi belum tentu optimal.
- Semua sinyal di bawah threshold live, termasuk `<0.30`, sebaiknya dipantau sebagai shadow, bukan dieksekusi.
- Memory `1000` kurang ideal untuk multi-timeframe.
- Naik ke `5000` lebih masuk akal, asal balanced.
- Jangan langsung `10000+` sebelum ada balancing dan report, karena bisa membawa noise regime lama.
- Jangan berharap confidence mencapai 100%; lebih penting confidence calibrated dan expectancy positif.

## Batasan Yang Harus Diingat

Tidak ada bot yang bisa menjamin next trade profit.
Target realistis adalah membuat sistem yang:

- Lebih sering mengambil setup yang expectancy-nya positif.
- Cepat tahu kalau edge hilang.
- Tidak menghancurkan akun saat loss streak.
- Belajar dari win/loss real dan shadow tanpa mencemari model.
