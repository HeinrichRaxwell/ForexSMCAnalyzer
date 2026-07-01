# ML Phase 4 Calibration Report - 2026-06-07

Phase 4 menambahkan report offline untuk mengukur apakah confidence model benar-benar selaras dengan outcome historis.

Tujuan utamanya: threshold live tidak dipilih dari feeling, tetapi dari winrate, expectancy, sample count, dan drawdown per confidence bucket.

## Yang Diimplementasikan

- Modul baru:
  - `src/calibration_report.py`
- Test baru:
  - `tests/test_calibration_report.py`
- Output report:
  - `data/calibration_report.json`

Report membaca:

- `data/labeled_setups.csv`
- `data/shadow_labeled_setups.csv`
- model aktif di `models/smc_xgb_classifier.joblib`
- model LightGBM jika tersedia di `models/smc_lgb_classifier.joblib`

Report ini offline. Ia tidak membuka MT5, tidak menjalankan scanner, dan tidak mengirim order.

## Metrik Yang Dihitung

- Overall sample count
- Win count dan loss count
- Winrate
- Expectancy dalam R
- Max drawdown dalam R
- Profit factor
- Max consecutive losses
- Average confidence
- Average win R dan average loss R
- Breakdown per:
  - threshold
  - confidence bucket
  - timeframe
  - hour
  - session/killzone
  - setup type
  - direction
  - source (`real` vs `shadow`)

## Rekomendasi Threshold Otomatis

Report memilih threshold paling longgar yang masih memenuhi rule konservatif:

- sample count minimal `50`
- expectancy minimal `1.0R`
- max drawdown maksimal `3.0R`

Hasil report saat ini:

```text
recommended threshold: 0.50
sample count: 328
expectancy: 1.54R
max drawdown: 2.0R
profit factor: 36.56
max consecutive losses: 2
reason: lowest_threshold_meeting_rules
```

Artinya threshold `0.50` masih masuk rule Phase 4 saat ini.

## Angka Threshold Saat Ini

```text
Threshold  Samples  Winrate   Expectancy  Max DD
0.30       473      75.05%    1.05R       7.0R
0.40       387      89.15%    1.44R       4.0R
0.50       328      95.12%    1.54R       2.0R
0.60       273      98.53%    1.61R       1.0R
0.70       190      98.42%    1.46R       1.0R
0.80       116      100.00%   1.44R       0.0R
```

Interpretasi:

- `0.30` dan `0.40` terlalu longgar untuk live karena drawdown historis lebih besar.
- `0.50` adalah pilihan paling longgar yang masih memenuhi rule konservatif.
- `0.60` lebih ketat dan metriknya bagus, tetapi jumlah trade yang lolos lebih sedikit.
- `0.80` terlihat sempurna, tetapi sample lebih kecil. Jangan mengejar 100% winrate sebagai target utama.

## Angka Confidence Bucket Saat Ini

```text
Bucket      Samples  Winrate   Expectancy  Max DD
0.00-0.30   547      3.11%     -0.92R      505.89R
0.30-0.40   86       11.63%    -0.69R      64.93R
0.40-0.50   59       55.93%    0.91R       7.0R
0.50-0.60   55       78.18%    1.16R       3.0R
0.60-0.70   83       98.80%    1.96R       1.0R
0.70-0.80   74       95.95%    1.48R       1.0R
0.80-0.90   95       100.00%   1.49R       0.0R
0.90-1.00   21       100.00%   1.22R       0.0R
```

Interpretasi:

- Bucket `<0.50` belum layak dieksekusi live berdasarkan report saat ini.
- Bucket `0.40-0.50` punya expectancy sedikit positif, tetapi drawdown terlalu besar.
- Bucket `0.50-0.60` mulai masuk akal, tetapi drawdown pas di batas.
- Bucket `0.60+` terlihat lebih stabil di data historis.

## Source Breakdown Saat Ini

```text
Source  Samples  Winrate   Expectancy  Max DD
real    1000     35.80%    -0.01R      81.38R
shadow  20       70.00%    0.38R       3.0R
```

Shadow baru `20` sample, jadi belum cukup untuk dijadikan dasar agresif menurunkan threshold.

## Manual PowerShell

Dari PowerShell:

```powershell
cd 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer'
```

Generate calibration report:

```powershell
& '.\.venv\Scripts\python.exe' -m src.calibration_report
```

Cek output:

```powershell
Get-Content .\data\calibration_report.json
```

Run test:

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests\test_calibration_report.py -q
& '.\.venv\Scripts\python.exe' -m pytest -q
```

## Batasan

- Ini bukan jaminan next trade profit.
- Report ini masih berbasis historical/scored dataset, bukan forward live sample murni.
- Hasil bagus pada threshold tinggi bisa overfit jika sample terlalu kecil.
- Phase 5 masih diperlukan untuk rollout policy: kapan scanner boleh live, kapan harus paper/one-shot, kapan threshold dinaikkan, dan kapan bot harus dihentikan sementara.
