# PineScript Translation Audit - 2026-06-07

Audit ini membandingkan file di `PineScripts/` dengan modul Python yang ada di project.
Tujuannya bukan klaim profit guaranteed, tapi memastikan formula yang dipakai bot tidak salah port.

## Ringkasan Verdict

| PineScript | Python terkait | Status |
| --- | --- | --- |
| `Floop PRO.txt` | `src/indicators/floop.py`, `src/indicators/pivots.py` | Core FLoOP + pivot classic sudah diperbaiki mendekati Pine formula. |
| `AI-SuperTrend (KNN Machine Learning).txt` | `src/indicators/knn_classifier.py` | Core KNN probability engine sudah ter-port besar, tapi bukan full visual/signal-state TradingView. |
| `Clusters Volume Profile [LuxAlgo].txt` | `src/indicators/volume_clusters.py` | Core K-Means cluster + POC numerik ada, visual/profile box Pine tidak dipakai. |
| `Machine Learning RSI AI Classification & Ranking (Zeiierman).txt` | Tidak ada port penuh khusus | Belum full port. KNN SuperTrend bukan ML RSI Zeiierman. |
| `Multi-Timeframe Volume Profiles.txt` | Tidak ada port penuh khusus | Belum full port. Python belum punya VAH/VAL/POC multi-HTF seperti Pine ini. |

## FLoOP Settings Yang Dicek

| Setting user | Status Python |
| --- | --- |
| Pivot timeframe Daily | OK, memakai data D1 untuk pivot lower timeframe. |
| Pivot formula Classic | OK, rumus PP/R1-R4/S1-S4 cocok dengan Pine classic. |
| Sensitivity 6 | OK. |
| ATR Length 14 | OK, sekarang memakai Wilder/RMA seperti `ta.atr(14)`. |
| ATR Multiplier 0.8 | OK. |
| HTF Timeframe H4 | OK, trend H4 dipakai sebagai HTF bias kalau tersedia di `tf_trends`. |
| HTF MA Length 90 | Input ada di Pine, tapi tidak dipakai lagi di formula FLoOP yang ada di file. Tidak ada logic Python yang perlu meniru ini. |
| Enable ADX Filter | OK. |
| ADX Length 14 | OK. |
| ADX Threshold 20 | OK. |
| Enable Choppiness Index Filter | OK. |
| Choppiness Length 14 | OK. |
| Choppiness Max 61.8 | OK. |
| Enable Signal Cooldown | OK, sudah disamakan dengan Pine. |
| Cooldown Bars 5 | OK. |
| Disable show signal when EMA aligned | OK sesuai request: Python memakai `ema_filter=False`. Pine default-nya true, tapi request kamu disabled. |

## Perbaikan FLoOP Yang Sudah Dilakukan

1. ATR FLoOP sebelumnya memakai rolling SMA. Pine `ta.atr()` memakai Wilder/RMA. Sudah diganti ke RMA seed SMA ala Pine.
2. Precompute trend MTF/HTF di `main.py`, `scanner_worker.py`, dan `labeler.py` sekarang juga memakai ATR RMA, bukan rolling SMA.
3. Cooldown sebelumnya bisa reset dari raw signal yang sebenarnya gagal gate ADX/CHOP/EMA. Sekarang cooldown reset hanya setelah long/short signal benar-benar diterima, sama seperti Pine.
4. MTF score sekarang hanya menghitung timeframe yang memang diskor Pine: M5, M15, H1, H4. M1 tidak dihitung, D1/M30 tidak ikut menambah score FLoOP.
5. Volatility percentile sekarang memakai nearest-rank style seperti Pine `ta.percentile_nearest_rank`, bukan linear quantile.
6. Choppiness zero-range sekarang return 50.0 seperti Pine guard, bukan membesar karena divide-by-tiny.

## Catatan Penting Per Script

### FLoOP PRO

FLoOP sekarang jauh lebih dekat ke Pine dibanding sebelumnya. Pivot daily classic sudah cocok. Bagian visual dashboard, label, alert, dan drawing pivot TradingView memang tidak relevan untuk bot Python.

Yang tetap perlu diingat: hasil 1:1 bar-by-bar dengan TradingView baru bisa dibuktikan kalau kita punya export OHLCV TradingView yang sama dan compare output signal per bar. Secara formula, mismatch besar yang kelihatan sudah diperbaiki.

### AI-SuperTrend KNN

Default parameter Python cocok dengan Pine utama:

- ATR period 10
- factor 2.0
- K neighbors 10
- sampling window 1000
- stride/momentum window 10
- RSI length 20
- MA length 20
- signal length 10
- CHOP length 14
- normalizing window 1000
- Minkowski p 2.0
- Gaussian shape 2.0
- PCA enabled

Python saat ini mengembalikan `prob_up` dan `prob_down`. Pine juga punya state signal visual (`last_stdir`, labels, plotshape, bar color). Jadi untuk fitur ML probability, port ini usable; untuk full TradingView indicator behavior, belum 100% lengkap.

### Clusters Volume Profile LuxAlgo

Python sudah mengerjakan bagian penting untuk bot:

- lookback
- K-Means cluster
- VWAP centroid update
- volume profile rows per cluster
- POC per cluster
- current cluster dan distance ke POC

Yang tidak di-port: drawing histogram, label total volume, dot size, offset, max width, dan semua visual object TradingView.

### Machine Learning RSI Zeiierman

Belum full port. Script ini jauh lebih besar dari sekadar RSI biasa: ada 8 feature RSI, memory bank, KNN analog, auto feature weight, rank/confidence gate, adaptive SuperTrend, dan ML RSI output. Di Python sekarang belum ada modul khusus yang mereplikasi ini.

Kalau strategy kamu memang mengandalkan ML RSI Zeiierman, ini perlu phase tambahan.

### Multi-Timeframe Volume Profiles

Belum full port. Script Pine ini menghitung profile beberapa HTF dengan lower timeframe volume, POC, VAH, VAL, value area 70%, dan delta/regular VP. Python sekarang belum punya output VAH/VAL multi-HTF equivalent.

Kalau bot butuh key level dari MTF Volume Profiles, ini perlu phase tambahan.

## Verifikasi Saat Audit

Focused indicator tests setelah patch:

```text
12 passed, 2 warnings
```

Full test suite setelah patch:

```text
150 passed, 38 warnings
```
