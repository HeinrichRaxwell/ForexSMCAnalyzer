# ML Phase 5 Rollout Policy - 2026-06-07

Phase 5 adalah aturan operasional untuk menjalankan bot secara disiplin setelah Phase 1-4 aktif.

Tujuan Phase 5 bukan menjanjikan bot perfect atau next trade pasti profit. Tujuannya membuat bot lebih aman dipakai live: hanya trade saat threshold dan calibration masuk akal, terus belajar dari real trade dan shadow signal, dan berhenti sementara saat kondisi memburuk.

## Yang Diimplementasikan

- Dokumen policy:
  - `docs/ML_PHASE_5_ROLLOUT_POLICY_2026-06-07.md`
- Utility offline:
  - `src/rollout_status.py`
- Test:
  - `tests/test_rollout_status.py`

Utility rollout status membaca `data/calibration_report.json` dan `.env`, lalu memberi status `READY` atau `BLOCKED` sebelum live scanner dijalankan. Utility ini offline, tidak membuka MT5, dan tidak mengirim order.

## Status Phase

Total phase saat ini: 5.

- Phase 1: shadow tracking untuk semua signal di bawah accept threshold.
- Phase 2: resolver virtual TP/SL untuk shadow signal.
- Phase 3: source-aware training dari real trade + shadow label berbobot rendah.
- Phase 4: calibration report untuk threshold, bucket, expectancy, drawdown, dan source.
- Phase 5: rollout policy, runbook, stop rule, dan cadence retraining/report.

## Mode Operasional

### 1. Offline Check

Dipakai sebelum market buka atau sebelum live scanner.

Run:

```powershell
cd 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer'
& '.\.venv\Scripts\python.exe' -m src.calibration_report
& '.\.venv\Scripts\python.exe' -m src.rollout_status --threshold 0.50
& '.\.venv\Scripts\python.exe' -m pytest -q
```

Lanjut live hanya jika:

- Test suite pass.
- Calibration report berhasil dibuat.
- Rollout status tidak `BLOCKED`.
- Recommendation threshold ada.
- Max drawdown pada threshold live masih dalam batas policy.

### 2. One-Shot Scanner

Dipakai untuk cek market sekali, tanpa loop panjang.

Run:

```powershell
cd 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer'
& '.\.venv\Scripts\python.exe' -m src.scanner_worker --symbol XAUUSD --threshold 0.50
```

Gunakan mode ini dulu saat market baru buka, setelah restart MT5, atau setelah mengganti model/config.

### 3. Live Loop Scanner

Mode ini bisa mengirim order real jika `.env` berisi:

```text
MT5_EXECUTE_TRADES=True
```

Jalankan hanya saat sudah sadar ini live risk:

```powershell
cd 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer'
& '.\.venv\Scripts\python.exe' -m src.scanner_worker --symbol XAUUSD --threshold 0.50 --loop --interval 5
```

Stop:

```powershell
Ctrl+C
```

## Threshold Policy

Threshold awal: `0.50`.

Alasannya: Phase 4 report terakhir merekomendasikan `0.50` sebagai threshold paling longgar yang masih memenuhi rule konservatif.

Rule Phase 4:

- Minimum sample: `50`.
- Minimum expectancy: `1.0R`.
- Maximum drawdown: `3.0R`.

Jika live forward result memburuk, naikkan threshold dulu sebelum mengubah logic strategy.

Urutan konservatif:

- Normal: `0.50`.
- Defensive: `0.60`.
- Very defensive: `0.70`.
- Jangan turunkan ke `0.40` atau `0.30` untuk live sampai calibration forward membuktikan drawdown aman.

## Stop Rules

Hentikan live loop sementara jika salah satu terjadi:

- `3` loss beruntun pada hari yang sama.
- Drawdown harian mencapai `3R`.
- MT5 error, requote, invalid volume, invalid stops, atau koneksi tidak stabil.
- Calibration report gagal dibuat.
- Model trainer menolak challenger dan performa live champion sedang menurun.
- Spread XAUUSD melebar ekstrem dibanding kondisi normal.
- Ada news besar high-impact dalam waktu dekat dan strategy belum punya news filter eksplisit.

Setelah stop, jangan langsung menaikkan lot atau balas dendam trade. Lakukan:

```powershell
& '.\.venv\Scripts\python.exe' -m src.calibration_report
& '.\.venv\Scripts\python.exe' -m src.model_trainer
& '.\.venv\Scripts\python.exe' -m pytest -q
```

Jika retrain menolak challenger, itu bukan error. Itu safety gate yang berarti model aktif lama masih lebih baik dari challenger pada evaluasi saat itu.

## Learning Cadence

Bot belajar dari:

- Real labeled trades di `data/labeled_setups.csv`.
- Resolved shadow signals di `data/shadow_labeled_setups.csv`.

Shadow signal mencakup confidence di bawah threshold, termasuk di bawah `30%`, selama `ML_SHADOW_MIN_CONFIDENCE` tidak dinaikkan.

Cadence yang disarankan:

- Setelah sesi market selesai: generate calibration report.
- Setelah ada closed real trade baru atau shadow resolved baru: retrain non-live.
- Setelah retrain: cek apakah challenger accepted atau rejected.
- Jika accepted: jalankan one-shot scanner dulu sebelum loop.
- Jika rejected: tetap pakai champion lama.

Command:

```powershell
cd 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer'
& '.\.venv\Scripts\python.exe' -m src.calibration_report
& '.\.venv\Scripts\python.exe' -m src.model_trainer
& '.\.venv\Scripts\python.exe' -m pytest -q
```

## Memory Policy

Trainer saat ini membatasi dataset terbaru sampai `1000` setup untuk mengurangi overfit pada market regime lama.

Policy saat ini:

- `1000` setup cukup untuk start dan lebih stabil daripada memakai semua data lama tanpa filter.
- Jangan langsung naik ke `5000` atau `10000` sebelum ada walk-forward validation yang jelas.
- Data banyak bisa membantu hanya kalau market regime relevan dan label bersih.
- Data terlalu banyak dari regime lama bisa membuat model lambat adaptasi.

Jika mau upgrade memory nanti, lakukan bertahap:

- Phase 6 kandidat: bandingkan window `1000`, `2500`, `5000` dengan walk-forward split.
- Pilih window berdasarkan expectancy, drawdown, dan forward stability, bukan winrate tertinggi saja.

## Confidence Policy

Confidence boleh naik mendekati `100%` hanya jika model memang mempelajari pola yang konsisten dari data valid.

Namun confidence `100%` bukan target. Jika terlalu sering muncul, itu harus dicurigai sebagai:

- overfit,
- leakage fitur,
- sample terlalu kecil,
- atau market regime yang belum diuji cukup lama.

Bot yang sehat tidak harus selalu memberi `100%`. Bot yang sehat harus:

- menolak setup jelek,
- menaikkan confidence pada pola yang berulang menang,
- menurunkan confidence pada pola yang sering gagal,
- dan tetap punya stop rule saat market berubah.

## Pre-Market Checklist

Sebelum live loop:

- MT5 login benar.
- Symbol XAUUSD aktif dan bisa fetch candle.
- `.env` dicek, terutama `MT5_EXECUTE_TRADES`.
- `pytest -q` pass.
- `src.calibration_report` berhasil.
- Threshold live mengikuti recommendation atau lebih konservatif.
- Lot/risk sesuai batas akun.
- Tidak ada news besar yang belum diantisipasi.

## Manual Runbook

### Full Safe Check

```powershell
cd 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer'
& '.\.venv\Scripts\python.exe' -m py_compile src\shadow_tracker.py src\scanner_worker.py src\model_trainer.py src\calibration_report.py src\inference.py
& '.\.venv\Scripts\python.exe' -m pytest -q
& '.\.venv\Scripts\python.exe' -m src.calibration_report
& '.\.venv\Scripts\python.exe' -m src.rollout_status --threshold 0.50
```

### Retrain Non-Live

```powershell
cd 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer'
& '.\.venv\Scripts\python.exe' -m src.model_trainer
```

### One-Shot Scanner

```powershell
cd 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer'
& '.\.venv\Scripts\python.exe' -m src.scanner_worker --symbol XAUUSD --threshold 0.50
```

### Live Loop Scanner

```powershell
cd 'C:\Users\WINDOWS 11 PRO\forex-smc-analyzer'
& '.\.venv\Scripts\python.exe' -m src.scanner_worker --symbol XAUUSD --threshold 0.50 --loop --interval 5
```

## Kesimpulan Phase 5

Dengan Phase 5, bot tidak dianggap perfect. Bot dianggap punya prosedur pakai yang lebih benar:

- signal rendah tetap dipantau,
- outcome shadow masuk learning dengan bobot kecil,
- model baru tidak otomatis menimpa champion,
- threshold dipilih dari calibration,
- live loop punya stop rule,
- dan retrain dilakukan setelah ada data baru yang valid.

Ini membuat sistem lebih siap tempur dan lebih terukur, tetapi tetap tidak bisa menjamin profit pada trade berikutnya.
