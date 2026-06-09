# ML Phase 2 Shadow Resolver - 2026-06-07

Phase 2 menambahkan resolver outcome virtual untuk shadow signal.
Tujuannya: sinyal yang tidak dieksekusi karena confidence di bawah threshold tetap bisa diketahui apakah secara virtual mencapai TP, SL, atau expired.

## Yang Diimplementasikan

- Resolver di `src/shadow_tracker.py`:
  - `resolve_shadow_record()`
  - `process_shadow_signal_outcomes()`
  - `append_shadow_labeled_rows()`
- Integrasi scanner di `src/scanner_worker.py`:
  - `process_existing_shadow_outcomes()`
  - Dipanggil setelah candle multi-timeframe berhasil difetch dari MT5.
- Output label shadow:
  - `data/shadow_labeled_setups.csv`

## Rule Outcome

Untuk BUY:

- Entry aktif jika candle low menyentuh `entry_price`.
- TP jika candle high menyentuh `tp_price`.
- SL jika candle low menyentuh `sl_price`.

Untuk SELL:

- Entry aktif jika candle high menyentuh `entry_price`.
- TP jika candle low menyentuh `tp_price`.
- SL jika candle high menyentuh `sl_price`.

Jika TP dan SL kena di candle yang sama:

- Resolver pakai aturan konservatif: hitung sebagai SL.
- Ini mencegah data shadow terlalu optimis saat urutan intrabar tidak diketahui.

Jika tidak ada entry/exit sampai batas `max_bars`:

- Status menjadi `expired`.
- Expired tidak masuk CSV label training karena tidak punya label menang/kalah yang jelas.

## CSV Shadow Label

Row TP/SL yang resolved masuk `data/shadow_labeled_setups.csv` dengan:

- `signal_id`
- `sample_source=shadow`
- fitur setup dari record shadow
- `confidence`
- `accept_threshold`
- `result`
- `pnl_relative`
- `label`

Append dibuat idempotent berdasarkan `signal_id`, jadi rerun resolver tidak menggandakan row yang sama.

## Verifikasi

Focused tests:

```text
20 passed
```

Full test suite:

```text
103 passed, 32 warnings
```

## Batasan Yang Masih Ada

- Shadow label belum dipakai untuk training model.
- Shadow sample belum diberi bobot rendah di trainer.
- Belum ada calibration report per confidence bucket.
- Resolver candle-level belum memakai tick order; same-candle TP/SL selalu dianggap SL.

Itu masuk Phase 3 dan Phase 4.
