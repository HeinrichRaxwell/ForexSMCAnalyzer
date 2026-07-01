# ML Phase 1 Shadow Tracking - 2026-06-07

Phase 1 menambahkan observability untuk sinyal yang tidak lolos confidence live.
Sinyal ini tidak dieksekusi, tapi disimpan agar Phase 2 bisa memantau virtual TP/SL.

## Yang Diimplementasikan

- File store baru: `data/shadow_signals.json`.
- Modul baru: `src/shadow_tracker.py`.
- Helper scanner:
  - `get_accept_threshold()`
  - `register_shadow_candidate()`
  - `register_low_confidence_lead()`
- Scanner sekarang menyimpan low-confidence lead ke:
  - `data/sent_signals.json` untuk kompatibilitas registry lama.
  - `data/shadow_signals.json` untuk monitoring virtual.

## Policy Confidence

- Live accepted: `confidence >= 0.50` secara default.
- Shadow tracked: `0.00 <= confidence < 0.50` secara default.
- Override live threshold:
  - `ML_ACCEPT_THRESHOLD=0.50`
  - atau CLI `--threshold 0.55`.
- Override shadow minimum jika noise terlalu banyak:
  - `ML_SHADOW_MIN_CONFIDENCE=0.10`
  - `ML_SHADOW_MIN_CONFIDENCE=0.30`

Jika `ML_SHADOW_MIN_CONFIDENCE` tidak diset, default adalah `0.00`, sesuai request agar sinyal di bawah 30% juga ikut dipantau.

## Format Shadow Signal

Setiap shadow record menyimpan:

- `signal_id`
- `source=shadow`
- `status=open`
- `result=null`
- `label=null`
- `symbol`
- `time`
- `timeframe`
- `strategy`
- `direction`
- `entry_price`
- `sl_price`
- `tp_price`
- `confidence`
- `accept_threshold`
- `filtered_reason=below_accept_threshold`
- `ticket_id=null`
- `features`

Dual entry disimpan sebagai dua leg terpisah:

- `_0.5`
- `_0.618`

Ini penting supaya outcome virtual tiap entry bisa dinilai sendiri.

## Batasan Phase 1

- Status ini sudah dilanjutkan oleh Phase 2; lihat `docs/ML_PHASE_2_SHADOW_RESOLVER_2026-06-07.md`.
- Phase 1 sendiri hanya membuat capture/registry shadow signal.
- Belum memasukkan shadow sample ke training.
- Belum ada calibration report per confidence bucket.

Training shadow sample dan calibration report tetap masuk Phase 3 sampai Phase 4.

## Catatan Profit

Sinyal yang sering win bisa membuat pola serupa diberi confidence lebih tinggi setelah training dan validation.
Tapi confidence 100% bukan target yang sehat. Jika model sering memberi 100%, itu perlu dicurigai sebagai overfit sampai terbukti lewat out-of-sample dan live metrics.
