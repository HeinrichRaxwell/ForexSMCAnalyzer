# Aturan SMC/ICT Fibo & Multi-Timeframe (Penyempurnaan Strategy)

Dokumen ini merinci aturan masuk (*entry*) dan keluar (*exit*) yang diperbarui menggunakan penarikan Fibonacci pada area FVG, hierarki Multi-Timeframe (MTF), deteksi konfirmasi penolakan harga (*rejection*), serta target TP dinamis.

---

## 1. Penarikan Fibonacci pada FVG

Untuk setiap FVG yang terdeteksi, kita mendefinisikan batas area berdasarkan lilin pertama (Candle 1) dan lilin ketiga (Candle 3) dari formasi 3-lilin.

```
Bearish FVG:
Candle 1 (High)  ======================= [Fibo 1.0 / SL + 20 Pips]
                       FVG GAP
Candle 3 (Low)   ======================= [Fibo 0.0 / Target TP Awal]
```

### Formula Perhitungan Level Fibonacci:
Jarak FVG (*Gap Range*) = `|Candle 1 High - Candle 3 Low|`

* **Fibo 1.0 (Batas Ekstrem / Invalidasi)**:
  * Bullish FVG: `Candle 1 Low` (Stop Loss diatur di: `Fibo 1.0 - 20 Pips`).
  * Bearish FVG: `Candle 1 High` (Stop Loss diatur di: `Fibo 1.0 + 20 Pips`).
  * *Catatan*: Untuk XAUUSD, 20 Pips setara dengan **2.0 USD** (1 Pip emas = 0.1 USD).

* **Fibo 0.0 (Batas Awal / Target TP Standar)**:
  * Bullish FVG: `Candle 3 High`.
  * Bearish FVG: `Candle 3 Low`.

* **Fibo 0.5 (Midpoint / Entry Agresif)**:
  * Formula: `Fibo 0.0 + 0.5 * (Fibo 1.0 - Fibo 0.0)`
  * *Syarat*: Menunggu reaksi penolakan (*rejection*) di timeframe lebih kecil (LTF).

* **Fibo 0.618 (Golden Pocket / Safe Entry)**:
  * Formula: `Fibo 0.0 + 0.618 * (Fibo 1.0 - Fibo 0.0)`
  * *Syarat*: Area entri terbaik dengan risiko lebih kecil.

---

## 2. Hierarki Multi-Timeframe (MTF)

Sistem akan menganalisis data pada timeframe berikut: `M15`, `M30`, `H1`, `H4`, dan `D1`.

### Aturan Prioritas (Top-Down Analysis):
1. **HTF Bias**: Deteksi FVG dan OB pada timeframe besar (`D1`, `H4`, `H1`) secara real-time.
2. **Prioritas Sinyal**: Jika harga saat ini berada di dalam FVG/OB milik `H4` or `D1`, maka seluruh pencarian setup di timeframe kecil (`M15`, `M30`) wajib diselaraskan dengan arah HTF tersebut (misal: hanya mencari buy jika harga memantul di FVG Bullish H4).
3. **Weighting Model ML**: Model XGBoost akan dipasok dengan fitur tambahan `htf_fvg_mitigated` (apakah harga sedang berada di dalam FVG timeframe besar) untuk menaikkan skor *confidence*.

---

## 3. Deteksi Penolakan Harga (Rejection Check)

Bagaimana cara kode memverifikasi adanya "rejection" saat harga menyentuh Fibo 0.5 / 0.618?

1. **Wick Rejection (Shadow Panjang)**:
   * Lilin menyentuh level entri (`Low` menembus ke bawah Fibo 0.5 untuk Buy, atau `High` menembus ke atas Fibo 0.5 untuk Sell), tetapi **Close** ditutup jauh dari level tersebut.
   * Aturan Wick: Panjang shadow minimal 50% dari total panjang badan candle (*body*).
2. **LTF Structure Shift (CHoCH Konfirmasi)**:
   * Jika harga menyentuh FVG H1, kita memeriksa apakah terjadi CHoCH berlawanan arah di M15/M5 dalam 3 bar berikutnya. Ini adalah tanda paling valid bahwa pembalikan arah telah dimulai.

---

## 4. Take Profit (TP) Dinamis

Kita tidak membatasi target keluar hanya di Fibo 0.0. Jika tren sedang kuat (ditandai oleh BOS beruntun):

* **TP 1 (Fibo 0.0)**: Mengamankan 50% posisi (*Partial Take Profit*).
* **TP 2 (Next Target)**: Memindahkan sisa posisi ke area ketidakseimbangan berikutnya, yaitu **FVG yang belum termitigasi berikutnya** atau **Order Block berlawanan terdekat**.
* **Trailing Stop**: Menggeser Stop Loss ke tingkat Breakeven (BE) setelah TP 1 tersentuh.
