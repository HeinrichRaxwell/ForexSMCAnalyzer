# 📊 Real-Tick Backtest Report: Verified Winrate & Execution Matrix
**Date Range:** January 2, 2026 – June 7, 2026 (133 Days, 100% Real Ticks Coverage)  
**Symbol:** XAUUSDm | **Starting Capital:** $100.00 | **Max Concurrent:** 3  
**Formula Applied:** $\text{Winrate (\%)} = \frac{\text{Wins}}{\text{Wins} + \text{Losses}} \times 100$ (Missed / Unfilled setups are excluded)

---

## 📈 Verified Winrate Performance Table

| Timeframe | Strategy / Mode | Setup Count | Total Resolved | Wins | Losses | Missed | Winrate (%) | Final Balance ($) | Max Drawdown (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **H4** | **OB (WatchZone)** | 37 | 12 | 12 | 0 | 25 | **100.00%** | **$1,006.84** | **0.00%** |
| **H1** | **OB (WatchZone)** | 191 | 70 | 60 | 10 | 121 | **85.71%** | **$1,931.44** | **18.73%** |
| **H4** | **COMBINED (WatchZone)** | 261 | 81 | 67 | 14 | 180 | **82.72%** | **$2,540.09** | **9.95%** |
| **H1** | **OB (Limit Order)** | 428 | 172 | 140 | 32 | 256 | **81.40%** | **$4,719.31** | **17.93%** |
| **H1** | **IC (Limit Order)** | 976 | 488 | 395 | 93 | 488 | **80.94%** | **$3,912.79** | **13.41%** |
| **M30** | **IC (Limit Order)** | 1,946 | 997 | 806 | 191 | 949 | **80.84%** | **$6,958.83** | **6.55%** |
| **H1** | **COMBINED (WatchZone)** | 1,089 | 424 | 340 | 84 | 665 | **80.19%** | **$5,883.89** | **18.99%** |
| **M30** | **OB (WatchZone)** | 339 | 127 | 101 | 26 | 212 | **79.53%** | **$2,128.13** | **13.36%** |
| **M30** | **COMBINED (WatchZone)** | 2,233 | 948 | 713 | 235 | 1,285 | **75.21%** | **$8,088.71** | **14.50%** |
| **H1** | **BPR (Limit Order)** | 270 | 43 | 31 | 12 | 227 | **72.09%** | **$641.10** | 28.42% |
| **H1** | **FVG (Limit Order)** | 961 | 190 | 136 | 54 | 771 | **71.58%** | **$2,895.37** | 14.60% |
| **H4** | **OB (Limit Order)** | 111 | 42 | 30 | 12 | 69 | **71.43%** | **$1,724.28** | 7.84% |
| **H1** | **COMBINED (Limit Order)** | 3,374 | 1,268 | 874 | 394 | 2,106 | **68.93%** | **$15,005.00** | 11.46% |
| **H4** | **FVG (Limit Order)** | 294 | 64 | 44 | 20 | 230 | **68.75%** | **$1,930.47** | 24.30% |
| **M30** | **COMBINED (Limit Order)** | 6,989 | 2,695 | 1,818 | 877 | 4,294 | **67.46%** | **$22,122.99** | 5.95% |
| **M30** | **FVG (Limit Order)** | 1,971 | 486 | 320 | 166 | 1,485 | **65.84%** | **$4,061.42** | 28.29% |
| **H4** | **COMBINED (Limit Order)** | 904 | 313 | 201 | 112 | 591 | **64.22%** | **$7,681.87** | 29.52% |
| **M30** | **BPR (Limit Order)** | 596 | 130 | 76 | 54 | 466 | **58.46%** | **$1,029.67** | 32.92% |
| **H4** | **Swapzone (Limit Order)** | 176 | 86 | 44 | 42 | 90 | **51.16%** | **$2,788.76** | 25.69% |
| **H1** | **Swapzone (Limit Order)** | 675 | 374 | 185 | 189 | 301 | **49.47%** | **$3,691.53** | 31.02% |
| **M30** | **Swapzone (Limit Order)** | 1,409 | 833 | 410 | 423 | 576 | **49.22%** | **$5,269.32** | 28.88% |
| **H4** | **BPR (Limit Order)** | 96 | 24 | 10 | 14 | 72 | **41.67%** | **$530.33** | 42.12% |

---

## 🧮 Audit Perhitungan Formula Winrate

Formula yang digunakan dalam seluruh simulasi:
$$\text{Winrate (\%)} = \left( \frac{\text{Jumlah Trade Menang (Wins)}}{\text{Jumlah Trade Menang (Wins)} + \text{Jumlah Trade Kalah (Losses)}} \right) \times 100$$

* **Metode Evaluasi**: Hanya trade yang benar-benar **tereksekusi dan selesai (*Resolved = Wins + Losses*)** yang dihitung ke dalam penyebut (*denominator*).
* **Setup Tidak Terisi (*Missed*)**: Setup yang harganya tidak pernah menyentuh titik entri atau tidak sempat terpasang karena batas *max concurrent* **dikeluarkan dari penyebut**. Hal ini sesuai dengan standar institusional global, karena setup yang tidak pernah terisi bukan merupakan trade (tidak menghasilkan profit maupun loss).
