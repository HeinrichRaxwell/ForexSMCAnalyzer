# Backtest $50 / $100 Results - 2026-06-07

## Scope

Backtest dijalankan ulang pada 2026-06-07 memakai:

- symbol: XAUUSD / XAUUSDm local historical files,
- threshold: `0.50`,
- live-relevant max concurrent: `3`,
- capital: `$50` dan `$100`,
- sizing:
  - `equal`: 0.01 / 0.01,
  - `weighted`: 0.01 / 0.02,
- output CSV: `data/backtest_simulation_results.csv`,
- markdown report lama: `brain\aade0c14-67d6-4b69-a8a6-5834a430a34c\backtest_analysis_results.md`.

## Data Range

Data candle lokal yang dipakai:

| File | First | Last | Rows |
| --- | --- | --- | ---: |
| `historical_xauusdm_15.csv` | 2025-12-07 23:00:00 | 2026-06-04 10:15:00 | 11504 |
| `historical_xauusdm_30.csv` | 2025-12-07 23:00:00 | 2026-06-04 10:00:00 | 5753 |
| `historical_xauusdm_1h.csv` | 2025-12-07 23:00:00 | 2026-06-04 10:00:00 | 2879 |
| `historical_xauusdm_4h.csv` | 2025-12-07 20:00:00 | 2026-06-04 08:00:00 | 780 |
| `historical_xauusdm_1d.csv` | 2025-12-07 | 2026-06-04 | 154 |

Ini kira-kira 6 bulan data. Untuk validasi lebih kuat, idealnya minimal 12 bulan plus forward validation.

## Important Limitation

Backtest ini belum full 100% tick-by-tick.

Yang benar:

- setup dan trade path utama masih berbasis candle OHLC,
- MT5 ticks hanya dipakai saat candle ambiguous, yaitu ketika entry/SL/TP bisa kena dalam candle yang sama,
- untuk H4/D1 tick resolver sengaja skip range lebih dari 1 jam supaya tidak timeout,
- margin requirement, slippage, commission, swap, requote, dan broker execution latency belum disimulasikan penuh.

Jadi hasil ini valid sebagai matrix candle-based strategy test dengan partial tick disambiguation, bukan final institutional tick backtest.

## Best Result - Threshold 0.50, Max Concurrent 3

### Capital $50

Top result:

- timeframe: `H4`,
- strategy: `COMBINED`,
- sizing: `weighted`,
- trades resolved: 164,
- wins/losses: 116W / 48L,
- winrate: 70.73%,
- final balance: `$5,188.36`,
- max drawdown: 37.53%,
- blown: False.

More conservative H4 option:

- timeframe: `H4`,
- strategy: `COMBINED`,
- sizing: `equal`,
- trades resolved: 164,
- wins/losses: 116W / 48L,
- winrate: 70.73%,
- final balance: `$3,602.91`,
- max drawdown: 33.78%,
- blown: False.

### Capital $100

Top result:

- timeframe: `H4`,
- strategy: `COMBINED`,
- sizing: `weighted`,
- trades resolved: 164,
- wins/losses: 116W / 48L,
- winrate: 70.73%,
- final balance: `$5,238.36`,
- max drawdown: 35.63%,
- blown: False.

More conservative H4 option:

- timeframe: `H4`,
- strategy: `COMBINED`,
- sizing: `equal`,
- trades resolved: 164,
- wins/losses: 116W / 48L,
- winrate: 70.73%,
- final balance: `$3,652.91`,
- max drawdown: 31.63%,
- blown: False.

## H4 Combined Weighted Comparison

Threshold 0.50, strategy `COMBINED`, sizing `weighted`:

| Capital | Max Concurrent | Trades | W/L | Winrate | Final Balance | Max DD | Blown |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| $50 | 1 | 122 | 95/27 | 77.87% | $4,779.52 | 22.92% | False |
| $50 | 3 | 164 | 116/48 | 70.73% | $5,188.36 | 37.53% | False |
| $50 | 5 | 164 | 116/48 | 70.73% | $5,188.36 | 37.53% | False |
| $100 | 1 | 122 | 95/27 | 77.87% | $4,829.52 | 19.04% | False |
| $100 | 3 | 164 | 116/48 | 70.73% | $5,238.36 | 35.63% | False |
| $100 | 5 | 164 | 116/48 | 70.73% | $5,238.36 | 35.63% | False |

Interpretasi:

- Max concurrent 1 lebih stabil secara drawdown.
- Max concurrent 3/5 final balance lebih tinggi, tapi drawdown naik besar.
- Karena live `.env` saat ini `MT5_MAX_CONCURRENT_TRADES=3`, hasil max concurrent 3 paling dekat ke live config.

## Risk Notes

Ada 560 kombinasi threshold 0.50 di matrix:

- 416 tidak blown,
- 144 blown.

Artinya strategy tidak boleh dipakai sembarang di semua timeframe/strategy/sizing. Hasil terbaik didominasi H4 Combined/FVG dan M30 FVG, sementara beberapa BPR/IC low timeframe bisa blow account pada fixed lot.

Modal $50/$100 dengan fixed 0.01 lot juga sangat sensitif. Backtester memakai fixed lot, bukan compound/risk-percent sizing, sehingga final balance $50 dan $100 sering beda sekitar $50 saja. Real broker bisa punya margin rule yang membuat sebagian trade tidak bisa dibuka walaupun simulasi balance belum nol.

## Recommendation

Untuk forward live:

- tetap threshold `0.50`,
- prioritaskan H4 Combined/FVG style setups,
- pertimbangkan mulai max concurrent `1` dulu untuk forward validation,
- jangan percaya hasil $5k sebagai target pasti,
- kumpulkan forward data dan bandingkan dengan backtest.

Full tick backtest berikutnya perlu engine khusus:

- download/cache MT5 tick range per symbol,
- simulate entry/SL/TP dari bid/ask tick stream penuh,
- include spread, commission, swap, slippage, margin, and broker fill rules,
- validasi minimal 12 bulan.
