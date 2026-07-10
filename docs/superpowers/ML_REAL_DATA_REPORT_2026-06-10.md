# ML Real Data Report - 2026-06-10

Source file: `data/calibration_report.json`
Scoring mode: `walk_forward`
Out-of-sample: `true`
Total rows: `5337`
Scored rows: `4445`
Unscored rows: `892`

## Verdict

Trade logic is not yet broadly good enough for real-money confidence.

The pipeline is more honest now: cost is included, confidence scoring is walk-forward out-of-sample, and calibration uses OOF data. However, the real trading edge is still weak. Combined performance is negative, real-source performance is negative, and rollout remains blocked.

Forward-test demo is acceptable as a controlled experiment. Real/live aggressive use is not ready.

## Combined Result

| Metric | Value |
|---|---:|
| Samples | 5337 |
| Wins | 1968 |
| Losses | 3369 |
| Winrate | 36.87% |
| Expectancy | -0.19R |
| Profit Factor | 0.72 |
| Avg Win | 1.34R |
| Avg Loss | -1.08R |
| Max Drawdown | 1202.56R |
| Max Consecutive Losses | 48 |
| Avg Confidence | 0.35 |

## Source Breakdown

| Source | Samples | Winrate | Expectancy | PF | Max DD | Notes |
|---|---:|---:|---:|---:|---:|---|
| real | 5000 | 34.98% | -0.24R | 0.66 | 1218.04R | Weak. This is the main reason system is not ready. |
| shadow | 337 | 64.99% | 0.57R | 2.64 | 14.00R | Promising, but smaller and not enough alone. |

## Confidence Thresholds

| Threshold | Samples | Winrate | Expectancy | PF | Max DD | Max Loss Streak |
|---:|---:|---:|---:|---:|---:|---:|
| 0.30 | 2590 | 42.12% | -0.10R | 0.84 | 390.26R | 35 |
| 0.40 | 1472 | 45.18% | -0.05R | 0.92 | 128.32R | 24 |
| 0.50 | 727 | 46.49% | -0.02R | 0.97 | 64.13R | 22 |
| 0.60 | 302 | 49.67% | 0.06R | 1.12 | 15.10R | 14 |
| 0.70 | 80 | 55.00% | 0.14R | 1.31 | 13.57R | 8 |
| 0.80 | 11 | 45.45% | -0.09R | 0.85 | 6.29R | 6 |

Interpretation:
- Below 0.60 is not good enough.
- 0.60 and 0.70 show a small positive edge, but drawdown is still too high.
- 0.80 has too few samples and turns negative again.
- Report recommendation reason is `no_threshold_met_all_rules`.

## Confidence Buckets

| Bucket | Samples | Winrate | Expectancy | PF | Max DD |
|---|---:|---:|---:|---:|---:|
| 0.00-0.30 | 1855 | 30.19% | -0.36R | 0.55 | 754.79R |
| 0.30-0.40 | 1118 | 38.10% | -0.17R | 0.75 | 278.68R |
| 0.40-0.50 | 745 | 43.89% | -0.08R | 0.87 | 95.15R |
| 0.50-0.60 | 425 | 44.24% | -0.07R | 0.88 | 57.32R |
| 0.60-0.70 | 222 | 47.75% | 0.03R | 1.07 | 17.34R |
| 0.70-0.80 | 69 | 56.52% | 0.18R | 1.40 | 9.17R |
| 0.80-0.90 | 11 | 45.45% | -0.09R | 0.85 | 6.29R |
| unscored | 892 | 35.54% | -0.09R | 0.86 | 91.99R |

Interpretation:
- Calibration is more honest now: higher confidence mostly improves quality until 0.70-0.80.
- The best bucket is 0.70-0.80, but sample is only 69. This is promising, not proven.

## Strategy Breakdown

| Strategy | Samples | Winrate | Expectancy | PF | Max DD | Verdict |
|---|---:|---:|---:|---:|---:|---|
| BPR | 701 | 37.80% | -0.31R | 0.52 | 225.84R | Bad |
| FVG | 1090 | 41.74% | -0.22R | 0.64 | 251.73R | Bad |
| FVG_OR_BPR | 201 | 72.14% | 0.41R | 2.77 | 10.39R | Good candidate |
| IC | 1144 | 34.09% | -0.08R | 0.88 | 151.11R | Weak |
| OB | 369 | 25.47% | -0.11R | 0.85 | 96.65R | Weak |
| OB_OR_SWAPZONE_IC_SND | 169 | 50.30% | 0.56R | 2.13 | 11.00R | Good candidate |
| PIVOT_REJECTION | 20 | 60.00% | 0.83R | 3.21 | 2.00R | Promising but too small |
| Pivot | 391 | 41.94% | -0.77R | 0.22 | 302.85R | Very bad |
| SND | 466 | 29.83% | -0.27R | 0.64 | 141.34R | Bad |
| Swapzone | 786 | 27.86% | -0.23R | 0.69 | 207.04R | Bad |

Interpretation:
- Most base strategies are still negative.
- The best candidates are combo strategies and new `PIVOT_REJECTION`, but sample size matters.
- `Pivot` legacy logic is very bad and should probably be disabled or heavily filtered.
- `PIVOT_REJECTION` looks good after reaction routing, but 20 samples is not enough proof.

## Timeframe Breakdown

| Timeframe | Samples | Winrate | Expectancy | PF | Max DD | Verdict |
|---:|---:|---:|---:|---:|---:|---|
| 15 | 2891 | 38.74% | -0.18R | 0.73 | 625.57R | Bad |
| 30 | 1449 | 33.68% | -0.28R | 0.61 | 440.57R | Bad |
| 60 | 741 | 33.74% | -0.16R | 0.77 | 176.84R | Bad |
| 240 | 211 | 42.65% | 0.14R | 1.24 | 25.64R | Candidate |
| 1440 | 45 | 44.44% | 0.29R | 1.52 | 12.82R | Candidate but small |

Interpretation:
- M15, M30, and H1 are dragging the system down.
- H4 and D1 look better, but samples are smaller and drawdown still needs control.

## Direction Breakdown

| Direction | Samples | Winrate | Expectancy | PF | Max DD |
|---:|---:|---:|---:|---:|---:|
| Sell (-1) | 2773 | 38.80% | -0.13R | 0.81 | 583.35R |
| Buy (1) | 2564 | 34.79% | -0.26R | 0.64 | 725.46R |

Interpretation:
- Sell side is less bad than buy side.
- Buy logic needs stronger filtering or regime rules.

## Setup Type Breakdown

| Setup Type | Samples | Winrate | Expectancy | PF | Max DD |
|---:|---:|---:|---:|---:|---:|
| 0 | 1992 | 43.42% | -0.19R | 0.67 | 454.94R |
| 1 | 2934 | 31.60% | -0.12R | 0.83 | 529.87R |
| 2 | 411 | 42.82% | -0.69R | 0.28 | 302.85R |

Interpretation:
- Setup type 2 is the worst by expectancy and PF.
- Setup type 1 has low winrate but bigger average wins, still negative.

## Trade Logic Quality

What is good:
- The ML/reporting pipeline is now much more honest.
- Confidence is more meaningful after 0.60.
- Reaction-routed `PIVOT_REJECTION` is promising.
- Shadow data is strong and may reveal better filters than current accepted-live logic.

What is not good:
- Combined result is still negative.
- Real-source result is clearly negative.
- Most base strategies are negative.
- Drawdown is far too large for live use.
- M15/M30/H1 generate too much bad flow.
- Legacy `Pivot` is very weak.

## Forward Test Recommendation

Demo forward test only.

Suggested constraints:
1. Do not use threshold 0.50 for real confidence.
2. If forward testing, prefer threshold 0.60 or 0.70 with very small lot.
3. Add or enforce max daily loss and max trades per day.
4. Consider disabling or heavily filtering weak strategies: `Pivot`, `BPR`, `FVG`, `SND`, `Swapzone`.
5. Watch high-confidence bucket 0.70-0.80 separately.
6. Restart scanner before testing so latest schema guards and calibration are loaded.

Final answer: the logic is improved, but not a masterpiece yet. The honest data says this system is still in research/forward-test mode, not real-money ready.
