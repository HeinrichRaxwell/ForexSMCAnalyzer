# ML Confidence Predictive & Expectancy Repair — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bikin confidence ML bot jadi predictive dan expectancy jujur, lewat 4 workstream: cost modeling, feature engineering, rework PIVOT_REJECTION jadi reaction-based order routing, dan isotonic confidence calibration.

**Architecture:** Label dihitung di `src/labeler.py` (first-touch + tick resolution). Cost modeling menyesuaikan `pnl_relative` (R aktual). Fitur baru ditambah konsisten di tiap blok setup. Rejection logic baru jadi modul terpisah `src/reaction_router.py` yang dikonsumsi blok pivot. Kalibrasi isotonic di modul baru `src/calibrator.py`, dilatih dari walk-forward OOF (sudah out-of-sample). Verifikasi pakai walk-forward calibration report yang sudah ada.

**Tech Stack:** Python, pandas, numpy, scikit-learn (IsotonicRegression), xgboost, lightgbm, pytest, joblib.

**Urutan wajib:** cost → fitur → rework rejection → kalibrasi → re-score. Mengerjakan kalibrasi sebelum cost/fitur = mengalibrasi ke angka yang akan berubah.

**Spec:** `docs/superpowers/specs/2026-06-09-ml-confidence-predictive-design.md`

---

## Workstream 1: Cost Modeling

### Task 1: Helper `compute_cost_r` di labeler

**Files:**
- Modify: `src/labeler.py` (tambah helper + env readers di dekat atas modul)
- Test: `tests/test_cost_modeling.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cost_modeling.py
import os
import pytest
from src.labeler import compute_cost_r


def test_cost_r_basic():
    # spread 0.30 + slippage 0.0, risk 3.0 USD => 0.10 R
    assert compute_cost_r(risk_pips=3.0, spread_usd=0.30, slippage_usd=0.0) == pytest.approx(0.10)


def test_cost_r_small_risk_larger_fraction():
    # risk kecil => cost_R lebih besar
    big = compute_cost_r(risk_pips=10.0, spread_usd=0.30, slippage_usd=0.0)
    small = compute_cost_r(risk_pips=1.0, spread_usd=0.30, slippage_usd=0.0)
    assert small > big
    assert small == pytest.approx(0.30)


def test_cost_r_includes_slippage():
    assert compute_cost_r(risk_pips=2.0, spread_usd=0.30, slippage_usd=0.10) == pytest.approx(0.20)


def test_cost_r_zero_risk_no_div_zero():
    # risk_pips=0 tidak boleh ZeroDivisionError; kembalikan 0.0 (tidak bisa hitung)
    assert compute_cost_r(risk_pips=0.0, spread_usd=0.30, slippage_usd=0.0) == 0.0


def test_cost_r_negative_risk_treated_as_abs():
    assert compute_cost_r(risk_pips=-3.0, spread_usd=0.30, slippage_usd=0.0) == pytest.approx(0.10)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cost_modeling.py -v`
Expected: FAIL with `ImportError: cannot import name 'compute_cost_r'`

- [ ] **Step 3: Write minimal implementation**

Tambah di `src/labeler.py` (dekat atas, setelah import):

```python
def _read_float_env(name: str, default: float) -> float:
    import os
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def get_spread_usd() -> float:
    return _read_float_env("ML_SPREAD_USD", 0.30)


def get_slippage_usd() -> float:
    return _read_float_env("ML_SLIPPAGE_USD", 0.0)


def compute_cost_r(risk_pips: float, spread_usd: float = None, slippage_usd: float = None) -> float:
    """Cost per trade dalam satuan R. cost_R = (spread + slippage) / risk_usd.

    risk_pips di sini adalah |entry - sl| dalam USD (sesuai pemakaian di labeler).
    Mengembalikan 0.0 bila risk tidak valid (hindari div-by-zero).
    """
    if spread_usd is None:
        spread_usd = get_spread_usd()
    if slippage_usd is None:
        slippage_usd = get_slippage_usd()
    risk = abs(float(risk_pips))
    if risk <= 0.0:
        return 0.0
    return (float(spread_usd) + float(slippage_usd)) / risk
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cost_modeling.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add tests/test_cost_modeling.py src/labeler.py
git commit -m "feat: add compute_cost_r helper for XAUUSD cost modeling"
```

---

### Task 2: Terapkan cost ke pnl_relative saat labeling

**Files:**
- Modify: `src/labeler.py` (fungsi `simulate_trade` return R, atau pembungkus yang menulis `pnl_relative`)
- Test: `tests/test_cost_modeling.py` (tambah test)

> **Catatan konteks:** `simulate_trade` saat ini return `1.0` (win) / `0.0` (loss) / `None` (unresolved) — ini dipakai sebagai label biner. `pnl_relative` adalah kolom terpisah. Kita TIDAK mengubah label biner. Kita menambah perhitungan `pnl_relative` ber-cost di titik setup di-append. Base R: win = `(tp-entry)/(entry-sl)` magnitude, loss = `-1.0`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cost_modeling.py (tambahkan)
from src.labeler import compute_pnl_relative


def test_pnl_relative_win_subtracts_cost():
    # RR 2:1, risk 3 USD, cost 0.30/3=0.10 => 2.0 - 0.10 = 1.90
    r = compute_pnl_relative(label=1, entry=2000.0, sl=1997.0, tp=2006.0,
                             spread_usd=0.30, slippage_usd=0.0)
    assert r == pytest.approx(1.90)


def test_pnl_relative_loss_is_deeper_than_minus_one():
    # loss => -1.0 - cost
    r = compute_pnl_relative(label=0, entry=2000.0, sl=1997.0, tp=2006.0,
                             spread_usd=0.30, slippage_usd=0.0)
    assert r == pytest.approx(-1.10)


def test_pnl_relative_zero_risk_safe():
    r = compute_pnl_relative(label=1, entry=2000.0, sl=2000.0, tp=2006.0,
                             spread_usd=0.30, slippage_usd=0.0)
    # risk 0 => base R 0, cost 0 => 0.0
    assert r == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cost_modeling.py::test_pnl_relative_win_subtracts_cost -v`
Expected: FAIL with `ImportError: cannot import name 'compute_pnl_relative'`

- [ ] **Step 3: Write minimal implementation**

Tambah di `src/labeler.py`:

```python
def compute_pnl_relative(label: int, entry: float, sl: float, tp: float,
                         spread_usd: float = None, slippage_usd: float = None) -> float:
    """R aktual per trade, sudah dikurangi cost.

    Win  -> +RR  - cost_R  (RR = |tp-entry| / |entry-sl|)
    Loss -> -1.0 - cost_R
    """
    risk = abs(float(entry) - float(sl))
    cost_r = compute_cost_r(risk, spread_usd=spread_usd, slippage_usd=slippage_usd)
    if risk <= 0.0:
        return 0.0
    if int(label) == 1:
        rr = abs(float(tp) - float(entry)) / risk
        return rr - cost_r
    return -1.0 - cost_r
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cost_modeling.py -v`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add tests/test_cost_modeling.py src/labeler.py
git commit -m "feat: compute cost-adjusted pnl_relative"
```

---

### Task 3: Tulis pnl_relative ke setiap blok setup

**Files:**
- Modify: `src/labeler.py` (tiap blok `setups.append({...})` — FVG, OB, BPR, Swap, Pivot)
- Test: `tests/test_labeler_pnl_integration.py`

> **Catatan:** Saat ini blok setup tidak menulis `pnl_relative`. Tambah key `'pnl_relative': compute_pnl_relative(int(label), entry, sl, tp)` ke SETIAP dict yang di-append. Ada 5+ blok; pastikan semua diubah konsisten.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_labeler_pnl_integration.py
import pandas as pd
import numpy as np
from src.labeler import label_smc_setups


def _synthetic_uptrend_df(n=120):
    idx = pd.date_range("2025-01-01", periods=n, freq="15min")
    base = np.linspace(2000, 2020, n)
    noise = np.sin(np.linspace(0, 12, n)) * 2.0
    close = base + noise
    high = close + 1.0
    low = close - 1.0
    openp = close - 0.2
    return pd.DataFrame({
        "time": idx, "Open": openp, "High": high, "Low": low,
        "Close": close, "Volume": np.full(n, 100.0),
    })


def test_labeled_setups_have_pnl_relative_column():
    df = _synthetic_uptrend_df()
    out = label_smc_setups(df, symbol="XAUUSD")
    if out.empty:
        # Sintetik mungkin tak menghasilkan setup; skip jika kosong
        import pytest
        pytest.skip("no setups produced by synthetic data")
    assert "pnl_relative" in out.columns
    assert out["pnl_relative"].notna().all()


def test_winning_setups_pnl_below_raw_rr():
    df = _synthetic_uptrend_df()
    out = label_smc_setups(df, symbol="XAUUSD")
    if out.empty:
        import pytest
        pytest.skip("no setups produced")
    wins = out[out["label"] == 1]
    if wins.empty:
        import pytest
        pytest.skip("no winning setups")
    # pnl_relative untuk win harus < RR mentah (karena cost dikurangi)
    rr_raw = (wins["tp_price"] - wins["entry_price"]).abs() / \
             (wins["entry_price"] - wins["sl_price"]).abs()
    assert (wins["pnl_relative"] <= rr_raw + 1e-9).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_labeler_pnl_integration.py -v`
Expected: FAIL — `pnl_relative` belum ada di output (KeyError/assert false)

- [ ] **Step 3: Write minimal implementation**

Di SETIAP blok `setups.append({...})` di `src/labeler.py`, tambah baris sebelum append:

```python
                    pnl_rel = compute_pnl_relative(int(label), entry, sl, tp)
```

dan tambah key di dict:

```python
                        'pnl_relative': pnl_rel,
```

Lakukan untuk blok: FVG (line ~424), OB (~465), BPR (~507), Swap (~547), dan Pivot. Pastikan `entry`, `sl`, `tp`, `label` sudah terdefinisi di scope tiap blok (sudah ada).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_labeler_pnl_integration.py -v`
Expected: PASS (atau SKIP bila sintetik tak hasilkan setup — jalankan juga full suite di Step 5)

- [ ] **Step 5: Commit**

```bash
git add tests/test_labeler_pnl_integration.py src/labeler.py
git commit -m "feat: write cost-adjusted pnl_relative to all setup blocks"
```

---

## Workstream 2: Feature Engineering

### Task 4: Modul fitur `src/setup_features.py`

**Files:**
- Create: `src/setup_features.py`
- Test: `tests/test_setup_features.py`

> Tiap fungsi fitur murni (pure) dan testable terpisah, dipanggil dari labeler. Hipotesis tiap fitur ada di spec.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_setup_features.py
import numpy as np
import pandas as pd
import pytest
from src.setup_features import (
    rr_ratio, atr_percentile, body_to_range_ratio,
    dist_to_recent_swing_norm, htf_trend_aligned,
)


def test_rr_ratio_buy():
    # entry 2000, sl 1997 (risk 3), tp 2006 (reward 6) => 2.0
    assert rr_ratio(entry=2000.0, sl=1997.0, tp=2006.0) == pytest.approx(2.0)


def test_rr_ratio_zero_risk_safe():
    assert rr_ratio(entry=2000.0, sl=2000.0, tp=2006.0) == 0.0


def test_atr_percentile_midrange():
    series = pd.Series(np.arange(1, 101, dtype=float))  # 1..100
    # current atr = 50 => percentile ~0.49-0.50
    p = atr_percentile(series, current_atr=50.0)
    assert 0.45 <= p <= 0.55


def test_atr_percentile_max():
    series = pd.Series(np.arange(1, 11, dtype=float))
    assert atr_percentile(series, current_atr=100.0) == pytest.approx(1.0)


def test_body_to_range_ratio_full_body():
    # open 2000 close 2010 high 2010 low 2000 => body 10 range 10 => 1.0
    assert body_to_range_ratio(open_=2000.0, high=2010.0, low=2000.0, close=2010.0) == pytest.approx(1.0)


def test_body_to_range_ratio_doji():
    # body 0 range 10 => 0.0
    assert body_to_range_ratio(open_=2005.0, high=2010.0, low=2000.0, close=2005.0) == pytest.approx(0.0)


def test_body_to_range_zero_range_safe():
    assert body_to_range_ratio(open_=2000.0, high=2000.0, low=2000.0, close=2000.0) == 0.0


def test_dist_to_recent_swing_norm():
    # entry 2000, swing 2012, atr 4 => 3.0
    assert dist_to_recent_swing_norm(entry=2000.0, swing_price=2012.0, atr=4.0) == pytest.approx(3.0)


def test_dist_to_recent_swing_zero_atr_safe():
    assert dist_to_recent_swing_norm(entry=2000.0, swing_price=2012.0, atr=0.0) == 0.0


def test_htf_trend_aligned_match():
    assert htf_trend_aligned(direction=1, htf_trend=1) == 1
    assert htf_trend_aligned(direction=-1, htf_trend=-1) == 1


def test_htf_trend_aligned_mismatch():
    assert htf_trend_aligned(direction=1, htf_trend=-1) == 0
    assert htf_trend_aligned(direction=1, htf_trend=0) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_setup_features.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.setup_features'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/setup_features.py
"""Pure feature functions for SMC setups. Each has an explicit hypothesis
for why it should separate winning trades (TP) from losing trades (SL)."""
import numpy as np
import pandas as pd


def rr_ratio(entry: float, sl: float, tp: float) -> float:
    """Reward-to-risk ratio. Higher RR setups need lower winrate to be profitable."""
    risk = abs(float(entry) - float(sl))
    if risk <= 0.0:
        return 0.0
    return abs(float(tp) - float(entry)) / risk


def atr_percentile(atr_window: pd.Series, current_atr: float) -> float:
    """Where current volatility sits in its recent distribution (0..1).
    Hypothesis: extreme regimes change SMC reliability."""
    s = pd.to_numeric(pd.Series(atr_window), errors="coerce").dropna()
    if s.empty:
        return 0.0
    return float((s <= float(current_atr)).mean())


def body_to_range_ratio(open_: float, high: float, low: float, close: float) -> float:
    """Candle conviction. Hypothesis: strong bodies at entry signal momentum."""
    rng = float(high) - float(low)
    if rng <= 0.0:
        return 0.0
    return abs(float(close) - float(open_)) / rng


def dist_to_recent_swing_norm(entry: float, swing_price: float, atr: float) -> float:
    """Room to target in ATR units. Hypothesis: enough space to reach TP."""
    if float(atr) <= 0.0:
        return 0.0
    return abs(float(swing_price) - float(entry)) / float(atr)


def htf_trend_aligned(direction: int, htf_trend: int) -> int:
    """1 if trade direction matches higher-timeframe trend, else 0."""
    if int(htf_trend) == 0:
        return 0
    return 1 if int(direction) == int(htf_trend) else 0


def confluence_score(flags: list) -> int:
    """Count of SMC elements stacking at the entry (FVG+OB+pivot+...).
    Hypothesis: more confluence => higher-quality setup."""
    return int(sum(1 for f in flags if bool(f)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_setup_features.py -v`
Expected: PASS (12 passed)

- [ ] **Step 5: Commit**

```bash
git add tests/test_setup_features.py src/setup_features.py
git commit -m "feat: add pure setup feature functions with hypotheses"
```

---

### Task 5: Test `confluence_score` + integrasi fitur ke labeler

**Files:**
- Modify: `src/labeler.py` (tiap blok setup: hitung & tulis 6 fitur baru)
- Test: `tests/test_setup_features.py` (confluence), `tests/test_labeler_pnl_integration.py` (kolom fitur ada)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_setup_features.py (tambahkan)
from src.setup_features import confluence_score


def test_confluence_score_counts_true_flags():
    assert confluence_score([True, False, True, True]) == 3
    assert confluence_score([False, False]) == 0
    assert confluence_score([]) == 0
```

```python
# tests/test_labeler_pnl_integration.py (tambahkan)
def test_labeled_setups_have_new_features():
    df = _synthetic_uptrend_df()
    out = label_smc_setups(df, symbol="XAUUSD")
    if out.empty:
        import pytest
        pytest.skip("no setups produced")
    for col in ["rr_ratio", "atr_percentile", "body_to_range_ratio",
                "dist_to_recent_swing", "htf_trend_aligned", "confluence_score"]:
        assert col in out.columns, f"missing feature {col}"
        assert out[col].notna().all(), f"NaN leaked in {col}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_setup_features.py::test_confluence_score_counts_true_flags tests/test_labeler_pnl_integration.py::test_labeled_setups_have_new_features -v`
Expected: confluence test PASS langsung; integration test FAIL (kolom belum ada)

- [ ] **Step 3: Write minimal implementation**

Di `src/labeler.py`, import di atas:

```python
from src.setup_features import (
    rr_ratio, atr_percentile, body_to_range_ratio,
    dist_to_recent_swing_norm, htf_trend_aligned, confluence_score,
)
```

Sebelum loop `for i in range(len(df))`, siapkan ATR window helper:

```python
    atr_series_full = df['ATR_14']
```

Di dalam tiap blok setup, sebelum `setups.append`, hitung fitur (contoh untuk blok FVG; ulang pola sama di OB/BPR/Swap/Pivot dengan flag confluence yang sesuai):

```python
                    sig_o = float(df['Open'].iloc[i]); sig_h = float(df['High'].iloc[i])
                    sig_l = float(df['Low'].iloc[i]);  sig_c = float(df['Close'].iloc[i])
                    atr_win = atr_series_full.iloc[max(0, i-100):i+1]
                    htf_tr = int(df['floop_trend'].iloc[i]) if 'floop_trend' in df.columns else 0
                    # nearest swing high/low sebagai proxy target room
                    swing_col = 'Swing_High' if direction == 1 else 'Swing_Low'
                    swing_px = float(df[swing_col].iloc[i]) if swing_col in df.columns and pd.notna(df[swing_col].iloc[i]) else tp
                    feat_rr = rr_ratio(entry, sl, tp)
                    feat_atrp = atr_percentile(atr_win, atr_val)
                    feat_body = body_to_range_ratio(sig_o, sig_h, sig_l, sig_c)
                    feat_swing = dist_to_recent_swing_norm(entry, swing_px, atr_val)
                    feat_align = htf_trend_aligned(direction, htf_tr)
                    # confluence: FVG selalu True di blok FVG; cek OB/BPR/pivot co-occur
                    feat_conf = confluence_score([
                        True,
                        pd.notna(df.get('OB_Type', pd.Series([None]*len(df))).iloc[i]),
                        pd.notna(df.get('BPR_Type', pd.Series([None]*len(df))).iloc[i]),
                    ])
```

Tambah key di dict append:

```python
                        'rr_ratio': feat_rr,
                        'atr_percentile': feat_atrp,
                        'body_to_range_ratio': feat_body,
                        'dist_to_recent_swing': feat_swing,
                        'htf_trend_aligned': feat_align,
                        'confluence_score': feat_conf,
```

Untuk blok non-FVG, ganti elemen pertama confluence flags sesuai setup (mis. OB block: `[pd.notna(ob_type), pd.notna(fvg co-occur), ...]`).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_setup_features.py tests/test_labeler_pnl_integration.py -v`
Expected: PASS (atau SKIP bila sintetik kosong — jalankan full suite nanti)

- [ ] **Step 5: Commit**

```bash
git add tests/test_setup_features.py tests/test_labeler_pnl_integration.py src/labeler.py
git commit -m "feat: integrate 6 new entry features into all setup blocks"
```

---

## Workstream 3: Rework PIVOT_REJECTION → Reaction-Based Order Routing

### Task 6: Modul `src/reaction_router.py` — klasifikasi reaksi & order type

**Files:**
- Create: `src/reaction_router.py`
- Test: `tests/test_reaction_router.py`

> State machine: baca reaksi market di level (key level / S / R), pilih order type. Reuse `detect_rejection_at_level` dari `rejection_detector`. Output: `(reaction_state, order_type, reaction_strength)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_reaction_router.py
import pandas as pd
import numpy as np
import pytest
from src.reaction_router import (
    classify_reaction, ORDER_MARKET, ORDER_LIMIT, ORDER_STOP,
    STATE_CONFIRMED, STATE_APPROACHING, STATE_BREAKOUT, reaction_strength,
)


def _candle(o, h, l, c):
    return {"Open": o, "High": h, "Low": l, "Close": c}


def test_reaction_strength_pinbar():
    # bullish pinbar di support: low jauh di bawah, close balik atas
    # range 10, lower wick 6 => 0.6
    s = reaction_strength(_candle(2005, 2006, 1996, 2005), level=1997.0, direction=1)
    assert s == pytest.approx(0.6, abs=0.05)


def test_confirmed_rejection_buy_returns_market():
    # harga sudah menyentuh level & wick rejection kuat & close balik arah
    df = pd.DataFrame([
        _candle(2002, 2003, 2001, 2002),
        _candle(2000, 2001, 1996, 2000),   # touch level 1997 dgn lower wick
        _candle(2000, 2004, 1999, 2003),   # close balik naik
    ])
    state, order, strength = classify_reaction(df, level=1997.0, direction=1)
    assert state == STATE_CONFIRMED
    assert order == ORDER_MARKET


def test_approaching_returns_limit():
    # harga masih jauh DI ATAS support level, belum menyentuh
    df = pd.DataFrame([
        _candle(2010, 2011, 2009, 2010),
        _candle(2010, 2012, 2009, 2011),
        _candle(2011, 2013, 2010, 2012),
    ])
    state, order, strength = classify_reaction(df, level=1997.0, direction=1)
    assert state == STATE_APPROACHING
    assert order == ORDER_LIMIT


def test_breakout_momentum_returns_stop():
    # harga menembus resistance dgn momentum (close di atas level, body besar)
    df = pd.DataFrame([
        _candle(1995, 1996, 1994, 1995),
        _candle(1996, 1998, 1995, 1997),
        _candle(1997, 2003, 1997, 2002),   # break di atas level 1998 dgn body besar
    ])
    state, order, strength = classify_reaction(df, level=1998.0, direction=1)
    assert state == STATE_BREAKOUT
    assert order == ORDER_STOP
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_reaction_router.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.reaction_router'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/reaction_router.py
"""Reaction-based order routing for price interaction with key levels.

Reads how price reacts at a key level / support / resistance and decides the
appropriate order type, the way an institutional trader would: not every
touch is treated the same.
"""
import pandas as pd

# Order types (kategorikal, dipakai sebagai fitur model)
ORDER_MARKET = 0
ORDER_LIMIT = 1
ORDER_STOP = 2

# Reaction states
STATE_CONFIRMED = "CONFIRMED_REJECTION"
STATE_APPROACHING = "APPROACHING"
STATE_BREAKOUT = "BREAKOUT_MOMENTUM"
STATE_NONE = "NONE"


def reaction_strength(candle: dict, level: float, direction: int) -> float:
    """Rasio wick penolakan terhadap range candle di level (0..1)."""
    o = float(candle["Open"]); h = float(candle["High"])
    l = float(candle["Low"]);  c = float(candle["Close"])
    rng = h - l
    if rng <= 0.0:
        return 0.0
    if direction == 1:
        lower_wick = min(o, c) - l
        return max(0.0, min(lower_wick / rng, 1.0))
    upper_wick = h - max(o, c)
    return max(0.0, min(upper_wick / rng, 1.0))


def _touched(candle: dict, level: float, direction: int) -> bool:
    l = float(candle["Low"]); h = float(candle["High"])
    return l <= level <= h


def classify_reaction(df: pd.DataFrame, level: float, direction: int,
                      breakout_body_ratio: float = 0.6,
                      strong_wick_ratio: float = 0.5):
    """Return (state, order_type, strength).

    - CONFIRMED_REJECTION: candle terakhir menyentuh level, wick rejection >=
      strong_wick_ratio, dan close balik arah -> MARKET.
    - BREAKOUT_MOMENTUM: candle terakhir close menembus level searah dgn body
      besar (>= breakout_body_ratio) -> STOP.
    - APPROACHING: harga belum menyentuh level -> LIMIT.
    """
    if df is None or len(df) == 0:
        return STATE_NONE, ORDER_LIMIT, 0.0

    last = df.iloc[-1]
    candle = {"Open": float(last["Open"]), "High": float(last["High"]),
              "Low": float(last["Low"]), "Close": float(last["Close"])}
    rng = candle["High"] - candle["Low"]
    body_ratio = (abs(candle["Close"] - candle["Open"]) / rng) if rng > 0 else 0.0
    strength = reaction_strength(candle, level, direction)

    closed_back = (candle["Close"] > level) if direction == 1 else (candle["Close"] < level)
    broke_through = (candle["Close"] > level) if direction == 1 else (candle["Close"] < level)

    # CONFIRMED: touch + strong wick + close back in trade direction
    if _touched(candle, level, direction) and strength >= strong_wick_ratio and closed_back:
        return STATE_CONFIRMED, ORDER_MARKET, strength

    # BREAKOUT: closed beyond level with strong body, level was below(buy)/above(sell)
    prior_beyond = (float(df.iloc[-2]["Close"]) <= level) if (direction == 1 and len(df) >= 2) else \
                   (float(df.iloc[-2]["Close"]) >= level) if (direction == -1 and len(df) >= 2) else False
    if broke_through and body_ratio >= breakout_body_ratio and prior_beyond:
        return STATE_BREAKOUT, ORDER_STOP, strength

    # APPROACHING: price not yet at level
    if not _touched(candle, level, direction):
        return STATE_APPROACHING, ORDER_LIMIT, strength

    # default
    return STATE_APPROACHING, ORDER_LIMIT, strength
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_reaction_router.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add tests/test_reaction_router.py src/reaction_router.py
git commit -m "feat: add reaction-based order routing state machine"
```

---

### Task 7: Entry/SL/TP per order_type

**Files:**
- Modify: `src/reaction_router.py` (tambah `compute_levels`)
- Test: `tests/test_reaction_router.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_reaction_router.py (tambahkan)
from src.reaction_router import compute_levels


def test_market_levels_buy():
    # MARKET buy: entry = harga konfirmasi (close terakhir), SL di luar wick (low),
    # TP ke target.
    levels = compute_levels(ORDER_MARKET, direction=1,
                            confirm_price=2003.0, level=1997.0,
                            wick_extreme=1996.0, target=2010.0)
    assert levels["entry"] == pytest.approx(2003.0)
    assert levels["sl"] < 1997.0          # SL di luar level/wick
    assert levels["tp"] == pytest.approx(2010.0)


def test_limit_levels_buy():
    # LIMIT buy: entry = level, SL di luar level
    levels = compute_levels(ORDER_LIMIT, direction=1,
                            confirm_price=2005.0, level=1997.0,
                            wick_extreme=1996.0, target=2010.0)
    assert levels["entry"] == pytest.approx(1997.0)
    assert levels["sl"] < 1997.0


def test_stop_levels_buy():
    # STOP buy: entry = trigger breakout (di atas level), SL sisi berlawanan
    levels = compute_levels(ORDER_STOP, direction=1,
                            confirm_price=2002.0, level=1998.0,
                            wick_extreme=1997.0, target=2010.0)
    assert levels["entry"] >= 1998.0
    assert levels["sl"] < levels["entry"]
    assert levels["tp"] > levels["entry"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_reaction_router.py::test_market_levels_buy -v`
Expected: FAIL with `ImportError: cannot import name 'compute_levels'`

- [ ] **Step 3: Write minimal implementation**

Tambah di `src/reaction_router.py`:

```python
def compute_levels(order_type: int, direction: int, confirm_price: float,
                   level: float, wick_extreme: float, target: float,
                   sl_buffer: float = 0.2) -> dict:
    """Tentukan entry/SL/TP konsisten dengan order_type.

    direction 1=buy, -1=sell. wick_extreme = low(buy)/high(sell) candle reaksi.
    """
    if order_type == ORDER_MARKET:
        entry = float(confirm_price)
        sl = float(wick_extreme) - sl_buffer if direction == 1 else float(wick_extreme) + sl_buffer
    elif order_type == ORDER_LIMIT:
        entry = float(level)
        sl = float(wick_extreme) - sl_buffer if direction == 1 else float(wick_extreme) + sl_buffer
    else:  # ORDER_STOP
        entry = float(confirm_price)
        sl = float(level) - sl_buffer if direction == 1 else float(level) + sl_buffer
    return {"entry": entry, "sl": sl, "tp": float(target)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_reaction_router.py -v`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add tests/test_reaction_router.py src/reaction_router.py
git commit -m "feat: compute entry/SL/TP per order type"
```

---

### Task 8: Pakai reaction_router di blok pivot labeler

**Files:**
- Modify: `src/labeler.py` (blok PIVOT_REJECTION), tambah fitur `order_type`, `reaction_strength`
- Modify: `src/model_trainer.py` (pastikan `order_type`/`reaction_strength` bukan di NON_FEATURE_COLUMNS — keduanya FITUR, jadi tidak ditambahkan ke list itu)
- Test: `tests/test_labeler_pnl_integration.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_labeler_pnl_integration.py (tambahkan)
def test_pivot_setups_have_order_type_feature():
    df = _synthetic_uptrend_df(200)
    out = label_smc_setups(df, symbol="XAUUSD")
    if out.empty:
        import pytest
        pytest.skip("no setups produced")
    # order_type & reaction_strength harus ada sbg kolom (default utk non-pivot)
    assert "order_type" in out.columns
    assert "reaction_strength" in out.columns
    assert out["order_type"].notna().all()
    assert out["reaction_strength"].notna().all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_labeler_pnl_integration.py::test_pivot_setups_have_order_type_feature -v`
Expected: FAIL — kolom belum ada

- [ ] **Step 3: Write minimal implementation**

Di `src/labeler.py`:
1. Import: `from src.reaction_router import classify_reaction, compute_levels, reaction_strength`
2. Tambah `'order_type': 0` dan `'reaction_strength': 0.0` sebagai default di SEMUA blok non-pivot (FVG/OB/BPR/Swap) agar kolom konsisten.
3. Di blok PIVOT_REJECTION, ganti logika entry lama dengan:

```python
            window = df.iloc[max(0, i-5):i+1]
            state, order_type, rstrength = classify_reaction(window, level=pivot_level, direction=direction)
            wick_extreme = float(df['Low'].iloc[i]) if direction == 1 else float(df['High'].iloc[i])
            confirm_price = float(df['Close'].iloc[i])
            lv = compute_levels(order_type, direction, confirm_price, pivot_level, wick_extreme, target=pivot_target)
            entry, sl, tp = lv["entry"], lv["sl"], lv["tp"]
            risk_pips = abs(entry - sl)
            if risk_pips <= 0:
                continue
            label = simulate_trade(df, i + 1, direction, sl, tp, entry=entry, symbol=symbol)
            if label is not None:
                pnl_rel = compute_pnl_relative(int(label), entry, sl, tp)
                # ... append dict dgn semua fitur + 'order_type': order_type, 'reaction_strength': rstrength
```

(`pivot_level` dan `pivot_target` ambil dari logika pivot existing; bila nama beda, sesuaikan.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_labeler_pnl_integration.py -v`
Expected: PASS (atau SKIP bila kosong)

- [ ] **Step 5: Commit**

```bash
git add tests/test_labeler_pnl_integration.py src/labeler.py
git commit -m "feat: route pivot setups through reaction-based order logic"
```

---

## Workstream 4: Confidence Calibration (Isotonic)

### Task 9: Modul `src/calibrator.py`

**Files:**
- Create: `src/calibrator.py`
- Test: `tests/test_calibrator.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_calibrator.py
import numpy as np
import os
import tempfile
import pytest
from src.calibrator import fit_calibrator, apply_calibrator, save_calibrator, load_calibrator


def test_calibrator_monotonic_nondecreasing():
    rng = np.random.default_rng(0)
    raw = rng.uniform(0, 1, 500)
    # winrate naik seiring raw prob (noisy)
    y = (rng.uniform(0, 1, 500) < raw).astype(int)
    cal = fit_calibrator(raw, y)
    grid = np.linspace(0, 1, 50)
    mapped = apply_calibrator(cal, grid)
    diffs = np.diff(mapped)
    assert (diffs >= -1e-9).all(), "calibration must be non-decreasing"


def test_calibrator_maps_into_unit_interval():
    rng = np.random.default_rng(1)
    raw = rng.uniform(0, 1, 300)
    y = (rng.uniform(0, 1, 300) < raw).astype(int)
    cal = fit_calibrator(raw, y)
    out = apply_calibrator(cal, np.array([0.0, 0.5, 1.0]))
    assert (out >= 0.0).all() and (out <= 1.0).all()


def test_fit_calibrator_insufficient_data_returns_none():
    cal = fit_calibrator(np.array([0.5, 0.6]), np.array([1, 0]), min_samples=50)
    assert cal is None


def test_apply_none_calibrator_is_identity():
    out = apply_calibrator(None, np.array([0.2, 0.8]))
    assert out[0] == pytest.approx(0.2)
    assert out[1] == pytest.approx(0.8)


def test_save_and_load_roundtrip():
    rng = np.random.default_rng(2)
    raw = rng.uniform(0, 1, 200)
    y = (rng.uniform(0, 1, 200) < raw).astype(int)
    cal = fit_calibrator(raw, y)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "cal.joblib")
        save_calibrator(cal, path)
        loaded = load_calibrator(path)
    a = apply_calibrator(cal, np.array([0.3, 0.7]))
    b = apply_calibrator(loaded, np.array([0.3, 0.7]))
    assert np.allclose(a, b)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_calibrator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.calibrator'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/calibrator.py
"""Isotonic confidence calibration. Maps raw ensemble probability to empirical
winrate, trained on walk-forward OOF data (already out-of-sample, no leakage)."""
import os
import numpy as np
import joblib
from sklearn.isotonic import IsotonicRegression


def fit_calibrator(raw_probs, labels, min_samples: int = 50):
    """Fit isotonic regression raw_prob -> winrate. Returns None if too few rows."""
    raw = np.asarray(raw_probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    mask = np.isfinite(raw) & np.isfinite(y)
    raw, y = raw[mask], y[mask]
    if len(raw) < int(min_samples):
        return None
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(raw, y)
    return iso


def apply_calibrator(calibrator, raw_probs):
    """Apply calibrator. If None, identity (fallback)."""
    raw = np.asarray(raw_probs, dtype=float)
    if calibrator is None:
        return raw
    return np.clip(calibrator.predict(raw), 0.0, 1.0)


def save_calibrator(calibrator, path: str):
    if calibrator is None:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    joblib.dump(calibrator, path)


def load_calibrator(path: str):
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_calibrator.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add tests/test_calibrator.py src/calibrator.py
git commit -m "feat: add isotonic confidence calibrator"
```

---

### Task 10: Fit calibrator dari walk-forward OOF + simpan

**Files:**
- Modify: `src/calibration_report.py` (setelah OOF di-score, fit calibrator dari kolom OOF confidence + label, simpan ke `models/confidence_calibrator.joblib`)
- Test: `tests/test_calibration_report_calibrator.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_calibration_report_calibrator.py
import numpy as np
import pandas as pd
from src.calibration_report import fit_calibrator_from_scored


def test_fit_calibrator_from_scored_df():
    rng = np.random.default_rng(3)
    n = 300
    raw = rng.uniform(0, 1, n)
    label = (rng.uniform(0, 1, n) < raw).astype(int)
    df = pd.DataFrame({"confidence": raw, "label": label,
                       "confidence_source": ["walk_forward_oof"] * n})
    cal = fit_calibrator_from_scored(df)
    assert cal is not None
    # monotonic check
    grid = np.linspace(0, 1, 20)
    mapped = cal.predict(grid)
    assert (np.diff(mapped) >= -1e-9).all()


def test_fit_calibrator_ignores_unscored_rows():
    df = pd.DataFrame({
        "confidence": [np.nan, np.nan, 0.5],
        "label": [1, 0, 1],
        "confidence_source": ["unscored", "unscored", "walk_forward_oof"],
    })
    # hanya 1 baris ter-score => di bawah min_samples => None
    cal = fit_calibrator_from_scored(df, min_samples=50)
    assert cal is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_calibration_report_calibrator.py -v`
Expected: FAIL with `ImportError: cannot import name 'fit_calibrator_from_scored'`

- [ ] **Step 3: Write minimal implementation**

Tambah di `src/calibration_report.py`:

```python
from src.calibrator import fit_calibrator, save_calibrator


def fit_calibrator_from_scored(scored_df, min_samples: int = 50):
    """Fit isotonic calibrator dari baris yang ter-score out-of-fold."""
    import pandas as pd
    df = scored_df
    conf = pd.to_numeric(df.get("confidence"), errors="coerce")
    label = pd.to_numeric(df.get("label"), errors="coerce")
    mask = conf.notna() & label.isin([0, 1])
    if int(mask.sum()) < int(min_samples):
        return None
    return fit_calibrator(conf[mask].values, label[mask].astype(int).values,
                          min_samples=min_samples)
```

Di `main()` setelah `score_outcome_dataset(...)` (mode walk_forward), tambah:

```python
    if getattr(args, "save_calibrator", True):
        cal = fit_calibrator_from_scored(scored)
        cal_path = _resolve_project_path("models/confidence_calibrator.joblib")
        if cal is not None:
            save_calibrator(cal, cal_path)
            print(f"[Calibration] Isotonic calibrator saved to {cal_path}")
        else:
            print("[Calibration] Not enough OOF rows to fit calibrator; skipped.")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_calibration_report_calibrator.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add tests/test_calibration_report_calibrator.py src/calibration_report.py
git commit -m "feat: fit and save isotonic calibrator from walk-forward OOF"
```

---

### Task 11: Terapkan calibrator di inference

**Files:**
- Modify: `src/inference.py` (setelah hitung raw ensemble prob, apply calibrator bila ada)
- Test: `tests/test_inference_calibration.py`

> **Catatan:** Baca dulu bagaimana `inference.py` menghitung probabilitas ensemble saat ini (cari `predict_proba`). Sisipkan apply calibrator tepat setelah raw prob dihitung, sebelum threshold check.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_inference_calibration.py
import numpy as np
from src.inference import apply_confidence_calibration


def test_apply_calibration_uses_calibrator(tmp_path):
    from src.calibrator import fit_calibrator, save_calibrator
    rng = np.random.default_rng(4)
    raw = rng.uniform(0, 1, 200)
    y = (rng.uniform(0, 1, 200) < raw).astype(int)
    cal = fit_calibrator(raw, y)
    p = tmp_path / "cal.joblib"
    save_calibrator(cal, str(p))
    # raw 0.9 => calibrated value finite in [0,1]
    out = apply_confidence_calibration(0.9, calibrator_path=str(p))
    assert 0.0 <= out <= 1.0


def test_apply_calibration_fallback_identity_when_missing():
    out = apply_confidence_calibration(0.73, calibrator_path="does/not/exist.joblib")
    assert out == 0.73
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_inference_calibration.py -v`
Expected: FAIL with `ImportError: cannot import name 'apply_confidence_calibration'`

- [ ] **Step 3: Write minimal implementation**

Tambah di `src/inference.py`:

```python
from src.calibrator import load_calibrator, apply_calibrator
import numpy as np

_CALIBRATOR_CACHE = {}


def apply_confidence_calibration(raw_prob, calibrator_path="models/confidence_calibrator.joblib"):
    """Map raw ensemble prob -> calibrated confidence. Identity fallback if no calibrator."""
    if calibrator_path not in _CALIBRATOR_CACHE:
        _CALIBRATOR_CACHE[calibrator_path] = load_calibrator(calibrator_path)
    cal = _CALIBRATOR_CACHE[calibrator_path]
    return float(apply_calibrator(cal, np.array([float(raw_prob)]))[0])
```

Lalu di titik prob ensemble dihitung (setelah `(probs_xgb + probs_lgb)/2`), bungkus dengan `apply_confidence_calibration(...)` sebelum dibandingkan ke threshold. Resolusi path absolut pakai pola `_resolve_project_path` bila ada.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_inference_calibration.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add tests/test_inference_calibration.py src/inference.py
git commit -m "feat: apply isotonic calibration to inference confidence"
```

---

## Verifikasi Akhir

### Task 12: Regenerasi data + re-score walk-forward + bandingkan

**Files:**
- No code change (eksekusi + observasi). Bila perlu, simpan ringkasan ke `docs/`.

- [ ] **Step 1: Jalankan full unit suite**

Run: `python -m pytest -q`
Expected: semua hijau (atau hanya SKIP yang sudah diketahui).

- [ ] **Step 2: Re-label data historis dengan cost + fitur baru**

> Gunakan skrip pelabelan existing yang membaca `data/historical_xauusdm*.csv` → menulis `data/labeled_setups.csv`. (Cari entrypoint labeler di `src/` — mis. fungsi yang memanggil `label_smc_setups` per timeframe.) Backup dulu:

```bash
cp data/labeled_setups.csv data/labeled_setups_precostfeat_backup.csv
```

Jalankan pelabelan ulang (sesuaikan dengan entrypoint yang ada).

- [ ] **Step 3: Re-train model dengan fitur baru**

Run: `python -m src.model_trainer`
Expected: training jalan, fitur baru muncul di `Features: [...]`, model tersimpan.

- [ ] **Step 4: Re-score walk-forward + fit calibrator**

Run: `python -m src.calibration_report`
Expected: `scoring_mode=walk_forward`, calibrator tersimpan, report ke `data/calibration_report.json`.

- [ ] **Step 5: Bandingkan sebelum/sesudah**

Periksa:
- Expectancy per threshold (target: minimal satu threshold > 0 SETELAH cost).
- Calibration curve: `avg_confidence` per bucket vs `winrate_pct` aktual — harus makin sejajar (monotonik).
- PF per source (`real` vs `shadow`).

Laporkan apa adanya. Jika edge tetap tipis/negatif setelah cost, itu temuan valid: strategi dasar perlu revisi — bukan ditutupi.

- [ ] **Step 6: Commit hasil**

```bash
git add data/calibration_report.json models/ docs/
git commit -m "chore: re-score walk-forward with cost+features+calibration"
```

---

## Self-Review Notes

- **Spec coverage:** WS1 cost → Task 1-3. WS2 fitur → Task 4-5. WS3 rework → Task 6-8. WS4 kalibrasi → Task 9-11. Verifikasi → Task 12. ✅
- **Type consistency:** `compute_cost_r`, `compute_pnl_relative`, fitur di `setup_features.py`, `classify_reaction`/`compute_levels`/`reaction_strength` (konstanta ORDER_*/STATE_*), `fit_calibrator`/`apply_calibrator`/`save`/`load`, `fit_calibrator_from_scored`, `apply_confidence_calibration` — nama konsisten lintas task. ✅
- **No placeholders:** semua step punya kode/perintah konkret. Integrasi labeler (Task 3,5,8) memberi pola eksplisit untuk diulang di tiap blok karena struktur inline existing. ✅
- **Risiko diketahui:** cost menurunkan expectancy terlihat (disengaja); sintetik test bisa SKIP (full suite + Task 12 jadi verifikasi nyata).
