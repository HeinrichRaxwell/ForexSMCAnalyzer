import numpy as np
import pandas as pd

def calculate_tr(df: pd.DataFrame) -> pd.Series:
    high = df['High']
    low = df['Low']
    close_prev = df['Close'].shift(1)
    
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def rma(s: pd.Series, l: int) -> pd.Series:
    if l <= 0:
        return pd.Series(np.nan, index=s.index, dtype=float)

    src = pd.Series(s, index=s.index, dtype=float)
    out = pd.Series(np.nan, index=src.index, dtype=float)
    valid = src.dropna()
    if len(valid) < l:
        return out

    seed_label = valid.index[l - 1]
    seed_pos = src.index.get_loc(seed_label)
    prev = valid.iloc[:l].mean()
    out.iloc[seed_pos] = prev

    alpha = 1.0 / l
    for pos in range(seed_pos + 1, len(src)):
        val = src.iloc[pos]
        if pd.isna(val):
            out.iloc[pos] = prev
            continue
        prev = alpha * val + (1.0 - alpha) * prev
        out.iloc[pos] = prev

    return out

def calculate_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    return rma(calculate_tr(df), length)

def calculate_adx(df: pd.DataFrame, len_adx: int = 14) -> tuple:
    high = df['High']
    low = df['Low']
    
    up = high.diff()
    down = low.shift(1) - low
    
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    
    tr = calculate_tr(df)
    
    plus_di = 100 * rma(pd.Series(plus_dm, index=df.index), len_adx) / rma(tr, len_adx).replace(0, 0.00001)
    minus_di = 100 * rma(pd.Series(minus_dm, index=df.index), len_adx) / rma(tr, len_adx).replace(0, 0.00001)
    
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 0.00001) * 100
    adx_val = rma(dx, len_adx)
    
    return adx_val, plus_di, minus_di

def calculate_chop(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = calculate_tr(df)
    sum_tr = tr.rolling(period).sum()
    
    highest_high = df['High'].rolling(period).max()
    lowest_low = df['Low'].rolling(period).min()
    
    range_high_low = highest_high - lowest_low
    chop = pd.Series(50.0, index=df.index, dtype=float)
    valid_range = range_high_low > 0
    chop.loc[valid_range] = (
        100 * np.log10(sum_tr.loc[valid_range] / range_high_low.loc[valid_range]) / np.log10(period)
    )
    return chop

def _rolling_percentile_nearest_rank(src: pd.Series, window: int, percentile: float) -> pd.Series:
    rank = int(np.ceil(window * percentile / 100.0))
    rank = max(1, min(window, rank))

    def nearest_rank(values):
        clean = pd.Series(values).dropna().sort_values().to_numpy()
        if len(clean) < window:
            return np.nan
        return clean[rank - 1]

    return src.rolling(window).apply(nearest_rank, raw=False)

def calculate_range_filter(src: pd.Series, atr: pd.Series, sensitivity: float, atr_multiplier: float) -> tuple:
    n = len(src)
    filt = np.zeros(n)
    trend = np.zeros(n)
    sig = np.zeros(n)
    
    src_vals = src.values
    atr_vals = atr.values
    
    if n > 0:
        filt[0] = src_vals[0]
        
    for i in range(1, n):
        rng = atr_vals[i] * atr_multiplier * (sensitivity / 8.0)
        
        # Range Filter calculation
        if src_vals[i] > filt[i-1] + rng:
            filt[i] = src_vals[i] - rng
        elif src_vals[i] < filt[i-1] - rng:
            filt[i] = src_vals[i] + rng
        else:
            filt[i] = filt[i-1]
            
        # Trend
        if filt[i] > filt[i-1]:
            trend[i] = 1
        elif filt[i] < filt[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]
            
        # Signal
        prev_trend = trend[i-1]
        if trend[i] != prev_trend:
            sig[i] = trend[i]
            
    return pd.Series(filt, index=src.index), pd.Series(trend, index=src.index), pd.Series(sig, index=src.index)

def _apply_signal_gates(
    rf_sig: pd.Series,
    ema_gate,
    adx_trending,
    chop_clear,
    use_adx: bool,
    use_chop: bool,
    use_cooldown: bool,
    cooldown_len: int,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    index = rf_sig.index
    ema_gate = pd.Series(ema_gate, index=index).fillna(False).astype(bool)
    adx_trending = pd.Series(adx_trending, index=index).fillna(False).astype(bool)
    chop_clear = pd.Series(chop_clear, index=index).fillna(False).astype(bool)

    long_sig = []
    short_sig = []
    cooldown_clear = []
    chop_gate = []

    bars_since_sig = 999
    for i in range(len(rf_sig)):
        bars_since_sig += 1
        cooldown_ok = (not use_cooldown) or bars_since_sig >= cooldown_len
        base_gate = ((not use_adx) or bool(adx_trending.iloc[i])) and ((not use_chop) or bool(chop_clear.iloc[i]))
        gate_open = base_gate and cooldown_ok

        raw_sig = rf_sig.iloc[i]
        is_long = bool(raw_sig == 1 and ema_gate.iloc[i] and gate_open)
        is_short = bool(raw_sig == -1 and ema_gate.iloc[i] and gate_open)

        if is_long or is_short:
            bars_since_sig = 0

        long_sig.append(is_long)
        short_sig.append(is_short)
        cooldown_clear.append(bool(cooldown_ok))
        chop_gate.append(bool(gate_open))

    return (
        pd.Series(long_sig, index=index),
        pd.Series(short_sig, index=index),
        pd.Series(cooldown_clear, index=index),
        pd.Series(chop_gate, index=index),
    )

def _score_mtf_confluence(rf_trend: pd.Series, mtf_trends) -> pd.Series:
    score_mtf = pd.Series(0, index=rf_trend.index, dtype=int)
    if mtf_trends is None:
        return score_mtf

    if isinstance(mtf_trends, dict):
        aliases = (
            ("M5", "5m", "5min", "5"),
            ("M15", "15m", "15min", "15"),
            ("H1", "1h", "60min", "60"),
            ("H4", "4h", "240min", "240"),
        )
        trend_sources = []
        for alias_group in aliases:
            selected = None
            for key in alias_group:
                if key in mtf_trends and mtf_trends[key] is not None:
                    selected = mtf_trends[key]
                    break
            if selected is not None:
                trend_sources.append(selected)
    else:
        trend_sources = [trend for trend in mtf_trends if trend is not None][:4]

    for m_trend in trend_sources:
        aligned_m = m_trend.reindex(rf_trend.index).ffill()
        score_mtf += (aligned_m == rf_trend).fillna(False).astype(int)

    return score_mtf

def run_floop_pro(
    df: pd.DataFrame,
    sensitivity: int = 6,
    atr_len: int = 14,
    atr_mult: float = 0.8,
    use_adx: bool = True,
    adx_len: int = 14,
    adx_thresh: float = 20.0,
    use_chop: bool = True,
    chop_len: int = 14,
    chop_thresh: float = 61.8,
    use_cooldown: bool = True,
    cooldown_len: int = 5,
    ema_filter: bool = False,
    ema_fast_len: int = 60,
    ema_slow_len: int = 200,
    htf_trend_series: pd.Series = None,  # Custom HTF trend mapping
    mtf_trends: list = None             # List of series from other timeframes
):
    df = df.copy()
    original_index = df.index
    has_time_col = 'time' in df.columns
    if has_time_col:
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        
    close = df['Close']
    
    # 1. ATR calculation
    tr = calculate_tr(df)
    atr = calculate_atr(df, atr_len)
    atr_norm = atr / close * 100
    
    # 2. Volatility Ranking (last 60 bars)
    atr_vals = atr.values
    atr_rank = np.zeros(len(df))
    for i in range(len(df)):
        start = max(0, i - 59)
        window = atr_vals[start:i+1]
        if len(window) > 0:
            atr_rank[i] = np.sum(atr_vals[i] >= window) / len(window) * 100
            
    # Volatility scoring
    atr_norm_percentile_65 = _rolling_percentile_nearest_rank(atr_norm, 60, 65)
    score_vol = (np.where(atr_rank < 80, 1, 0) + np.where(atr_norm < atr_norm_percentile_65, 1, 0))
    
    # 3. Range Filter calculations (Sensitivity = 6)
    rf_filt, rf_trend, rf_sig = calculate_range_filter(close, atr, sensitivity, atr_mult)
    
    # 4. Momentum Calculation
    roc5 = (close - close.shift(5)) / close.shift(5).replace(0, 0.00001) * 100
    roc10 = (close - close.shift(10)) / close.shift(10).replace(0, 0.00001) * 100
    roc20 = (close - close.shift(20)) / close.shift(20).replace(0, 0.00001) * 100
    
    mom_bull = (roc5 > 0) & (roc10 > 0) & (roc20 > 0)
    mom_bear = (roc5 < 0) & (roc10 < 0) & (roc20 < 0)
    mom_aligned = np.where((rf_trend == 1) & mom_bull, 1, np.where((rf_trend == -1) & mom_bear, 1, 0))
    mom_partial = np.where((rf_trend == 1) & (roc5 > 0), 1, np.where((rf_trend == -1) & (roc5 < 0), 1, 0))
    
    # 5. EMA Alignment
    ema_fast = close.ewm(span=ema_fast_len, adjust=False).mean()
    ema_slow = close.ewm(span=ema_slow_len, adjust=False).mean()
    
    ema_cross_bull = ema_fast > ema_slow
    ema_price_above = close > ema_fast
    ema_slope_up = ema_fast > ema_fast.shift(5)
    ema_all_bull = ema_cross_bull & ema_price_above & ema_slope_up
    
    ema_cross_bear = ema_fast < ema_slow
    ema_price_below = close < ema_fast
    ema_slope_down = ema_fast < ema_fast.shift(5)
    ema_all_bear = ema_cross_bear & ema_price_below & ema_slope_down
    
    ema_cond1 = np.where(rf_trend == 1, ema_cross_bull, ema_cross_bear)
    ema_cond2 = np.where(rf_trend == 1, ema_price_above, ema_price_below)
    score_ema = np.minimum(ema_cond1.astype(int) + ema_cond2.astype(int) + mom_aligned + mom_partial, 4)
    
    ema_fully_aligned = np.where(rf_trend == 1, ema_all_bull, ema_all_bear)
    
    # 6. Anti-Chop: ADX Filter
    adx_val, _, _ = calculate_adx(df, adx_len)
    adx_trending = adx_val >= adx_thresh
    
    # Anti-Chop: Choppiness Index
    chop_index = calculate_chop(df, chop_len)
    chop_clear = chop_index <= chop_thresh
    
    chop_penalty = (np.where((use_adx) & np.logical_not(adx_trending), -1, 0) + 
                    np.where((use_chop) & np.logical_not(chop_clear), -1, 0))
    
    # 7. Gated Signals
    ema_gate = ema_fully_aligned if ema_filter else np.ones(len(df), dtype=bool)
    long_sig, short_sig, cooldown_clear, chop_gate = _apply_signal_gates(
        rf_sig=rf_sig,
        ema_gate=ema_gate,
        adx_trending=adx_trending,
        chop_clear=chop_clear,
        use_adx=use_adx,
        use_chop=use_chop,
        use_cooldown=use_cooldown,
        cooldown_len=cooldown_len,
    )
    
    # 8. Multi-Timeframe Bias
    score_htf = np.zeros(len(df))
    if htf_trend_series is not None:
        # Align HTF trend onto current DataFrame index
        htf_aligned = htf_trend_series.reindex(df.index).ffill().fillna(0)
        score_htf = np.where(rf_trend == htf_aligned, 1, 0)
        
    score_mtf = _score_mtf_confluence(rf_trend, mtf_trends)
            
    # 9. Sensitivity Cross-Checks
    # Sensitivity 12 and 16 trends
    _, sC_trend, _ = calculate_range_filter(close, atr, 12, atr_mult)
    _, sD_trend, _ = calculate_range_filter(close, atr, 16, atr_mult)
    score_sens = np.where(sC_trend == rf_trend, 2, 0) + np.where(sD_trend == rf_trend, 1, 0)
    
    # 10. Combined Signal Strength (0-14)
    signal_strength_raw = score_htf + score_mtf + score_sens + score_ema + score_vol
    signal_strength = np.maximum(0, np.minimum(signal_strength_raw + chop_penalty, 14))
    
    sig_series = pd.Series(np.where(long_sig, 1, np.where(short_sig, -1, 0)), index=df.index)
    strength_series = pd.Series(signal_strength, index=df.index)
    trend_series = pd.Series(rf_trend, index=df.index)
    
    if has_time_col:
        sig_series.index = original_index
        strength_series.index = original_index
        trend_series.index = original_index
        
    return sig_series, strength_series, trend_series
