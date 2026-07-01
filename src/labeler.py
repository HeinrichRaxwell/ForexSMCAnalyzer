import os
import sys
import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

# Add project root to python path if not present
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.smc_detector import detect_swing_points, detect_structures, detect_fvg_and_ob, detect_bpr, detect_snr_and_swapzones, detect_indecision_candles, detect_supply_demand_zones
from src.rejection_detector import is_near_psychological_level
from src.setup_features import (
    rr_ratio, atr_percentile, body_to_range_ratio,
    dist_to_recent_swing_norm, htf_trend_aligned, confluence_score,
)
from src.indicators.knn_classifier import run_knn_classifier, calculate_knn_probability_at_bar
from src.indicators.volume_clusters import calculate_volume_clusters

def _read_float_env(name: str, default: float) -> float:
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


def get_killzone(hour: int) -> int:
    """
    Get session killzone code relative to MT5 server time:
    1 for London (~09:00-12:00 MT5 time / 07:00-10:00 GMT)
    2 for NY (~14:00-17:00 MT5 time / 12:00-15:00 GMT)
    3 for Asia (~02:00-07:00 MT5 time / 00:00-05:00 GMT)
    0 for none.
    """
    if 9 <= hour < 12:
        return 1
    elif 14 <= hour < 17:
        return 2
    elif 2 <= hour < 7:
        return 3
    else:
        return 0

def get_timeframe_delta(df: pd.DataFrame) -> pd.Timedelta:
    """Estimate timeframe delta from the first two candles."""
    if len(df) >= 2:
        return df['time'].iloc[1] - df['time'].iloc[0]
    return pd.Timedelta(minutes=15) # Default fallback

_TICK_RESOLUTION_CACHE = {}

def resolve_ambiguity_with_ticks(symbol: str, start_time: pd.Timestamp, end_time: pd.Timestamp, direction: int, entry: float, sl: float, tp: float, is_filled: bool) -> tuple:
    """
    Fetch tick data from MT5 for the specified time range and simulate the exact path.
    Uses a global cache to avoid redundant network queries for identical trade scenarios.
    Returns: (is_filled_now, resolved_outcome)
    resolved_outcome is 1.0 (win), 0.0 (loss), or None (still open/no tick data).
    """
    global _TICK_RESOLUTION_CACHE
    cache_key = (symbol, start_time, end_time, direction, entry, sl, tp, is_filled)
    if cache_key in _TICK_RESOLUTION_CACHE:
        return _TICK_RESOLUTION_CACHE[cache_key]
        
    # Avoid fetching huge tick ranges for HTF (H4, D1) to prevent API timeouts
    if (end_time - start_time) > pd.Timedelta(hours=1):
        return is_filled, None

    if mt5 is None or "PYTEST_CURRENT_TEST" in os.environ:
        return is_filled, None # Fallback: let caller handle candle-level logic
        
    # Make sure MT5 is initialized
    if not mt5.initialize():
        return is_filled, None
        
    # Try to resolve symbol suffix
    symbols_to_try = [symbol, symbol + "m", symbol + ".", "GOLD"]
    active_sym = None
    for sym in symbols_to_try:
        mt5.symbol_select(sym, True)
        rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M15, 0, 1)
        if rates is not None and len(rates) > 0:
            active_sym = sym
            break
            
    if not active_sym:
        res_val = (is_filled, None)
        _TICK_RESOLUTION_CACHE[cache_key] = res_val
        return res_val
        
    dt_from = start_time.to_pydatetime()
    dt_to = end_time.to_pydatetime()
    
    ticks = mt5.copy_ticks_range(active_sym, dt_from, dt_to, mt5.COPY_TICKS_ALL)
    if ticks is None or len(ticks) == 0:
        res_val = (is_filled, None)
        _TICK_RESOLUTION_CACHE[cache_key] = res_val
        return res_val

    # Verify price scale compatibility (avoid mixing synthetic test data with real ticks)
    first_tick_price = ticks[0]['bid']
    if abs(first_tick_price - entry) > entry * 0.5:
        res_val = (is_filled, None)
        _TICK_RESOLUTION_CACHE[cache_key] = res_val
        return res_val
        
    filled = is_filled
    for tick in ticks:
        price = tick['bid']
        ask_price = tick['ask']
        
        if not filled:
            if direction == 1: # Buy
                if ask_price <= entry:
                    filled = True
            else: # Sell
                if price >= entry:
                    filled = True
                    
            if not filled:
                if direction == 1:
                    if price <= sl: # Invalidated before entry
                        res_val = (False, None)
                        _TICK_RESOLUTION_CACHE[cache_key] = res_val
                        return res_val
                    if price >= tp: # Mitigated before entry
                        res_val = (False, None)
                        _TICK_RESOLUTION_CACHE[cache_key] = res_val
                        return res_val
                else:
                    if price >= sl: # Invalidated
                        res_val = (False, None)
                        _TICK_RESOLUTION_CACHE[cache_key] = res_val
                        return res_val
                    if price <= tp: # Mitigated
                        res_val = (False, None)
                        _TICK_RESOLUTION_CACHE[cache_key] = res_val
                        return res_val
                        
        if filled:
            if direction == 1:
                if price <= sl:
                    res_val = (True, 0.0) # Loss
                    _TICK_RESOLUTION_CACHE[cache_key] = res_val
                    return res_val
                if price >= tp:
                    res_val = (True, 1.0) # Win
                    _TICK_RESOLUTION_CACHE[cache_key] = res_val
                    return res_val
            else:
                if price >= sl:
                    res_val = (True, 0.0) # Loss
                    _TICK_RESOLUTION_CACHE[cache_key] = res_val
                    return res_val
                if price <= tp:
                    res_val = (True, 1.0) # Win
                    _TICK_RESOLUTION_CACHE[cache_key] = res_val
                    return res_val
                    
    res_val = (filled, None)
    _TICK_RESOLUTION_CACHE[cache_key] = res_val
    return res_val

def simulate_trade(df: pd.DataFrame, start_idx: int, direction: int, sl: float, tp: float, entry: float = None, symbol: str = "XAUUSD") -> float:
    """
    Simulate trade entry and outcome using tick data for ambiguous candles.
    If entry is None, assumes immediate entry at start_idx (for backward compatibility).
    
    Returns:
        1.0 if Win (TP hit first)
        0.0 if Loss (SL hit first)
        None if trade doesn't resolve by the end of the dataframe
    """
    if entry is None:
        # Backward compatible mode (no entry trigger required, assumes filled immediately)
        for j in range(start_idx, len(df)):
            high = df['High'].iloc[j]
            low = df['Low'].iloc[j]
            
            if direction == 1:  # Bullish (Buy)
                is_sl_hit = (low <= sl)
                is_tp_hit = (high >= tp)
                if is_sl_hit and is_tp_hit:
                    return 0.0
                elif is_sl_hit:
                    return 0.0
                elif is_tp_hit:
                    return 1.0
            else:  # Bearish (Sell)
                is_sl_hit = (high >= sl)
                is_tp_hit = (low <= tp)
                if is_sl_hit and is_tp_hit:
                    return 0.0
                elif is_sl_hit:
                    return 0.0
                elif is_tp_hit:
                    return 1.0
        return None

    # Full retracement & tick resolution simulation
    tf_delta = get_timeframe_delta(df)
    filled = False
    
    # Initialize connection once if needed
    mt5_active = False
    if mt5 is not None and "PYTEST_CURRENT_TEST" not in os.environ:
        mt5_active = mt5.initialize()
        
    for j in range(start_idx, len(df)):
        high = df['High'].iloc[j]
        low = df['Low'].iloc[j]
        candle_time = df['time'].iloc[j]
        
        if not filled:
            # Check entry conditions
            if direction == 1: # Buy
                has_entry = (low <= entry)
                has_sl = (low <= sl)
                has_tp = (high >= tp)
                
                if has_entry and (has_sl or has_tp):
                    outcome = None
                    if mt5_active:
                        filled, outcome = resolve_ambiguity_with_ticks(symbol, candle_time, candle_time + tf_delta, direction, entry, sl, tp, False)
                    if outcome is not None:
                        return outcome
                    if has_sl:
                        return 0.0
                    if has_tp:
                        return 1.0
                elif has_entry:
                    filled = True
                elif has_sl or has_tp:
                    return None
            else: # Sell
                has_entry = (high >= entry)
                has_sl = (high >= sl)
                has_tp = (low <= tp)
                
                if has_entry and (has_sl or has_tp):
                    outcome = None
                    if mt5_active:
                        filled, outcome = resolve_ambiguity_with_ticks(symbol, candle_time, candle_time + tf_delta, direction, entry, sl, tp, False)
                    if outcome is not None:
                        return outcome
                    if has_sl:
                        return 0.0
                    if has_tp:
                        return 1.0
                elif has_entry:
                    filled = True
                elif has_sl or has_tp:
                    return None
                    
        # If filled
        if filled:
            if direction == 1:
                is_sl_hit = (low <= sl)
                is_tp_hit = (high >= tp)
                
                if is_sl_hit and is_tp_hit:
                    outcome = None
                    if mt5_active:
                        _, outcome = resolve_ambiguity_with_ticks(symbol, candle_time, candle_time + tf_delta, direction, entry, sl, tp, True)
                    if outcome is not None:
                        return outcome
                    return 0.0
                elif is_sl_hit:
                    return 0.0
                elif is_tp_hit:
                    return 1.0
            else:
                is_sl_hit = (high >= sl)
                is_tp_hit = (low <= tp)
                
                if is_sl_hit and is_tp_hit:
                    outcome = None
                    if mt5_active:
                        _, outcome = resolve_ambiguity_with_ticks(symbol, candle_time, candle_time + tf_delta, direction, entry, sl, tp, True)
                    if outcome is not None:
                        return outcome
                    return 0.0
                elif is_sl_hit:
                    return 0.0
                elif is_tp_hit:
                    return 1.0

                    
    return None


def label_smc_setups(df: pd.DataFrame, buffer: float = 0.5, symbol: str = "XAUUSD", tf_trends: dict = None, df_d1: pd.DataFrame = None) -> pd.DataFrame:
    """
    Load historical data, run SMC detection algorithms, and simulate trade setups
    to label them with outcomes.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing OHLCV.
        buffer (float): Distance in USD to add to SL (below Low for Buy, above High for Sell).
        symbol (str): Symbol name (e.g., "XAUUSD", "EURUSD").
        tf_trends (dict): Pre-calculated trends from other timeframes for FLOOP Pro.
        df_d1 (pd.DataFrame): Daily DataFrame to calculate key pivot levels.
        
    Returns:
        pd.DataFrame: A DataFrame of labeled setups with features.
    """
    df = df.copy()
    
    # Ensure time is datetime
    df['time'] = pd.to_datetime(df['time'])
    
    # Estimate timeframe
    tf_delta = get_timeframe_delta(df)
    timeframe_minutes = int(tf_delta.total_seconds() / 60)
    
    # Run SMC detectors
    df = detect_swing_points(df, window=5)
    df = detect_structures(df)
    df = detect_fvg_and_ob(df, symbol=symbol)
    df = detect_snr_and_swapzones(df, symbol=symbol)
    df = detect_bpr(df, symbol=symbol)
    df = detect_indecision_candles(df, symbol=symbol)
    df = detect_supply_demand_zones(df, symbol=symbol)
    
    # Align daily pivots if df_d1 is provided
    if df_d1 is not None:
        from src.indicators.pivots import align_daily_pivots
        try:
            df = align_daily_pivots(df, df_d1)
        except Exception as e:
            print(f"Error aligning daily pivots: {e}")
            
    # Pre-calculate KNN features for lazy execution
    try:
        print("Pre-calculating KNN features...")
        pc1, pc2, pc3, pc4, target_clean = run_knn_classifier(
            df,
            atr_period=10, factor=2.0,
            k_neighbors=10, sampling_window_size=1000, momentum_window=10,
            normalizing_window_size=1000,
            lazy=True
        )
        pc1_vals = pc1.values
        pc2_vals = pc2.values
        pc3_vals = pc3.values
        pc4_vals = pc4.values
        tgt_vals = target_clean.values
        has_knn = True
    except Exception as e:
        print(f"Error pre-calculating KNN features: {e}")
        has_knn = False
    
    # Calculate ATR_14
    close_prev = df['Close'].shift(1).fillna(df['Open'])
    tr = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            np.abs(df['High'] - close_prev),
            np.abs(df['Low'] - close_prev)
        )
    )
    df['ATR_14'] = tr.rolling(window=14, min_periods=1).mean()
    
    # Calculate FLOOP Pro signals
    from src.indicators.floop import run_floop_pro
    htf_trend_series = None
    mtf_trends_list = None
    if tf_trends is not None:
        htf_trend_series = tf_trends.get('H4')
        if htf_trend_series is None:
            htf_trend_series = tf_trends.get('4h')
        mtf_trends_list = tf_trends
        
    floop_signals, floop_strengths, floop_trends = run_floop_pro(
        df,
        sensitivity=6,
        atr_len=14,
        atr_mult=0.8,
        use_adx=True,
        adx_thresh=20.0,
        use_chop=True,
        chop_thresh=61.8,
        use_cooldown=True,
        cooldown_len=5,
        ema_filter=False,
        htf_trend_series=htf_trend_series,
        mtf_trends=mtf_trends_list
    )
    df['floop_signal'] = floop_signals
    df['floop_strength'] = floop_strengths
    df['floop_trend'] = floop_trends
    
    setups = []
    from src.indicators.pivots import get_pivot_features_at_idx
    
    for i in range(len(df)):
        # Skip if ATR_14 is NaN
        if pd.isna(df['ATR_14'].iloc[i]):
            continue
            
        t_val = df['time'].iloc[i]
        hour_val = int(t_val.hour)
        day_of_week_val = int(t_val.dayofweek)
        trend_val = int(df['Trend'].iloc[i])
        killzone_val = get_killzone(hour_val)
        atr_val = df['ATR_14'].iloc[i]
        floop_sig_val = int(df['floop_signal'].iloc[i]) if 'floop_signal' in df.columns else 0
        floop_strength_val = float(df['floop_strength'].iloc[i]) if 'floop_strength' in df.columns else 0.0
        
        # 1. Check FVG Setup
        fvg_type = df['FVG_Type'].iloc[i] if 'FVG_Type' in df.columns else None
        if pd.notna(fvg_type) and fvg_type is not None:
            fvg_top = df['FVG_Top'].iloc[i]
            fvg_bottom = df['FVG_Bottom'].iloc[i]
            fvg_sl = df['FVG_SL'].iloc[i]
            
            if fvg_type == 'BULLISH':
                direction = 1
                fvg_width = df['Low'].iloc[i] - df['High'].iloc[i-2]
            else: # BEARISH
                direction = -1
                fvg_width = df['Low'].iloc[i-2] - df['High'].iloc[i]
                
            for entry_col in ['FVG_Fibo_0.5', 'FVG_Fibo_0.618']:
                entry = df[entry_col].iloc[i]
                sl = fvg_sl
                tp = df['FVG_Fibo_0.0'].iloc[i]
                
                risk_pips = abs(entry - sl)
                p_feat = get_pivot_features_at_idx(df, i, entry)
                    
                label = simulate_trade(df, i + 1, direction, sl, tp, entry=entry, symbol=symbol)
                if label is not None:
                    setups.append({
                        'time': t_val,
                        'timeframe': timeframe_minutes,
                        'hour': hour_val,
                        'day_of_week': day_of_week_val,
                        'setup_type': 0,  # FVG
                        'strategy': 'FVG',
                        'direction': direction,
                        'entry_price': entry,
                        'sl_price': sl,
                        'tp_price': tp,
                        'risk_pips': risk_pips,
                        'atr_14': atr_val,
                        'trend': trend_val,
                        'relative_risk': risk_pips / atr_val,
                        'killzone': killzone_val,
                        'fvg_width': fvg_width,
                        'relative_fvg_width': fvg_width / atr_val,
                        'floop_signal': floop_sig_val,
                        'floop_strength': floop_strength_val,
                        'dist_entry_to_pp': p_feat['dist_entry_to_pp'],
                        'dist_entry_to_nearest_pivot': p_feat['dist_entry_to_nearest_pivot'],
                        'label': int(label)
                    })
                    
        # 2. Check OB Setup
        ob_type = df['OB_Type'].iloc[i] if 'OB_Type' in df.columns else None
        if pd.notna(ob_type) and ob_type is not None:
            ob_top = df['OB_Top'].iloc[i]
            ob_bottom = df['OB_Bottom'].iloc[i]
            ob_sl = df['OB_SL'].iloc[i]
            direction = 1 if ob_type == 'BULLISH' else -1
                
            for entry_col in ['OB_Fibo_0.5', 'OB_Fibo_0.618']:
                entry = df[entry_col].iloc[i]
                sl = ob_sl
                tp = df['OB_Fibo_0.0'].iloc[i]
                risk_pips = abs(entry - sl)
                p_feat = get_pivot_features_at_idx(df, i, entry)
                
                label = simulate_trade(df, i + 1, direction, sl, tp, entry=entry, symbol=symbol)
                if label is not None:
                    setups.append({
                        'time': t_val,
                        'timeframe': timeframe_minutes,
                        'hour': hour_val,
                        'day_of_week': day_of_week_val,
                        'setup_type': 1,  # OB
                        'strategy': 'OB',
                        'direction': direction,
                        'entry_price': entry,
                        'sl_price': sl,
                        'tp_price': tp,
                        'risk_pips': risk_pips,
                        'atr_14': atr_val,
                        'trend': trend_val,
                        'relative_risk': risk_pips / atr_val,
                        'killzone': killzone_val,
                        'fvg_width': 0.0,
                        'relative_fvg_width': 0.0,
                        'floop_signal': floop_sig_val,
                        'floop_strength': floop_strength_val,
                        'dist_entry_to_pp': p_feat['dist_entry_to_pp'],
                        'dist_entry_to_nearest_pivot': p_feat['dist_entry_to_nearest_pivot'],
                        'label': int(label)
                    })
 
        # 3. Check BPR Setup
        bpr_type = df['BPR_Type'].iloc[i] if 'BPR_Type' in df.columns else None
        if pd.notna(bpr_type) and bpr_type is not None:
            bpr_top = df['BPR_Top'].iloc[i]
            bpr_bottom = df['BPR_Bottom'].iloc[i]
            bpr_sl = df['BPR_SL'].iloc[i]
            direction = 1 if bpr_type == 'BULLISH' else -1
            
            for entry_col in ['BPR_Fibo_0.5', 'BPR_Fibo_0.618']:
                entry = df[entry_col].iloc[i]
                sl = bpr_sl
                tp = df['BPR_Fibo_0.0'].iloc[i]
                risk_pips = abs(entry - sl)
                p_feat = get_pivot_features_at_idx(df, i, entry)
                
                label = simulate_trade(df, i + 1, direction, sl, tp, entry=entry, symbol=symbol)
                if label is not None:
                    setups.append({
                        'time': t_val,
                        'timeframe': timeframe_minutes,
                        'hour': hour_val,
                        'day_of_week': day_of_week_val,
                        'setup_type': 0,  # Treat as FVG for ML
                        'strategy': 'BPR',
                        'direction': direction,
                        'entry_price': entry,
                        'sl_price': sl,
                        'tp_price': tp,
                        'risk_pips': risk_pips,
                        'atr_14': atr_val,
                        'trend': trend_val,
                        'relative_risk': risk_pips / atr_val,
                        'killzone': killzone_val,
                        'fvg_width': abs(bpr_top - bpr_bottom),
                        'relative_fvg_width': abs(bpr_top - bpr_bottom) / atr_val,
                        'floop_signal': floop_sig_val,
                        'floop_strength': floop_strength_val,
                        'dist_entry_to_pp': p_feat['dist_entry_to_pp'],
                        'dist_entry_to_nearest_pivot': p_feat['dist_entry_to_nearest_pivot'],
                        'label': int(label)
                    })
                    
        # 4. Check Swapzone Setup
        swap_type = df['Swap_Type'].iloc[i] if 'Swap_Type' in df.columns else None
        if pd.notna(swap_type) and swap_type is not None:
            swap_sl = df['Swap_SL'].iloc[i]
            direction = 1 if swap_type == 'SUPPORT' else -1
            
            for entry_col in ['Swap_Fibo_0.5', 'Swap_Fibo_0.618']:
                entry = df[entry_col].iloc[i]
                sl = swap_sl
                tp = df['Swap_Fibo_0.0'].iloc[i]
                risk_pips = abs(entry - sl)
                p_feat = get_pivot_features_at_idx(df, i, entry)
                
                label = simulate_trade(df, i + 1, direction, sl, tp, entry=entry, symbol=symbol)
                if label is not None:
                    setups.append({
                        'time': t_val,
                        'timeframe': timeframe_minutes,
                        'hour': hour_val,
                        'day_of_week': day_of_week_val,
                        'setup_type': 1,  # Treat as OB for ML
                        'strategy': 'Swapzone',
                        'direction': direction,
                        'entry_price': entry,
                        'sl_price': sl,
                        'tp_price': tp,
                        'risk_pips': risk_pips,
                        'atr_14': atr_val,
                        'trend': trend_val,
                        'relative_risk': risk_pips / atr_val,
                        'killzone': killzone_val,
                        'fvg_width': 0.0,
                        'relative_fvg_width': 0.0,
                        'floop_signal': floop_sig_val,
                        'floop_strength': floop_strength_val,
                        'dist_entry_to_pp': p_feat['dist_entry_to_pp'],
                        'dist_entry_to_nearest_pivot': p_feat['dist_entry_to_nearest_pivot'],
                        'label': int(label)
                    })
                    
        # 5. Check Indecision Candle Setup
        ic_type = df['IC_Type'].iloc[i] if 'IC_Type' in df.columns else None
        if pd.notna(ic_type) and ic_type is not None:
            ic_sl = df['IC_SL'].iloc[i]
            direction = 1 if ic_type == 'BULLISH' else -1
            
            for entry_col in ['IC_Fibo_0.5', 'IC_Fibo_0.618']:
                entry = df[entry_col].iloc[i]
                sl = ic_sl
                tp = df['IC_Fibo_0.0'].iloc[i]
                risk_pips = abs(entry - sl)
                p_feat = get_pivot_features_at_idx(df, i, entry)
                
                label = simulate_trade(df, i + 1, direction, sl, tp, entry=entry, symbol=symbol)
                if label is not None:
                    setups.append({
                        'time': t_val,
                        'timeframe': timeframe_minutes,
                        'hour': hour_val,
                        'day_of_week': day_of_week_val,
                        'setup_type': 1,  # Treat as OB for ML
                        'strategy': 'IC',
                        'direction': direction,
                        'entry_price': entry,
                        'sl_price': sl,
                        'tp_price': tp,
                        'risk_pips': risk_pips,
                        'atr_14': atr_val,
                        'trend': trend_val,
                        'relative_risk': risk_pips / atr_val,
                        'killzone': killzone_val,
                        'fvg_width': 0.0,
                        'relative_fvg_width': 0.0,
                        'floop_signal': floop_sig_val,
                        'floop_strength': floop_strength_val,
                        'dist_entry_to_pp': p_feat['dist_entry_to_pp'],
                        'dist_entry_to_nearest_pivot': p_feat['dist_entry_to_nearest_pivot'],
                        'label': int(label)
                    })
                    
        # 6. Check Supply & Demand Setup
        sd_type = df['SD_Type'].iloc[i] if 'SD_Type' in df.columns else None
        if pd.notna(sd_type) and sd_type is not None:
            sd_sl = df['SD_SL'].iloc[i]
            direction = 1 if 'DEMAND' in sd_type else -1
            
            for entry_col in ['SD_Fibo_0.5', 'SD_Fibo_0.618']:
                entry = df[entry_col].iloc[i]
                sl = sd_sl
                tp = df['SD_Fibo_0.0'].iloc[i]
                risk_pips = abs(entry - sl)
                p_feat = get_pivot_features_at_idx(df, i, entry)
                
                label = simulate_trade(df, i + 1, direction, sl, tp, entry=entry, symbol=symbol)
                if label is not None:
                    setups.append({
                        'time': t_val,
                        'timeframe': timeframe_minutes,
                        'hour': hour_val,
                        'day_of_week': day_of_week_val,
                        'setup_type': 1,  # Treat as OB for ML
                        'strategy': 'SND',
                        'direction': direction,
                        'entry_price': entry,
                        'sl_price': sl,
                        'tp_price': tp,
                        'risk_pips': risk_pips,
                        'atr_14': atr_val,
                        'trend': trend_val,
                        'relative_risk': risk_pips / atr_val,
                        'killzone': killzone_val,
                        'fvg_width': 0.0,
                        'relative_fvg_width': 0.0,
                        'floop_signal': floop_sig_val,
                        'floop_strength': floop_strength_val,
                        'dist_entry_to_pp': p_feat['dist_entry_to_pp'],
                        'dist_entry_to_nearest_pivot': p_feat['dist_entry_to_nearest_pivot'],
                        'label': int(label)
                    })

        # 7. Check Pivot Rejection Setup
        from src.indicators.pivots import detect_pivot_rejection_setups_at_idx
        pivot_setups = detect_pivot_rejection_setups_at_idx(df, i, symbol=symbol)
        for ps in pivot_setups:
            direction = ps['direction']
            entry = ps['entry_price']
            sl = ps['sl_price']
            tp = ps['tp_price']
            risk_pips = abs(entry - sl)
            p_feat = get_pivot_features_at_idx(df, i, entry)
            
            label = simulate_trade(df, i + 1, direction, sl, tp, entry=entry, symbol=symbol)
            if label is not None:
                setups.append({
                    'time': t_val,
                    'timeframe': timeframe_minutes,
                    'hour': hour_val,
                    'day_of_week': day_of_week_val,
                    'setup_type': 2,  # 2: Pivot Rejection
                    'strategy': 'Pivot',
                    'direction': direction,
                    'entry_price': entry,
                    'sl_price': sl,
                    'tp_price': tp,
                    'risk_pips': risk_pips,
                    'atr_14': atr_val,
                    'trend': trend_val,
                    'relative_risk': risk_pips / atr_val,
                    'killzone': killzone_val,
                    'fvg_width': 0.0,
                    'relative_fvg_width': 0.0,
                    'floop_signal': floop_sig_val,
                    'floop_strength': floop_strength_val,
                    'dist_entry_to_pp': p_feat['dist_entry_to_pp'],
                    'dist_entry_to_nearest_pivot': p_feat['dist_entry_to_nearest_pivot'],
                    'label': int(label)
                })
                    
    time_to_idx = {t: idx for idx, t in enumerate(df['time'])}
    
    print("Appending ML indicator features...")
    vp_cache = {}
    
    for s in setups:
        t_val = s['time']
        s['near_psychological_level'] = int(is_near_psychological_level(s['entry_price'], symbol))
        
        # Calculate simulated PnL relative to risk, adjusted for trading cost
        # (spread + slippage). Win/loss label is unchanged; only the R magnitude
        # is reduced by cost_R so expectancy reflects real-world friction.
        s['pnl_relative'] = compute_pnl_relative(
            int(s['label']), s['entry_price'], s['sl_price'], s['tp_price']
        )
        
        idx = time_to_idx.get(t_val)
        if idx is not None:
            floop_trend_val = int(df['floop_trend'].iloc[idx]) if 'floop_trend' in df.columns else 0
            s['floop_trend'] = floop_trend_val
            s['floop_trend_aligned'] = 1 if floop_trend_val == s['direction'] else 0

            # --- New entry-quality features (model inputs only; strategy rules unchanged) ---
            entry_px = s['entry_price']; sl_px = s['sl_price']; tp_px = s['tp_price']
            atr_now = s.get('atr_14', 0.0) or 0.0
            direction = s['direction']

            # rr_ratio: explicit reward-to-risk at entry
            s['rr_ratio'] = rr_ratio(entry_px, sl_px, tp_px)

            # atr_percentile: where current volatility sits in its trailing window
            atr_win = df['ATR_14'].iloc[max(0, idx - 100):idx + 1] if 'ATR_14' in df.columns else pd.Series(dtype=float)
            s['atr_percentile'] = atr_percentile(atr_win, atr_now)

            # body_to_range_ratio: conviction of the signal candle at idx
            s['body_to_range_ratio'] = body_to_range_ratio(
                float(df['Open'].iloc[idx]), float(df['High'].iloc[idx]),
                float(df['Low'].iloc[idx]), float(df['Close'].iloc[idx])
            )

            # dist_to_recent_swing: room to the most recent opposing swing (ATR-normalized)
            swing_col = 'Swing_High' if direction == 1 else 'Swing_Low'
            swing_px = tp_px  # fallback to target if no swing found
            if swing_col in df.columns:
                prior = df[swing_col].iloc[max(0, idx - 100):idx + 1].dropna()
                if not prior.empty:
                    swing_px = float(prior.iloc[-1])
            s['dist_to_recent_swing'] = dist_to_recent_swing_norm(entry_px, swing_px, atr_now)

            # htf_trend_aligned: direction matches FLOOP higher-timeframe trend
            s['htf_trend_aligned'] = htf_trend_aligned(direction, floop_trend_val)

            # confluence_score: how many SMC elements co-occur at this bar
            def _present(col):
                return col in df.columns and pd.notna(df[col].iloc[idx])
            s['confluence_score'] = confluence_score([
                _present('FVG_Type'), _present('OB_Type'), _present('BPR_Type'),
                _present('Swap_Type'), _present('SND_Type'),
                s.get('near_psychological_level', 0) == 1,
            ])

            # KNN (lazy evaluation only on index)
            if has_knn:
                knn_up, knn_down = calculate_knn_probability_at_bar(
                    idx, pc1_vals, pc2_vals, pc3_vals, pc4_vals, tgt_vals,
                    k=10, sampling_window=1000, stride=10
                )
            else:
                knn_up, knn_down = 0.0, 0.0
            s['knn_prob_sig'] = knn_up if s['direction'] == 1 else knn_down
            s['knn_prob_opp'] = knn_down if s['direction'] == 1 else knn_up
            
            # K-Means Volume Profile
            if idx >= 200:
                if idx not in vp_cache:
                    try:
                        vp_cache[idx] = calculate_volume_clusters(
                            df.iloc[:idx+1], lookback=200, k=5, iterations=20, rows=20
                        )
                    except Exception:
                        vp_cache[idx] = {}
                
                clusters_data = vp_cache[idx]
                if clusters_data and 'current_poc' in clusters_data:
                    curr_poc = clusters_data['current_poc']
                    entry = s['entry_price']
                    s['dist_entry_to_poc'] = (entry - curr_poc) / curr_poc if curr_poc > 0 else 0.0
                    
                    pocs = clusters_data.get('pocs', [])
                    if pocs:
                        s['dist_entry_to_nearest_poc'] = min(abs(entry - poc) for poc in pocs) / entry
                    else:
                        s['dist_entry_to_nearest_poc'] = 0.0
                else:
                    s['dist_entry_to_poc'] = 0.0
                    s['dist_entry_to_nearest_poc'] = 0.0
            else:
                s['dist_entry_to_poc'] = 0.0
                s['dist_entry_to_nearest_poc'] = 0.0
        else:
            s['knn_prob_sig'] = 0.0
            s['knn_prob_opp'] = 0.0
            s['dist_entry_to_poc'] = 0.0
            s['dist_entry_to_nearest_poc'] = 0.0
            # New entry-quality feature defaults (idx not found -> no NaN leak)
            s['floop_trend'] = s.get('floop_trend', 0)
            s['floop_trend_aligned'] = s.get('floop_trend_aligned', 0)
            s['rr_ratio'] = rr_ratio(s['entry_price'], s['sl_price'], s['tp_price'])
            s['atr_percentile'] = 0.0
            s['body_to_range_ratio'] = 0.0
            s['dist_to_recent_swing'] = 0.0
            s['htf_trend_aligned'] = 0
            s['confluence_score'] = 0

    return pd.DataFrame(setups)

def main():
    print("=== Trade Simulation & Dataset Labeling Engine ===")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    output_path = os.path.join(data_dir, 'labeled_setups.csv')
    
    # We look for timeframe-specific files first
    tf_files = {
        '15': 'historical_xauusdm_15.csv',
        '30': 'historical_xauusdm_30.csv',
        '1h': 'historical_xauusdm_1h.csv',
        '4h': 'historical_xauusdm_4h.csv',
        '1d': 'historical_xauusdm_1d.csv'
    }
    
    loaded_dfs = {}
    for tf, fname in tf_files.items():
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            print(f"Loading historical data for TF {tf} from {fpath}...")
            loaded_dfs[tf] = pd.read_csv(fpath)
            
    # Pre-calculate trends for FLOOP Pro MTF/HTF
    tf_trends = {}
    from src.indicators.floop import calculate_atr, calculate_range_filter
    for tf, df_tf in loaded_dfs.items():
        try:
            df_tf_copy = df_tf.copy()
            df_tf_copy['time'] = pd.to_datetime(df_tf_copy['time'])
            df_tf_copy.set_index('time', inplace=True)
            
            atr = calculate_atr(df_tf_copy, 14)
            _, trend, _ = calculate_range_filter(df_tf_copy['Close'], atr, sensitivity=6, atr_multiplier=0.8)
            tf_trends[tf] = pd.Series(trend, index=df_tf_copy.index)
        except Exception as e:
            print(f"Error calculating RF trend for TF {tf}: {e}")
            tf_trends[tf] = None

    all_labeled_dfs = []
    files_processed = 0
    symbol = "XAUUSD" # Define symbol here
    for tf, df in loaded_dfs.items():
        print(f"Simulating setups for TF {tf}...")
        labeled_df = label_smc_setups(df, symbol=symbol, tf_trends=tf_trends, df_d1=loaded_dfs.get('1d'))
        print(f"Generated {len(labeled_df)} labeled setups.")
        all_labeled_dfs.append(labeled_df)
        files_processed += 1
        
    # Fall back to historical_xauusdm.csv if no timeframe files exist
    if files_processed == 0:
        fallback_path = os.path.join(data_dir, 'historical_xauusdm.csv')
        if os.path.exists(fallback_path):
            print(f"No timeframe-specific files found. Falling back to default: {fallback_path}")
            df = pd.read_csv(fallback_path)
            labeled_df = label_smc_setups(df, symbol=symbol)
            all_labeled_dfs.append(labeled_df)
        else:
            print(f"Error: No historical data files found in {data_dir}")
            return
            
    # Combine all setups
    combined_df = pd.concat(all_labeled_dfs, ignore_index=True)
    
    if combined_df.empty:
        print("No valid resolved setups detected across any timeframe.")
        return
        
    total_trades = len(combined_df)
    wins = (combined_df['label'] == 1).sum()
    losses = (combined_df['label'] == 0).sum()
    winrate = (wins / total_trades) * 100 if total_trades > 0 else 0.0
    
    print("\n--- Simulation Summary ---")
    print(f"Total resolved setups: {total_trades}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Raw SMC Winrate (Tick-Accurate Retracement): {winrate:.2f}%")
    print("--------------------------\n")
    
    # Save the output CSV
    combined_df.to_csv(output_path, index=False)
    print(f"Combined labeled setups saved to: {output_path}")

if __name__ == "__main__":
    main()
