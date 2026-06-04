import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.labeler import label_smc_setups, get_killzone, simulate_trade

def test_get_killzone():
    # London
    assert get_killzone(9) == 1
    assert get_killzone(10) == 1
    assert get_killzone(11) == 1
    # NY
    assert get_killzone(14) == 2
    assert get_killzone(15) == 2
    assert get_killzone(16) == 2
    # Asia
    assert get_killzone(2) == 3
    assert get_killzone(5) == 3
    assert get_killzone(6) == 3
    # None
    assert get_killzone(13) == 0
    assert get_killzone(8) == 0

def test_simulate_trade_bullish_win():
    # Simple df for simulation
    data = {
        'High': [10.0, 11.0, 13.0],
        'Low':  [9.0, 9.5, 9.2]
    }
    df = pd.DataFrame(data)
    # Entry = 10, SL = 8, TP = 12
    # Candle 0: High = 10, Low = 9 (no hit)
    # Candle 1: High = 11, Low = 9.5 (no hit)
    # Candle 2: High = 13, Low = 9.2 (High >= 12, Low > 8) -> TP hit -> Win (1.0)
    label = simulate_trade(df, start_idx=0, direction=1, sl=8.0, tp=12.0)
    assert label == 1.0

def test_simulate_trade_bullish_loss():
    data = {
        'High': [10.5, 11.0, 11.5],
        'Low':  [9.0, 8.5, 7.5]
    }
    df = pd.DataFrame(data)
    # Entry = 10, SL = 8, TP = 12
    # Candle 0: High = 10.5, Low = 9.0 (no hit)
    # Candle 1: High = 11.0, Low = 8.5 (no hit)
    # Candle 2: High = 11.5, Low = 7.5 (Low <= 8.0) -> SL hit -> Loss (0.0)
    label = simulate_trade(df, start_idx=0, direction=1, sl=8.0, tp=12.0)
    assert label == 0.0

def test_simulate_trade_bearish_win():
    data = {
        'High': [11.0, 10.5, 10.2],
        'Low':  [9.5, 9.0, 7.0]
    }
    df = pd.DataFrame(data)
    # Entry = 10, SL = 12, TP = 8
    # Candle 0: High = 11.0, Low = 9.5 (no hit)
    # Candle 1: High = 10.5, Low = 9.0 (no hit)
    # Candle 2: High = 10.2, Low = 7.0 (Low <= 8.0, High < 12.0) -> TP hit -> Win (1.0)
    label = simulate_trade(df, start_idx=0, direction=-1, sl=12.0, tp=8.0)
    assert label == 1.0

def test_simulate_trade_bearish_loss():
    data = {
        'High': [11.0, 12.5, 11.0],
        'Low':  [9.5, 9.0, 9.0]
    }
    df = pd.DataFrame(data)
    # Entry = 10, SL = 12, TP = 8
    # Candle 0: High = 11.0, Low = 9.5 (no hit)
    # Candle 1: High = 12.5, Low = 9.0 (High >= 12.0) -> SL hit -> Loss (0.0)
    label = simulate_trade(df, start_idx=0, direction=-1, sl=12.0, tp=8.0)
    assert label == 0.0

def test_label_smc_setups_fvg_bullish():
    # Build a DataFrame with at least 15 candles to compute ATR_14
    # Candle i-2 is index 11, candle i-1 is index 12, candle i is index 13
    times = [datetime(2026, 6, 1, 8, 0) + timedelta(minutes=15*i) for i in range(20)]
    
    # Base flat candles
    opens = [10.0] * 20
    highs = [10.5] * 20
    lows = [9.5] * 20
    closes = [10.0] * 20
    
    # Configure FVG pattern
    # Candle 11 (i-2): High = 10.5
    highs[11] = 10.5
    
    # Candle 12 (i-1): Bullish expansion
    opens[12] = 10.0
    closes[12] = 14.0
    highs[12] = 14.5
    lows[12] = 10.0
    
    # Candle 13 (i): Low = 12.0. So 10.5 < 12.0 -> Bullish FVG!
    opens[13] = 14.0
    closes[13] = 14.5
    highs[13] = 15.0
    lows[13] = 12.0
    
    # Subsequent candles: index 14 goes up to hit TP
    opens[14] = 14.5
    closes[14] = 25.0
    highs[14] = 30.0
    lows[14] = 11.0
    
    df = pd.DataFrame({
        'time': times,
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': closes,
        'Volume': [100] * 20
    })
    
    labeled_df = label_smc_setups(df, buffer=0.5)
    
    # Should find 2 setups now
    assert len(labeled_df) >= 2
    fvg_trades = labeled_df[labeled_df['setup_type'] == 0]
    assert len(fvg_trades) == 2
    
    # Option A (Midpoint): Entry = 12.25, SL = 8.0, TP = 14.5
    trade = fvg_trades.iloc[0]
    assert trade['direction'] == 1
    assert np.isclose(trade['entry_price'], 12.25)
    assert np.isclose(trade['sl_price'], 8.0)
    assert np.isclose(trade['tp_price'], 14.5)
    assert trade['label'] == 1
    
    # Option B (Golden Pocket): Entry = 11.719, SL = 8.0, TP = 14.5
    trade2 = fvg_trades.iloc[1]
    assert trade2['direction'] == 1
    assert np.isclose(trade2['entry_price'], 11.719)
    assert np.isclose(trade2['sl_price'], 8.0)
    assert np.isclose(trade2['tp_price'], 14.5, atol=1e-3)
    assert trade2['label'] == 1

def test_label_smc_setups_fvg_bearish():
    times = [datetime(2026, 6, 1, 14, 0) + timedelta(minutes=15*i) for i in range(20)]
    
    opens = [20.0] * 20
    highs = [20.5] * 20
    lows = [19.5] * 20
    closes = [20.0] * 20
    
    # Configure Bearish FVG pattern
    # Candle 11 (i-2): Low = 19.5
    lows[11] = 19.5
    
    # Candle 12 (i-1): Bearish expansion
    opens[12] = 20.0
    closes[12] = 16.0
    highs[12] = 20.0
    lows[12] = 15.5
    
    # Candle 13 (i): High = 18.0. So 19.5 > 18.0 -> Bearish FVG!
    opens[13] = 16.0
    closes[13] = 15.5
    highs[13] = 18.0
    lows[13] = 15.0
    
    # Subsequent candles: index 14 goes down to hit TP
    opens[14] = 15.5
    closes[14] = 10.5
    highs[14] = 19.0
    lows[14] = 8.0
    
    df = pd.DataFrame({
        'time': times,
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': closes,
        'Volume': [100] * 20
    })
    
    labeled_df = label_smc_setups(df, buffer=0.5)
    
    assert len(labeled_df) >= 2
    fvg_trades = labeled_df[labeled_df['setup_type'] == 0]
    assert len(fvg_trades) == 2
    
    # Option A (Midpoint): Entry = 17.75, SL = 22.0, TP = 15.5
    trade = fvg_trades.iloc[0]
    assert trade['direction'] == -1
    assert np.isclose(trade['entry_price'], 17.75)
    assert np.isclose(trade['sl_price'], 22.0)
    assert np.isclose(trade['tp_price'], 15.5)
    assert trade['label'] == 1
    
    # Option B (Golden Pocket): Entry = 18.281, SL = 22.0, TP = 15.5
    trade2 = fvg_trades.iloc[1]
    assert trade2['direction'] == -1
    assert np.isclose(trade2['entry_price'], 18.281)
    assert np.isclose(trade2['sl_price'], 22.0)
    assert np.isclose(trade2['tp_price'], 15.5, atol=1e-3)
    assert trade2['label'] == 1
