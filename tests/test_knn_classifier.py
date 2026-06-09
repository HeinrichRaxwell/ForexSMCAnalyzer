import pandas as pd
import numpy as np
import pytest
from src.indicators.knn_classifier import run_knn_classifier, calculate_supertrend

def test_calculate_supertrend():
    # Create a simple synthetic upward trend
    dates = pd.date_range(start="2026-01-01", periods=50, freq="15min")
    high = np.arange(100, 150) + 1.0
    low = np.arange(100, 150) - 1.0
    close = np.arange(100, 150)
    volume = np.ones(50) * 1000
    
    df = pd.DataFrame({
        'time': dates,
        'Open': close,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    })
    
    supertrend, direction = calculate_supertrend(df, period=10, factor=2.0)
    assert len(supertrend) == 50
    assert len(direction) == 50
    # Because price goes up, direction should eventually flip to bullish (-1)
    assert -1 in direction.values

def test_run_knn_classifier():
    # Create 1100 bars of synthetic data to satisfy the 1000 sampling window limit
    dates = pd.date_range(start="2026-01-01", periods=1100, freq="15min")
    # Generates a trending sinewave
    t = np.linspace(0, 10, 1100)
    close = 100 + 10 * np.sin(t) + t
    high = close + 0.5
    low = close - 0.5
    volume = np.random.randint(100, 1000, 1100).astype(float)
    
    df = pd.DataFrame({
        'time': dates,
        'Open': close,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    })
    
    prob_up, prob_down = run_knn_classifier(
        df,
        atr_period=10, factor=2.0,
        k_neighbors=10, sampling_window_size=100, momentum_window=5,
        normalizing_window_size=100
    )
    
    assert len(prob_up) == 1100
    assert len(prob_down) == 1100
    # Check that probabilities are bounded in [0, 1]
    assert prob_up.min() >= 0.0
    assert prob_up.max() <= 1.0
    assert prob_down.min() >= 0.0
    assert prob_down.max() <= 1.0
