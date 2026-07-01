import pandas as pd
import numpy as np
import pytest
from src.indicators.volume_clusters import calculate_volume_clusters

def test_calculate_volume_clusters():
    # Create synthetic data with clearly distinct clusters (two regimes)
    dates = pd.date_range(start="2026-01-01", periods=100, freq="15min")
    # First 50 bars are around 100, next 50 bars are around 200
    close = np.concatenate([np.random.normal(100, 1, 50), np.random.normal(200, 1, 50)])
    high = close + 0.5
    low = close - 0.5
    volume = np.random.randint(100, 500, 100).astype(float)
    
    df = pd.DataFrame({
        'time': dates,
        'Open': close,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    })
    
    clusters = calculate_volume_clusters(df, lookback=100, k=2, iterations=10, rows=10)
    
    assert 'centroids' in clusters
    assert 'pocs' in clusters
    assert 'total_volumes' in clusters
    assert len(clusters['centroids']) == 2
    assert len(clusters['pocs']) == 2
    
    # Check current cluster details
    assert 'current_cluster' in clusters
    assert clusters['current_cluster'] in [0, 1]
    assert clusters['current_poc'] > 0
    assert abs(clusters['current_dist_to_poc']) < 1.0
