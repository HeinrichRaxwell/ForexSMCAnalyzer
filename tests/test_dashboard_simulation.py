import pytest
import pandas as pd
from src.dashboard import (
    MockTick,
    MockInfo,
    generate_synthetic_data,
    fetch_simulated_data,
)

def test_mock_tick():
    tick = MockTick(1.2345, 1.2346)
    assert tick.bid == 1.2345
    assert tick.ask == 1.2346

def test_mock_info():
    info = MockInfo(3, 0.01)
    assert info.digits == 3
    assert info.point == 0.01

def test_generate_synthetic_data():
    df = generate_synthetic_data("XAUUSD", "M15", 50)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 50
    assert list(df.columns) == ['time', 'Open', 'High', 'Low', 'Close', 'Volume']
    assert not df.isnull().values.any()

def test_fetch_simulated_data_fallback():
    # Test fallback to synthetic when file doesn't exist
    df = fetch_simulated_data("NONEXISTENT", "H1", 30)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 30
    assert not df.isnull().values.any()
