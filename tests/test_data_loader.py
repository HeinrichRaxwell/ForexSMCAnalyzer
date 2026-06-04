import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from src.data_loader import connect_mt5, fetch_historical_data

@patch('src.data_loader.mt5.initialize')
def test_connect_mt5_success(mock_initialize):
    """Test successful connection initialization to MT5."""
    mock_initialize.return_value = True
    assert connect_mt5() is True
    mock_initialize.assert_called_once()

@patch('src.data_loader.mt5.initialize')
@patch('src.data_loader.mt5.last_error')
def test_connect_mt5_failure(mock_last_error, mock_initialize):
    """Test failed connection initialization to MT5."""
    mock_initialize.return_value = False
    mock_last_error.return_value = (1, "Mock connection error")
    assert connect_mt5() is False
    mock_initialize.assert_called_once()

@patch('src.data_loader.mt5.copy_rates_from_pos', create=True)
def test_fetch_historical_data_success(mock_copy_rates):
    """Test successful historical data retrieval and column formatting."""
    # Mock data structured array as typically returned by MT5 API
    mock_rates = np.array(
        [
            (1685748000, 1960.0, 1965.0, 1958.0, 1962.0, 100),
            (1685748900, 1962.0, 1970.0, 1961.0, 1968.0, 150)
        ],
        dtype=[
            ('time', '<i8'),
            ('open', '<f8'),
            ('high', '<f8'),
            ('low', '<f8'),
            ('close', '<f8'),
            ('tick_volume', '<i8')
        ]
    )
    mock_copy_rates.return_value = mock_rates
    
    df = fetch_historical_data("XAUUSD", 15, 2)
    
    mock_copy_rates.assert_called_once_with("XAUUSD", 15, 0, 2)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ['time', 'Open', 'High', 'Low', 'Close', 'Volume']
    assert len(df) == 2
    assert df['Open'].iloc[0] == 1960.0
    assert df['High'].iloc[0] == 1965.0
    assert df['Low'].iloc[0] == 1958.0
    assert df['Close'].iloc[0] == 1962.0
    assert df['Volume'].iloc[0] == 100
    assert df['time'].iloc[0] == pd.to_datetime(1685748000, unit='s')

@patch('src.data_loader.mt5.copy_rates_from_pos', create=True)
def test_fetch_historical_data_failure_none(mock_copy_rates):
    """Test data fetching when copy_rates_from_now returns None."""
    mock_copy_rates.return_value = None
    with pytest.raises(ValueError) as excinfo:
        fetch_historical_data("XAUUSD", 15, 2)
    assert "Failed to fetch historical data" in str(excinfo.value)

@patch('src.data_loader.mt5.copy_rates_from_pos', create=True)
def test_fetch_historical_data_failure_empty(mock_copy_rates):
    """Test data fetching when copy_rates_from_now returns an empty sequence."""
    mock_copy_rates.return_value = np.array([])
    with pytest.raises(ValueError) as excinfo:
        fetch_historical_data("XAUUSD", 15, 2)
    assert "Failed to fetch historical data" in str(excinfo.value)
