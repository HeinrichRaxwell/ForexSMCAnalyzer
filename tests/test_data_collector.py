import os
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from src.data_collector import download_bulk_data, get_mt5_timeframe

def test_get_mt5_timeframe():
    """Test mapping of different timeframe types to MT5 constants."""
    with patch('src.data_collector.mt5') as mock_mt5:
        mock_mt5.TIMEFRAME_M15 = 15
        mock_mt5.TIMEFRAME_H1 = 16385
        
        assert get_mt5_timeframe(15) == 15
        assert get_mt5_timeframe("15") == 15
        assert get_mt5_timeframe("1h") == 16385

@patch('src.data_collector.connect_mt5')
@patch('src.data_collector.fetch_historical_data')
@patch('src.data_collector.mt5.symbol_select')
@patch('src.data_collector.mt5.copy_rates_from_pos')
@patch('src.data_collector.mt5.shutdown')
def test_download_bulk_data_success(mock_shutdown, mock_copy_rates, mock_symbol_select, mock_fetch_data, mock_connect, tmp_path):
    """Test successful data collection and saving to CSV."""
    mock_connect.return_value = True
    
    # Mock finding active symbol: copy_rates_from_pos returns non-empty array
    mock_copy_rates.return_value = np.array([(1685748000, 1960.0)], dtype=[('time', '<i8'), ('close', '<f8')])
    
    # Mock fetched dataframe
    df_mock = pd.DataFrame({
        'time': pd.to_datetime([1685748000, 1685748900], unit='s'),
        'Open': [1960.0, 1962.0],
        'High': [1965.0, 1970.0],
        'Low': [1958.0, 1961.0],
        'Close': [1962.0, 1968.0],
        'Volume': [100, 150]
    })
    mock_fetch_data.return_value = df_mock
    
    output_dir = tmp_path / "data"
    
    # Run the download function
    saved_file = download_bulk_data(
        symbol="XAUUSD",
        timeframe=15,
        num_candles=2,
        output_dir=str(output_dir)
    )
    
    # Verify outputs
    expected_file = os.path.join(str(output_dir), "historical_xauusd_15.csv")
    assert saved_file == expected_file
    assert os.path.exists(expected_file)
    
    # Verify data in CSV matches
    saved_df = pd.read_csv(expected_file)
    assert len(saved_df) == 2

    assert list(saved_df.columns) == ['time', 'Open', 'High', 'Low', 'Close', 'Volume']
    assert saved_df['Close'].iloc[0] == 1962.0
    
    # Verify calls
    mock_connect.assert_called_once()
    mock_fetch_data.assert_called_once_with("XAUUSD", 15, 2)
    mock_shutdown.assert_called_once()

@patch('src.data_collector.connect_mt5')
def test_download_bulk_data_connection_failure(mock_connect):
    """Test behavior when MT5 connection fails."""
    mock_connect.return_value = False
    
    with pytest.raises(RuntimeError) as excinfo:
        download_bulk_data(symbol="XAUUSD", timeframe=15, num_candles=2)
    assert "Failed to connect to MetaTrader 5" in str(excinfo.value)
