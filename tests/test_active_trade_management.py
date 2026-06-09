import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

# Import our active trade management functions
from src.execution import modify_position_sltp, manage_active_trades

@patch('MetaTrader5.symbol_info')
@patch('MetaTrader5.order_send')
def test_modify_position_sltp_success(mock_order_send, mock_symbol_info):
    """Test successful position SL/TP modification."""
    # Mock symbol info
    mock_sym = MagicMock()
    mock_sym.digits = 3
    mock_symbol_info.return_value = mock_sym
    
    # Mock successful order send response
    mock_res = MagicMock()
    mock_res.retcode = 10009  # TRADE_RETCODE_DONE
    mock_order_send.return_value = mock_res
    
    success = modify_position_sltp(12345, "XAUUSDm", 4450.1234, 4500.5678)
    assert success is True
    
    # Verify mock calls and digit rounding
    mock_order_send.assert_called_once()
    sent_request = mock_order_send.call_args[0][0]
    assert sent_request['position'] == 12345
    assert sent_request['sl'] == 4450.123  # Rounded to 3 digits
    assert sent_request['tp'] == 4500.568  # Rounded to 3 digits

@patch('MetaTrader5.symbol_info')
@patch('MetaTrader5.positions_get')
@patch('MetaTrader5.symbol_info_tick')
@patch('MetaTrader5.order_send')
@patch('src.execution.load_sent_signals')
def test_manage_active_trades_bep_trigger(mock_load_signals, mock_order_send, mock_tick, mock_positions, mock_symbol_info):
    """Test that Break Even (BEP) triggers when price reaches 1:1 R:R."""
    # Mock symbol info
    mock_sym = MagicMock()
    mock_sym.digits = 3
    mock_symbol_info.return_value = mock_sym
    
    # Mock position
    p_buy = MagicMock()
    p_buy.ticket = 9999
    p_buy.price_open = 100.0
    p_buy.sl = 90.0
    p_buy.tp = 130.0
    p_buy.type = 0  # POSITION_TYPE_BUY
    p_buy.comment = "SMC M30 Option A"
    p_buy.symbol = "XAUUSDm"
    mock_positions.return_value = [p_buy]
    
    # Mock current price (reached 110.0, which is exactly 1:1 risk: risk is 10.0, price is at 110.0)
    tick = MagicMock()
    tick.bid = 110.5
    tick.ask = 110.7
    mock_tick.return_value = tick
    
    # Mock sent signals database (lookup original SL)
    mock_load_signals.return_value = {
        "some_sig_key": {
            "ticket_a": 9999,
            "features_0.5": {
                "sl_price": 90.0
            }
        }
    }
    
    # Mock successful modification
    mock_res = MagicMock()
    mock_res.retcode = 10009  # TRADE_RETCODE_DONE
    mock_order_send.return_value = mock_res
    
    # We pass empty timeframes_data to avoid trailing logic, focusing only on BEP
    manage_active_trades("XAUUSD", 202606, {})
    
    # Order send should be called once to modify the SL to BEP (100.0 + spread buffer)
    mock_order_send.assert_called_once()
    sent_request = mock_order_send.call_args[0][0]
    assert sent_request['position'] == 9999
    assert sent_request['sl'] == 100.2  # 100.0 + 2 pips buffer (Gold pip is 0.1, so buffer is 0.2)

@patch('MetaTrader5.symbol_info')
@patch('MetaTrader5.positions_get')
@patch('MetaTrader5.symbol_info_tick')
@patch('MetaTrader5.order_send')
@patch('src.execution.load_sent_signals')
def test_manage_active_trades_structural_trailing(mock_load_signals, mock_order_send, mock_tick, mock_positions, mock_symbol_info):
    """Test that structural trailing stop moves SL to the latest swing low."""
    # Mock symbol info
    mock_sym = MagicMock()
    mock_sym.digits = 3
    mock_symbol_info.return_value = mock_sym
    
    # Mock position
    p_buy = MagicMock()
    p_buy.ticket = 8888
    p_buy.price_open = 100.0
    p_buy.sl = 100.2  # Already at BEP
    p_buy.tp = 130.0
    p_buy.type = 0  # POSITION_TYPE_BUY
    p_buy.comment = "SMC M30 Option A"
    p_buy.symbol = "XAUUSDm"
    mock_positions.return_value = [p_buy]
    
    # Mock current price
    tick = MagicMock()
    tick.bid = 120.0
    tick.ask = 120.2
    mock_tick.return_value = tick
    
    # Mock sent signals database
    mock_load_signals.return_value = {}
    
    # Mock successful modification
    mock_res = MagicMock()
    mock_res.retcode = 10009
    mock_order_send.return_value = mock_res
    
    # Mock timeframes data containing a new swing low at 112.0
    df_m30 = pd.DataFrame({
        'time': pd.date_range("2026-06-01", periods=10, freq="30min"),
        'High': [115.0]*10,
        'Low': [110.0]*10,
        'Swing_Low': [np.nan]*9 + [112.0]  # Swing low of 112.0 on the latest candle
    })
    
    timeframes_data = {'M30': df_m30}
    
    manage_active_trades("XAUUSD", 202606, timeframes_data)
    
    # Order send should be called to trail the SL to 112.0 - 0.2 (buffer) = 111.8
    mock_order_send.assert_called_once()
    sent_request = mock_order_send.call_args[0][0]
    assert sent_request['position'] == 8888
    assert sent_request['sl'] == 111.8


@patch('MetaTrader5.symbol_info')
@patch('MetaTrader5.positions_get')
@patch('MetaTrader5.symbol_info_tick')
@patch('MetaTrader5.order_send')
@patch('src.execution.load_sent_signals')
@patch('src.telegram_bot.send_telegram_alert')
def test_manage_active_trades_emergency_exit_only_on_closed_candle(
    mock_send_alert, mock_load_signals, mock_order_send, mock_tick, mock_positions, mock_symbol_info
):
    """Test that trend reversal emergency exit is only triggered if confirmed on a closed candle (index -2)."""
    # Mock symbol info
    mock_sym = MagicMock()
    mock_sym.digits = 3
    mock_symbol_info.return_value = mock_sym
    
    # Mock position (Buy position)
    p_buy = MagicMock()
    p_buy.ticket = 7777
    p_buy.price_open = 100.0
    p_buy.sl = 90.0
    p_buy.tp = 130.0
    p_buy.volume = 0.01
    p_buy.type = 0  # POSITION_TYPE_BUY
    p_buy.comment = "SMC M30 Option A"
    p_buy.symbol = "XAUUSDm"
    mock_positions.return_value = [p_buy]
    
    # Mock current price (no SL/TP or BE trigger)
    tick = MagicMock()
    tick.bid = 102.0
    tick.ask = 102.2
    mock_tick.return_value = tick
    
    # Mock sent signals database
    mock_load_signals.return_value = {}
    
    # Mock order_send return code
    mock_res = MagicMock()
    mock_res.retcode = 10009  # TRADE_RETCODE_DONE
    mock_order_send.return_value = mock_res
    
    # Scenario 1: Trend on last closed candle is 1 (Bullish), but active candle is -1 (Bearish)
    # The bot should NOT exit the position.
    df_m30_temp = pd.DataFrame({
        'time': pd.date_range("2026-06-01", periods=5, freq="30min"),
        'High': [105.0]*5,
        'Low': [95.0]*5,
        'Close': [100.0]*5,
        'Trend': [1, 1, 1, 1, -1],  # Last row is -1 (active), second last is 1 (closed)
        'Swing_High': [np.nan]*5,
        'Swing_Low': [np.nan]*5
    })
    
    manage_active_trades("XAUUSD", 202606, {'M30': df_m30_temp})
    
    # Close position (order_send with TRADE_ACTION_DEAL) should NOT have been called
    for call in mock_order_send.call_args_list:
        req = call[0][0]
        # Verify it wasn't a close action (TRADE_ACTION_DEAL is 1, TRADE_ACTION_SLTP is 6)
        assert req.get('action') != 1  
        
    # Scenario 2: M30 has only one closed opposite candle.
    # The bot should wait for one more confirmation candle instead of exiting immediately.
    df_m30_single_reversal = pd.DataFrame({
        'time': pd.date_range("2026-06-01", periods=5, freq="30min"),
        'High': [105.0]*5,
        'Low': [95.0]*5,
        'Close': [100.0]*5,
        'Trend': [1, 1, 1, -1, -1],  # Second last row is -1 (closed)
        'Swing_High': [np.nan]*5,
        'Swing_Low': [np.nan]*5
    })
    
    # Reset mock call counts
    mock_order_send.reset_mock()
    
    manage_active_trades("XAUUSD", 202606, {'M30': df_m30_single_reversal})
    
    # Close position should NOT be called on one LTF reversal candle.
    for call in mock_order_send.call_args_list:
        req = call[0][0]
        assert req.get('action') != 1

    # Scenario 3: M30 has two closed opposite candles.
    # The bot should exit because the reversal is confirmed.
    df_m30_confirmed_reversal = pd.DataFrame({
        'time': pd.date_range("2026-06-01", periods=6, freq="30min"),
        'High': [105.0]*6,
        'Low': [95.0]*6,
        'Close': [100.0]*6,
        'Trend': [1, 1, 1, -1, -1, -1],  # Last two closed candles are -1
        'Swing_High': [np.nan]*6,
        'Swing_Low': [np.nan]*6
    })

    mock_order_send.reset_mock()

    manage_active_trades("XAUUSD", 202606, {'M30': df_m30_confirmed_reversal})

    # Close position should be called after confirmed LTF reversal.
    mock_order_send.assert_called_once()
    sent_request = mock_order_send.call_args[0][0]
    assert sent_request['position'] == 7777
    assert sent_request['action'] == 1  # mt5.TRADE_ACTION_DEAL to close
    assert sent_request['volume'] == 0.01


@patch('MetaTrader5.symbol_info')
@patch('MetaTrader5.positions_get')
@patch('MetaTrader5.symbol_info_tick')
@patch('MetaTrader5.order_send')
@patch('src.execution.load_sent_signals')
@patch('src.telegram_bot.send_telegram_alert')
def test_manage_active_trades_h1_emergency_exit_allows_one_closed_reversal(
    mock_send_alert, mock_load_signals, mock_order_send, mock_tick, mock_positions, mock_symbol_info
):
    """H1/H4 reversal is heavier than LTF noise, so one closed opposite candle can close."""
    mock_sym = MagicMock()
    mock_sym.digits = 3
    mock_symbol_info.return_value = mock_sym

    p_buy = MagicMock()
    p_buy.ticket = 7778
    p_buy.price_open = 100.0
    p_buy.sl = 90.0
    p_buy.tp = 130.0
    p_buy.volume = 0.01
    p_buy.type = 0
    p_buy.comment = "SMC H1 Option A"
    p_buy.symbol = "XAUUSDm"
    mock_positions.return_value = [p_buy]

    tick = MagicMock()
    tick.bid = 102.0
    tick.ask = 102.2
    mock_tick.return_value = tick
    mock_load_signals.return_value = {}

    mock_res = MagicMock()
    mock_res.retcode = 10009
    mock_order_send.return_value = mock_res

    df_h1_reversed = pd.DataFrame({
        'time': pd.date_range("2026-06-01", periods=5, freq="1h"),
        'High': [105.0]*5,
        'Low': [95.0]*5,
        'Close': [100.0]*5,
        'Trend': [1, 1, 1, -1, -1],
        'Swing_High': [np.nan]*5,
        'Swing_Low': [np.nan]*5
    })

    manage_active_trades("XAUUSD", 202606, {'H1': df_h1_reversed})

    mock_order_send.assert_called_once()
    sent_request = mock_order_send.call_args[0][0]
    assert sent_request['position'] == 7778
    assert sent_request['action'] == 1
