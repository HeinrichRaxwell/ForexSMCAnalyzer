import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

# Import our active trade management functions
from src.execution import modify_position_sltp, manage_active_trades

@pytest.fixture(autouse=True)
def disable_custom_staged_profit_lock_by_default(monkeypatch):
    monkeypatch.setenv("MT5_CUSTOM_STAGED_PROFIT_LOCK", "False")


@patch('MetaTrader5.symbol_info')
@patch('MetaTrader5.positions_get')
@patch('MetaTrader5.symbol_info_tick')
@patch('MetaTrader5.order_send')
@patch('src.execution.load_sent_signals')
def test_manage_active_trades_filters_magic_without_unsupported_mt5_kwarg(
    mock_load_signals, mock_order_send, mock_tick, mock_positions, mock_symbol_info
):
    mock_sym = MagicMock()
    mock_sym.digits = 3
    mock_symbol_info.return_value = mock_sym

    wrong_magic_position = MagicMock()
    wrong_magic_position.magic = 999999
    wrong_magic_position.ticket = 1111

    def positions_get(**kwargs):
        assert kwargs == {"symbol": "XAUUSD"}
        return [wrong_magic_position]

    mock_positions.side_effect = positions_get

    manage_active_trades("XAUUSD", 202606, {})

    mock_tick.assert_not_called()
    mock_load_signals.assert_not_called()
    mock_order_send.assert_not_called()


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
    p_buy.magic = 202606
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
def test_manage_active_trades_buy_locks_200_pips_after_350_pips_profit(
    mock_load_signals, mock_order_send, mock_tick, mock_positions, mock_symbol_info, monkeypatch
):
    """XAUUSD +350 pips should protect +200 pips, not only BEP."""
    monkeypatch.setenv("MT5_PROFIT_LOCK_ENABLED", "True")
    monkeypatch.setenv("MT5_PROFIT_LOCK_STEP_PIPS", "100")
    monkeypatch.setenv("MT5_PROFIT_LOCK_GAP_PIPS", "100")

    mock_sym = MagicMock()
    mock_sym.digits = 3
    mock_symbol_info.return_value = mock_sym

    p_buy = MagicMock()
    p_buy.ticket = 9998
    p_buy.price_open = 4000.0
    p_buy.sl = 3990.0
    p_buy.tp = 4050.0
    p_buy.type = 0
    p_buy.comment = "SMC M30 Option A"
    p_buy.symbol = "XAUUSDm"
    p_buy.magic = 202606
    mock_positions.return_value = [p_buy]

    tick = MagicMock()
    tick.bid = 4035.0
    tick.ask = 4035.2
    mock_tick.return_value = tick
    mock_load_signals.return_value = {}

    mock_res = MagicMock()
    mock_res.retcode = 10009
    mock_order_send.return_value = mock_res

    manage_active_trades("XAUUSD", 202606, {})

    mock_order_send.assert_called_once()
    sent_request = mock_order_send.call_args[0][0]
    assert sent_request['position'] == 9998
    assert sent_request['sl'] == 4025.0


@patch('MetaTrader5.symbol_info')
@patch('MetaTrader5.positions_get')
@patch('MetaTrader5.symbol_info_tick')
@patch('MetaTrader5.order_send')
@patch('src.execution.load_sent_signals')
def test_manage_active_trades_sell_locks_200_pips_after_350_pips_profit(
    mock_load_signals, mock_order_send, mock_tick, mock_positions, mock_symbol_info, monkeypatch
):
    """XAUUSD sell +350 pips should protect +200 pips below entry."""
    monkeypatch.setenv("MT5_PROFIT_LOCK_ENABLED", "True")
    monkeypatch.setenv("MT5_PROFIT_LOCK_STEP_PIPS", "100")
    monkeypatch.setenv("MT5_PROFIT_LOCK_GAP_PIPS", "100")

    mock_sym = MagicMock()
    mock_sym.digits = 3
    mock_symbol_info.return_value = mock_sym

    p_sell = MagicMock()
    p_sell.ticket = 9997
    p_sell.price_open = 4000.0
    p_sell.sl = 4010.0
    p_sell.tp = 3950.0
    p_sell.type = 1
    p_sell.comment = "SMC M30 Option A"
    p_sell.symbol = "XAUUSDm"
    p_sell.magic = 202606
    mock_positions.return_value = [p_sell]

    tick = MagicMock()
    tick.bid = 3964.8
    tick.ask = 3965.0
    mock_tick.return_value = tick
    mock_load_signals.return_value = {}

    mock_res = MagicMock()
    mock_res.retcode = 10009
    mock_order_send.return_value = mock_res

    manage_active_trades("XAUUSD", 202606, {})

    mock_order_send.assert_called_once()
    sent_request = mock_order_send.call_args[0][0]
    assert sent_request['position'] == 9997
    assert sent_request['sl'] == 3975.0

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
    p_buy.magic = 202606
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
    p_buy.magic = 202606
    mock_positions.return_value = [p_buy]
    
    # Mock current price (in loss to allow emergency exit)
    tick = MagicMock()
    tick.bid = 98.0
    tick.ask = 98.2
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
def test_manage_active_trades_h1_emergency_exit_requires_two_closed_reversals(
    mock_send_alert, mock_load_signals, mock_order_send, mock_tick, mock_positions, mock_symbol_info
):
    """H1/H4 reversal requires two closed opposite candles to confirm a true reversal."""
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
    p_buy.magic = 202606
    mock_positions.return_value = [p_buy]

    # Mock current price in loss to allow emergency exit
    tick = MagicMock()
    tick.bid = 98.0
    tick.ask = 98.2
    mock_tick.return_value = tick
    mock_load_signals.return_value = {}

    mock_res = MagicMock()
    mock_res.retcode = 10009
    mock_order_send.return_value = mock_res

    # Require 2 closed opposite trend candles ([1, 1, -1, -1, -1] -> closed trends: [1, 1, -1, -1])
    df_h1_reversed = pd.DataFrame({
        'time': pd.date_range("2026-06-01", periods=5, freq="1h"),
        'High': [105.0]*5,
        'Low': [95.0]*5,
        'Close': [100.0]*5,
        'Trend': [1, 1, -1, -1, -1],
        'Swing_High': [np.nan]*5,
        'Swing_Low': [np.nan]*5
    })

    manage_active_trades("XAUUSD", 202606, {'H1': df_h1_reversed})

    mock_order_send.assert_called_once()
    sent_request = mock_order_send.call_args[0][0]
    assert sent_request['position'] == 7778
    assert sent_request['action'] == 1


@patch('MetaTrader5.symbol_info')
@patch('MetaTrader5.positions_get')
@patch('MetaTrader5.symbol_info_tick')
@patch('src.execution.close_position')
@patch('src.execution.load_sent_signals')
def test_manage_active_trades_m30_does_not_exit_from_htf_override_without_same_tf_confirmation(
    mock_load_signals, mock_close_position, mock_tick, mock_positions, mock_symbol_info, monkeypatch
):
    """M30 mitigation must wait for M30 closed-candle confirmation; H1/H4 cannot be the default trigger."""
    monkeypatch.setenv("MT5_PROFIT_LOCK_ENABLED", "False")
    monkeypatch.setenv("MT5_EARLY_MITIGATION_ENABLED", "False")

    mock_sym = MagicMock()
    mock_sym.digits = 3
    mock_symbol_info.return_value = mock_sym

    p_buy = MagicMock()
    p_buy.ticket = 7788
    p_buy.price_open = 100.0
    p_buy.sl = 90.0
    p_buy.tp = 130.0
    p_buy.volume = 0.01
    p_buy.type = 0
    p_buy.comment = "SMC M30 Option A"
    p_buy.symbol = "XAUUSDm"
    p_buy.magic = 202606
    mock_positions.return_value = [p_buy]

    tick = MagicMock()
    tick.bid = 102.0
    tick.ask = 102.2
    mock_tick.return_value = tick
    mock_load_signals.return_value = {}
    mock_close_position.return_value = True

    df_m30_one_closed_reversal = pd.DataFrame({
        'time': pd.date_range("2026-06-11 13:00", periods=5, freq="30min"),
        'High': [105.0]*5,
        'Low': [95.0]*5,
        'Close': [100.0]*5,
        'Trend': [1, 1, 1, -1, -1],
        'Swing_High': [np.nan]*5,
        'Swing_Low': [np.nan]*5,
        'Volume': [100, 100, 100, 100, 100],
    })
    df_h1_opposite = pd.DataFrame({'Trend': [1, 1, -1, -1]})
    df_h4_opposite = pd.DataFrame({'Trend': [1, 1, -1, -1]})

    manage_active_trades(
        "XAUUSD",
        202606,
        {'M30': df_m30_one_closed_reversal, 'H1': df_h1_opposite, 'H4': df_h4_opposite},
    )

    mock_close_position.assert_not_called()


@patch('MetaTrader5.symbol_info')
@patch('MetaTrader5.positions_get')
@patch('MetaTrader5.symbol_info_tick')
@patch('src.execution.close_position')
@patch('src.execution.load_sent_signals')
@patch('src.execution.save_sent_signals', create=True)
@patch('src.telegram_bot.send_telegram_alert')
def test_manage_active_trades_early_mitigation_requires_same_tf_volume_structure_and_bad_exit_price(
    mock_send_alert, mock_save_signals, mock_load_signals, mock_close_position, mock_tick, mock_positions, mock_symbol_info, monkeypatch
):
    """Early mitigation is allowed only from strong same-TF closed-candle evidence plus adverse bid/ask."""
    monkeypatch.setenv("MT5_PROFIT_LOCK_ENABLED", "False")
    monkeypatch.setenv("MT5_EARLY_MITIGATION_ENABLED", "True")
    monkeypatch.setenv("MT5_EARLY_MITIGATION_MIN_ADVERSE_PIPS", "20")
    monkeypatch.setenv("MT5_EARLY_MITIGATION_RISK_FRACTION", "0.30")
    monkeypatch.setenv("MT5_EARLY_MITIGATION_VOLUME_MULTIPLIER", "2.0")

    mock_sym = MagicMock()
    mock_sym.digits = 3
    mock_symbol_info.return_value = mock_sym

    p_buy = MagicMock()
    p_buy.ticket = 7789
    p_buy.price_open = 100.0
    p_buy.sl = 90.0
    p_buy.tp = 130.0
    p_buy.volume = 0.01
    p_buy.type = 0
    p_buy.comment = "SMC M30 Option A"
    p_buy.symbol = "XAUUSDm"
    p_buy.magic = 202606
    mock_positions.return_value = [p_buy]

    tick = MagicMock()
    tick.bid = 95.0
    tick.ask = 95.2
    mock_tick.return_value = tick

    registry = {
        "sig-early": {
            "ticket_a": 7789,
            "features_0.5": {
                "entry_price": 100.0,
                "sl_price": 90.0,
                "tp_price": 130.0,
            },
        }
    }
    mock_load_signals.return_value = registry
    mock_close_position.return_value = True

    df_m30_one_closed_reversal = pd.DataFrame({
        'time': pd.date_range("2026-06-11 13:00", periods=6, freq="30min"),
        'Open': [99.0, 100.0, 100.5, 101.0, 101.0, 96.0],
        'High': [103.0]*6,
        'Low': [94.0]*6,
        'Close': [100.0, 100.4, 100.8, 101.2, 96.0, 95.5],
        'Trend': [1, 1, 1, 1, -1, -1],
        'Swing_High': [np.nan]*6,
        'Swing_Low': [np.nan]*6,
        'Volume': [100, 110, 90, 100, 350, 350],
    })

    manage_active_trades("XAUUSD", 202606, {'M30': df_m30_one_closed_reversal})

    mock_close_position.assert_called_once_with(7789, "XAUUSD")
    assert registry["sig-early"]["manager_exit_trigger_a"] == "early_market_deterioration"
    assert registry["sig-early"]["manager_exit_timeframe_a"] == "M30"
    assert "same-TF" in registry["sig-early"]["manager_exit_detail_a"]
    assert "volume" in registry["sig-early"]["manager_exit_detail_a"]
    mock_save_signals.assert_called_once_with(registry)


@patch('MetaTrader5.symbol_info')
@patch('MetaTrader5.positions_get')
@patch('MetaTrader5.symbol_info_tick')
@patch('MetaTrader5.order_send')
@patch('src.execution.load_sent_signals')
@patch('src.execution.save_sent_signals', create=True)
@patch('src.telegram_bot.send_telegram_alert')
def test_manage_active_trades_records_emergency_exit_trigger_in_registry(
    mock_send_alert, mock_save_signals, mock_load_signals, mock_order_send, mock_tick, mock_positions, mock_symbol_info
):
    mock_sym = MagicMock()
    mock_sym.digits = 3
    mock_symbol_info.return_value = mock_sym

    p_sell = MagicMock()
    p_sell.ticket = 7780
    p_sell.price_open = 4098.247
    p_sell.sl = 4105.851
    p_sell.tp = 4067.831
    p_sell.volume = 0.02
    p_sell.type = 1
    p_sell.comment = "SMC M15 BPR Gold"
    p_sell.symbol = "XAUUSDm"
    p_sell.magic = 202606
    mock_positions.return_value = [p_sell]

    tick = MagicMock()
    tick.bid = 4098.300
    tick.ask = 4098.560
    mock_tick.return_value = tick

    registry = {
        "sig-bpr": {
            "ticket_b": 7780,
            "features_0.618": {
                "entry_price": 4098.247,
                "sl_price": 4105.851,
                "tp_price": 4089.181,
            },
        }
    }
    mock_load_signals.return_value = registry

    mock_res = MagicMock()
    mock_res.retcode = 10009
    mock_order_send.return_value = mock_res

    df_m15_reversed = pd.DataFrame({
        'time': pd.date_range("2026-06-11 13:00", periods=6, freq="15min"),
        'High': [4100.0]*6,
        'Low': [4090.0]*6,
        'Close': [4095.0]*6,
        'Trend': [-1, -1, -1, 1, 1, 1],
        'Swing_High': [np.nan]*6,
        'Swing_Low': [np.nan]*6,
    })

    manage_active_trades("XAUUSD", 202606, {'M15': df_m15_reversed})

    assert registry["sig-bpr"]["manager_exit_trigger_b"] == "emergency_reversal"
    assert registry["sig-bpr"]["manager_exit_timeframe_b"] == "M15"
    assert "opposite CHoCH" in registry["sig-bpr"]["manager_exit_detail_b"]
    mock_save_signals.assert_called_once_with(registry)


@patch('MetaTrader5.symbol_info')
@patch('MetaTrader5.positions_get')
@patch('MetaTrader5.symbol_info_tick')
@patch('src.execution.modify_position_sltp')
@patch('src.execution.load_sent_signals')
@patch('src.execution.save_sent_signals', create=True)
def test_manage_active_trades_removes_tp_for_strong_runner_near_fibo_zero(
    mock_save_signals, mock_load_signals, mock_modify_sltp, mock_tick, mock_positions, mock_symbol_info, monkeypatch
):
    monkeypatch.setenv("MT5_RUNNER_ENABLED", "True")
    monkeypatch.setenv("MT5_RUNNER_ARM_DISTANCE_PIPS", "10")
    monkeypatch.setenv("MT5_PROFIT_LOCK_ENABLED", "False")

    mock_sym = MagicMock()
    mock_sym.digits = 3
    mock_symbol_info.return_value = mock_sym

    p_buy = MagicMock()
    p_buy.ticket = 7781
    p_buy.price_open = 100.0
    p_buy.sl = 95.0
    p_buy.tp = 110.0
    p_buy.volume = 0.01
    p_buy.type = 0
    p_buy.comment = "SMC M15 BPR Midp"
    p_buy.symbol = "XAUUSDm"
    p_buy.magic = 202606
    mock_positions.return_value = [p_buy]

    tick = MagicMock()
    tick.bid = 109.35
    tick.ask = 109.61
    tick.volume = 10
    tick.volume_real = 10.0
    mock_tick.return_value = tick

    registry = {
        "sig-runner": {
            "ticket_a": 7781,
            "features_0.5": {
                "entry_price": 100.0,
                "sl_price": 95.0,
                "tp_price": 110.0,
                "direction": 1,
                "floop_trend_aligned": 1,
            },
        }
    }
    mock_load_signals.return_value = registry
    mock_modify_sltp.return_value = True

    manage_active_trades("XAUUSD", 202606, {})

    mock_modify_sltp.assert_called_once()
    _ticket, _symbol, protected_sl, tp = mock_modify_sltp.call_args.args
    assert protected_sl == pytest.approx(100.2)
    assert tp == pytest.approx(0.0)
    assert registry["sig-runner"]["runner_active_a"] is True
    assert registry["sig-runner"]["manager_exit_trigger_a"] == "runner_tp_removed"
    assert "TP1" in registry["sig-runner"]["manager_exit_detail_a"]
    mock_save_signals.assert_called()


@patch('MetaTrader5.symbol_info')
@patch('MetaTrader5.positions_get')
@patch('MetaTrader5.symbol_info_tick')
@patch('src.execution.close_position')
@patch('src.execution.load_sent_signals')
@patch('src.execution.save_sent_signals', create=True)
def test_manage_active_trades_cuts_runner_profit_on_exhaustion_retrace(
    mock_save_signals, mock_load_signals, mock_close_position, mock_tick, mock_positions, mock_symbol_info, monkeypatch
):
    monkeypatch.setenv("MT5_RUNNER_ENABLED", "True")
    monkeypatch.setenv("MT5_RUNNER_EXHAUSTION_RETRACE_PIPS", "25")
    monkeypatch.setenv("MT5_PROFIT_LOCK_ENABLED", "False")

    mock_sym = MagicMock()
    mock_sym.digits = 3
    mock_symbol_info.return_value = mock_sym

    p_buy = MagicMock()
    p_buy.ticket = 7782
    p_buy.price_open = 100.0
    p_buy.sl = 100.2
    p_buy.tp = 0.0
    p_buy.volume = 0.01
    p_buy.type = 0
    p_buy.comment = "SMC M15 BPR Midp"
    p_buy.symbol = "XAUUSDm"
    p_buy.magic = 202606
    mock_positions.return_value = [p_buy]

    tick = MagicMock()
    tick.bid = 108.8
    tick.ask = 109.06
    mock_tick.return_value = tick

    registry = {
        "sig-runner": {
            "ticket_a": 7782,
            "runner_active_a": True,
            "runner_best_price_a": 112.0,
            "features_0.5": {
                "entry_price": 100.0,
                "sl_price": 95.0,
                "tp_price": 110.0,
                "direction": 1,
                "floop_trend_aligned": 1,
            },
        }
    }
    mock_load_signals.return_value = registry
    mock_close_position.return_value = True

    manage_active_trades("XAUUSD", 202606, {})

    mock_close_position.assert_called_once_with(7782, "XAUUSD")
    assert registry["sig-runner"]["manager_exit_trigger_a"] == "runner_exhaustion_cut"
    assert registry["sig-runner"]["runner_active_a"] is False
    assert "retraced" in registry["sig-runner"]["manager_exit_detail_a"]
    mock_save_signals.assert_called()


@patch('MetaTrader5.symbol_info')
@patch('MetaTrader5.positions_get')
@patch('MetaTrader5.symbol_info_tick')
@patch('src.execution.modify_position_sltp')
@patch('src.execution.load_sent_signals')
def test_manage_active_trades_custom_staged_profit_lock_50_pips_bep(
    mock_load_signals, mock_modify_sltp, mock_tick, mock_positions, mock_symbol_info, monkeypatch
):
    monkeypatch.setenv("MT5_PROFIT_LOCK_ENABLED", "True")
    monkeypatch.setenv("MT5_CUSTOM_STAGED_PROFIT_LOCK", "True")

    mock_sym = MagicMock()
    mock_sym.digits = 3
    mock_symbol_info.return_value = mock_sym

    # Position Buy
    p_buy = MagicMock()
    p_buy.ticket = 8881
    p_buy.price_open = 100.0
    p_buy.sl = 95.0
    p_buy.tp = 120.0
    p_buy.volume = 0.01
    p_buy.type = 0  # BUY
    p_buy.comment = "SMC M15"
    p_buy.symbol = "XAUUSDm"
    p_buy.magic = 202606
    p_buy.profit = 2.0  # Floating profit under $10
    mock_positions.return_value = [p_buy]

    # Current Price: reached 105.0 (50 pips profit for Gold, pip_multiplier=0.1)
    tick = MagicMock()
    tick.bid = 105.0
    tick.ask = 105.2
    mock_tick.return_value = tick

    mock_load_signals.return_value = {}
    mock_modify_sltp.return_value = True

    manage_active_trades("XAUUSD", 202606, {})

    # SL should be set to BEP / entry_price + spread_buffer = 100.0 + 0.2 = 100.2
    mock_modify_sltp.assert_called_once_with(8881, "XAUUSD", pytest.approx(100.2), 120.0)


@patch('MetaTrader5.symbol_info')
@patch('MetaTrader5.positions_get')
@patch('MetaTrader5.symbol_info_tick')
@patch('src.execution.modify_position_sltp')
@patch('src.execution.load_sent_signals')
def test_manage_active_trades_custom_staged_profit_lock_100_pips_lock_50(
    mock_load_signals, mock_modify_sltp, mock_tick, mock_positions, mock_symbol_info, monkeypatch
):
    monkeypatch.setenv("MT5_PROFIT_LOCK_ENABLED", "True")
    monkeypatch.setenv("MT5_CUSTOM_STAGED_PROFIT_LOCK", "True")

    mock_sym = MagicMock()
    mock_sym.digits = 3
    mock_symbol_info.return_value = mock_sym

    # Position Buy
    p_buy = MagicMock()
    p_buy.ticket = 8882
    p_buy.price_open = 100.0
    p_buy.sl = 95.0
    p_buy.tp = 120.0
    p_buy.volume = 0.01
    p_buy.type = 0  # BUY
    p_buy.comment = "SMC M15"
    p_buy.symbol = "XAUUSDm"
    p_buy.magic = 202606
    p_buy.profit = 8.0  # Floating profit under $10
    mock_positions.return_value = [p_buy]

    # Current Price: reached 110.0 (100 pips profit for Gold, pip_multiplier=0.1)
    tick = MagicMock()
    tick.bid = 110.0
    tick.ask = 110.2
    mock_tick.return_value = tick

    mock_load_signals.return_value = {}
    mock_modify_sltp.return_value = True

    manage_active_trades("XAUUSD", 202606, {})

    # SL should be set to lock 50 pips = entry_price + 5.0 = 105.0
    mock_modify_sltp.assert_called_once_with(8882, "XAUUSD", pytest.approx(105.0), 120.0)


@patch('MetaTrader5.symbol_info')
@patch('MetaTrader5.positions_get')
@patch('MetaTrader5.symbol_info_tick')
@patch('src.execution.modify_position_sltp')
@patch('src.execution.load_sent_signals')
def test_manage_active_trades_custom_staged_profit_lock_gold_10_dollars_lock_50(
    mock_load_signals, mock_modify_sltp, mock_tick, mock_positions, mock_symbol_info, monkeypatch
):
    monkeypatch.setenv("MT5_PROFIT_LOCK_ENABLED", "True")
    monkeypatch.setenv("MT5_CUSTOM_STAGED_PROFIT_LOCK", "True")

    mock_sym = MagicMock()
    mock_sym.digits = 3
    mock_symbol_info.return_value = mock_sym

    # Position Buy on Gold
    p_buy = MagicMock()
    p_buy.ticket = 8883
    p_buy.price_open = 100.0
    p_buy.sl = 95.0
    p_buy.tp = 120.0
    p_buy.volume = 0.1
    p_buy.type = 0  # BUY
    p_buy.comment = "SMC M15"
    p_buy.symbol = "XAUUSDm"
    p_buy.magic = 202606
    p_buy.profit = 12.0  # Floating profit over $10
    mock_positions.return_value = [p_buy]

    # Current Price: reached 106.0 (60 pips profit - normally only BEP, but profit is $12)
    tick = MagicMock()
    tick.bid = 106.0
    tick.ask = 106.2
    mock_tick.return_value = tick

    mock_load_signals.return_value = {}
    mock_modify_sltp.return_value = True

    manage_active_trades("XAUUSD", 202606, {})

    # SL should be set to lock 50 pips = entry_price + 5.0 = 105.0 (since profit is $12.0)
    mock_modify_sltp.assert_called_once_with(8883, "XAUUSD", pytest.approx(105.0), 120.0)

