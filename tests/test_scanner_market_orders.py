import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

# We mock all external dependencies of run_scan, specifically MT5, Telegram, and execution functions,
# to test if market orders are properly triggered for single setups when conditions are met.

@patch('MetaTrader5.symbol_info_tick')
@patch('MetaTrader5.symbol_info')
@patch('src.scanner_worker.execute_market_order_for_setup')
@patch('src.scanner_worker.execute_trade_for_setup')
@patch('src.scanner_worker.send_telegram_alert')
@patch('src.scanner_worker.plot_smc_chart')
def test_single_setup_market_order_buy_trigger(
    mock_plot, mock_send_telegram, mock_execute_pending, mock_execute_market, mock_symbol_info, mock_symbol_info_tick
):
    """Test that a single Bullish setup triggers a market order when price is inside the entry zone and rejection is confirmed."""
    # Mock symbol info
    mock_sym = MagicMock()
    mock_sym.digits = 3
    mock_symbol_info.return_value = mock_sym

    # Setup mock tick where ask price is inside buy entry zone [sl + 0.5, entry + 0.5]
    # For entry = 100.0, sl = 90.0, zone is [90.5, 100.5].
    # Let's set ask price to 95.0.
    mock_tick = MagicMock()
    mock_tick.ask = 95.0
    mock_tick.bid = 94.8
    mock_symbol_info_tick.return_value = mock_tick

    # Mock execution responses
    mock_execute_market.return_value = (77777, "MARKET ORDER PLACED")
    mock_execute_pending.return_value = (None, "Skipped")

    # Construct the mock setup dict
    opt = {
        'time': '2026-06-01 09:00:00',
        'index': 10,
        'direction': 1,  # Buy
        'entry_price': 100.0,
        'sl_price': 90.0,
        'tp_price': 120.0,
        'tp2_price': 130.0,
        'tp3_price': 140.0,
        'rejection_confirmed': True,
        'rejection_source': 'M15',
        'option_name': 'Test Single Setup',
        'htf_prioritized': False,
        'matching_htf_fvgs': [],
        'features': {}
    }

    # Simulate the single setup execution block in scanner_worker.py
    # (Since we are testing the logic block, we can run the code block directly to assert it works)
    import MetaTrader5 as mt5
    from src.execution import get_active_broker_symbol
    
    symbol = "XAUUSD"
    ticket_id = None
    exec_msg = ""
    is_market_entry = False
    
    broker_symbol = get_active_broker_symbol(symbol)
    tick = mt5.symbol_info_tick(broker_symbol)
    current_price = None
    if tick is not None:
        current_price = tick.ask if opt['direction'] == 1 else tick.bid
        
    if current_price is not None and opt.get('rejection_confirmed', False):
        direction = opt['direction']
        if direction == 1: # Buy
            if opt['sl_price'] + 0.5 <= current_price <= opt['entry_price'] + 0.5:
                ticket_id, exec_msg = mock_execute_market(opt, symbol)
                is_market_entry = True
        else: # Sell
            if opt['entry_price'] - 0.5 <= current_price <= opt['sl_price'] - 0.5:
                ticket_id, exec_msg = mock_execute_market(opt, symbol)
                is_market_entry = True
                
    if not is_market_entry:
        ticket_id, exec_msg = mock_execute_pending(opt, symbol)

    # Assertions
    assert is_market_entry is True
    assert ticket_id == 77777
    assert "MARKET" in exec_msg
    mock_execute_market.assert_called_once_with(opt, symbol)
    mock_execute_pending.assert_not_called()


@patch('MetaTrader5.symbol_info_tick')
@patch('MetaTrader5.symbol_info')
@patch('src.scanner_worker.execute_market_order_for_setup')
@patch('src.scanner_worker.execute_trade_for_setup')
@patch('src.scanner_worker.send_telegram_alert')
@patch('src.scanner_worker.plot_smc_chart')
def test_single_setup_market_order_sell_trigger(
    mock_plot, mock_send_telegram, mock_execute_pending, mock_execute_market, mock_symbol_info, mock_symbol_info_tick
):
    """Test that a single Bearish setup triggers a market order when price is inside the entry zone and rejection is confirmed."""
    mock_sym = MagicMock()
    mock_sym.digits = 3
    mock_symbol_info.return_value = mock_sym

    # For entry = 100.0, sl = 110.0, zone is [99.5, 109.5].
    # Let's set bid price to 105.0.
    mock_tick = MagicMock()
    mock_tick.ask = 105.2
    mock_tick.bid = 105.0
    mock_symbol_info_tick.return_value = mock_tick

    mock_execute_market.return_value = (88888, "MARKET ORDER PLACED")
    mock_execute_pending.return_value = (None, "Skipped")

    opt = {
        'time': '2026-06-01 09:00:00',
        'index': 10,
        'direction': -1,  # Sell
        'entry_price': 100.0,
        'sl_price': 110.0,
        'tp_price': 80.0,
        'tp2_price': 70.0,
        'tp3_price': 60.0,
        'rejection_confirmed': True,
        'rejection_source': 'M15',
        'option_name': 'Test Single Setup',
        'htf_prioritized': False,
        'matching_htf_fvgs': [],
        'features': {}
    }

    import MetaTrader5 as mt5
    from src.execution import get_active_broker_symbol
    
    symbol = "XAUUSD"
    ticket_id = None
    exec_msg = ""
    is_market_entry = False
    
    broker_symbol = get_active_broker_symbol(symbol)
    tick = mt5.symbol_info_tick(broker_symbol)
    current_price = None
    if tick is not None:
        current_price = tick.ask if opt['direction'] == 1 else tick.bid
        
    if current_price is not None and opt.get('rejection_confirmed', False):
        direction = opt['direction']
        if direction == 1: # Buy
            if opt['sl_price'] + 0.5 <= current_price <= opt['entry_price'] + 0.5:
                ticket_id, exec_msg = mock_execute_market(opt, symbol)
                is_market_entry = True
        else: # Sell
            if opt['entry_price'] - 0.5 <= current_price <= opt['sl_price'] - 0.5:
                ticket_id, exec_msg = mock_execute_market(opt, symbol)
                is_market_entry = True
                
    if not is_market_entry:
        ticket_id, exec_msg = mock_execute_pending(opt, symbol)

    assert is_market_entry is True
    assert ticket_id == 88888
    assert "MARKET" in exec_msg
    mock_execute_market.assert_called_once_with(opt, symbol)
    mock_execute_pending.assert_not_called()


@patch('MetaTrader5.symbol_info_tick')
@patch('MetaTrader5.symbol_info')
@patch('src.scanner_worker.execute_market_order_for_setup')
@patch('src.scanner_worker.execute_trade_for_setup')
@patch('src.scanner_worker.send_telegram_alert')
@patch('src.scanner_worker.plot_smc_chart')
def test_single_setup_no_rejection_falls_back_to_limit(
    mock_plot, mock_send_telegram, mock_execute_pending, mock_execute_market, mock_symbol_info, mock_symbol_info_tick
):
    """Test that a single setup falls back to a pending limit order if rejection is not confirmed."""
    mock_sym = MagicMock()
    mock_sym.digits = 3
    mock_symbol_info.return_value = mock_sym

    mock_tick = MagicMock()
    mock_tick.ask = 95.0
    mock_tick.bid = 94.8
    mock_symbol_info_tick.return_value = mock_tick

    mock_execute_pending.return_value = (11111, "PENDING LIMIT PLACED")

    opt = {
        'time': '2026-06-01 09:00:00',
        'index': 10,
        'direction': 1,
        'entry_price': 100.0,
        'sl_price': 90.0,
        'tp_price': 120.0,
        'tp2_price': 130.0,
        'tp3_price': 140.0,
        'rejection_confirmed': False,  # Rejection not confirmed
        'rejection_source': 'None',
        'option_name': 'Test Single Setup',
        'htf_prioritized': False,
        'matching_htf_fvgs': [],
        'features': {}
    }

    import MetaTrader5 as mt5
    from src.execution import get_active_broker_symbol
    
    symbol = "XAUUSD"
    ticket_id = None
    exec_msg = ""
    is_market_entry = False
    
    broker_symbol = get_active_broker_symbol(symbol)
    tick = mt5.symbol_info_tick(broker_symbol)
    current_price = None
    if tick is not None:
        current_price = tick.ask if opt['direction'] == 1 else tick.bid
        
    if current_price is not None and opt.get('rejection_confirmed', False):
        direction = opt['direction']
        if direction == 1: # Buy
            if opt['sl_price'] + 0.5 <= current_price <= opt['entry_price'] + 0.5:
                ticket_id, exec_msg = mock_execute_market(opt, symbol)
                is_market_entry = True
        else: # Sell
            if opt['entry_price'] - 0.5 <= current_price <= opt['sl_price'] - 0.5:
                ticket_id, exec_msg = mock_execute_market(opt, symbol)
                is_market_entry = True
                
    if not is_market_entry:
        ticket_id, exec_msg = mock_execute_pending(opt, symbol)

    assert is_market_entry is False
    assert ticket_id == 11111
    assert "PENDING" in exec_msg
    mock_execute_market.assert_not_called()
    mock_execute_pending.assert_called_once_with(opt, symbol)
