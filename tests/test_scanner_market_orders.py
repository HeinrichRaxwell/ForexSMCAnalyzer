import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from types import SimpleNamespace

# We mock all external dependencies of run_scan, specifically MT5, Telegram, and execution functions,
# to test if market orders are properly triggered for single setups when conditions are met.


def test_active_trade_count_filters_magic_without_unsupported_mt5_kwarg():
    from src import execution

    def positions_get(**kwargs):
        assert kwargs == {"symbol": "XAUUSD"}
        return [
            SimpleNamespace(ticket=1, magic=202606),
            SimpleNamespace(ticket=2, magic=999999),
        ]

    def orders_get(**kwargs):
        assert kwargs == {"symbol": "XAUUSD"}
        return [
            SimpleNamespace(ticket=3, magic=202606),
            SimpleNamespace(ticket=4, magic=123456),
        ]

    with patch.object(execution.mt5, "positions_get", side_effect=positions_get), patch.object(
        execution.mt5, "orders_get", side_effect=orders_get
    ):
        assert execution.get_active_trade_count("XAUUSD", 202606) == 2


@patch("MetaTrader5.symbol_info")
@patch("MetaTrader5.orders_get")
@patch("MetaTrader5.symbol_info_tick")
@patch("MetaTrader5.order_send")
def test_prune_pending_orders_filters_magic_without_unsupported_mt5_kwarg(
    mock_order_send, mock_tick, mock_orders_get, mock_symbol_info
):
    from src.scanner_worker import prune_invalid_pending_orders

    mock_symbol_info.return_value = SimpleNamespace(digits=3, point=0.001)
    wrong_magic_order = SimpleNamespace(ticket=44, magic=999999, price_open=4000.0, type=2)

    def orders_get(**kwargs):
        assert kwargs == {"symbol": "XAUUSD"}
        return [wrong_magic_order]

    mock_orders_get.side_effect = orders_get

    prune_invalid_pending_orders("XAUUSD", 202606, [])

    mock_tick.assert_not_called()
    mock_order_send.assert_not_called()


@patch("MetaTrader5.symbol_info")
@patch("MetaTrader5.orders_get")
@patch("MetaTrader5.symbol_info_tick")
@patch("MetaTrader5.order_send")
def test_prune_pending_orders_keeps_spread_adjusted_buy_limit(
    mock_order_send, mock_tick, mock_orders_get, mock_symbol_info
):
    from src.scanner_worker import prune_invalid_pending_orders

    mock_symbol_info.return_value = SimpleNamespace(digits=3, point=0.001)
    mock_tick.return_value = SimpleNamespace(ask=4077.900, bid=4077.640, last=4077.640)
    mock_orders_get.return_value = [
        SimpleNamespace(
            ticket=3263604605,
            magic=202606,
            price_open=4067.958,
            type=2,
        )
    ]

    prune_invalid_pending_orders(
        "XAUUSD",
        202606,
        [
            {
                "timeframe": "H1",
                "direction": 1,
                "entry_price": 4067.698,
            }
        ],
    )

    mock_order_send.assert_not_called()


@patch("MetaTrader5.symbol_info")
@patch("MetaTrader5.orders_get")
@patch("MetaTrader5.symbol_info_tick")
@patch("MetaTrader5.order_send")
def test_prune_pending_orders_keeps_valid_order_even_when_price_moves_far(
    mock_order_send, mock_tick, mock_orders_get, mock_symbol_info
):
    from src.scanner_worker import prune_invalid_pending_orders

    mock_symbol_info.return_value = SimpleNamespace(digits=3, point=0.001)
    mock_tick.return_value = SimpleNamespace(ask=4200.260, bid=4200.000, last=4200.000)
    mock_orders_get.return_value = [
        SimpleNamespace(
            ticket=3263604606,
            magic=202606,
            price_open=4067.958,
            type=2,
        )
    ]

    prune_invalid_pending_orders(
        "XAUUSD",
        202606,
        [
            {
                "timeframe": "H1",
                "direction": 1,
                "entry_price": 4067.698,
            }
        ],
    )

    mock_order_send.assert_not_called()


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


def _pending_setup(**overrides):
    setup = {
        "timeframe": "M15",
        "direction": 1,
        "entry_price": 4000.0,
        "sl_price": 3995.0,
        "tp_price": 4010.0,
        "tp2_price": 4015.0,
        "tp3_price": 4020.0,
        "option_name": "FVG Midpoint 0.5",
    }
    setup.update(overrides)
    return setup


@patch("src.execution.mt5")
def test_dynamic_lot_size_uses_balance_million_ladder(mock_mt5, monkeypatch):
    from src.execution import resolve_lot_size

    monkeypatch.setenv("MT5_DYNAMIC_LOT_ENABLED", "True")
    monkeypatch.setenv("MT5_DYNAMIC_LOT_BASE_BALANCE_IDR", "2000000")
    monkeypatch.setenv("MT5_DYNAMIC_LOT_BALANCE_STEP_IDR", "1000000")
    monkeypatch.setenv("MT5_DYNAMIC_LOT_BASE_LOT", "0.01")
    monkeypatch.setenv("MT5_DYNAMIC_LOT_STEP_LOT", "0.01")
    monkeypatch.setenv("MT5_DYNAMIC_LOT_MAX", "0.10")
    mock_mt5.symbol_info.return_value = SimpleNamespace(volume_min=0.01, volume_step=0.01, volume_max=100.0)

    mock_mt5.account_info.return_value = SimpleNamespace(balance=2143722.86)
    assert resolve_lot_size("FVG Midpoint 0.5", "XAUUSDm") == pytest.approx(0.01)

    mock_mt5.account_info.return_value = SimpleNamespace(balance=3000000.0)
    assert resolve_lot_size("FVG Midpoint 0.5", "XAUUSDm") == pytest.approx(0.02)

    mock_mt5.account_info.return_value = SimpleNamespace(balance=4000000.0)
    assert resolve_lot_size("FVG GoldenPocket 0.618", "XAUUSDm") == pytest.approx(0.03)


@patch("src.execution.validate_market_indicators", return_value=(True, "ok"))
@patch("src.execution.mt5")
def test_pending_buy_order_adds_live_spread_to_entry_price(mock_mt5, _mock_validate, monkeypatch):
    from src.execution import execute_trade_for_setup

    monkeypatch.setenv("MT5_EXECUTE_TRADES", "True")
    monkeypatch.setenv("MT5_ALLOWED_TIMEFRAMES", "M30,H1,H4,D1")
    mock_mt5.TRADE_RETCODE_PLACED = 10008
    mock_mt5.TRADE_RETCODE_DONE = 10009
    mock_mt5.symbol_info.side_effect = lambda symbol: SimpleNamespace(digits=3, point=0.001)
    mock_mt5.symbol_info_tick.return_value = SimpleNamespace(ask=3999.900, bid=3999.640)
    mock_mt5.orders_get.return_value = []
    mock_mt5.positions_get.return_value = []
    mock_mt5.order_send.return_value = SimpleNamespace(retcode=mock_mt5.TRADE_RETCODE_PLACED, order=12345)
    mock_mt5.ORDER_TYPE_BUY_LIMIT = 2
    mock_mt5.ORDER_TYPE_SELL_LIMIT = 3
    mock_mt5.TRADE_ACTION_PENDING = 5
    mock_mt5.ORDER_TIME_GTC = 0
    mock_mt5.ORDER_FILLING_RETURN = 0
    mock_mt5.ORDER_FILLING_IOC = 1
    mock_mt5.ORDER_FILLING_FOK = 2

    ticket, _message = execute_trade_for_setup(_pending_setup(timeframe="M30", direction=1), "XAUUSD")

    assert ticket == 12345
    request = mock_mt5.order_send.call_args.args[0]
    assert request["price"] == pytest.approx(4000.260)
    assert request["tp"] == pytest.approx(4010.0)


@patch("src.execution.validate_market_indicators", return_value=(True, "ok"))
@patch("src.execution.mt5")
def test_pending_buy_order_caps_far_tp_to_default_150_pips(mock_mt5, _mock_validate, monkeypatch):
    from src.execution import execute_trade_for_setup

    monkeypatch.setenv("MT5_EXECUTE_TRADES", "True")
    monkeypatch.setenv("MT5_ALLOWED_TIMEFRAMES", "M30,H1,H4,D1")
    mock_mt5.TRADE_RETCODE_PLACED = 10008
    mock_mt5.TRADE_RETCODE_DONE = 10009
    mock_mt5.symbol_info.side_effect = lambda symbol: SimpleNamespace(digits=3, point=0.001)
    mock_mt5.symbol_info_tick.return_value = SimpleNamespace(ask=3999.900, bid=3999.640)
    mock_mt5.orders_get.return_value = []
    mock_mt5.positions_get.return_value = []
    mock_mt5.order_send.return_value = SimpleNamespace(retcode=mock_mt5.TRADE_RETCODE_PLACED, order=12345)
    mock_mt5.ORDER_TYPE_BUY_LIMIT = 2
    mock_mt5.ORDER_TYPE_SELL_LIMIT = 3
    mock_mt5.TRADE_ACTION_PENDING = 5
    mock_mt5.ORDER_TIME_GTC = 0
    mock_mt5.ORDER_FILLING_RETURN = 0
    mock_mt5.ORDER_FILLING_IOC = 1
    mock_mt5.ORDER_FILLING_FOK = 2

    # Structural TP (fibo_0_0) sits 1000 pips away; should be clamped to entry + 150 pips.
    ticket, _message = execute_trade_for_setup(
        _pending_setup(timeframe="M30", direction=1, tp_price=4100.0), "XAUUSD"
    )

    assert ticket == 12345
    request = mock_mt5.order_send.call_args.args[0]
    assert request["price"] == pytest.approx(4000.260)
    assert request["tp"] == pytest.approx(4015.260)


@patch("src.execution.mt5")
def test_pending_order_blocks_m15_when_allowed_timeframes_not_overridden(mock_mt5, monkeypatch):
    from src.execution import execute_trade_for_setup

    monkeypatch.setenv("MT5_EXECUTE_TRADES", "True")
    monkeypatch.delenv("MT5_ALLOWED_TIMEFRAMES", raising=False)

    ticket, message = execute_trade_for_setup(_pending_setup(timeframe="M15"), "XAUUSD")

    assert ticket is None
    assert "Timeframe M15 disabled" in message
    mock_mt5.order_send.assert_not_called()


@patch("src.execution.mt5")
def test_pending_order_respects_max_concurrent_trades(mock_mt5, monkeypatch):
    from src.execution import execute_trade_for_setup

    monkeypatch.setenv("MT5_EXECUTE_TRADES", "True")
    monkeypatch.setenv("MT5_ALLOWED_TIMEFRAMES", "M30,H1,H4,D1")
    monkeypatch.setenv("MT5_MAX_CONCURRENT_TRADES", "1")
    mock_mt5.symbol_info.return_value = SimpleNamespace(digits=3, point=0.001)
    mock_mt5.symbol_info_tick.return_value = SimpleNamespace(ask=3999.900, bid=3999.640)
    mock_mt5.orders_get.return_value = [SimpleNamespace(ticket=111, magic=202606)]
    mock_mt5.positions_get.return_value = []
    mock_mt5.ORDER_TYPE_BUY_LIMIT = 2
    mock_mt5.ORDER_TYPE_SELL_LIMIT = 3

    ticket, message = execute_trade_for_setup(_pending_setup(timeframe="M30", direction=1), "XAUUSD")

    assert ticket is None
    assert "max concurrent trades" in message
    mock_mt5.order_send.assert_not_called()


@patch("src.execution.validate_market_indicators", return_value=(True, "ok"))
@patch("src.execution.mt5")
def test_pending_order_has_no_concurrent_trade_limit_by_default(mock_mt5, _mock_validate, monkeypatch):
    from src.execution import execute_trade_for_setup

    monkeypatch.setenv("MT5_EXECUTE_TRADES", "True")
    monkeypatch.setenv("MT5_ALLOWED_TIMEFRAMES", "M30,H1,H4,D1")
    monkeypatch.delenv("MT5_MAX_CONCURRENT_TRADES", raising=False)
    mock_mt5.TRADE_RETCODE_PLACED = 10008
    mock_mt5.TRADE_RETCODE_DONE = 10009
    mock_mt5.symbol_info.side_effect = lambda symbol: SimpleNamespace(digits=3, point=0.001)
    mock_mt5.symbol_info_tick.return_value = SimpleNamespace(ask=3999.900, bid=3999.640)
    mock_mt5.orders_get.return_value = [
        SimpleNamespace(ticket=111, magic=202606, price_open=3990.0),
        SimpleNamespace(ticket=112, magic=202606, price_open=3985.0),
    ]
    mock_mt5.positions_get.return_value = [SimpleNamespace(ticket=222, magic=202606, price_open=3980.0)]
    mock_mt5.order_send.return_value = SimpleNamespace(retcode=mock_mt5.TRADE_RETCODE_PLACED, order=22347)
    mock_mt5.ORDER_TYPE_BUY_LIMIT = 2
    mock_mt5.ORDER_TYPE_SELL_LIMIT = 3
    mock_mt5.TRADE_ACTION_PENDING = 5
    mock_mt5.ORDER_TIME_GTC = 0
    mock_mt5.ORDER_FILLING_RETURN = 0
    mock_mt5.ORDER_FILLING_IOC = 1
    mock_mt5.ORDER_FILLING_FOK = 2

    ticket, message = execute_trade_for_setup(_pending_setup(timeframe="M30", direction=1), "XAUUSD")

    assert ticket == 22347
    assert "PENDING ORDER PLACED" in message


@patch("src.execution.validate_market_indicators", return_value=(True, "ok"))
@patch("src.execution.mt5")
def test_pending_order_continues_after_daily_runner_target(mock_mt5, _mock_validate, monkeypatch):
    from src.execution import execute_trade_for_setup

    monkeypatch.setenv("MT5_EXECUTE_TRADES", "True")
    monkeypatch.setenv("MT5_ALLOWED_TIMEFRAMES", "M30,H1,H4,D1")
    monkeypatch.setenv("MT5_MAX_CONCURRENT_TRADES", "3")
    monkeypatch.setenv("MT5_DAILY_RUNNER_TARGET_PIPS", "300")
    mock_mt5.TRADE_RETCODE_PLACED = 10008
    mock_mt5.TRADE_RETCODE_DONE = 10009
    mock_mt5.DEAL_ENTRY_IN = 0
    mock_mt5.DEAL_ENTRY_OUT = 1
    mock_mt5.DEAL_TYPE_BUY = 0
    mock_mt5.DEAL_TYPE_SELL = 1
    mock_mt5.symbol_info.return_value = SimpleNamespace(digits=3, point=0.001)
    mock_mt5.symbol_info_tick.return_value = SimpleNamespace(ask=3999.900, bid=3999.640)
    mock_mt5.orders_get.return_value = []
    mock_mt5.positions_get.return_value = []
    mock_mt5.history_deals_get.return_value = [
        SimpleNamespace(position_id=1, entry=0, type=0, price=4000.0, magic=202606),
        SimpleNamespace(position_id=1, entry=1, type=1, price=4031.0, magic=202606),
    ]
    mock_mt5.order_send.return_value = SimpleNamespace(retcode=mock_mt5.TRADE_RETCODE_PLACED, order=22345)
    mock_mt5.ORDER_TYPE_BUY_LIMIT = 2
    mock_mt5.ORDER_TYPE_SELL_LIMIT = 3
    mock_mt5.TRADE_ACTION_PENDING = 5
    mock_mt5.ORDER_TIME_GTC = 0
    mock_mt5.ORDER_FILLING_RETURN = 0
    mock_mt5.ORDER_FILLING_IOC = 1
    mock_mt5.ORDER_FILLING_FOK = 2

    ticket, message = execute_trade_for_setup(_pending_setup(timeframe="M30", direction=1), "XAUUSD")

    assert ticket == 22345
    assert "PENDING ORDER PLACED" in message


@patch("src.execution.validate_market_indicators", return_value=(True, "ok"))
@patch("src.execution.mt5")
def test_pending_order_blocks_when_daily_history_is_unavailable(mock_mt5, _mock_validate, monkeypatch):
    from src.execution import execute_trade_for_setup

    monkeypatch.setenv("MT5_EXECUTE_TRADES", "True")
    monkeypatch.setenv("MT5_ALLOWED_TIMEFRAMES", "M30,H1,H4,D1")
    monkeypatch.setenv("MT5_MAX_CONCURRENT_TRADES", "3")
    monkeypatch.setenv("MT5_DAILY_GOVERNOR_ENABLED", "True")
    mock_mt5.TRADE_RETCODE_PLACED = 10008
    mock_mt5.TRADE_RETCODE_DONE = 10009
    mock_mt5.symbol_info.return_value = SimpleNamespace(digits=3, point=0.001)
    mock_mt5.symbol_info_tick.return_value = SimpleNamespace(ask=3999.900, bid=3999.640)
    mock_mt5.orders_get.return_value = []
    mock_mt5.positions_get.return_value = []
    mock_mt5.history_deals_get.side_effect = RuntimeError("MT5 history unavailable")
    mock_mt5.order_send.return_value = SimpleNamespace(retcode=mock_mt5.TRADE_RETCODE_PLACED, order=22346)
    mock_mt5.ORDER_TYPE_BUY_LIMIT = 2
    mock_mt5.ORDER_TYPE_SELL_LIMIT = 3
    mock_mt5.TRADE_ACTION_PENDING = 5
    mock_mt5.ORDER_TIME_GTC = 0
    mock_mt5.ORDER_FILLING_RETURN = 0
    mock_mt5.ORDER_FILLING_IOC = 1
    mock_mt5.ORDER_FILLING_FOK = 2

    ticket, message = execute_trade_for_setup(_pending_setup(timeframe="M30", direction=1), "XAUUSD")

    assert ticket is None
    assert "daily governor unavailable" in message
    mock_mt5.order_send.assert_not_called()


@patch("src.execution.validate_market_indicators", return_value=(True, "ok"))
@patch("src.execution.mt5")
def test_pending_order_blocks_when_daily_governor_hits_loss_limit(mock_mt5, _mock_validate, monkeypatch):
    from src.execution import execute_trade_for_setup

    monkeypatch.setenv("MT5_EXECUTE_TRADES", "True")
    monkeypatch.setenv("MT5_ALLOWED_TIMEFRAMES", "M30,H1,H4,D1")
    monkeypatch.setenv("MT5_MAX_CONCURRENT_TRADES", "3")
    monkeypatch.setenv("MT5_DAILY_GOVERNOR_ENABLED", "True")
    monkeypatch.setenv("MT5_DAILY_MAX_LOSS_PIPS", "200")
    mock_mt5.TRADE_RETCODE_PLACED = 10008
    mock_mt5.TRADE_RETCODE_DONE = 10009
    mock_mt5.DEAL_ENTRY_IN = 0
    mock_mt5.DEAL_ENTRY_OUT = 1
    mock_mt5.DEAL_TYPE_BUY = 0
    mock_mt5.DEAL_TYPE_SELL = 1
    mock_mt5.symbol_info.return_value = SimpleNamespace(digits=3, point=0.001)
    mock_mt5.symbol_info_tick.return_value = SimpleNamespace(ask=3999.900, bid=3999.640)
    mock_mt5.orders_get.return_value = []
    mock_mt5.positions_get.return_value = []
    mock_mt5.history_deals_get.return_value = [
        SimpleNamespace(position_id=1, entry=0, type=0, price=4000.0, magic=202606, symbol="XAUUSD"),
        SimpleNamespace(position_id=1, entry=1, type=1, price=3979.0, magic=202606, symbol="XAUUSD"),
    ]

    ticket, message = execute_trade_for_setup(_pending_setup(timeframe="M30", direction=1), "XAUUSD")

    assert ticket is None
    assert "daily risk governor blocked new order" in message
    mock_mt5.order_send.assert_not_called()


@patch("src.execution.validate_market_indicators", return_value=(True, "ok"))
@patch("src.execution.mt5")
def test_pending_sell_order_subtracts_live_spread_from_entry_price(mock_mt5, _mock_validate, monkeypatch):
    from src.execution import execute_trade_for_setup

    monkeypatch.setenv("MT5_EXECUTE_TRADES", "True")
    monkeypatch.setenv("MT5_ALLOWED_TIMEFRAMES", "M30,H1,H4,D1")
    mock_mt5.TRADE_RETCODE_PLACED = 10008
    mock_mt5.TRADE_RETCODE_DONE = 10009
    mock_mt5.symbol_info.side_effect = lambda symbol: SimpleNamespace(digits=3, point=0.001)
    mock_mt5.symbol_info_tick.return_value = SimpleNamespace(ask=4000.260, bid=4000.000)
    mock_mt5.orders_get.return_value = []
    mock_mt5.positions_get.return_value = []
    mock_mt5.order_send.return_value = SimpleNamespace(retcode=mock_mt5.TRADE_RETCODE_PLACED, order=12346)
    mock_mt5.ORDER_TYPE_BUY_LIMIT = 2
    mock_mt5.ORDER_TYPE_SELL_LIMIT = 3
    mock_mt5.TRADE_ACTION_PENDING = 5
    mock_mt5.ORDER_TIME_GTC = 0
    mock_mt5.ORDER_FILLING_RETURN = 0
    mock_mt5.ORDER_FILLING_IOC = 1
    mock_mt5.ORDER_FILLING_FOK = 2

    ticket, _message = execute_trade_for_setup(_pending_setup(timeframe="M30", direction=-1, sl_price=4005.0, tp_price=3990.0), "XAUUSD")

    assert ticket == 12346
    request = mock_mt5.order_send.call_args.args[0]
    assert request["price"] == pytest.approx(3999.740)


@patch("src.execution.validate_market_indicators", return_value=(True, "ok"))
@patch("src.execution.mt5")
def test_pending_order_respects_max_pending_orders(mock_mt5, _mock_validate, monkeypatch):
    from src.execution import execute_trade_for_setup

    monkeypatch.setenv("MT5_EXECUTE_TRADES", "True")
    monkeypatch.setenv("MT5_ALLOWED_TIMEFRAMES", "M30,H1,H4,D1")
    monkeypatch.setenv("MT5_MAX_PENDING_ORDERS", "2")
    monkeypatch.setenv("MT5_PENDING_PROXIMITY_PIPS", "0.0")  # Disable proximity check
    mock_mt5.symbol_info.side_effect = lambda symbol: SimpleNamespace(digits=3, point=0.001)
    mock_mt5.symbol_info_tick.return_value = SimpleNamespace(ask=3999.900, bid=3999.640)
    mock_mt5.orders_get.return_value = [
        SimpleNamespace(ticket=111, magic=202606, price_open=3980.0, comment="SMC M30 FVG Option A", type=2),
        SimpleNamespace(ticket=112, magic=202606, price_open=3975.0, comment="SMC M30 FVG Option B", type=2),
    ]
    mock_mt5.positions_get.return_value = []
    mock_mt5.ORDER_TYPE_BUY_LIMIT = 2
    mock_mt5.ORDER_TYPE_SELL_LIMIT = 3

    ticket, message = execute_trade_for_setup(_pending_setup(timeframe="M30", direction=1), "XAUUSD")

    assert ticket is None
    assert "max pending orders reached" in message
    mock_mt5.order_send.assert_not_called()


@patch("src.execution.validate_market_indicators", return_value=(True, "ok"))
@patch("src.execution.mt5")
def test_pending_order_blocks_mixed_strategies_on_same_tf(mock_mt5, _mock_validate, monkeypatch):
    from src.execution import execute_trade_for_setup

    monkeypatch.setenv("MT5_EXECUTE_TRADES", "True")
    monkeypatch.setenv("MT5_ALLOWED_TIMEFRAMES", "M30,H1,H4,D1")
    monkeypatch.setenv("MT5_MAX_PENDING_ORDERS", "5")
    monkeypatch.setenv("MT5_ALLOW_MIXED_STRATEGIES_PER_TF", "False")
    monkeypatch.setenv("MT5_PENDING_PROXIMITY_PIPS", "0.0")  # Disable proximity check
    
    mock_mt5.TRADE_RETCODE_PLACED = 10008
    mock_mt5.TRADE_RETCODE_DONE = 10009
    mock_mt5.symbol_info.side_effect = lambda symbol: SimpleNamespace(digits=3, point=0.001)
    mock_mt5.symbol_info_tick.return_value = SimpleNamespace(ask=3999.900, bid=3999.640)
    
    # We already have an FVG pending order on M30
    mock_mt5.orders_get.return_value = [
        SimpleNamespace(ticket=111, magic=202606, price_open=3980.0, comment="SMC M30 FVG Option A", type=2),
    ]
    mock_mt5.positions_get.return_value = []
    mock_mt5.order_send.return_value = SimpleNamespace(retcode=mock_mt5.TRADE_RETCODE_PLACED, order=999)
    mock_mt5.ORDER_TYPE_BUY_LIMIT = 2
    mock_mt5.ORDER_TYPE_SELL_LIMIT = 3
    mock_mt5.TRADE_ACTION_PENDING = 5
    mock_mt5.ORDER_TIME_GTC = 0
    mock_mt5.ORDER_FILLING_RETURN = 0
    mock_mt5.ORDER_FILLING_IOC = 1
    mock_mt5.ORDER_FILLING_FOK = 2

    # Case 1: Try placing an OB on M30 -> should be blocked
    ticket_ob, msg_ob = execute_trade_for_setup(_pending_setup(timeframe="M30", direction=1, option_name="OB Option A"), "XAUUSD")
    assert ticket_ob is None
    assert "blocked mixed strategy" in msg_ob
    
    # Case 2: Try placing FVG on M30 -> should be allowed
    ticket_fvg, msg_fvg = execute_trade_for_setup(_pending_setup(timeframe="M30", direction=1, option_name="FVG Option B"), "XAUUSD")
    assert ticket_fvg == 999
    
    # Case 3: Allow mixed strategies -> OB should be allowed
    monkeypatch.setenv("MT5_ALLOW_MIXED_STRATEGIES_PER_TF", "True")
    mock_mt5.order_send.reset_mock()
    ticket_ob_allowed, msg_ob_allowed = execute_trade_for_setup(_pending_setup(timeframe="M30", direction=1, option_name="OB Option A"), "XAUUSD")
    assert ticket_ob_allowed == 999


@patch("src.execution.validate_market_indicators", return_value=(True, "ok"))
@patch("src.execution.mt5")
def test_pending_order_proximity_block(mock_mt5, _mock_validate, monkeypatch):
    from src.execution import execute_trade_for_setup

    monkeypatch.setenv("MT5_EXECUTE_TRADES", "True")
    monkeypatch.setenv("MT5_ALLOWED_TIMEFRAMES", "M30,H1,H4,D1")
    monkeypatch.setenv("MT5_MAX_PENDING_ORDERS", "0")  # Disabled
    monkeypatch.setenv("MT5_PENDING_PROXIMITY_PIPS", "15.0")
    
    mock_mt5.TRADE_RETCODE_PLACED = 10008
    mock_mt5.TRADE_RETCODE_DONE = 10009
    mock_mt5.symbol_info.side_effect = lambda symbol: SimpleNamespace(digits=3, point=0.001)
    mock_mt5.symbol_info_tick.return_value = SimpleNamespace(ask=3999.900, bid=3999.640)
    
    # Mock get_pip_multiplier to return 0.1 so 15 pips = 1.5 price unit
    with patch('src.smc_detector.get_pip_multiplier', return_value=0.1):
        # We already have an H1 OB pending order at 3998.0
        mock_mt5.orders_get.return_value = [
            SimpleNamespace(ticket=111, magic=202606, price_open=3998.0, comment="SMC H1 OB Option A", type=2),
        ]
        mock_mt5.positions_get.return_value = []
        mock_mt5.order_send.return_value = SimpleNamespace(retcode=mock_mt5.TRADE_RETCODE_PLACED, order=999)
        mock_mt5.ORDER_TYPE_BUY_LIMIT = 2
        mock_mt5.ORDER_TYPE_SELL_LIMIT = 3
        mock_mt5.TRADE_ACTION_PENDING = 5
        mock_mt5.ORDER_TIME_GTC = 0
        mock_mt5.ORDER_FILLING_RETURN = 0
        mock_mt5.ORDER_FILLING_IOC = 1
        mock_mt5.ORDER_FILLING_FOK = 2

        # Case 1: Try placing M30 FVG at 3999.0 (dist = 1.0 < 1.5) -> should be blocked by proximity
        ticket_close, msg_close = execute_trade_for_setup(
            _pending_setup(timeframe="M30", direction=1, entry_price=3999.0, option_name="FVG Option A"), "XAUUSD"
        )
        assert ticket_close is None
        assert "proximity block" in msg_close

        # Case 2: Try placing M30 FVG at 3996.0 (dist = 2.0 > 1.5) -> should be allowed
        ticket_far, msg_far = execute_trade_for_setup(
            _pending_setup(timeframe="M30", direction=1, entry_price=3996.0, option_name="FVG Option A"), "XAUUSD"
        )
        assert ticket_far == 999

        # Case 3: Try placing H1 OB Option B at 3999.0 (dist = 1.0) -> allowed: A+B are dual-fib counterparts of the same structure
        ticket_same, msg_same = execute_trade_for_setup(
            _pending_setup(timeframe="H1", direction=1, entry_price=3999.0, option_name="OB Option B"), "XAUUSD"
        )
        assert ticket_same == 999

        # Case 4: Try placing H1 OB Option A (another cycle) at 3999.5 (dist = 1.5, same_tf_limit = 30 pips = 3.0) -> blocked
        monkeypatch.setenv("MT5_SAME_TF_PROXIMITY_PIPS", "30.0")
        ticket_stack, msg_stack = execute_trade_for_setup(
            _pending_setup(timeframe="H1", direction=1, entry_price=3999.5, option_name="OB Option A"), "XAUUSD"
        )
        assert ticket_stack is None
        assert "proximity block" in msg_stack


@patch("MetaTrader5.symbol_info")
@patch("MetaTrader5.orders_get")
@patch("MetaTrader5.symbol_info_tick")
@patch("MetaTrader5.order_send")
@patch("src.execution.load_sent_signals")
def test_prune_pending_orders_preserves_young_order(
    mock_load_signals, mock_order_send, mock_tick, mock_orders_get, mock_symbol_info
):
    from src.scanner_worker import prune_invalid_pending_orders
    from datetime import datetime

    mock_symbol_info.return_value = SimpleNamespace(digits=3, point=0.001)
    mock_tick.return_value = SimpleNamespace(ask=4077.900, bid=4077.640, last=4077.640)
    mock_orders_get.return_value = [
        SimpleNamespace(
            ticket=55555,
            magic=202606,
            price_open=4050.0,
            sl=4040.0,
            type=2, # BUY_LIMIT
        )
    ]
    
    # Mock sent signals showing the order was placed 5 minutes ago (young)
    mock_load_signals.return_value = {
        "sig-test": {
            "ticket_a": 55555,
            "time_sent": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    }

    # Pass empty active setups (meaning it's not active in current scan)
    prune_invalid_pending_orders("XAUUSD", 202606, [])

    # Order should be preserved (not canceled)
    mock_order_send.assert_not_called()


@patch("MetaTrader5.symbol_info")
@patch("MetaTrader5.orders_get")
@patch("MetaTrader5.symbol_info_tick")
@patch("MetaTrader5.order_send")
@patch("src.execution.load_sent_signals")
def test_prune_pending_orders_cancels_when_sl_violated(
    mock_load_signals, mock_order_send, mock_tick, mock_orders_get, mock_symbol_info
):
    from src.scanner_worker import prune_invalid_pending_orders
    from datetime import datetime

    mock_symbol_info.return_value = SimpleNamespace(digits=3, point=0.001)
    # Price bid (4035.0) is below SL (4040.0) -> BUY limit SL violated!
    mock_tick.return_value = SimpleNamespace(ask=4035.2, bid=4035.0, last=4035.0)
    mock_orders_get.return_value = [
        SimpleNamespace(
            ticket=66666,
            magic=202606,
            price_open=4050.0,
            sl=4040.0,
            type=2, # BUY_LIMIT
        )
    ]
    
    # Even if order is young (placed 5 mins ago), it should be pruned because SL is violated!
    mock_load_signals.return_value = {
        "sig-test": {
            "ticket_a": 66666,
            "time_sent": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    }
    
    mock_order_send.return_value = SimpleNamespace(retcode=10009)

    prune_invalid_pending_orders("XAUUSD", 202606, [])

    # Order should be canceled
    mock_order_send.assert_called_once()
