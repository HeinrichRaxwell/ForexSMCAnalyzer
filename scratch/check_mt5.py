import MetaTrader5 as mt5
import os
import sys

# Connect to MT5
if not mt5.initialize():
    print("MT5 initialize failed")
    sys.exit(1)

print("MT5 initialized successfully")
account_info = mt5.account_info()
if account_info is not None:
    print(f"Account: {account_info.login}, Company: {account_info.company}, Balance: {account_info.balance}")
else:
    print("Failed to get account info")

# Find the broker symbol for Gold
symbols_to_try = ["XAUUSD", "XAUUSDm", "XAUUSD.", "GOLD"]
active_sym = None
for sym in symbols_to_try:
    if mt5.symbol_select(sym, True):
        info = mt5.symbol_info(sym)
        if info is not None:
            print(f"Symbol found: {sym}, Bid: {info.bid}, Ask: {info.ask}")
            active_sym = sym
            break

if active_sym:
    orders = mt5.orders_get(symbol=active_sym)
    if orders is not None and len(orders) > 0:
        print(f"Active pending orders for {active_sym}:")
        for o in orders:
            d = o._asdict()
            print(f"  Ticket: {d.get('ticket')}, Type: {d.get('type')}, Price: {d.get('price_open')}, Volume: {d.get('volume_initial')}, SL: {d.get('sl')}, TP: {d.get('tp')}, Comment: {d.get('comment')}")
    else:
        print(f"No active pending orders for {active_sym}")
        
    positions = mt5.positions_get(symbol=active_sym)
    if positions is not None and len(positions) > 0:
        print(f"Active positions for {active_sym}:")
        for p in positions:
            d = p._asdict()
            print(f"  Ticket: {d.get('ticket')}, Type: {d.get('type')}, Price: {d.get('price_open')}, Volume: {d.get('volume')}, Profit: {d.get('profit')}")
    else:
        print(f"No active positions for {active_sym}")
else:
    print("No Gold symbol found")

mt5.shutdown()
