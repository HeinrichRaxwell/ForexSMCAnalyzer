import os, sys
sys.path.append("C:\\Users\\WINDOWS 11 PRO\\forex-smc-analyzer")
os.chdir("C:\\Users\\WINDOWS 11 PRO\\forex-smc-analyzer")
from dotenv import load_dotenv
load_dotenv()
import MetaTrader5 as mt5
from src.data_loader import connect_mt5
from src.execution import get_active_broker_symbol

connect_mt5()
sym = get_active_broker_symbol("XAUUSD")
tick = mt5.symbol_info_tick(sym)
print(f"Price: {tick.bid:.3f}")
orders = mt5.orders_get(symbol=sym)
print(f"Pending orders: {len(orders) if orders else 0}")
if orders:
    for o in orders:
        otype = "BUY LIMIT" if o.type == 2 else "SELL LIMIT"
        dist = abs(o.price_open - tick.bid)
        print(f"  #{o.ticket} | {otype} | {o.volume_current} lot @ {o.price_open:.3f} | dist={dist:.1f}")
mt5.shutdown()
