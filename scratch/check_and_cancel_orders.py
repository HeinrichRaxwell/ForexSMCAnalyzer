"""Check and cancel ALL pending orders from our bot on MT5."""
import os, sys
sys.path.append("C:\\Users\\WINDOWS 11 PRO\\forex-smc-analyzer")
os.chdir("C:\\Users\\WINDOWS 11 PRO\\forex-smc-analyzer")

from dotenv import load_dotenv
load_dotenv()

import MetaTrader5 as mt5
from src.data_loader import connect_mt5
from src.execution import get_active_broker_symbol

def main():
    if not connect_mt5():
        print("Failed to connect.")
        return

    symbol = get_active_broker_symbol("XAUUSD")
    magic = int(os.getenv("MT5_MAGIC_NUMBER", "202606"))
    
    tick = mt5.symbol_info_tick(symbol)
    print(f"Current price: {tick.bid:.3f}")
    
    # Check ALL pending orders (not just our magic)
    all_orders = mt5.orders_get(symbol=symbol)
    if all_orders is None or len(all_orders) == 0:
        print("No pending orders found.")
    else:
        print(f"\n=== {len(all_orders)} PENDING ORDERS ===")
        for o in all_orders:
            o_type = {2: "BUY LIMIT", 3: "SELL LIMIT", 4: "BUY STOP", 5: "SELL STOP"}.get(o.type, f"TYPE_{o.type}")
            vol = o.volume_current
            dist = abs(o.price_open - tick.bid)
            print(f"  #{o.ticket} | {o_type} | {vol} lot @ {o.price_open:.3f} | SL: {o.sl:.3f} | TP: {o.tp:.3f} | Dist: {dist:.1f} | Magic: {o.magic}")

    # Check positions
    positions = mt5.positions_get(symbol=symbol)
    if positions and len(positions) > 0:
        print(f"\n=== {len(positions)} OPEN POSITIONS ===")
        for p in positions:
            p_type = "BUY" if p.type == 0 else "SELL"
            print(f"  #{p.ticket} | {p_type} | {p.volume} lot @ {p.price_open:.3f} | SL: {p.sl:.3f} | TP: {p.tp:.3f} | Profit: ${p.profit:.2f} | Magic: {p.magic}")
    else:
        print("\nNo open positions.")
    
    # Cancel ALL our bot's pending orders
    our_orders = mt5.orders_get(symbol=symbol)
    if our_orders:
        cancelled = 0
        for o in our_orders:
            if o.magic == magic:
                req = {"action": mt5.TRADE_ACTION_REMOVE, "order": o.ticket}
                res = mt5.order_send(req)
                if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                    print(f"  [CANCELLED] #{o.ticket}")
                    cancelled += 1
                else:
                    print(f"  [FAILED] #{o.ticket}: {res}")
        print(f"\nCancelled {cancelled} pending orders from our bot.")
    
    mt5.shutdown()

if __name__ == "__main__":
    main()
