import os
import sys
import json
import MetaTrader5 as mt5

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import connect_mt5
from src.execution import get_active_broker_symbol

def main():
    if not connect_mt5():
        print("Failed to connect to MT5")
        return
        
    symbol = get_active_broker_symbol("XAUUSD")
    print(f"Active symbol for Gold: {symbol}")
    
    # Get all pending orders
    orders = mt5.orders_get(symbol=symbol)
    if orders is None or len(orders) == 0:
        print("No pending orders found on the account.")
    else:
        print(f"Found {len(orders)} pending orders. Cancelling all of them...")
        cancelled = 0
        for o in orders:
            request = {
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": o.ticket
            }
            res = mt5.order_send(request)
            if res is not None and res.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"  [CANCELLED] Ticket #{o.ticket} | Price: {o.price_open:.3f} | Comment: '{o.comment}'")
                cancelled += 1
            else:
                print(f"  [FAILED] Ticket #{o.ticket} | Error: {res.comment if res else 'Unknown error'}")
        print(f"Successfully cancelled {cancelled} pending orders.")
        
    # Clear data/sent_signals.json
    sent_signals_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sent_signals.json")
    if os.path.exists(sent_signals_file):
        try:
            with open(sent_signals_file, "w") as f:
                json.dump({}, f, indent=4)
            print("Successfully cleared sent_signals.json registry.")
        except Exception as e:
            print(f"Failed to clear sent_signals.json: {e}")
            
    mt5.shutdown()

if __name__ == "__main__":
    main()
