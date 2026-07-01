import os
import sys
import MetaTrader5 as mt5

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import connect_mt5

def main():
    if not connect_mt5():
        print("Failed to connect to MT5")
        return
        
    print("\n--- ACTIVE POSITIONS ---")
    positions = mt5.positions_get()
    if positions:
        for p in positions:
            print(f"Ticket: {p.ticket} | Symbol: {p.symbol} | Type: {'BUY' if p.type==0 else 'SELL'} | Vol: {p.volume} | Price Open: {p.price_open} | SL: {p.sl} | TP: {p.tp} | Comment: {p.comment}")
    else:
        print("No active positions.")
        
    print("\n--- PENDING ORDERS ---")
    orders = mt5.orders_get()
    if orders:
        for o in orders:
            order_type_str = "BUY_LIMIT" if o.type==2 else "SELL_LIMIT" if o.type==3 else str(o.type)
            print(f"Ticket: {o.ticket} | Symbol: {o.symbol} | Type: {order_type_str} | Vol: {o.volume} | Price: {o.price_open} | SL: {o.sl} | TP: {o.tp} | Comment: {o.comment}")
    else:
        print("No pending orders.")
        
    mt5.shutdown()

if __name__ == "__main__":
    main()
