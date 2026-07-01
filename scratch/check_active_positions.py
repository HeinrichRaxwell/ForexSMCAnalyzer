import os
import sys
import MetaTrader5 as mt5

sys.path.append("C:\\Users\\WINDOWS 11 PRO\\forex-smc-analyzer")
os.chdir("C:\\Users\\WINDOWS 11 PRO\\forex-smc-analyzer")

from src.data_loader import connect_mt5
from src.execution import get_active_broker_symbol

def main():
    if not connect_mt5():
        print("Failed to connect to MT5")
        return
        
    symbol = "XAUUSD"
    symbols_to_try = [symbol, symbol + "m", symbol + ".", "GOLD"]
    active_symbol = symbol
    for sym in symbols_to_try:
        if mt5.symbol_info(sym) is not None:
            active_symbol = sym
            break
            
    # Get active positions
    positions = mt5.positions_get(symbol=active_symbol)
    print(f"Active positions for {active_symbol}: {len(positions) if positions else 0}")
    if positions:
        for p in positions:
            ptype = "BUY" if p.type == 0 else "SELL"
            print(f"  Ticket #{p.ticket} | {ptype} | {p.volume} lot | entry={p.price_open:.3f} | current={p.price_current:.3f} | SL={p.sl:.3f} | TP={p.tp:.3f} | profit={p.profit:,.2f}")
            
    mt5.shutdown()

if __name__ == "__main__":
    main()
