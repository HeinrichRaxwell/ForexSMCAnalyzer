import MetaTrader5 as mt5

def main():
    if not mt5.initialize():
        print("Failed to initialize MT5")
        return
        
    symbol = "XAUUSDm"
    mt5.symbol_select(symbol, True)
    tick = mt5.symbol_info_tick(symbol)
    if tick is not None:
        print(f"Current Tick for {symbol}:")
        print(f"  Bid: {tick.bid:.3f}")
        print(f"  Ask: {tick.ask:.3f}")
        print(f"  Last: {tick.last:.3f}")
    else:
        print("Failed to get tick info")
        
    mt5.shutdown()

if __name__ == "__main__":
    main()
