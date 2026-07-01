import os
import sys
import pandas as pd
from datetime import datetime, timedelta
import MetaTrader5 as mt5

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import connect_mt5

def check_history():
    print("Connecting to MT5...")
    if not connect_mt5():
        print("Failed to connect.")
        return
        
    # Get history from last 24 hours
    from_date = datetime.now() - timedelta(days=2)
    to_date = datetime.now() + timedelta(days=1)
    
    print(f"Fetching deals from {from_date.strftime('%Y-%m-%d %H:%M:%S')} to {to_date.strftime('%Y-%m-%d %H:%M:%S')}...")
    deals = mt5.history_deals_get(from_date, to_date)
    
    if deals is None:
        print("No deals found or failed to fetch. Error:", mt5.last_error())
        mt5.shutdown()
        return
        
    print(f"Total deals found: {len(deals)}")
    
    if len(deals) > 0:
        df_deals = pd.DataFrame(list(deals), columns=deals[0]._asdict().keys())
        df_deals['time'] = pd.to_datetime(df_deals['time'], unit='s')
        
        # Display relevant columns
        cols = ['ticket', 'time', 'symbol', 'type', 'entry', 'volume', 'price', 'profit', 'comment', 'magic']
        # Filter columns that exist
        available_cols = [c for c in cols if c in df_deals.columns]
        
        # Print deals sorted by time
        df_sorted = df_deals.sort_values(by='time', ascending=False)
        print("\n--- RECENT TRADES HISTORY ---")
        for idx, row in df_sorted.head(30).iterrows():
            # type: 0 = Buy, 1 = Sell
            trade_type = "BUY" if row['type'] == 0 else ("SELL" if row['type'] == 1 else f"TYPE_{row['type']}")
            entry_type = "IN" if row.get('entry') == 0 else ("OUT" if row.get('entry') == 1 else "")
            
            print(f"Ticket: {row['ticket']} | Time: {row['time']} | Symbol: {row['symbol']} | {trade_type} {entry_type} | Vol: {row['volume']} | Price: {row['price']:.3f} | Profit: ${row['profit']:.2f} | Magic: {row.get('magic')} | Comment: {row.get('comment')}")
            
        print("\nSummary:")
        total_profit = df_deals['profit'].sum()
        print(f"Total Net Profit/Loss: ${total_profit:.2f}")
    else:
        print("History is empty.")
        
    mt5.shutdown()

if __name__ == "__main__":
    check_history()
