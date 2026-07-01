import os
import json
import MetaTrader5 as mt5

def main():
    sent_signals_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sent_signals.json")
    
    if not os.path.exists(sent_signals_file):
        print("sent_signals.json not found!")
        return
        
    with open(sent_signals_file, "r") as f:
        sent_signals = json.load(f)
        
    if not mt5.initialize():
        print("MT5 initialize failed")
        return
        
    orders = mt5.orders_get()
    if not orders:
        print("No pending orders found on MT5.")
        mt5.shutdown()
        return
        
    print(f"Total active pending orders on MT5: {len(orders)}")
    print("=" * 100)
    
    # Let's map ticket numbers to sent_signals entries
    ticket_map = {}
    for key, signal in sent_signals.items():
        # A signal could have ticket_a, ticket_b, or ticket_id
        if 'ticket_a' in signal and signal['ticket_a'] is not None:
            ticket_map[signal['ticket_a']] = (key, 'Option A (0.5)', signal)
        if 'ticket_b' in signal and signal['ticket_b'] is not None:
            ticket_map[signal['ticket_b']] = (key, 'Option B (0.618)', signal)
        if 'ticket_id' in signal and signal['ticket_id'] is not None:
            ticket_map[signal['ticket_id']] = (key, 'Single', signal)
            
    matched_count = 0
    unmatched = []
    
    for o in orders:
        ticket = o.ticket
        price = o.price_open
        sl = o.sl
        tp = o.tp
        comment = o.comment
        symbol = o.symbol
        order_type = "BUY_LIMIT" if o.type == 2 else "SELL_LIMIT" if o.type == 3 else str(o.type)
        
        if ticket in ticket_map:
            key, option_name, sig = ticket_map[ticket]
            matched_count += 1
            print(f"Ticket: #{ticket} | Symbol: {symbol} | Type: {order_type} | Vol: {o.volume_initial} | Comment: {comment}")
            print(f"  -> Match Key: {key}")
            print(f"  -> Timeframe: {sig.get('timeframe')} | Setup Type: {sig.get('type')} | Direction: {sig.get('direction')}")
            print(f"  -> Placed Time: {sig.get('time_sent')}")
            
            # Print features / confluences if stored
            if 'features_0.5' in sig or 'features_0.618' in sig or 'features' in sig:
                feat = sig.get('features_0.5') or sig.get('features_0.618') or sig.get('features', {})
                print(f"  -> AI Prob: {sig.get('probability_0.5', 0)*100:.1f}% (0.5), {sig.get('probability_0.618', 0)*100:.1f}% (0.618)")
                print(f"  -> Key Features: Killzone={feat.get('killzone')}, NearPsych={feat.get('near_psychological_level')}, FloopSig={feat.get('floop_signal')}")
            
            # Let's see if we can construct confluences/reasons from the signal details
            # Or if they are described in the key
            print("-" * 100)
        else:
            unmatched.append(o)
            
    if unmatched:
        print(f"\nUnmatched Pending Orders ({len(unmatched)}):")
        for o in unmatched:
            order_type = "BUY_LIMIT" if o.type == 2 else "SELL_LIMIT" if o.type == 3 else str(o.type)
            print(f"  Ticket: #{o.ticket} | Symbol: {o.symbol} | Type: {order_type} | Price: {o.price_open} | Comment: '{o.comment}'")
            
    mt5.shutdown()

if __name__ == "__main__":
    main()
