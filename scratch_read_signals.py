import json
import os

filepath = os.path.join("data", "sent_signals.json")
if not os.path.exists(filepath):
    print("sent_signals.json not found")
    exit()

try:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    print("Total keys in sent_signals:", len(data))
    
    count = 0
    # Let's search all keys for any failed execution attempts or watch status
    for k, item in reversed(list(data.items())):
        # print non-null watch_last_execution_message or execution_message or retries > 0
        msg_a = item.get("watch_last_execution_message_0.5") or item.get("execution_message_0.5")
        msg_b = item.get("watch_last_execution_message_0.618") or item.get("execution_message_0.618")
        retries_a = item.get("execution_retries_0.5")
        retries_b = item.get("execution_retries_0.618")
        
        has_msg = msg_a or msg_b
        has_retries = (retries_a is not None and retries_a > 0) or (retries_b is not None and retries_b > 0)
        
        if has_msg or has_retries or item.get("watch_status"):
            print("="*60)
            print(f"Key: {k}")
            print(f"Timeframe: {item.get('timeframe')} | Strategy: {item.get('strategy')}")
            print(f"Ticket A: {item.get('ticket_a')} | Ticket B: {item.get('ticket_b')}")
            print(f"Message A: {msg_a}")
            print(f"Message B: {msg_b}")
            print(f"Retries A: {retries_a} | Retries B: {retries_b}")
            print(f"Watch Status: {item.get('watch_status')} | Watch Reason: {item.get('watch_reason')}")
            count += 1
            if count >= 10:
                break
except Exception as e:
    print("Error:", e)
