import json
import os

file_path = "C:\\Users\\WINDOWS 11 PRO\\forex-smc-analyzer\\data\\sent_signals.json"

if os.path.exists(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
        
    original_count = len(data)
    # Remove keys sent during the failed run (after 20:10:00 today)
    keys_to_remove = []
    for k, v in data.items():
        time_sent = v.get("time_sent", "")
        if time_sent.startswith("2026-06-04 20:10"):
            keys_to_remove.append(k)
            
    for k in keys_to_remove:
        del data[k]
        
    print(f"Removed {len(keys_to_remove)} keys from {original_count} total keys.")
    
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
else:
    print("No sent_signals.json found.")
