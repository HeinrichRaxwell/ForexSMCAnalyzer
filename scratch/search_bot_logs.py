import os

def main():
    log_path = r"C:\Users\WINDOWS 11 PRO\.gemini\antigravity-cli\brain\aade0c14-67d6-4b69-a8a6-5834a430a34c\.system_generated\tasks\task-3202.log"
    if not os.path.exists(log_path):
        print(f"Log path does not exist: {log_path}")
        return
        
    print(f"Reading logs from {log_path}...")
    with open(log_path, 'r', errors='ignore') as f:
        lines = f.readlines()
        
    print(f"Total lines: {len(lines)}")
    
    keywords = ["Order", "Placed", "Cancel", "Risk", "zombie", "Signal", "indicator", "rejection", "BPR"]
    print("\n--- RELEVANT LOG ENTRIES ---")
    for idx, line in enumerate(lines):
        line_lower = line.lower()
        if any(kw.lower() in line_lower for kw in keywords):
            print(f"Line {idx+1}: {line.strip()}")

if __name__ == "__main__":
    main()
