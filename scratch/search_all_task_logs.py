import os
import glob

def main():
    tasks_dir = r"C:\Users\WINDOWS 11 PRO\.gemini\antigravity-cli\brain\aade0c14-67d6-4b69-a8a6-5834a430a34c\.system_generated\tasks"
    log_files = glob.glob(os.path.join(tasks_dir, "task-*.log"))
    
    print(f"Found {len(log_files)} log files. Searching for sent alerts...")
    
    search_terms = ["SMC SIGNAL", "HIGH CONFIDENCE", "PLACED", "FAILED", "Option A", "Option B", "BPR"]
    
    alerts_found = []
    for log_path in log_files:
        with open(log_path, 'r', errors='ignore') as f:
            content = f.read()
            if any(term in content for term in search_terms):
                # Print lines containing terms
                f.seek(0)
                for line_no, line in enumerate(f):
                    if "SMC SIGNAL" in line or "PLACED" in line or "Option" in line or "BPR" in line:
                        alerts_found.append((os.path.basename(log_path), line_no+1, line.strip()))
                        
    print(f"\nFound {len(alerts_found)} matching log lines:")
    for file, line_no, text in alerts_found[:30]:
        print(f"File: {file} | Line: {line_no} | {text}")

if __name__ == "__main__":
    main()
