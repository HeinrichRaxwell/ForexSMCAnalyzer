import json
import os

def main():
    transcript_path = r"C:\Users\WINDOWS 11 PRO\.gemini\antigravity-cli\brain\aade0c14-67d6-4b69-a8a6-5834a430a34c\.system_generated\logs\transcript_full.jsonl"
    if not os.path.exists(transcript_path):
        print("Transcript file not found.")
        return
        
    print(f"Reading from: {transcript_path}")
    count = 0
    with open(transcript_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                content = str(data.get('content', ''))
                if "telegram" in content.lower():
                    count += 1
                    # We print matches 1 to 5 to see the original plan from the beginning!
                    if count <= 5:
                        print(f"\n================ Match {count} (Type: {data.get('type')}, Source: {data.get('source')}) ================")
                        print(content)
                        print("==============================================")
            except Exception as e:
                pass

if __name__ == "__main__":
    main()
