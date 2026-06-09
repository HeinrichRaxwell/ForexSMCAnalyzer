import json
import sys

def find_phase3():
    sys.stdout.reconfigure(encoding='utf-8')
    transcript_path = r"C:\Users\WINDOWS 11 PRO\.gemini\antigravity-cli\brain\aade0c14-67d6-4b69-a8a6-5834a430a34c\.system_generated\logs\transcript.jsonl"
    
    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            content = data.get("content", "")
            if "Kelanjutan Proyek (Fase 3: Telegram Alert Bot)" in content or "Telegram Alert Bot" in content:
                print(f"=== Found in Step {data.get('step_index')} ({data.get('type')}) ===")
                print(content)
                print("="*60)
                break # Just print the first one

if __name__ == "__main__":
    find_phase3()
