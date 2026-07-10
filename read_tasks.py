import os

tasks_dir = r"C:\Users\WINDOWS 11 PRO\.gemini\antigravity-cli\brain\47b9acea-1f4b-4c2c-8ad0-0ec9fd8added\.system_generated\tasks"

print("Tasks Dir:", tasks_dir)
if os.path.exists(tasks_dir):
    files = os.listdir(tasks_dir)
    print("Files in tasks:", files)
    matching = [f for f in files if "778" in f]
    if matching:
        print("Content of:", matching[0])
        with open(os.path.join(tasks_dir, matching[0]), "r", encoding="utf-8") as f:
            print(f.read())
else:
    print("Directory does not exist")
