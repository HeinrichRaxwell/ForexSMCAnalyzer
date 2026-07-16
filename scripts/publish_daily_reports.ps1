param(
    [string]$Python = ".\.venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
& $Python scripts\update_public_reports.py

git add reports requirements.txt scripts\update_public_reports.py scripts\publish_daily_reports.ps1 README.md
if (git diff --cached --quiet) {
    Write-Host "No public report changes to publish."
    exit 0
}

$stamp = Get-Date -Format "yyyy-MM-dd"
git commit -m "docs: update public forward report $stamp"
git push origin main
