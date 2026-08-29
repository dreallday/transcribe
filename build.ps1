# Windows build. Run in PowerShell from a copy of this repo on a Windows drive:
#   powershell -ExecutionPolicy Bypass -File build.ps1
$ErrorActionPreference = "Stop"
if (-not (Test-Path .venv)) { python -m venv .venv }
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\pip install -r requirements.txt pywebview nicegui pyinstaller
.\.venv\Scripts\pyinstaller --noconfirm --clean morbo.spec
Write-Output "built: $(Resolve-Path .\dist\morbo\morbo.exe)"
