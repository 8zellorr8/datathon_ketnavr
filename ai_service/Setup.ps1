param([string]$PythonCommand = 'python')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.\.venv\Scripts\python.exe')) {
    & $PythonCommand -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python environment creation failed' }
}
& '.\.venv\Scripts\python.exe' -c 'import sys; raise SystemExit(0 if sys.maxsize > 2**32 and sys.version_info >= (3,10) else 1)'
if ($LASTEXITCODE -ne 0) { throw 'Use a 64-bit Python 3.10+ environment (Python 3.12 recommended).' }
$backendRequirements = Join-Path (Split-Path -Parent $PSScriptRoot) 'backend\requirements.txt'
& '.\.venv\Scripts\python.exe' -m pip install -r $backendRequirements
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
& '.\.venv\Scripts\python.exe' -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw 'Local tests failed' }
Write-Host 'Python setup complete. Next: ollama pull qwen3-vl:2b-instruct-q4_K_M'
