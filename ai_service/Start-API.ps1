param([int]$Port = 8001)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not $env:FORTE_API_KEY) {
    $keyPath = Join-Path $PSScriptRoot 'data\runtime\api-key.txt'
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $keyPath) | Out-Null
    if (-not (Test-Path -LiteralPath $keyPath)) {
        $generatedKey = & '.\.venv\Scripts\python.exe' -c 'import secrets; print(secrets.token_urlsafe(32))'
        if ($LASTEXITCODE -ne 0) { throw 'Could not create local API key.' }
        Set-Content -LiteralPath $keyPath -Value $generatedKey -Encoding Ascii -NoNewline
    }
    $env:FORTE_API_KEY = (Get-Content -LiteralPath $keyPath -Raw).Trim()
    Write-Host "API key is stored locally: $keyPath"
}
$pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$configPath = Join-Path $PSScriptRoot 'config\local.json'
Push-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
try {
    & $pythonPath -m backend serve --config $configPath --port $Port
    $backendExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $backendExitCode
