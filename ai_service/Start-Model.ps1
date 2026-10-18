param([int]$Port = 11435)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if ($Port -lt 1024 -or $Port -gt 65535) { throw 'Invalid localhost port' }
$env:OLLAMA_NO_CLOUD = '1'
$env:OLLAMA_HOST = "127.0.0.1:$Port"
$env:OLLAMA_NUM_PARALLEL = '1'
$env:OLLAMA_MAX_LOADED_MODELS = '1'
$env:OLLAMA_CONTEXT_LENGTH = '8192'
$env:OLLAMA_FLASH_ATTENTION = '1'
$env:OLLAMA_KV_CACHE_TYPE = 'q8_0'
$taskCacheMiB = 128
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (Test-Path -LiteralPath $taskPython -PathType Leaf) {
    $taskDetectedCache = & $taskPython -c 'from portable_hardware import detect_hardware,select_profile; print(select_profile(detect_hardware())["cache_ram_mib"])'
    if ($LASTEXITCODE -eq 0 -and "$taskDetectedCache" -match '^(128|512)$') {
        $taskCacheMiB = [int]$taskDetectedCache
    }
}
$env:LLAMA_ARG_CACHE_RAM = "$taskCacheMiB"
$taskOllamaCommand = Get-Command ollama -ErrorAction SilentlyContinue
$taskOllama = if ($taskOllamaCommand) { $taskOllamaCommand.Source } else { Join-Path $env:LOCALAPPDATA 'Programs\Ollama\ollama.exe' }
if (-not (Test-Path -LiteralPath $taskOllama -PathType Leaf)) { throw 'Install Ollama first' }
Write-Host "Starting local Ollama on 127.0.0.1:$Port with cloud disabled; prompt RAM cache $taskCacheMiB MiB. Keep this terminal open."
& $taskOllama serve
