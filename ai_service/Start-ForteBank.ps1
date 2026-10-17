[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$LauncherArguments
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

$candidates = @()
foreach ($environmentName in @('.venv-portable', '.venv')) {
    $localPython = Join-Path $PSScriptRoot "$environmentName\Scripts\python.exe"
    if (Test-Path -LiteralPath $localPython -PathType Leaf) {
        $candidates += [pscustomobject]@{ Program = $localPython; Prefix = @() }
    }
}
$pythonLauncher = Get-Command py -ErrorAction SilentlyContinue
if ($pythonLauncher) {
    $candidates += [pscustomobject]@{ Program = $pythonLauncher.Source; Prefix = @('-3.12') }
    $candidates += [pscustomobject]@{ Program = $pythonLauncher.Source; Prefix = @('-3') }
}
foreach ($commandName in @('python3', 'python')) {
    $pythonCommand = Get-Command $commandName -ErrorAction SilentlyContinue
    if ($pythonCommand -and $pythonCommand.Source -notlike '*\WindowsApps\*') {
        $candidates += [pscustomobject]@{ Program = $pythonCommand.Source; Prefix = @() }
    }
}

$selectedPython = $null
foreach ($candidate in $candidates) {
    try {
        & $candidate.Program @($candidate.Prefix) -c 'import sys; raise SystemExit(0 if sys.maxsize > 2**32 and sys.version_info >= (3, 10) else 1)' 2>$null
        if ($LASTEXITCODE -eq 0) {
            $selectedPython = $candidate
            break
        }
    } catch {
        # Continue with another installed interpreter.
    }
}

if (-not $selectedPython) {
    Write-Host 'Python 3.10+ (64 bit) is required. Python 3.12 is recommended.'
    Write-Host 'Install it from https://www.python.org/downloads/windows/ and enable Add Python to PATH.'
    Write-Host 'Then run Start-ForteBank.cmd again.'
    exit 1
}

& $selectedPython.Program @($selectedPython.Prefix) (Join-Path $PSScriptRoot 'Start-ForteBank.py') @LauncherArguments
exit $LASTEXITCODE
