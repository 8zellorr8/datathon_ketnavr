param(
    [string]$Workbook = 'data/labels_full.xlsx',
    [string]$VideosDir = 'data/videos'
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath $Workbook -PathType Leaf)) {
    throw 'Workbook not found. Use the full handoff archive or pass -Workbook.'
}
if (-not (Test-Path -LiteralPath $VideosDir -PathType Container)) {
    throw 'Video directory not found. Use the full handoff archive or pass -VideosDir.'
}
& '.\.venv\Scripts\python.exe' -m forte_ai prepare --workbook $Workbook --videos-dir $VideosDir --output data
if ($LASTEXITCODE -ne 0) { throw 'Dataset preparation failed' }
Write-Host 'Dataset paths and grouped partitions have been prepared for this computer.'
