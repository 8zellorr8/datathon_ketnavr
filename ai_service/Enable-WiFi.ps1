#Requires -RunAsAdministrator
$ErrorActionPreference = 'Stop'
$interfaces = @(Get-NetIPConfiguration | Where-Object { $_.IPv4DefaultGateway -and $_.IPv4Address -and $_.NetAdapter.Status -eq 'Up' })
foreach ($network in $interfaces) {
    $address = $network.IPv4Address.IPAddress
    if ($address -notmatch '^(10\.|192\.168\.|172\.(1[6-9]|2[0-9]|3[01])\.)') { continue }
    $ruleName = 'ForteBank WiFi 8001 - ' + $network.InterfaceIndex
    if (-not (Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue)) {
        New-NetFirewallRule -DisplayName $ruleName -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8001 -LocalAddress $address -RemoteAddress LocalSubnet -InterfaceAlias $network.InterfaceAlias -Profile Any | Out-Null
    }
    Write-Host ('Allowed local subnet only: http://' + $address + ':8001')
}
