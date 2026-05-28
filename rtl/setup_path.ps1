# Add Quartus 18.1 + ModelSim 18.1 bin folders to the *User* PATH.
# Idempotent: existing entries are not duplicated. Run from any shell.
# Open a fresh PowerShell after this to pick up the changes.

$ErrorActionPreference = 'Stop'

$adds = @(
    'E:\Quart\quartus\bin64',
    'E:\Quart\modelsim_ase\win32aloem'
)

$cur = [Environment]::GetEnvironmentVariable('Path', 'User')
if ($null -eq $cur) { $cur = '' }
$parts = @($cur.Split(';') | Where-Object { $_ -ne '' })

$added = @()
foreach ($p in $adds) {
    if ($parts -notcontains $p) {
        $parts += $p
        $added += $p
    }
}

$new = ($parts -join ';')
[Environment]::SetEnvironmentVariable('Path', $new, 'User')

Write-Host '--- Added this run ---'
if ($added.Count -eq 0) {
    Write-Host '  (nothing new; both paths were already on PATH)'
} else {
    $added | ForEach-Object { Write-Host "  + $_" }
}

Write-Host ''
Write-Host '--- Current User PATH ---'
$parts | ForEach-Object { Write-Host "  $_" }
