# Removes the OpenJarvis application checkout, its user config, and old installer
# scripts. It intentionally keeps shared tools (Git, uv, Python, Rust) and the
# original C:\Users\Mithilesh\Jarvis project untouched.
$ErrorActionPreference = 'Stop'

$targets = @(
    (Join-Path $env:LOCALAPPDATA 'OpenJarvis'),
    (Join-Path $env:USERPROFILE '.openjarvis'),
    (Join-Path $env:TEMP 'openjarvis-install.ps1'),
    (Join-Path $env:TEMP 'openjarvis-install-fixed.ps1'),
    (Join-Path $env:TEMP 'openjarvis-install.original.ps1')
)

Write-Host 'The following OpenJarvis-specific paths will be removed if present:' -ForegroundColor Yellow
$targets | ForEach-Object { Write-Host "  $_" }
Write-Host ''
Write-Host 'Shared utilities and your original Jarvis project/model files will not be touched.'
$confirmation = Read-Host 'Type REMOVE OPENJARVIS to continue'
if ($confirmation -cne 'REMOVE OPENJARVIS') {
    Write-Host 'Cancelled. No files were removed.'
    exit 0
}

foreach ($target in $targets) {
    if (Test-Path -LiteralPath $target) {
        Remove-Item -LiteralPath $target -Recurse -Force
        Write-Host "Removed: $target" -ForegroundColor Green
    } else {
        Write-Host "Not found: $target"
    }
}
Write-Host 'OpenJarvis cleanup completed.' -ForegroundColor Green
