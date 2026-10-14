param(
    [switch]$Remove
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$launcherPath = Join-Path $projectRoot 'jarvis_wakeword.py'
$pythonwPath = Join-Path $projectRoot '.venv\Scripts\pythonw.exe'
$startupFolder = [Environment]::GetFolderPath('Startup')
$shortcutPath = Join-Path $startupFolder 'Jarvis Wake Listener.lnk'

if ($Remove) {
    if (Test-Path -LiteralPath $shortcutPath) {
        Remove-Item -LiteralPath $shortcutPath
        Write-Output "Removed Jarvis startup shortcut: $shortcutPath"
    } else {
        Write-Output 'Jarvis startup shortcut was not installed.'
    }
    exit 0
}

if (-not (Test-Path -LiteralPath $launcherPath)) {
    throw "Wake listener not found: $launcherPath"
}
if (-not (Test-Path -LiteralPath $pythonwPath)) {
    throw "Virtual-environment pythonw.exe not found: $pythonwPath. Install the project dependencies first."
}
if (-not (Test-Path -LiteralPath $startupFolder)) {
    throw "Windows Startup folder is unavailable: $startupFolder"
}

$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $pythonwPath
$shortcut.Arguments = "`"$launcherPath`""
$shortcut.WorkingDirectory = $projectRoot
$shortcut.Description = 'Wait for the Jarvis wake hotkey and voice command.'
$shortcut.WindowStyle = 7
$shortcut.Save()
Write-Output "Installed Jarvis wake listener to run at sign-in: $shortcutPath"
Write-Output "Press F9 after sign-in, then say 'Hey Jarvis'."
