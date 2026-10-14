@echo off
setlocal
cd /d "%~dp0"

set "SERVER_EXE=%~dp0llama.cpp\llama-server.exe"
set "MODEL_PATH=%~dp0models\gemma-4-E2B-it-Q8_0.gguf"
set "MMPROJ_PATH=%~dp0models\mmproj-gemma-4-E2B-it-Q8_0.gguf"
set "HOST=127.0.0.1"
set "PORT=8080"
set "URL=http://%HOST%:%PORT%"
set "CHROME_EXE=C:\Program Files\Google\Chrome\Application\chrome.exe"

if not exist "%SERVER_EXE%" (
  echo [ERROR] llama-server.exe was not found: "%SERVER_EXE%"
  if not defined JARVIS_NO_PAUSE pause
  exit /b 1
)
if not exist "%MODEL_PATH%" (
  echo [ERROR] Model file was not found: "%MODEL_PATH%"
  if not defined JARVIS_NO_PAUSE pause
  exit /b 1
)
if not exist "%MMPROJ_PATH%" (
  echo [ERROR] mmproj file was not found: "%MMPROJ_PATH%"
  if not defined JARVIS_NO_PAUSE pause
  exit /b 1
)

echo Starting local Gemma 4 llama-server on %URL% ...
if /I not "%JARVIS_OPEN_MODEL_BROWSER%"=="false" echo Chrome will open %URL% automatically once the model is loaded.
echo.

if /I not "%JARVIS_OPEN_MODEL_BROWSER%"=="false" (
  start "" /b powershell -NoProfile -Command "for ($i = 0; $i -lt 90; $i++) { try { $r = Invoke-WebRequest -Uri '%URL%/health' -UseBasicParsing -TimeoutSec 1 -ErrorAction Stop; if ($r.StatusCode -eq 200) { break } } catch {}; Start-Sleep -Seconds 1 }; if (Test-Path '%CHROME_EXE%') { Start-Process '%CHROME_EXE%' '%URL%' } else { Start-Process 'chrome.exe' '%URL%' }"
)

"%SERVER_EXE%" ^
  -m "%MODEL_PATH%" ^
  --mmproj "%MMPROJ_PATH%" ^
  --host %HOST% ^
  --port %PORT% ^
  --ctx-size 4096 ^
  --n-gpu-layers 0 ^
  --reasoning off

if not defined JARVIS_NO_PAUSE pause
