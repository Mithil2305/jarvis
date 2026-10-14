@echo off
setlocal
cd /d "%~dp0"

echo =========================================================
echo            Starting Jarvis Telegram Remote Bot
echo =========================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment (.venv) not found!
    echo Please make sure you are in the Jarvis project directory.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" telegram_jarvis.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo Bot stopped or encountered an error.
    pause
)
