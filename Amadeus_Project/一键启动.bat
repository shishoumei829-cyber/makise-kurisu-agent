@echo off
setlocal EnableExtensions
title Amadeus one-click

set "AMADEUS_ROOT=D:\Amadeus_Trae\Amadeus_Project"
cd /d "%AMADEUS_ROOT%"
if not exist "package.json" (
  echo [X] Amadeus project folder is missing.
  pause
  exit /b 1
)

where node >nul 2>&1
if errorlevel 1 (
  echo [X] Node.js was not found in PATH.
  pause
  exit /b 1
)

set "AMADEUS_PORT=3000"
if exist ".env" (
  for /f "usebackq tokens=1,* delims==" %%A in (`findstr /b /i "AMADEUS_BACKEND_PORT=" ".env" 2^>nul`) do set "AMADEUS_PORT=%%B"
)

if not exist "node_modules\express\package.json" (
  echo [.] Dependencies are missing. Running first-time backend setup...
  call "%AMADEUS_ROOT%\run_backend.bat"
  exit /b %ERRORLEVEL%
)

if not exist "node_modules\electron\dist\electron.exe" (
  echo [X] Electron is missing. Please run npm install.
  pause
  exit /b 1
)

echo [.] Starting GPT-SoVITS...
powershell -NoProfile -ExecutionPolicy Bypass -File "%AMADEUS_ROOT%\scripts\start-sovits.ps1"
if errorlevel 1 (
  echo [!] GPT-SoVITS failed to start. Text chat will still be available.
)

echo [.] Starting the desktop client. Electron owns the backend lifecycle...
start "Amadeus" /D "%AMADEUS_ROOT%" "%AMADEUS_ROOT%\node_modules\electron\dist\electron.exe" "."
exit /b 0
endlocal
