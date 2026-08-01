@echo off
setlocal EnableExtensions
title Amadeus Pure Window

set "AMADEUS_ROOT=%~dp0"
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

if not exist "node_modules\electron\dist\electron.exe" (
  echo [X] Electron is missing. Please run npm install.
  pause
  exit /b 1
)

set "AMADEUS_UI_MODE=pure"
start "" /D "%AMADEUS_ROOT%" "%AMADEUS_ROOT%\node_modules\electron\dist\electron.exe" . --ui=pure
exit /b 0
