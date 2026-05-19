@echo off
setlocal EnableExtensions
title Amadeus one-click

cd /d "%~dp0"
if not exist "package.json" (
  echo [X] 请在本项目目录运行一键启动。
  pause
  exit /b 1
)

where node >nul 2>&1
if errorlevel 1 (
  echo [X] 未检测到 Node.js。
  pause
  exit /b 1
)

if not exist "node_modules\express\package.json" (
  echo [.] 依赖未安装，先运行 run_backend.bat 完成首次安装。
  call "%~dp0run_backend.bat"
  exit /b %ERRORLEVEL%
)

start "Amadeus-Backend" /D "%CD%" cmd /k "%~dp0run_backend.bat"

timeout /t 3 /nobreak >nul 2>&1
if errorlevel 1 ping -n 4 127.0.0.1 >nul

start "" "http://localhost:3000/amadeus_work.html"

echo.
echo 已打开浏览器。后端日志见窗口 Amadeus-Backend。
echo.
pause
endlocal
