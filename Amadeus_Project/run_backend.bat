@echo off
setlocal EnableExtensions
title Amadeus Backend

cd /d "%~dp0"
if not exist "package.json" (
  echo [X] 请在 Amadeus_Project 目录运行本脚本。
  pause
  exit /b 1
)

where node >nul 2>&1
if errorlevel 1 (
  echo [X] 未检测到 Node.js: https://nodejs.org
  pause
  exit /b 1
)

if not defined NPM_CONFIG_CACHE set "NPM_CONFIG_CACHE=D:\npm-cache"
if not exist "%NPM_CONFIG_CACHE%" mkdir "%NPM_CONFIG_CACHE%" 2>nul

set "AMADEUS_PORT=3000"
if exist ".env" (
  echo [OK] 已检测到 .env，将使用其中的配置。
  for /f "usebackq tokens=1,* delims==" %%A in (`findstr /b /i "AMADEUS_BACKEND_PORT=" ".env" 2^>nul`) do set "AMADEUS_PORT=%%B"
) else (
  echo [.] 未找到 .env，将使用默认配置。可复制 env.example 为 .env 后修改。
)

set AMADEUS_CHAT_MINIMAL=1
set AMADEUS_OLLAMA_NUM_CTX=4096
set AMADEUS_MAX_PROMPT_CHARS=4500
set AMADEUS_OLLAMA_KEEP_ALIVE=2m
set AMADEUS_RAG_MS=800

if exist "node_modules\express\package.json" goto :start_server

echo [.] 首次需要安装依赖（仅生产包，不装 Electron）...
call npm install --omit=dev --no-audit --no-fund
if errorlevel 1 (
  echo.
  echo [X] npm 安装失败。请检查磁盘空间与网络后重试。
  pause
  exit /b 1
)

:start_server
echo [OK] 启动 http://localhost:%AMADEUS_PORT%  ^(Ctrl+C 停止^)
echo [.] 启动后可在浏览器打开 /health 查看系统自检状态
node server.js
pause
endlocal
