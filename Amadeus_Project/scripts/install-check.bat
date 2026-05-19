@echo off
setlocal EnableExtensions
title Amadeus Install Check
cd /d "%~dp0\.."

echo === Amadeus 环境检查 ===
echo.

set FAIL=0

where node >nul 2>&1
if errorlevel 1 (
  echo [X] Node.js 未安装: https://nodejs.org
  set FAIL=1
) else (
  for /f "delims=" %%v in ('node -v') do echo [OK] Node.js %%v
)

where ollama >nul 2>&1
if errorlevel 1 (
  echo [X] Ollama 未安装: https://ollama.com
  set FAIL=1
) else (
  echo [OK] Ollama 已安装
  ollama list 2>nul | findstr /i "kurisu" >nul
  if errorlevel 1 (
    echo [!] 未检测到 kurisu 模型，请按 INSTALL.md 准备对话模型
  ) else (
    echo [OK] 检测到 kurisu 相关模型
  )
)

if exist "node_modules\express\package.json" (
  echo [OK] npm 依赖已安装
) else (
  echo [!] 依赖未安装，请运行: npm install
)

echo.
echo 若后端已启动，可访问: http://localhost:3000/health
echo 详细说明见 INSTALL.md
echo.

if %FAIL%==1 (
  echo 检查结果: 有必需项缺失
  pause
  exit /b 1
)

echo 检查结果: 基本环境就绪
pause
exit /b 0
