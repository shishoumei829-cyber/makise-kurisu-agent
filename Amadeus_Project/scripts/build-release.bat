@echo off
setlocal EnableExtensions
title Amadeus Release Build
cd /d "%~dp0\.."

if not exist "package.json" (
  echo [X] 请在 Amadeus_Project 目录运行。
  pause
  exit /b 1
)

echo [.] 运行测试...
call npm test
if errorlevel 1 (
  echo [X] 测试未通过，已中止打包。
  pause
  exit /b 1
)

echo [.] 开始构建 Windows 安装包...
set CSC_IDENTITY_AUTO_DISCOVERY=false
call npm run release:win
if errorlevel 1 (
  echo [X] 打包失败。
  pause
  exit /b 1
)

echo.
echo [OK] 安装包已生成: dist\Amadeus Setup *.exe
echo [.] 请在另一台 Windows 机器试装并验证 /health
pause
endlocal
