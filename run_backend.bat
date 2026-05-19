@echo off
rem -------------------------------------------------
rem  Amadeus 后端启动脚本
rem  双击此文件即可在控制台中看到日志并保持窗口打开
rem -------------------------------------------------

rem 1️⃣ 切换到项目根目录
cd /d "D:\Amadeus_Trae\Amadeus_Project"

rem 2️⃣ 检查 Node 是否可用
where node >nul 2>&1
if errorlevel 1 (
  echo [!] 未检测到 Node.js，请先安装 Node.js (https://nodejs.org)
  pause
  exit /b 1
)

rem 3️⃣ 安装依赖（第一次运行可以保留，后续可以注释掉）
echo 正在检查 / 安装 npm 依赖…
npm install --silent
if errorlevel 1 (
  echo [!] npm install 失败，请检查网络或权限。
  pause
  exit /b 1
)

rem 4️⃣ 启动后端
rem 使用 cmd /K 让子进程结束后窗口仍然保留
echo 正在启动 Amadeus 后端 (http://localhost:3000) …
cmd /K npm run dev

rem 当子窗口被手动关闭后，下面的 pause 会让主窗口保持打开
pause
