@echo off
title Amadeus Server Control
cd /d "%~dp0"
if not exist "server.js" (
  echo [!] 找不到 server.js，请确认你在 Amadeus_Project 目录内运行本脚本。
  pause
  exit /b 1
)
:: 8GB 显存友好（可被用户环境变量覆盖）
set AMADEUS_OLLAMA_NUM_CTX=2048
set AMADEUS_MAX_PROMPT_CHARS=6000
set AMADEUS_OLLAMA_KEEP_ALIVE=2m
:: RAG 检索超时（毫秒，越小越早放弃嵌入、加快首包；可删此行用服务端默认 900）
set AMADEUS_RAG_MS=800
echo 正在接入实验室服务器...
echo [提示] Ollama 若跑 CPU：请在 ollama 进程上设置 OLLAMA_NUM_GPU=999，见 docs\GPU_OLLAMA_SOVITS.md
npm run dev
pause