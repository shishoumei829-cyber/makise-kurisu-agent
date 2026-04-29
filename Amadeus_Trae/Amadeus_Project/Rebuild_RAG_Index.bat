@echo off
cd /d "%~dp0"
echo [RAG] Rebuilding vector index...
node ingest.js
echo.
if %ERRORLEVEL% EQU 0 (
  echo [RAG] Done. If server is running, restart server.js to load new index.
) else (
  echo [RAG] Failed. Please check error logs above.
)
echo.
pause
